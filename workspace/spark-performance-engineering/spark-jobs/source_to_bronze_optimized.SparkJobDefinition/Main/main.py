"""
Spark Job Definition: Seed LEGO Delta tables for the performance workshop.

This job:
1. Generates mixed-format (JSON + Parquet) landing files with LakeGen LegoDataGen.
2. Uses ArcFlow to ingest all generated tables into Bronze Delta tables.
3. Avoids Kafka/Event Hub entirely (file-based ingestion only).
"""

import argparse
import logging
import sys
import time

from pyspark.sql import SparkSession
import notebookutils

from arcflow import Controller
from lakegen.generators.lego import LegoDataGen
from pipeline_config import build_tables, OUTPUT_TYPE_MAP, ALL_TABLES


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events-per-second", type=int, default=15000)
    parser.add_argument("--generator-threads", type=int, default=1)
    parser.add_argument("--buffer-write-seconds", type=float, default=2.0)
    parser.add_argument("--run-for-n-minutes", type=float,
                        help="Stop the pipeline after N minutes (default: None).")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def configure_logging(debug):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("fsspec_wrapper.trident.core").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def create_spark(app_name, debug):
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.databricks.delta.autoCompact.enabled", True)
        .config("spark.microsoft.delta.targetFileSize.adaptive.enabled", True)
        .config("spark.microsoft.delta.autoCompact.onCheckpointOnly.enabled", True)
        .config("spark.microsoft.delta.optimize.fileLevelTarget.enabled", True)
        .config("spark.microsoft.delta.snapshot.driverMode.enabled", True)
        .config("spark.databricks.delta.properties.defaults.enableDeletionVectors", True)
        .config("spark.databricks.delta.optimizeWrite.enabled", True)
        .config("spark.native.enabled", True)
        .config(
            "spark.sql.streaming.stateStore.providerClass",
            "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("INFO" if debug else "ERROR")
    return spark


def main(argv):
    args = parse_args(argv)
    logger = configure_logging(args.debug)
    spark = create_spark("seed-lego-delta-tables", args.debug)

    logger.info("=" * 80)
    logger.info("Starting LakeGen: LegoDataGen")
    logger.info("=" * 80)

    default_workspace_id = notebookutils.runtime.context['currentWorkspaceId']
    default_lakehouse_id = notebookutils.runtime.context['defaultLakehouseId']
    onelake_endpoint = spark.sparkContext._jsc.hadoopConfiguration().get("trident.onelake.endpoint").split('//')[1]
    lakehouse_root_uri=f"abfss://{default_workspace_id}@{onelake_endpoint}/{default_lakehouse_id}"

    data_gen = LegoDataGen(
        target_folder_uri=f"{lakehouse_root_uri}/Files/landing",
        output_type_map=OUTPUT_TYPE_MAP,
        max_events_per_second=args.max_events_per_second,
        concurrenct_threads=args.generator_threads,
        buffer_write_by_seconds=args.buffer_write_seconds,
    )
    data_gen.start(verbose=False, background=True)

    # Wait for initial data to land so schema inference works
    logger.info("Waiting for initial landing data...")
    max_wait = 60
    start_wait = time.time()
    while time.time() - start_wait < max_wait:
        tables = build_tables(spark, landing_uri="Files/landing")
        if len(tables) >= len(ALL_TABLES):
            break
        missing = len(ALL_TABLES) - len(tables)
        logger.info("  %d/%d tables found, waiting for %d more...", len(tables), len(ALL_TABLES), missing)
        time.sleep(5)
    else:
        logger.warning("Timed out waiting for all tables — proceeding with %d/%d", len(tables), len(ALL_TABLES))

    logger.info("=" * 80)
    logger.info("Starting ArcFlow ingestion to Bronze")
    logger.info("=" * 80)

    config = {
        "streaming_enabled": True,
        "checkpoint_uri": "Files/checkpoints",
        "archive_uri": "Files/archive",
        "landing_uri": "Files/landing",
        "trigger_interval": "2 seconds",
        "await_termination": False,
        "autoset_spark_configs": True,
        "job_lock_enabled": True,
        "job_id": "seed_lego_delta_tables",
        "job_lock_timeout_seconds": 60,
        "job_lock_path": f"{lakehouse_root_uri}/Files/job_locks",
    }

    controller = Controller(
        spark=spark,
        config=config,
        table_registry=tables,
    )

    logger.info("Running full pipeline for bronze zone across all LEGO tables")
    controller.run_full_pipeline(zones=["bronze"])

    if args.run_for_n_minutes is not None:
        # Pipeline is non-blocking — wait for the requested duration, then stop gracefully
        run_seconds = args.run_for_n_minutes * 60
        logger.info(f"Pipeline will run for {args.run_for_n_minutes} minutes ({int(run_seconds)}s)")
    
        deadline = time.time() + run_seconds
        try:
            while time.time() < deadline:
                remaining = deadline - time.time()
                time.sleep(min(remaining, 30))
                elapsed = int(time.time() + run_seconds - deadline)
                logger.info(
                    f"  [{elapsed // 60}m {elapsed % 60:02d}s / "
                    f"{int(run_seconds // 60)}m elapsed]"
                )
        except KeyboardInterrupt:
            logger.info("Interrupted — stopping early")
    
        logger.info("Time limit reached — stopping data generator and streams")
        data_gen.stop()
        controller.stop_all()
        logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
