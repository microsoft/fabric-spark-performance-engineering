f"""
End-to-end test for the LEGO seed pipeline.

Generates LEGO data with LakeGen, processes it through ArcFlow into Bronze
Delta tables, and verifies every table was created with all columns intact.

Usage (from ArcFlow .venv):
    python test_lego_pipeline.py
    python test_lego_pipeline.py --debug
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time

from pyspark.sql import SparkSession

from lakegen.generators.lego import LegoDataGen

# Add the SJD Libs directory to the path so we can import pipeline_config
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__),
    "seed_lego_delta_tables.SparkJobDefinition", "Libs"
))
from pipeline_config import build_tables, OUTPUT_TYPE_MAP, ALL_TABLES, JSON_TABLES


def configure_logging(debug: bool) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("test_lego_pipeline")


def create_spark(debug: bool, warehouse_dir: str) -> SparkSession:
    from delta import configure_spark_with_delta_pip

    derby_dir = os.path.join(warehouse_dir, "derby")
    os.makedirs(derby_dir, exist_ok=True)

    builder = (
        SparkSession.builder
        .appName("test-lego-pipeline")
        .master("local[*]")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={derby_dir}")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.autoCompact.enabled", "false")
        .config("spark.ui.enabled", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN" if not debug else "INFO")
    return spark


def generate_data(landing_dir: str, logger: logging.Logger) -> None:
    """Generate a small batch of LEGO data into landing_dir."""
    logger.info("Generating LEGO data into %s", landing_dir)

    gen = LegoDataGen(
        target_folder_uri=landing_dir,
        output_type_map=OUTPUT_TYPE_MAP,
        max_events_per_second=50000,
        concurrenct_threads=1,
        buffer_write_by_seconds=0,
    )
    gen.start(verbose=False, background=True)
    time.sleep(3)
    gen.stop()

    for table in ALL_TABLES:
        table_dir = os.path.join(landing_dir, table)
        if not os.path.isdir(table_dir):
            logger.warning("  ✗ Missing directory: %s", table)
            continue
        files = os.listdir(table_dir)
        logger.info("  ✓ %s: %d file(s)", table, len(files))


def run_pipeline(spark, tables, output_dir, logger):
    """Run ArcFlow Controller to process all tables into bronze delta."""
    from arcflow import Controller

    config = {
        "streaming_enabled": True,
        "checkpoint_uri": os.path.join(output_dir, "checkpoints").replace("\\", "/"),
        "archive_uri": os.path.join(output_dir, "archive").replace("\\", "/"),
        "landing_uri": os.path.join(output_dir, "landing").replace("\\", "/"),
        "trigger_interval": "2 seconds",
        "await_termination": True,
        "autoset_spark_configs": False,
        "job_lock_enabled": False,
    }

    logger.info("Initializing ArcFlow Controller with %d tables", len(tables))
    controller = Controller(spark=spark, config=config, table_registry=tables)

    logger.info("Running bronze pipeline...")
    controller.run_full_pipeline(zones=["bronze"])
    logger.info("Pipeline complete.")


def verify_delta_tables(spark, logger):
    """Check that bronze delta tables exist and have data with full schemas."""
    succeeded = []
    failed = []

    # Minimum expected column counts (entity cols + OrganizationId + GeneratedAt + _processing_timestamp)
    min_cols = {
        # JSON tables: flattened entity columns (no _meta wrapper)
        "web_order": 7,
        "manufacturing_event": 11,
        "production_order": 14,
        "quality_inspection": 5,
        "product_return": 5,
        # Parquet tables: all entity cols + OrganizationId + GeneratedAt + _processing_timestamp
        "web_order_line": 10,
        "customer": 8,
        "inventory_transaction": 11,
        "set_price_history": 8,
        "colors": 10,
        "parts": 6,
        "sets": 8,
        "themes": 5,
        "part_categories": 4,
        "inventories": 5,
        "inventory_parts": 8,
        "inventory_sets": 5,
        "production_line": 8,
        "mold": 12,
    }

    for table in ALL_TABLES:
        try:
            df = spark.table(f"bronze.{table}")
            count = df.count()
            cols = len(df.columns)
            expected = min_cols.get(table, 3)

            if count > 0 and cols >= expected:
                logger.info("  ✓ bronze.%-25s  %6d rows, %3d cols (>=%d)", table, count, cols, expected)
                succeeded.append(table)
            elif count == 0:
                logger.warning("  ✗ bronze.%-25s  0 rows", table)
                failed.append(table)
            else:
                logger.warning("  ✗ bronze.%-25s  %d cols < expected %d", table, cols, expected)
                failed.append(table)
        except Exception as e:
            logger.error("  ✗ bronze.%-25s  %s", table, e)
            failed.append(table)

    return succeeded, failed


def main():
    parser = argparse.ArgumentParser(description="Test LEGO seed pipeline end-to-end")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep-temp", action="store_true", help="Don't delete temp dirs")
    args = parser.parse_args()

    logger = configure_logging(args.debug)
    logger.info("=" * 70)
    logger.info("LEGO Seed Pipeline — E2E Test")
    logger.info("=" * 70)

    base_tmp = tempfile.mkdtemp(prefix="lego_pipeline_test_")
    landing_dir = os.path.join(base_tmp, "landing").replace("\\", "/")
    output_dir = os.path.join(base_tmp, "output").replace("\\", "/")
    warehouse_dir = os.path.join(base_tmp, "warehouse").replace("\\", "/")
    os.makedirs(landing_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warehouse_dir, exist_ok=True)
    logger.info("Temp directory: %s", base_tmp)

    spark = None
    try:
        # Step 1: Generate data
        logger.info("\n--- Step 1: Generate LEGO data ---")
        generate_data(landing_dir, logger)

        # Step 2: Create Spark and build table registry using SJD's pipeline_config
        logger.info("\n--- Step 2: Build table registry (schema inference) ---")
        spark = create_spark(args.debug, warehouse_dir)
        tables = build_tables(spark, landing_uri=landing_dir, trigger_mode="availableNow")
        logger.info("Built %d FlowConfig entries", len(tables))

        if len(tables) < len(ALL_TABLES):
            missing = set(ALL_TABLES) - set(tables.keys())
            logger.warning("Missing tables: %s", missing)

        # Log discovered schemas
        for name, fc in tables.items():
            col_names = [f.name for f in fc.schema.fields]
            logger.info("  %s (%s): %d cols %s", name, fc.format, len(col_names), col_names)

        # Step 3: Run pipeline
        logger.info("\n--- Step 3: Run ArcFlow pipeline ---")
        run_pipeline(spark, tables, output_dir, logger)

        # Step 4: Verify
        logger.info("\n--- Step 4: Verify delta tables ---")
        succeeded, failed = verify_delta_tables(spark, logger)

        logger.info("\n" + "=" * 70)
        logger.info("RESULTS: %d succeeded, %d failed out of %d tables",
                     len(succeeded), len(failed), len(ALL_TABLES))
        if failed:
            logger.error("FAILED tables: %s", failed)
            sys.exit(1)
        else:
            logger.info("✓ All tables created successfully!")

    finally:
        if spark:
            spark.stop()
        if not args.keep_temp:
            shutil.rmtree(base_tmp, ignore_errors=True)
            logger.info("Cleaned up temp directory: %s", base_tmp)
        else:
            logger.info("Temp directory kept: %s", base_tmp)


if __name__ == "__main__":
    main()
