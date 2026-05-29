"""
ArcFlow table registry for LakeGen LEGO data.

Discovers schemas at runtime from landing zone files so that every column
produced by LakeGen is preserved in the Bronze Delta tables.  Works in both
local-batch and Fabric-streaming modes.
"""

import logging
import posixpath
from typing import Dict, Optional, Set

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from arcflow import FlowConfig, StageConfig
from arcflow.transformations.zone_transforms import register_zone_transformer

logger = logging.getLogger(__name__)

ALL_TABLES = [
    "web_order",
    "web_order_line",
    "customer",
    "manufacturing_event",
    "production_order",
    "inventory_transaction",
    "quality_inspection",
    "product_return",
    "set_price_history",
    "colors",
    "parts",
    "sets",
    "themes",
    "part_categories",
    "inventories",
    "inventory_parts",
    "inventory_sets",
    "production_line",
    "mold",
]

JSON_TABLES: Set[str] = {
    "production_order",
    "quality_inspection",
    "product_return",
}

OUTPUT_TYPE_MAP: Dict[str, str] = {
    t: ("json" if t in JSON_TABLES else "parquet") for t in ALL_TABLES
}


@register_zone_transformer
def explode_data(df: DataFrame) -> DataFrame:
    """Unwrap the _meta/data JSON envelope and flatten into one row per record."""
    return df.selectExpr("explode(data) as data").select("data.*")


def _infer_schema(
    spark: SparkSession,
    table_name: str,
    landing_uri: str,
) -> Optional[StructType]:
    """Read one batch from the landing directory to infer the schema."""
    fmt = OUTPUT_TYPE_MAP[table_name]
    path = posixpath.join(landing_uri, table_name)

    try:
        if fmt == "json":
            df = spark.read.option("multiline", "true").json(path)
        else:
            df = spark.read.parquet(path)
        return df.schema
    except Exception as exc:
        logger.warning("Schema inference failed for %s: %s", table_name, exc)
        return None


def build_tables(
    spark: SparkSession,
    landing_uri: str = "Files/landing",
    trigger_mode: str = "processingTime",
) -> Dict[str, FlowConfig]:
    """
    Build the full ArcFlow table registry by inferring schemas from
    landing-zone files that LakeGen has already written.

    Call this *after* LakeGen.start() has produced at least one batch.

    Args:
        spark: Active SparkSession for schema inference.
        landing_uri: Root landing directory containing per-table sub-folders.
        trigger_mode: Spark trigger mode — ``"processingTime"`` for continuous
            streaming (Fabric SJD) or ``"availableNow"`` for one-shot batch
            (local testing).
    """
    tables: Dict[str, FlowConfig] = {}

    for table_name in ALL_TABLES:
        schema = _infer_schema(spark, table_name, landing_uri)
        if schema is None:
            logger.warning("Skipping %s — no data found in landing zone", table_name)
            continue

        fmt = OUTPUT_TYPE_MAP[table_name]
        bronze_stage = StageConfig(enabled=True, mode="append")
        if fmt == "json":
            bronze_stage.custom_transform = "explode_data"

        tables[table_name] = FlowConfig(
            name=table_name,
            format=fmt,
            source_uri=posixpath.join(landing_uri, table_name),
            schema=schema,
            description=f"LakeGen LEGO: {table_name}",
            trigger_mode=trigger_mode,
            clean_source=False,
            zones={"bronze": bronze_stage},
        )

    logger.info(
        "Built FlowConfig for %d / %d LEGO tables", len(tables), len(ALL_TABLES)
    )
    return tables
