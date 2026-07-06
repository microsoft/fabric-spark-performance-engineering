# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "28f1e957-ea23-49e8-846b-be0d8a67412e",
# META       "default_lakehouse_name": "lego",
# META       "default_lakehouse_workspace_id": "7fc5eff4-7153-4da9-b909-54981a3ffcdb",
# META       "known_lakehouses": [
# META         {
# META           "id": "28f1e957-ea23-49e8-846b-be0d8a67412e"
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "99FB9CB3-86D3-4877-BB60-659B3CDD45C3",
# META       "workspaceId": "7fc5eff4-7153-4da9-b909-54981a3ffcdb"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Module 1 — Optimizing Code
# 
# Welcome to the first Fabric Jumpstart Spark performance lab for **Toy Brick Manufacturing**.
# 
# ## What this module teaches
# 
# This module teaches you to recognize and fix **code-level** anti-patterns: the table design and cluster are fine, but the query as written is wrong or wasteful. You will also start using the diagnostic toolkit that Modules 2 and 3 reuse: Spark UI, `explain()` and physical plans, Delta metadata from `DESCRIBE DETAIL` / `DESCRIBE HISTORY`, `inputFiles()`, and Native Execution Engine (NEE) fallback detection.
# 
# > Litmus test: if the fix is a diff to the transformation logic, it belongs here. Storage fixes are Module 2; execution, AQE, caching, and repartitioning fixes are Module 3.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Setup: imports, workspace schema reset, source table helpers, and diagnostic utilities
from collections import defaultdict
import json
import re
import time

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

SOURCE_SCHEMA = "bronze"
WORK_SCHEMA = "opt_code"
MODULE_RESULTS = []


def table_ref(name: str) -> str:
    return f"`{SOURCE_SCHEMA}`.`{name}`"


def plan_string(df: DataFrame) -> str:
    return df._jdf.queryExecution().executedPlan().toString()


def scan_filters(df: DataFrame) -> dict:
    file_scans = [node for node in plan_string(df).split("\n") if "FileScan" in node]
    last_file_scan = file_scans[-1].strip() if file_scans else ""
    data_filters = re.search(r"DataFilters: \[(.*?)\]", last_file_scan)
    pushed_filters = re.search(r"PushedFilters: \[(.*?)\]", last_file_scan)
    return {
        "fileScan": last_file_scan,
        "dataFilters": data_filters.group(1) if data_filters else "",
        "pushedFilters": pushed_filters.group(1) if pushed_filters else "",
    }


# Used for NEE fallback analysis. If the plan has no Velox transition blocks, this returns zeroes.
block_pattern = re.compile(
    r"(?ms)^\s*\+-\s*RowToVeloxColumnar\b[^\n]*\n(?P<block>.*?)^\s*\+-\s*VeloxColumnarToRow\b"
)
op_pattern = re.compile(
    r"(?m)^\s*\+-\s*(?:\^\(\d+\)\s*)?(?P<op>[A-Za-z][A-Za-z0-9]*)\b"
)


def extract_nee_fallbacks(plan: str) -> dict:
    fallback_blocks = []
    fallback_operations = []
    for match in block_pattern.finditer(plan):
        block_text = match.group("block")
        block_lines = [line.strip() for line in block_text.split("\n") if line.strip()]
        block_ops = op_pattern.findall(block_text)
        fallback_blocks.append({"operations": block_lines, "operatorNames": block_ops})
        fallback_operations.extend(block_ops)
    return {
        "blockCount": len(fallback_blocks),
        "operatorCount": len(fallback_operations),
        "operators": fallback_operations,
        "blocks": fallback_blocks,
    }


def table_metrics(name: str) -> dict:
    ref = table_ref(name)
    detail_metrics = get_table_metrics(ref)
    detail = spark.sql(f"DESCRIBE DETAIL {ref}").collect()[0].asDict()
    return {
        "table": f"{SOURCE_SCHEMA}.{name}",
        "rows": spark.table(ref).count(),
        "numFiles": int(detail.get("numFiles") or detail_metrics.get("num_files") or 0),
        "sizeMB": float(detail_metrics.get("size_mb") or 0),
        "avgFileKB": float(detail_metrics.get("avg_file_kb") or 0),
        "format": detail.get("format"),
        "partitions": spark.table(ref).rdd.getNumPartitions(),
    }


def recent_history(name: str, limit: int = 3) -> list:
    return [
        row.asDict()
        for row in spark.sql(f"DESCRIBE HISTORY {table_ref(name)}")
        .select("version", "timestamp", "operation")
        .limit(limit)
        .collect()
    ]


def record_result(exercise: str, phase: str, evidence: dict) -> None:
    row = {"exercise": exercise, "phase": phase, "evidence": evidence}
    MODULE_RESULTS.append(row)
    print("MODULE1_RESULT\n" + json.dumps(row, default=str, indent=2))


print("Spark application ID:", spark.sparkContext.applicationId)
print("Current database:", spark.catalog.currentDatabase())
print("Source schema:", SOURCE_SCHEMA)
print("Resetting work schema:", WORK_SCHEMA)
spark.sql(f"DROP SCHEMA IF EXISTS {WORK_SCHEMA} CASCADE")
spark.sql(f"CREATE SCHEMA {WORK_SCHEMA}")

spark.conf.set("spark.microsoft.delta.parallelSnapshotLoading.enabled", "true")
spark.conf.set("spark.microsoft.delta.snapshot.driverMode.enabled", "true")
spark.conf.set("spark.synapse.vegas.useCache", "false")
spark.catalog.clearCache()

expected_tables = [
    "manufacturing_event",
    "web_order",
    "inventory_transaction",
    "quality_inspection",
    "production_order",
]
available_tables = {row.tableName for row in spark.sql(f"SHOW TABLES IN `{SOURCE_SCHEMA}`").collect()}
missing = [name for name in expected_tables if name not in available_tables]
if missing:
    raise RuntimeError(f"Missing required Module 1 tables in schema {SOURCE_SCHEMA}: {missing}")

print("\n=== Delta table metrics from DESCRIBE DETAIL ===")
for table_name in expected_tables:
    show_metrics(table_ref(table_name), "source")

TABLE_METRICS = {name: table_metrics(name) for name in expected_tables}
print(json.dumps(TABLE_METRICS, default=str, indent=2))
print("Recent manufacturing_event history:")
print(json.dumps(recent_history("manufacturing_event"), default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 1 — Predicate pushdown
# 
# **Problem:** The daily defect-rate query transforms the timestamp into a string before filtering. That makes Spark scan more of `manufacturing_event` than the dashboard needs.
# 
# **Why it matters:** Full scans waste I/O and make a small daily dashboard behave like a whole-factory history query.
# 
# **Fix in one line:** Filter with a native timestamp/date expression first, then derive presentation columns.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# BENCHMARK / DIAGNOSE: string timestamp handling weakens pushdown
print("🐌 Running baseline predicate-pushdown query...\n")

mfg = spark.table(table_ref("manufacturing_event")).selectExpr("manufacturing_event.*")
latest_day = mfg.select(F.max(F.to_date("timestamp")).alias("d")).collect()[0]["d"]

result_predicate_before = (
    mfg
    .withColumn("event_day", F.substring(F.col("timestamp"), 1, 10))
    .filter(F.col("event_day") == F.lit(str(latest_day)))
    .groupBy("event_day", F.col("machine_id"))
    .agg(
        F.count("*").alias("events"),
        F.sum(F.col("defect_detected").cast("int")).alias("defects"),
        (F.sum(F.col("defect_detected").cast("int")) / F.count("*")).alias("defect_rate"),
    )
    .orderBy(F.desc("defect_rate"))
    .benchmark("Predicate Pushdown", "before")
)
predicate_before_pdf = result_predicate_before.toPandas()
display(predicate_before_pdf)

predicate_before_scan = scan_filters(result_predicate_before)
predicate_before_evidence = {
    "antiPattern": "String timestamp transformation before filtering",
    "dashboardDay": str(latest_day),
    "inputRows": TABLE_METRICS["manufacturing_event"]["rows"],
    "inputFiles": TABLE_METRICS["manufacturing_event"]["numFiles"],
    "readFiles": len(result_predicate_before.inputFiles()),
    "dataFilters": predicate_before_scan["dataFilters"],
    "pushedFilters": predicate_before_scan["pushedFilters"],
    "planHasSubstring": "substring" in plan_string(result_predicate_before).lower(),
}
record_result("predicate_pushdown", "before", predicate_before_evidence)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Inspect the formatted plan. Look for the `FileScan` node, `DataFilters`, `PushedFilters`, and the number of files from `inputFiles()`. Then rewrite the query so the filter is expressed before the aggregation.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Starter: inspect the baseline physical plan and a runnable native-filter sketch.
result_predicate_before.explain(mode="formatted")

starter_predicate_filter = mfg.filter(F.to_date("timestamp") == F.lit(latest_day))
print("Starter read files after native filter:", len(starter_predicate_filter.inputFiles()))
display(starter_predicate_filter.select("timestamp", "machine_id", "defect_detected").limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ✅ Solution: filter with native column functions before aggregating
print("✅ Running fixed predicate-pushdown query...\n")

result_predicate_after = (
    mfg
    .filter(F.to_date("timestamp") == F.lit(latest_day))
    .withColumn("event_day", F.to_date("timestamp"))
    .groupBy("event_day", F.col("machine_id"))
    .agg(
        F.count("*").alias("events"),
        F.sum(F.col("defect_detected").cast("int")).alias("defects"),
        (F.sum(F.col("defect_detected").cast("int")) / F.count("*")).alias("defect_rate"),
    )
    .orderBy(F.desc("defect_rate"))
    .benchmark("Predicate Pushdown", "after")
)
predicate_after_pdf = result_predicate_after.toPandas()
display(predicate_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Validation: benchmark comparison auto-printed; verify the fixed plan signals.
predicate_after_scan = scan_filters(result_predicate_after)
predicate_after_evidence = {
    "antiPattern": "String timestamp transformation before filtering",
    "baselineReadFiles": predicate_before_evidence["readFiles"],
    "fixedReadFiles": len(result_predicate_after.inputFiles()),
    "fixedDataFilters": predicate_after_scan["dataFilters"],
    "fixedPushedFilters": predicate_after_scan["pushedFilters"],
    "fixedPlanHasSubstring": "substring" in plan_string(result_predicate_after).lower(),
}
record_result("predicate_pushdown", "after", predicate_after_evidence)
result_predicate_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 2 — Python UDFs → native expressions / NEE
# 
# **Problem:** The top-customer query calculates line totals and extracts order days with Python UDFs.
# 
# **Why it matters:** Python UDFs force JVM↔Python serialization, usually show `BatchEvalPython` in the plan, and can trigger Native Execution Engine fallback.
# 
# **Fix in one line:** Replace scalar Python UDFs with built-in Spark SQL expressions.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# BENCHMARK / DIAGNOSE: Python UDFs add serialization and NEE fallback risk.
print("🐌 Running baseline query with Python UDFs...\n")

@F.udf(DoubleType())
def python_line_total(quantity, unit_price, extended_price):
    if extended_price is not None:
        return float(extended_price)
    if quantity is None or unit_price is None:
        return 0.0
    return float(quantity) * float(unit_price)


@F.udf("string")
def python_extract_day(timestamp_str):
    if timestamp_str is None:
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(timestamp_str))
    return match.group(1) if match else None


orders = spark.table(table_ref("web_order")).selectExpr("web_order.*")
exploded_orders = orders.select(
    F.col("customer_id"),
    F.col("order_date"),
    F.explode("order_lines").alias("line"),
)

result_udf_before = (
    exploded_orders
    .withColumn("line_total", python_line_total("line.quantity", "line.unit_price", "line.extended_price"))
    .withColumn("order_day", python_extract_day("order_date"))
    .groupBy("customer_id", "order_day")
    .agg(F.sum("line_total").alias("total_spend"), F.count("*").alias("line_count"))
    .groupBy("customer_id")
    .agg(F.sum("total_spend").alias("total_spend"), F.sum("line_count").alias("line_count"))
    .orderBy(F.desc("total_spend"))
    .limit(10)
    .benchmark("Python UDF Overhead", "before")
)
udf_before_pdf = result_udf_before.toPandas()
display(udf_before_pdf)

udf_before_plan = plan_string(result_udf_before)
udf_before_fallbacks = extract_nee_fallbacks(udf_before_plan)
udf_before_evidence = {
    "antiPattern": "Python UDFs instead of built-in functions",
    "sourceRows": TABLE_METRICS["web_order"]["rows"],
    "hasBatchEvalPython": "BatchEvalPython" in udf_before_plan or "PythonUDF" in udf_before_plan,
    "neeFallbackBlockCount": udf_before_fallbacks["blockCount"],
    "neeFallbackCount": udf_before_fallbacks["operatorCount"],
    "neeFallbackOperators": udf_before_fallbacks["operators"],
}
record_result("python_udfs", "before", udf_before_evidence)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Inspect the plan and find the Python boundary. Then replace both UDFs: use `coalesce` / arithmetic for line totals and a native date/string function for the order day. Watch whether NEE fallback blocks disappear.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Starter: look for BatchEvalPython / PythonUDF and preview the native columns to use.
result_udf_before.explain(mode="formatted")

starter_native_columns = exploded_orders.select(
    "customer_id",
    F.coalesce(
        F.col("line.extended_price").cast("double"),
        F.col("line.quantity").cast("double") * F.col("line.unit_price").cast("double"),
        F.lit(0.0),
    ).alias("line_total"),
    F.regexp_extract("order_date", r"(\d{4}-\d{2}-\d{2})", 1).alias("order_day"),
).limit(5)
display(starter_native_columns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ✅ Solution: replace Python UDFs with native Spark SQL expressions.
print("✅ Running fixed query with native expressions...\n")

line_total_col = F.coalesce(
    F.col("line.extended_price").cast("double"),
    F.col("line.quantity").cast("double") * F.col("line.unit_price").cast("double"),
    F.lit(0.0),
)
order_day_col = F.regexp_extract("order_date", r"(\d{4}-\d{2}-\d{2})", 1)

result_udf_after = (
    exploded_orders
    .withColumn("line_total", line_total_col)
    .withColumn("order_day", order_day_col)
    .groupBy("customer_id", "order_day")
    .agg(F.sum("line_total").alias("total_spend"), F.count("*").alias("line_count"))
    .groupBy("customer_id")
    .agg(F.sum("total_spend").alias("total_spend"), F.sum("line_count").alias("line_count"))
    .orderBy(F.desc("total_spend"))
    .limit(10)
    .benchmark("Python UDF Overhead", "after")
)
udf_after_pdf = result_udf_after.toPandas()
display(udf_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Validation: no Python UDF nodes should remain in the fixed plan.
udf_after_plan = plan_string(result_udf_after)
udf_after_fallbacks = extract_nee_fallbacks(udf_after_plan)
udf_after_evidence = {
    "antiPattern": "Python UDFs instead of built-in functions",
    "baselineHadBatchEvalPython": udf_before_evidence["hasBatchEvalPython"],
    "fixedHasBatchEvalPython": "BatchEvalPython" in udf_after_plan or "PythonUDF" in udf_after_plan,
    "baselineNeeFallbackCount": udf_before_evidence["neeFallbackCount"],
    "fixedNeeFallbackCount": udf_after_fallbacks["operatorCount"],
    "fixedNeeFallbackOperators": udf_after_fallbacks["operators"],
}
record_result("python_udfs", "after", udf_after_evidence)
result_udf_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 3 — Driver `collect()` / `toPandas()` and driver OOM
# 
# **Problem:** The inventory workflow pulls every transaction to the driver with `collect()` and aggregates in Python. `.toPandas()` has the same raw-data movement risk.
# 
# **Why it matters:** Pulling distributed data into one process can trip task-result transport limits, executor memory while serializing results, or `spark.driver.maxResultSize`.
# 
# **Fix in one line:** Keep the aggregation distributed and only bring the small final result to the driver.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# BENCHMARK / DIAGNOSE: collecting raw rows to the driver defeats Spark.
print("🐌 Running baseline query with driver-side collect...\n")

inv = spark.table(table_ref("inventory_transaction")).select(
    "line_id", "part_num", "quantity", "transaction_type"
)
print(f"About to collect {TABLE_METRICS['inventory_transaction']['rows']:,} inventory rows to the driver.")
print("spark.driver.maxResultSize =", spark.conf.get("spark.driver.maxResultSize"))

start = time.time()
collected_inventory = inv.benchmark("Driver Collect", "before").collect()
with benchmark_op("Driver Python Aggregation", "before", spark):
    inventory_by_line = defaultdict(int)
    for row in collected_inventory:
        qty = int(row["quantity"] or 0)
        if row["transaction_type"] in ("CONSUMPTION", "ORDER_PICK", "SCRAP"):
            qty = -abs(qty)
        inventory_by_line[row["line_id"] or "UNKNOWN"] += qty

driver_result_rows = [
    {"line_id": line_id, "net_quantity": qty}
    for line_id, qty in inventory_by_line.items()
]
driver_elapsed_ms = (time.time() - start) * 1000
print(f"Driver-side Python aggregation elapsed: {driver_elapsed_ms:.2f} ms")
display(driver_result_rows)

driver_before_evidence = {
    "antiPattern": "Driver-side collect and Python aggregation",
    "sourceRows": TABLE_METRICS["inventory_transaction"]["rows"],
    "collectedRows": len(collected_inventory),
    "resultRows": len(driver_result_rows),
    "driverOomRisk": "Raw rows are transferred to the driver; toPandas has the same risk profile.",
}
record_result("driver_collect", "before", driver_before_evidence)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Rewrite the workflow so executors compute the net inventory by line. The driver should receive only the grouped result, not every raw transaction row. Use the Spark UI to compare task result size before and after.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Starter: build the signed quantity column with Spark expressions, then aggregate it.
inv.explain(mode="formatted")

starter_signed_inventory = inv.withColumn(
    "signed_quantity",
    F.when(
        F.col("transaction_type").isin("CONSUMPTION", "ORDER_PICK", "SCRAP"),
        -F.abs(F.col("quantity").cast("int")),
    ).otherwise(F.col("quantity").cast("int")),
)
display(starter_signed_inventory.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ✅ Solution: distributed Spark aggregation, then small final result to pandas/display.
print("✅ Running fixed query with distributed aggregation...\n")

result_driver_after = (
    starter_signed_inventory
    .groupBy("line_id")
    .agg(F.sum("signed_quantity").alias("net_quantity"))
    .orderBy(F.desc("net_quantity"))
    .benchmark("Driver Collect", "after")
)
driver_after_pdf = result_driver_after.toPandas()
display(driver_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Validation: the fixed path collects only the final grouped rows.
record_result("driver_collect", "after", {
    "antiPattern": "Driver-side collect and Python aggregation",
    "baselineCollectedRows": driver_before_evidence["collectedRows"],
    "fixedRawRowsCollected": 0,
    "fixedResultRowsReturnedToDriver": len(driver_after_pdf),
    "improvement": "Aggregation runs on executors; only the small grouped result reaches the driver.",
})
result_driver_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 4 — Cartesian / missing join key
# 
# **Problem:** A pass-rate query combines `quality_inspection` with `production_order` without the production-order join key. A related cycle-time query uses an inequality self-join when it really needs the previous event per machine.
# 
# **Why it matters:** Missing equality keys create N × M pairs. In production this shows up as massive shuffle, spill, executor loss, or out-of-memory.
# 
# **Fix in one line:** Supply the correct equality join condition; when the intent is "previous row," use a window function instead of an inequality self-join.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# BENCHMARK / DIAGNOSE: missing join predicate creates a Cartesian product.
print("🐌 Running baseline query with missing join key...\n")

qi = spark.table(table_ref("quality_inspection")).select(
    F.col("quality_inspection.production_order_id").alias("qi_production_order_id"),
    F.col("quality_inspection.result").alias("inspection_result"),
    F.col("quality_inspection.pass_count").alias("pass_count"),
    F.col("quality_inspection.sample_size").alias("sample_size"),
)
po = spark.table(table_ref("production_order")).select(
    F.col("production_order.production_order_id").alias("po_production_order_id"),
    F.col("production_order.machine_id").alias("machine_id"),
    F.col("production_order.status").alias("order_status"),
)

estimated_pairs = TABLE_METRICS["quality_inspection"]["rows"] * TABLE_METRICS["production_order"]["rows"]
print(f"Estimated Cartesian pairs: {estimated_pairs:,}")

result_cartesian_before = (
    qi.crossJoin(po)
    .groupBy("machine_id")
    .agg(
        F.count("*").alias("joined_rows"),
        (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
    )
    .orderBy(F.desc("joined_rows"))
    .benchmark("Cartesian Join", "before")
)
cartesian_before_pdf = result_cartesian_before.toPandas()
display(cartesian_before_pdf)

cartesian_before_plan = plan_string(result_cartesian_before)
cartesian_before_join = (
    "BroadcastNestedLoopJoin" if "BroadcastNestedLoopJoin" in cartesian_before_plan
    else "CartesianProduct" if "CartesianProduct" in cartesian_before_plan
    else "None"
)
cartesian_before_evidence = {
    "antiPattern": "Missing join predicate / Cartesian join",
    "qualityRows": TABLE_METRICS["quality_inspection"]["rows"],
    "productionOrderRows": TABLE_METRICS["production_order"]["rows"],
    "estimatedCartesianPairs": estimated_pairs,
    "resultRows": len(cartesian_before_pdf),
    "expectedPlanSignal": "CartesianProduct or BroadcastNestedLoopJoin",
    "executedJoin": cartesian_before_join,
}
record_result("cartesian_join", "before", cartesian_before_evidence)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Inspect the plan for `CartesianProduct` or `BroadcastNestedLoopJoin`. Then rewrite the pass-rate query with the real key, `production_order_id`, so Spark can use an equi-join. For the cycle-time variant, replace the inequality self-join with `lag()` over a window.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Starter: inspect the missing-key plan and build the equi-join condition explicitly.
result_cartesian_before.explain(mode="formatted")

join_condition = F.col("qi_production_order_id") == F.col("po_production_order_id")
starter_join_preview = qi.join(po, join_condition).select(
    "qi_production_order_id", "po_production_order_id", "machine_id", "pass_count", "sample_size"
).limit(5)
display(starter_join_preview)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ✅ Solution: join quality inspections to production orders by production_order_id.
print("✅ Running fixed query with the correct join predicate...\n")

result_cartesian_after = (
    qi.join(po, join_condition)
    .groupBy("machine_id")
    .agg(
        F.count("*").alias("joined_rows"),
        (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
    )
    .orderBy(F.desc("joined_rows"))
    .benchmark("Cartesian Join", "after")
)
cartesian_after_pdf = result_cartesian_after.toPandas()
display(cartesian_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Validation: the fixed plan should use a real equi-join and process matched rows only.
cartesian_after_plan = plan_string(result_cartesian_after)
cartesian_after_join = (
    "SortMergeJoin" if "SortMergeJoin" in cartesian_after_plan
    else "BroadcastHashJoin" if "BroadcastHashJoin" in cartesian_after_plan
    else "ShuffledHashJoin" if "ShuffledHashJoin" in cartesian_after_plan
    else "None"
)
record_result("cartesian_join", "after", {
    "antiPattern": "Missing join predicate / Cartesian join",
    "baselineJoin": cartesian_before_evidence["executedJoin"],
    "fixedJoin": cartesian_after_join,
    "baselineResultRows": len(cartesian_before_pdf),
    "fixedResultRows": len(cartesian_after_pdf),
    "improvement": "Added equality join on production_order_id instead of crossJoin.",
})
result_cartesian_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Related code fix: inequality self-join for previous event should be a window function.
print("🔎 Diagnosing the cycle-time self-join variant without executing the Cartesian action...\n")

cycle_events = mfg.select("machine_id", "timestamp", "cycle_time_ms").filter(
    F.to_date("timestamp") == F.lit(latest_day)
)
a = cycle_events.alias("a")
b = cycle_events.alias("b")

bad_cycle_join = a.join(b, F.col("b.timestamp") < F.col("a.timestamp")).select(
    F.col("a.machine_id"),
    F.col("a.timestamp"),
    (F.col("a.cycle_time_ms") - F.col("b.cycle_time_ms")).alias("delta_ms"),
)

bad_cycle_plan = plan_string(bad_cycle_join)
cycle_rows = cycle_events.count()
record_result("cartesian_window_rewrite", "before", {
    "antiPattern": "Inequality self-join instead of previous-row window",
    "filteredRows": cycle_rows,
    "estimatedPairs": cycle_rows * cycle_rows,
    "planHasCartesianSignal": "CartesianProduct" in bad_cycle_plan or "BroadcastNestedLoopJoin" in bad_cycle_plan,
})
bad_cycle_join.explain(mode="formatted")

w = Window.partitionBy("machine_id").orderBy("timestamp")
fixed_cycle_delta = (
    cycle_events
    .withColumn("prev_cycle_time_ms", F.lag("cycle_time_ms").over(w))
    .withColumn("delta_ms", F.col("cycle_time_ms") - F.col("prev_cycle_time_ms"))
    .filter(F.col("delta_ms").isNotNull())
    .benchmark("Cartesian Window Rewrite", "after")
)
fixed_cycle_count = fixed_cycle_delta.count()
record_result("cartesian_window_rewrite", "after", {
    "antiPattern": "Inequality self-join instead of previous-row window",
    "fixedRows": fixed_cycle_count,
    "planHasCartesianSignal": "CartesianProduct" in plan_string(fixed_cycle_delta),
    "improvement": "Window lag computes the previous event per machine without creating N x M pairs.",
})
fixed_cycle_delta.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Summary
# 
# You fixed four code-level Spark anti-patterns: weak predicate pushdown, Python UDF serialization and NEE fallback, driver-side raw-row collection, and missing/non-equality join logic that created Cartesian work.
# 
# Carry the same workflow into the next modules: benchmark the symptom, inspect the Spark UI and physical plan, check Delta metadata, change the right lever, and validate the before/after result.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
