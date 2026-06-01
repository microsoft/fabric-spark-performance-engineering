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

# # 🏗️ **Module 1: Diagnostics – The Factory Dashboard is Slow**
#
# Learn how to identify common Spark performance anti‑patterns behind a slow Power BI / Fabric dashboard, diagnose them using Spark plans and the UI, apply targeted fixes, and validate the impact with before‑and‑after benchmarks.
#
# You’ll work through six realistic queries that power a LEGO manufacturing analytics dashboard:
#
# 1. **Daily defect rate by machine** – predicate pushdown and partition pruning
# 2. **Top customers by spend** – Python UDF overhead vs. built‑in functions
# 3. **Inventory levels by plant/line** – driver‑side `.collect()` vs. distributed aggregation
# 4. **Event fan‑out by defect type** – repeated scans vs. caching shared intermediates
# 5. **Quality inspection pass rates** – Cartesian joins from missing join predicates
# 6. **Event aggregation by day** – unnecessary caching and Native Execution Engine (NEE) fallback
#
# **Duration:** 60 minutes | **Level:** 300–400
#
# ---
#
# ### Scenario
#
# The LEGO manufacturing analytics team has built a factory quality and operations dashboard backed by Spark queries over Lakehouse tables (bronze layer). Over time, the dashboard has become noticeably slower as data volumes and usage have grown.
#
# Your investigation reveals that several visuals are powered by inefficient Spark patterns:
# - Full table scans without effective predicate pushdown
# - Python UDFs blocking NEE optimizations
# - Driver‑side `.collect()` and Python loops
# - Repeated scans when branching to multiple outputs
# - Cartesian joins due to missing join keys
# - Unnecessary caching that adds overhead and can trigger NEE fallbacks
#
# **Your mission:** For each query, benchmark the current behavior, diagnose the root cause using plans and metrics, implement a fix, and re‑benchmark to quantify the improvement.
#
# ### Lab Pattern
#
# Every exercise follows the same steps:
#
# | Step | What you do |
# |------|-------------|
# | 🐌 **Benchmark** | Run a query and capture the baseline time/metric |
# | 🔍 **Diagnose** | Inspect table metadata and Spark plans to prove the root cause |
# | 🔧 **Fix** | Apply the optimization using recommended Spark patterns |
# | 🚀 **Re-benchmark** | Run the same test and compare against the baseline |

# MARKDOWN ********************

# ## Expected diagnostic prompts
#
# | Prompt | Scenario | Expected signal |
# |---|---|---|
# | Q1 | Daily defect rate by machine | Full scan / poor predicate pushdown from string timestamp handling |
# | Q2 | Top customers by spend | Python UDF and Python serialization overhead |
# | Q3 | Inventory levels by plant/line | Driver-side `.collect()` anti-pattern |
# | Q4 | Manufacturing event fan-out by event type | Repeated scans from looped writes with no caching |
# | Q5 | Quality inspection pass rates | Cartesian/nested-loop join from missing join key |
# | Q6 | Event aggregation by day | Unnecessary cache causing extra materialization and possible NEE fallback |

# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# Setup: imports, table references, and read-only helpers
from collections import defaultdict
import json
import time
import regex

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Used for NEE fallback analysis
import re
block_pattern = re.compile(
    r'(?ms)^\s*\+-\s*RowToVeloxColumnar\b[^\n]*\n(?P<block>.*?)^\s*\+-\s*VeloxColumnarToRow\b'
)
op_pattern = re.compile(
    r'(?m)^\s*\+-\s*(?:\^\(\d+\)\s*)?(?P<op>[A-Za-z][A-Za-z0-9]*)\b'
)

def extract_nee_fallbacks(plan: str) -> dict:
    fallback_blocks = []
    fallback_operations = []

    for match in block_pattern.finditer(plan):
        block_text = match.group("block")
        block_lines = [line.strip() for line in block_text.split("\n") if line.strip()]
        block_ops = op_pattern.findall(block_text)
        fallback_blocks.append({
            "operations": block_lines,
            "operatorNames": block_ops,
        })
        fallback_operations.extend(block_ops)

    return {
        "blockCount": len(fallback_blocks),
        "operatorCount": len(fallback_operations),
        "operators": fallback_operations,
        "blocks": fallback_blocks,
    }

spark:SparkSession = spark

print("Spark application ID:", spark.sparkContext.applicationId)
print("Current database:", spark.catalog.currentDatabase())

# Minimize snapshot generation overhead
spark.conf.set('spark.microsoft.delta.parallelSnapshotLoading.enabled', True)
spark.conf.set('spark.microsoft.delta.snapshot.driverMode.enabled', True)
spark.conf.set("spark.synapse.vegas.useCache", "false")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# Read-only table discovery and baseline metrics
expected_tables = [
    "manufacturing_event", "web_order", "web_order_line", "sets", "themes",
    "inventory_transaction", "quality_inspection", "production_order"
]

available_tables = {row.tableName for row in spark.sql(f"SHOW TABLES IN `bronze`").collect()}
print("Available expected tables:")
for name in expected_tables:
    print(f"  {name}: {name in available_tables}")

missing = [name for name in expected_tables if name not in available_tables]
if missing:
    raise RuntimeError(f"Missing required Lab 1 tables in schema bronze: {missing}")

TABLE_METRICS = {}
TABLE_METRICS['metrics'] = {
    "rows": "row_count",
    "numFiles": "num_files",
    "sizeBytes": "size_bytes",
    "sizeMB": "size_mb",
    "avgFileMB": "avg_file_mb",
    "partitions": "num_partitions",
}
for name in expected_tables:
    ref = f"bronze.{name}"
    detail = spark.sql(f"DESCRIBE DETAIL {ref}").collect()[0].asDict()
    row_count = spark.table(ref).count()
    num_files = int(detail.get("numFiles") or 0)
    size_bytes = int(detail.get("sizeInBytes") or 0)
    avg_file_mb = (size_bytes / num_files / 1024 / 1024) if num_files else 0
    TABLE_METRICS[name] = {
        "rows": int(row_count),
        "numFiles": int(num_files),
        "sizeBytes": int(size_bytes),
        "sizeMB": float(size_bytes / 1024 / 1024),
        "avgFileMB": float(avg_file_mb),
        "partitions": int(spark.table(ref).rdd.getNumPartitions()),
    }

display(TABLE_METRICS)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ---
#
# # Query 1: Daily defect rate by machine
#
# **Table:** `manufacturing_event` — high-frequency IoT telemetry from the injection molding floor
#
# **What's wrong:** This query filters with string transformations on nested timestamp fields. The table is not partition-pruned for the time window, so Spark should scan substantially more input than the dashboard result needs.
#
# **Why it matters:**
# - Full table scans are expensive and slow, especially on large datasets
# - Increases I/O and memory usage, leading to potential performance bottlenecks
#
# **Fix:** Appropriate predicate pushdown
#
# ---

# CELL ********************

# Display table metrics before benchmark
print("\n=== Table Metrics: manufacturing_event ===")
show_metrics("bronze.manufacturing_event", "baseline")

# Example of reading the data to find the latest date, which will be used for filtering in the benchmark queries
mfg = spark.table("bronze.manufacturing_event").selectExpr("manufacturing_event.*")
latest_day = mfg.select(F.max(F.to_date("timestamp")).alias("d")).collect()[0]["d"]
latest_day

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 1️⃣ BENCHMARK — Capture baseline query time
# ============================================================

print("🐌 Running baseline query on manufacturing_event...\n")

with benchmark_op("Predicate Pushdown", "before", spark):
    result_q1 = (
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
    )
    data = result_q1.toPandas()

display(data)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# =================================================================================================
# 1️⃣ DIAGNOSE — Prove the root cause is the ammount of read files and the filter not being pushed
# =================================================================================================

read_files = len(result_q1.inputFiles())
plan = result_q1._jdf.queryExecution().executedPlan().toString()
file_scans = [node for node in result_q1._jdf.queryExecution().executedPlan().toString().split("\n") if "FileScan" in node]
last_file_scan = file_scans[-1].strip() if file_scans else ""
data_filters = regex.search(r"DataFilters: \[(.*?)\]", last_file_scan)
data_filters = data_filters.group(1) if data_filters else ""
pushed_filters = regex.search(r"PushedFilters: \[(.*?)\]", last_file_scan)
pushed_filters = pushed_filters.group(1) if pushed_filters else ""

print(json.dumps({
    "antiPattern": "Full scan / weak pushdown due to string timestamp transformation",
    "inputRows": TABLE_METRICS["manufacturing_event"]["rows"],
    "inputFiles": TABLE_METRICS["manufacturing_event"]["numFiles"],
    "readFiles": read_files,
    "lastDataFilter": data_filters,
    "lastPushedFilter": pushed_filters,
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Check the Spark Plan and Spark UI to diagnose the problem
#
# You've seen that the query is slow, and you suspect it's because of the large number of files and the filter not being pushed down. You want to confirm this by checking the Spark Plan and the Spark UI.
#
# **Your task:** Check the Spark Plan and Spark UI to confirm that the query is doing a full scan of all files and that the filter on `event_day` is not being pushed down to the file scan level.
#
# > 💡 Hint: **Explain** methods in Spark can help you understand the physical plan and see if filters are being pushed down.
#
# Try it in the cell below!

# CELL ********************

# =================================================================================================
# 1️⃣ DIAGNOSE — (Optional) Look for the Pushed Filters on the Physical Plan
# =================================================================================================

result_q1.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ==================================================================================================
# 1️⃣ FIX — Use Functions on the column instead of string manipulation to enable predicate pushdown
# ==================================================================================================

with benchmark_op("Predicate Pushdown", "after", spark):
    result_q1 = (
            mfg
            .filter(F.col("timestamp").startswith(F.lit(str(latest_day))))
            .withColumn("event_day", F.substring(F.col("timestamp"), 1, 10))
            .groupBy("event_day", F.col("machine_id").alias("machine_id"))
            .agg(
                F.count("*").alias("events"),
                F.sum(F.col("defect_detected").cast("int")).alias("defects"),
                (F.sum(F.col("defect_detected").cast("int")) / F.count("*")).alias("defect_rate"),
            )
            .orderBy(F.desc("defect_rate"))
    )
    data = result_q1.toPandas()

display(data)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************


# ============================================================
# 1️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

read_files = len(result_q1.inputFiles())
plan = result_q1._jdf.queryExecution().executedPlan().toString()
# Pick the last FileScan in the plan as a heuristic for the relevant scan node, and count the files read from it
file_scans = [node for node in result_q1._jdf.queryExecution().executedPlan().toString().split("\n") if "FileScan" in node]
last_file_scan = file_scans[-1].strip() if file_scans else ""
data_filters = regex.search(r"DataFilters: \[(.*?)\]", last_file_scan)
data_filters = data_filters.group(1) if data_filters else ""
pushed_filters = regex.search(r"PushedFilters: \[(.*?)\]", last_file_scan)
pushed_filters = pushed_filters.group(1) if pushed_filters else ""

print(json.dumps({
    "antiPattern": "Full scan / weak pushdown due to string timestamp transformation",
    "inputRows": TABLE_METRICS["manufacturing_event"]["rows"],
    "inputFiles": TABLE_METRICS["manufacturing_event"]["numFiles"],
    "readFiles": read_files,
    "lastDataFilter": data_filters,
    "lastPushedFilter": pushed_filters,
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

result_q1.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ---
#
# # Query 2: Top 10 customers by spend
#
# ---
#
# **Table:** `web_order` — customer orders with nested order line items
#
# **Fix:** Replace Python UDFs with native Spark SQL expressions
#
# **What's wrong:** This query uses Python UDFs to calculate line totals and extract date strings, forcing Python serialization overhead and preventing Native Execution Engine optimization.
#
# - Significantly slower than built-in Spark SQL functions
#
# **Why it matters:**- Cannot leverage Native Execution Engine (NEE) optimizations
# - Python UDFs require data serialization between JVM and Python processes

# CELL ********************

# Display table metrics before benchmark
print("\n=== Table Metrics: web_order ===")
show_metrics("bronze.web_order", "baseline")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Capture baseline query time with Python UDFs
# ============================================================

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
    """Extract day from timestamp using regex - intentionally inefficient"""
    if timestamp_str is None:
        return None
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(timestamp_str))
    return match.group(1) if match else None

with benchmark_op("Python UDF Overhead", "Python UDFs", spark):
    orders = spark.table("bronze.web_order").selectExpr("web_order.*")
    exploded = orders.select(
        F.col("customer_id"),
        F.col("order_date"),
        F.explode("order_lines").alias("line")
    )
    result_q2 = (
        exploded
        .withColumn("line_total", python_line_total("line.quantity", "line.unit_price", "line.extended_price"))
        .withColumn("order_day", python_extract_day("order_date"))
        .groupBy("customer_id", "order_day")
        .agg(F.sum("line_total").alias("total_spend"), F.count("*").alias("line_count"))
        .groupBy("customer_id")
        .agg(F.sum("total_spend").alias("total_spend"), F.sum("line_count").alias("line_count"))
        .orderBy(F.desc("total_spend"))
        .limit(10)
    )
    data = result_q2.toPandas()

display(data)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# =================================================================================================
# 2️⃣ DIAGNOSE — Prove the root cause is Python UDF serialization and NEE fallback
# =================================================================================================

plan = result_q2._jdf.queryExecution().executedPlan().toString()
has_batch_eval_python = "BatchEvalPython" in plan or "PythonUDF" in plan
nee_fallback_summary = extract_nee_fallbacks(plan)

print(json.dumps({
    "antiPattern": "Multiple Python UDFs instead of built-in functions",
    "hasBatchEvalPython": has_batch_eval_python,
    "sourceRows": TABLE_METRICS["web_order"]["rows"],
    "neeFallbackBlockCount": nee_fallback_summary["blockCount"],
    "neeFallbackCount": nee_fallback_summary["operatorCount"],
    "neeFallbackOperators": nee_fallback_summary["operators"],
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Check the Spark Plan to find Python UDF overhead
#
# You've seen that the query is slow, and you suspect it's because of Python UDF serialization overhead preventing Native Execution Engine optimization.
#
# **Your task:** Check the Spark Plan to confirm that:
# 1. The query uses Python UDFs (look for `BatchEvalPython` or `PythonUDF`)
# 2. There are Native Execution Engine fallbacks (look for `RowToVeloxColumnar` and `VeloxColumnarToRow` blocks)
#
# > 💡 Hint: **Explain** methods in Spark can help you understand the physical plan and identify Python UDF operations.
#
# Try it in the cell below!

# CELL ********************

# =================================================================================================
# 2️⃣ DIAGNOSE — (Optional) Look for Python UDF and NEE fallbacks in the Physical Plan
# =================================================================================================

result_q2.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ==================================================================================================
# 2️⃣ FIX — Replace Python UDFs with native Spark SQL expressions
# ==================================================================================================

# Use coalesce for cleaner null handling
line_total_col_simple = F.coalesce(
    F.col("line.extended_price").cast("double"),
    (F.col("line.quantity").cast("double") * F.col("line.unit_price").cast("double")),
    F.lit(0.0)
)

order_day = F.regexp_extract("order_date", r'(\d{4}-\d{2}-\d{2})', 1).alias("order_day")

with benchmark_op("Python UDF Overhead", "Built-in Functions", spark):
    orders = spark.table("bronze.web_order").selectExpr("web_order.*")
    exploded = orders.select(
        F.col("customer_id"),
        F.col("order_date"),
        F.explode("order_lines").alias("line")
    )
    result_q2_fixed = (
        exploded
        .withColumn("line_total", line_total_col_simple)
        .withColumn("order_day", order_day)
        .groupBy("customer_id", "order_day")
        .agg(F.sum("line_total").alias("total_spend"), F.count("*").alias("line_count"))
        .groupBy("customer_id")
        .agg(F.sum("total_spend").alias("total_spend"), F.sum("line_count").alias("line_count"))
        .orderBy(F.desc("total_spend"))
        .limit(10)
    )
    data = result_q2_fixed.toPandas()

display(data)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 2️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

plan = result_q2_fixed._jdf.queryExecution().executedPlan().toString()
has_batch_eval_python = "BatchEvalPython" in plan or "PythonUDF" in plan
nee_fallback_summary = extract_nee_fallbacks(plan)

print(json.dumps({
    "antiPattern": "Multiple Python UDFs instead of built-in functions",
    "hasBatchEvalPython": has_batch_eval_python,
    "sourceRows": TABLE_METRICS["web_order"]["rows"],
    "neeFallbackBlockCount": nee_fallback_summary["blockCount"],
    "neeFallbackCount": nee_fallback_summary["operatorCount"],
    "neeFallbackOperators": nee_fallback_summary["operators"],
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

result_q2_fixed.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ---
#
# # Query 3: Inventory levels by plant/line
#
# **Table:** `inventory_transaction` — inventory movements across production lines
#
# **What's wrong:** This query uses `.collect()` to pull all inventory transaction rows to the driver, then performs aggregation in Python. This defeats distributed processing and creates a bottleneck on the driver node.
#
# **Why it matters:**
# - Driver memory limits can cause OOM errors on larger datasets
# - Single-node processing wastes cluster resources
# - Network transfer overhead for moving data to driver
# - Does not scale as data volume grows
#
# **Fix:** Use distributed Spark aggregations instead of driver-side Python loops
#
# ---

# CELL ********************

# Display table metrics before benchmark
print("\n=== Table Metrics: inventory_transaction ===")
show_metrics("bronze.inventory_transaction", "baseline")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Capture baseline query time with driver collect
# ============================================================

print("🐌 Running baseline query with driver-side collect...\n")


with benchmark_op("Local Driver work", "before", spark):
    inv = spark.table("bronze.inventory_transaction").select("line_id", "part_num", "quantity", "transaction_type")
    print(f"About to collect {TABLE_METRICS['inventory_transaction']['rows']} inventory rows to the driver.")
    collected = inv.collect()
    inventory_by_line = defaultdict(int)
    for row in collected:
        qty = int(row["quantity"] or 0)
        if row["transaction_type"] in ("CONSUMPTION", "ORDER_PICK", "SCRAP"):
            qty = -abs(qty)
        inventory_by_line[row["line_id"] or "UNKNOWN"] += qty

    # Create pandas DataFrame directly from Python dict
    import pandas as pd
    result_q3 = pd.DataFrame([
        {"line_id": line_id, "net_quantity": qty} 
        for line_id, qty in inventory_by_line.items()
    ]).sort_values("net_quantity", ascending=False).reset_index(drop=True)

display(result_q3)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# =================================================================================================
# 3️⃣ DIAGNOSE — Prove the root cause is driver-side collect and Python aggregation
# =================================================================================================

print(json.dumps({
    "antiPattern": "Driver-side collect and Python aggregation",
    "sourceRows": TABLE_METRICS["inventory_transaction"]["rows"],
    "collectedRows": len(collected),
    "resultRows": len(result_q3),
    "impact": "All rows transferred to driver, aggregation in single-threaded Python",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Understand why `.collect()` is an anti-pattern
#
# You've seen that this query collects all data to the driver and aggregates in Python.
#
# **Your task:** Think about:
# 1. What happens when the table grows to millions or billions of rows?
# 2. Why does this approach waste the distributed cluster resources?
# 3. What are the risks of driver memory exhaustion?
#
# > 💡 Hint: Spark's power comes from **distributed processing**. Moving all data to a single node defeats this purpose.
#
# Ready to see the distributed solution? Run the cells below!

# CELL ********************

# ==================================================================================================
# 3️⃣ FIX — Use distributed Spark aggregation instead of driver-side collect
# ==================================================================================================

print("✅ Running fixed query with distributed Spark aggregation...\n")

with benchmark_op("Local Driver work", "after", spark):
    inv = spark.table("bronze.inventory_transaction").select("line_id", "part_num", "quantity", "transaction_type")
    result_q3_fixed = (
        inv
        .withColumn(
            "signed_quantity",
            F.when(
                F.col("transaction_type").isin("CONSUMPTION", "ORDER_PICK", "SCRAP"),
                -F.abs(F.col("quantity").cast("int"))
            ).otherwise(F.col("quantity").cast("int"))
        )
        .groupBy("line_id")
        .agg(F.sum("signed_quantity").alias("net_quantity"))
        .orderBy(F.desc("net_quantity"))
    )

    # Convert to pandas only for final display (small result set)
    data = result_q3_fixed.toPandas()

display(data)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 3️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

result_rows = len(data)

print(json.dumps({
    "antiPattern": "Driver-side collect and Python aggregation",
    "sourceRows": TABLE_METRICS["inventory_transaction"]["rows"],
    "collectedRows": 0,  # No collect in fixed version
    "resultRows": result_rows,
    "improvement": "Distributed aggregation across all executors, only final results to driver",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ---
#
# # Query 4: Manufacturing event fan-out by event type
#
# **Table:** `manufacturing_event` — IoT telemetry with multiple joins and aggregations
#
# **What's wrong:** This query performs expensive joins and aggregations, then filters and collects results multiple times for different defect types. Each action triggers a complete re-scan of the source data because the expensive intermediate result is not cached.
#
# **Why it matters:**
# - Repeated scans waste I/O and compute resources
# - Each action re-executes the entire join and aggregation pipeline
# - Multiplies query cost by the number of downstream actions
# - Does not scale with data volume or number of output branches
#
# **Fix:** Cache the expensive intermediate result before branching into multiple actions
#
# ---

# CELL ********************

# Display table metrics before benchmark
print("\n=== Table Metrics: manufacturing_event ===")
show_metrics("bronze.web_order", "baseline")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Capture baseline with repeated scans (no caching)
# ============================================================

print("🐌 Running baseline with repeated scans (no caching)...\n")

with benchmark_op("Repeated Scans", "before", spark):
    mfg = spark.table(f"bronze.manufacturing_event").selectExpr("manufacturing_event.*")
    #mfg = mfg.filter(F.col("defect_detected") == True)  # Focus on defective events
    mfg = mfg.join(spark.table("bronze.colors"), F.col("color_id")==F.col("colors.id"))
    mfg = mfg.join(spark.table("bronze.parts"), F.col("manufacturing_event.part_num")==F.col("parts.part_num"))
    mfg = mfg.join(spark.table("bronze.part_categories").alias("p"), F.col("parts.part_cat_id")==F.col("p.id"))
    mfg = mfg.groupBy("p.name", "defect_detected", "defect_type").count()

    defect_type_col = F.col("defect_type")
    known_defects = ['color_streak', 'warp', 'sink_mark', 'short_shot']

    # First action - triggers full scan
    no_defect_df = mfg.filter(F.col("defect_detected") == False)
    no_defects = no_defect_df.groupBy("p.name").agg(F.sum("count").alias("total_count")).collect()

    # Second action - triggers another full scan
    defect_df = mfg.filter(defect_type_col.isin(known_defects))
    defects = defect_df.groupBy("p.name").agg(F.sum("count").alias("total_count")).collect()

    # Third action - triggers yet another full scan
    quarantine_df = mfg.filter(~defect_type_col.isin(known_defects))
    quarantine_rows = quarantine_df.groupBy("p.name").agg(F.sum("count").alias("total_count")).collect()
    

plan_1 = no_defect_df._jdf.queryExecution().executedPlan().toString()
plan_2 = defect_df._jdf.queryExecution().executedPlan().toString()
plan_3 = quarantine_df._jdf.queryExecution().executedPlan().toString()
print(f"No defects: {len(no_defects)} rows")
print(f"Known defects: {len(defects)} rows")
print(f"Quarantine: {len(quarantine_rows)} rows")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# =================================================================================================
# 4️⃣ DIAGNOSE — Prove the root cause is repeated scans without caching
# =================================================================================================

# Count FileScan operators in each plan to show repeated scans
file_scans_1 = plan_1.count("FileScan")
file_scans_2 = plan_2.count("FileScan")
file_scans_3 = plan_3.count("FileScan")
total_file_scans = file_scans_1 + file_scans_2 + file_scans_3

print(json.dumps({
    "antiPattern": "Repeated table scans without caching",
    "actions": 3,
    "fileScansPlan1": file_scans_1,
    "fileScansPlan2": file_scans_2,
    "fileScansPlan3": file_scans_3,
    "totalFileScans": total_file_scans,
    "impact": "Each action re-executes full join and aggregation pipeline",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Understand why repeated scans are inefficient
#
# You've seen that this query performs the same expensive operations three times.
#
# **Your task:** Check the Spark UI Jobs tab and think about:
# 1. How many jobs were created? (Hint: one per `.collect()` action)
# 2. Each job reads and processes the same source data — what's the waste?
# 3. What happens as you add more output branches (e.g., 10 defect types instead of 3)?
#
# > 💡 Hint: Spark's **caching** can materialize expensive intermediate results so downstream actions reuse them instead of recomputing.
#
# Ready to see the cached version? Run the cells below!

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Cache expensive intermediate result before multiple actions
# ==================================================================================================

print("✅ Running fixed version with caching...\n")

with benchmark_op("Repeated Scans", "after", spark):
    mfg = spark.table(f"bronze.manufacturing_event").selectExpr("manufacturing_event.*")
    #mfg = mfg.filter(F.col("defect_detected") == True)
    mfg = mfg.join(spark.table("bronze.colors"), F.col("color_id")==F.col("colors.id"))
    mfg = mfg.join(spark.table("bronze.parts"), F.col("manufacturing_event.part_num")==F.col("parts.part_num"))
    mfg = mfg.join(spark.table("bronze.part_categories").alias("p"), F.col("parts.part_cat_id")==F.col("p.id"))
    mfg = mfg.groupBy("p.name", "defect_detected", "defect_type").count()

    # Cache before branching into multiple actions
    mfg = mfg.cache()

    defect_type_col = F.col("defect_type")
    known_defects = ['color_streak', 'warp', 'sink_mark', 'short_shot']

    # First action - triggers scan and populates cache
    no_defect_df = mfg.filter(F.col("defect_detected") == False)
    no_defects_fixed = no_defect_df.groupBy("p.name").agg(F.sum("count").alias("total_count")).collect()

    # Second action - reads from cache (no re-scan)
    defect_df = mfg.filter(defect_type_col.isin(known_defects))
    defects_fixed = defect_df.groupBy("p.name").agg(F.sum("count").alias("total_count")).collect()

    # Third action - reads from cache (no re-scan)
    quarantine_df = mfg.filter(~defect_type_col.isin(known_defects))
    quarantine_rows_fixed = quarantine_df.groupBy("p.name").agg(F.sum("count").alias("total_count")).collect()

    mfg.unpersist()

plan_1_fixed = no_defect_df._jdf.queryExecution().executedPlan().toString()
plan_2_fixed = defect_df._jdf.queryExecution().executedPlan().toString()
plan_3_fixed = quarantine_df._jdf.queryExecution().executedPlan().toString()

print(f"No defects: {len(no_defects_fixed)} rows")
print(f"Known defects: {len(defects_fixed)} rows")
print(f"Quarantine: {len(quarantine_rows_fixed)} rows")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 4️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Count FileScan operators in each plan to show reduction in scans due to caching
file_scans_1 = plan_1.count("FileScan")
file_scans_2 = plan_2.count("FileScan")
file_scans_3 = plan_3.count("FileScan")
total_file_scans = file_scans_1 + file_scans_2 + file_scans_3

# Count InMemoryTableScan operators to confirm cache usage
in_memory_scans_1 = plan_1_fixed.count("InMemoryTableScan")
in_memory_scans_2 = plan_2_fixed.count("InMemoryTableScan")
in_memory_scans_3 = plan_3_fixed.count("InMemoryTableScan")
total_in_memory = in_memory_scans_1 + in_memory_scans_2 + in_memory_scans_3

print(json.dumps({
    "antiPattern": "Repeated table scans without caching",
    "actions": 3,
    "fileScansPlan1": file_scans_1,
    "fileScansPlan2": file_scans_2,
    "fileScansPlan3": file_scans_3,
    "totalFileScans": total_file_scans,
    "inMemoryScansPlan1": in_memory_scans_1,
    "inMemoryScansPlan2": in_memory_scans_2,
    "inMemoryScansPlan3": in_memory_scans_3,
    "totalInMemoryScans": total_in_memory,
    "improvement": "After first action, subsequent actions read from in-memory cache",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# Show the plan for the second action (should show InMemoryTableScan instead of FileScan)
print("Plan for second action (should use InMemoryTableScan):")
defect_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ---
#
# # Query 5: Quality inspection pass rates
#
# **Tables:** `quality_inspection` and `production_order`
#
# **What's wrong:** This query performs a `crossJoin` without a proper join predicate between the two tables. This creates a Cartesian product where every row from one table is combined with every row from the other table, resulting in exponentially more rows than needed.
#
# **Why it matters:**
# - Cartesian joins create N × M rows, which can be massive
# - Causes extreme memory pressure and potential OOM errors
# - Very slow execution even on small datasets
# - Often indicates a logic bug (missing join condition)
#
# **Fix:** Add proper join predicate to create an equi-join
#
# ---

# CELL ********************

# Display table metrics before benchmark
print("\n=== Table Metrics: quality_inspection ===")
show_metrics("bronze.web_order", "baseline")
print("\n=== Table Metrics: production_order ===")
show_metrics("bronze.web_order", "baseline")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Capture baseline with Cartesian join
# ============================================================

print("🐌 Running baseline with Cartesian join (no join predicate)...\n")

qi = spark.table(f"bronze.quality_inspection").select(
    F.col("quality_inspection.production_order_id").alias("qi_production_order_id"),
    F.col("quality_inspection.result").alias("inspection_result"),
    F.col("quality_inspection.pass_count").alias("pass_count"),
    F.col("quality_inspection.sample_size").alias("sample_size"),
)
po = spark.table(f"bronze.production_order").select(
    F.col("production_order.production_order_id").alias("po_production_order_id"),
    F.col("production_order.machine_id").alias("machine_id"),
    F.col("production_order.status").alias("order_status"),
)
estimated_pairs = TABLE_METRICS["quality_inspection"]["rows"] * TABLE_METRICS["production_order"]["rows"]
print(f"Estimated Cartesian pairs: {estimated_pairs:,}")

with benchmark_op("Cartesian Join", "before", spark):
    result_q5 = (
        qi.crossJoin(po)
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate")
        )
        .orderBy(F.desc("joined_rows"))
    )
    rows_q5 = result_q5.toPandas()
display(rows_q5)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# =================================================================================================
# 5️⃣ DIAGNOSE — Prove the root cause is Cartesian join due to missing predicate
# =================================================================================================

executed_plan = result_q5._jdf.queryExecution().executedPlan().toString()
executed_join = "BroadcastNestedLoopJoin" if "BroadcastNestedLoopJoin" in executed_plan else "CartesianProduct" if "CartesianProduct" in executed_plan else "None"

print(json.dumps({
    "antiPattern": "Missing join predicate / Cartesian join",
    "qualityRows": TABLE_METRICS["quality_inspection"]["rows"],
    "productionOrderRows": TABLE_METRICS["production_order"]["rows"],
    "estimatedCartesianPairs": estimated_pairs,
    "resultRows": len(rows_q5),
    "expectedPlanSignal": "CartesianProduct or BroadcastNestedLoopJoin",
    "executedJoin": executed_join,
    "impact": f"Processing {estimated_pairs:,} row pairs instead of actual matches",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Identify the Cartesian join in the Spark Plan
#
# You've seen that this query processes way more rows than necessary.
#
# **Your task:** Check the Spark Plan to confirm:
# 1. Look for `CartesianProduct` or `BroadcastNestedLoopJoin` operators
# 2. Notice there's no join condition (no equality predicate)
# 3. Check the Spark UI to see the massive shuffle size
#
# > 💡 Hint: A proper join should show `SortMergeJoin` or `BroadcastHashJoin` with an equality condition.
#
# Try it in the cell below!

# CELL ********************

# =================================================================================================
# 5️⃣ DIAGNOSE — (Optional) Look for Cartesian join in the Physical Plan
# =================================================================================================

result_q5.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ==================================================================================================
# 5️⃣ FIX — Add proper join predicate to create equi-join
# ==================================================================================================

print("✅ Running fixed query with proper join predicate...\n")

qi = spark.table(f"bronze.quality_inspection").select(
    F.col("quality_inspection.production_order_id").alias("qi_production_order_id"),
    F.col("quality_inspection.result").alias("inspection_result"),
    F.col("quality_inspection.pass_count").alias("pass_count"),
    F.col("quality_inspection.sample_size").alias("sample_size"),
)
po = spark.table(f"bronze.production_order").select(
    F.col("production_order.production_order_id").alias("po_production_order_id"),
    F.col("production_order.machine_id").alias("machine_id"),
    F.col("production_order.status").alias("order_status"),
)

with benchmark_op("Cartesian Join", "after", spark):
    result_q5_fixed = (
        qi
        .crossJoin(po)
        .where(F.col("qi_production_order_id") == F.col("po_production_order_id"))
        #.join(po, F.col("qi_production_order_id") == F.col("po_production_order_id"))  Alternative option
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate")
        )
        .orderBy(F.desc("joined_rows"))
    )
    rows_q5_fixed = result_q5_fixed.toPandas()
display(rows_q5_fixed)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 5️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================


executed_plan_fixed = result_q5_fixed._jdf.queryExecution().executedPlan().toString()
executed_join_fixed = "SortMergeJoin" if "SortMergeJoin" in executed_plan_fixed else "BroadcastHashJoin" if "BroadcastHashJoin" in executed_plan_fixed else "None"

print(json.dumps({
    "antiPattern": "Missing join predicate / Cartesian join",
    "qualityRows": TABLE_METRICS["quality_inspection"]["rows"],
    "productionOrderRows": TABLE_METRICS["production_order"]["rows"],
    "baselineJoin": executed_join,
    "fixedJoin": executed_join_fixed,
    "baselineResultRows": len(rows_q5),
    "fixedResultRows": len(rows_q5_fixed),
    "improvement": f"Changed from {executed_join} to {executed_join_fixed} with proper join condition",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

result_q5_fixed.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ---
#
# # Query 6: Event aggregation with unnecessary caching
#
# **Table:** `manufacturing_event` — IoT telemetry data
#
# **What's wrong:** This query performs a simple transformation and aggregation but adds unnecessary caching in the middle. Caching forces materialization and can cause the Native Execution Engine (NEE) to fall back to slower row-based processing for certain operations.
#
# **Why it matters:**
# - Unnecessary cache() adds overhead without benefit
# - Can trigger Native Execution Engine fallbacks
# - Increases memory pressure on the cluster
# - Caching should only be used when the same data is reused multiple times
#
# **Fix:** Remove unnecessary cache when data is used only once
#
# ---

# CELL ********************

# Display table metrics before benchmark
print("\n=== Table Metrics: manufacturing_event ===")
show_metrics("bronze.manufacturing_event", "baseline")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 6️⃣ BENCHMARK — Capture baseline with unnecessary cache
# ============================================================

print("🐌 Running baseline with unnecessary cache...\n")

with benchmark_op("Unnecessary Cache", "before", spark):

    mfg = spark.table("bronze.manufacturing_event")
    transformed = mfg.withColumn("event_day", F.substring(F.col("manufacturing_event.timestamp"), 1, 10))
    transformed.cache()  # Unnecessary cache - data is only used once

    result_q6 = (
        transformed
        .groupBy("event_day")
        .agg(F.count("*").alias("events"))
        .orderBy("event_day")
    )
    rows_q6 = result_q6.toPandas()
    transformed.unpersist()

display(rows_q6)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# =================================================================================================
# 6️⃣ DIAGNOSE — Prove the root cause is unnecessary cache causing NEE fallback
# =================================================================================================

plan_q6 = result_q6._jdf.queryExecution().executedPlan().toString()
nee_fallback_summary_q6 = extract_nee_fallbacks(plan_q6)
has_in_memory_scan = "InMemoryTableScan" in plan_q6

print(json.dumps({
    "antiPattern": "Unnecessary cache leading to NEE fallback",
    "sourceRows": TABLE_METRICS["manufacturing_event"]["rows"],
    "hasInMemoryScan": has_in_memory_scan,
    "neeFallbackBlockCount": nee_fallback_summary_q6["blockCount"],
    "neeFallbackOperatorCount": nee_fallback_summary_q6["operatorCount"],
    "neeFallbackOperators": nee_fallback_summary_q6["operators"],
    "impact": "Cache forces materialization and may cause NEE to fall back to row-based processing",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Identify unnecessary cache and NEE fallback
#
# You've seen that caching doesn't always improve performance.
#
# **Your task:** Check the Spark Plan to understand:
# 1. Look for `InMemoryTableScan` — this indicates cached data
# 2. Look for `RowToVeloxColumnar` and `VeloxColumnarToRow` — these indicate NEE fallback
# 3. Consider: is the cached data being reused? If not, the cache is wasteful
#
# > 💡 Hint: Cache is beneficial when you perform **multiple actions** on the same expensive DataFrame. For single-use, it adds overhead.
#
# Try it in the cell below!

# CELL ********************

# =================================================================================================
# 6️⃣ DIAGNOSE — (Optional) Look for InMemoryTableScan and NEE fallbacks in the Physical Plan
# =================================================================================================

result_q6.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ==================================================================================================
# 6️⃣ FIX — Remove unnecessary cache for single-use transformation
# ==================================================================================================

print("✅ Running fixed query without unnecessary cache...\n")

with benchmark_op("Unnecessary Cache", "after", spark):

    mfg = spark.table("bronze.manufacturing_event")
    transformed = mfg.withColumn("event_day", F.substring(F.col("manufacturing_event.timestamp"), 1, 10))

    result_q6_fixed = (
        transformed
        .groupBy("event_day")
        .agg(F.count("*").alias("events"))
        .orderBy("event_day")
    )
    rows_q6_fixed = result_q6_fixed.toPandas()

display(rows_q6_fixed)

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# 6️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

plan_q6_fixed = result_q6_fixed._jdf.queryExecution().executedPlan().toString()
nee_fallback_summary_q6_fixed = extract_nee_fallbacks(plan_q6_fixed)
has_in_memory_scan_fixed = "InMemoryTableScan" in plan_q6_fixed

print(json.dumps({
    "antiPattern": "Unnecessary cache leading to NEE fallback",
    "sourceRows": TABLE_METRICS["manufacturing_event"]["rows"],
    "baselineHasInMemoryScan": has_in_memory_scan,
    "fixedHasInMemoryScan": has_in_memory_scan_fixed,
    "baselineNeeFallbackCount": nee_fallback_summary_q6["operatorCount"],
    "fixedNeeFallbackCount": nee_fallback_summary_q6_fixed["operatorCount"],
    "baselineNeeFallbackOperators": nee_fallback_summary_q6["operators"],
    "fixedNeeFallbackOperators": nee_fallback_summary_q6_fixed["operators"],
    "improvement": "Removed unnecessary cache, allowing streamlined execution",
}, indent=2))

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

result_q6_fixed.explain(mode="formatted")

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# ============================================================
# SUMMARY — All benchmark results
# ============================================================

print("=" * 62)
print("  🏆  PERFORMANCE IMPACT SUMMARY")
print("=" * 62)

print_all_scenarios()

# METADATA ********************

# META {
# META   "microsoft": {
# META     "language": "python",
# META     "language_group": "synapse_pyspark"
# META   }
# META }
