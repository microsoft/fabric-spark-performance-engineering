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

# MARKDOWN ********************

# ## Exercise summary
# 
# | Exercise | Scenario | Expected performance signal |
# |---|---|---|
# | 1 — Predicate pushdown | A daily defect-rate dashboard derives a string day before filtering `manufacturing_event`. | Fewer files read / filter pushed to FileScan; substring disappears from the filter path. |
# | 2 — Python UDFs → native expressions / NEE | A top-customer query calculates line totals and order days with Python UDFs. | No `BatchEvalPython` / Python boundary removed; NEE fallback blocks drop to 0. |
# | 3 — Driver `collect()` / `toPandas()` and driver OOM | An inventory workflow collects raw transactions to the driver and aggregates in Python. | Driver result size shrinks / raw-row collect avoided; no OOM risk from full-result transfer. |
# | 4 — Cartesian / missing join key | A pass-rate query omits the production-order join key, and a cycle-time variant uses an inequality self-join. | `CartesianProduct` / nested-loop work replaced by equi-join or window logic; runtime and pair counts drop. |

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

# Setup: reset the work schema, validate sources, and capture baseline metrics.
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SOURCE_SCHEMA = "bronze"
WORK_SCHEMA = "opt_code"

spark.conf.set("spark.microsoft.delta.parallelSnapshotLoading.enabled", "true")
spark.conf.set("spark.microsoft.delta.snapshot.driverMode.enabled", "true")
spark.conf.set("spark.synapse.vegas.useCache", "false")
spark.catalog.clearCache()

reset_work_schema(WORK_SCHEMA)

expected_tables = [
    "manufacturing_event",
    "web_order",
    "inventory_transaction",
    "quality_inspection",
    "production_order",
]
require_tables(expected_tables, SOURCE_SCHEMA)

print("Spark application ID:", spark.sparkContext.applicationId)
print("Source schema:", SOURCE_SCHEMA, "| Work schema:", WORK_SCHEMA)
print("\n=== Delta table metrics from DESCRIBE DETAIL ===")
for table_name in expected_tables:
    show_metrics(table_ref(table_name, SOURCE_SCHEMA), "source")

TABLE_METRICS = {name: table_metrics(name, SOURCE_SCHEMA) for name in expected_tables}
print(json.dumps(TABLE_METRICS, default=str, indent=2))
print("Recent manufacturing_event history:")
print(json.dumps(recent_history("manufacturing_event", SOURCE_SCHEMA), default=str, indent=2))

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

# ============================================================
# 1️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# The baseline derives a string day before filtering, which weakens pushdown.
print("🐌 Running baseline predicate-pushdown query...\n")

mfg = spark.table(table_ref("manufacturing_event")).selectExpr("manufacturing_event.*")
latest_day = mfg.select(F.max(F.to_date("timestamp")).alias("d")).collect()[0]["d"]

with benchmark_op("Predicate Pushdown", "before", spark):
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
    )
    predicate_before_pdf = result_predicate_before.toPandas()

display(predicate_before_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 1️⃣ DIAGNOSE — Prove the root cause is weak file pruning and missing pushdown
# =================================================================================================

# Capture the FileScan filters and read-file count that explain the slow baseline.
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
print(json.dumps(predicate_before_evidence, default=str, indent=2))

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

# ==================================================================================================
# 1️⃣ FIX — Filter with native column functions before aggregating
# ==================================================================================================

# The fixed query applies the date predicate before deriving presentation columns.
print("✅ Running fixed predicate-pushdown query...\n")

with benchmark_op("Predicate Pushdown", "after", spark):
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
    )
    predicate_after_pdf = result_predicate_after.toPandas()

display(predicate_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 1️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Verify the fixed plan signals and record the before/after evidence.
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

# ============================================================
# 2️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# The baseline uses Python UDFs, adding a JVM↔Python boundary to the plan.
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

with benchmark_op("Python UDF Overhead", "before", spark):
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
    )
    udf_before_pdf = result_udf_before.toPandas()

display(udf_before_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 2️⃣ DIAGNOSE — Prove the root cause is Python execution and NEE fallback risk
# =================================================================================================

# Look for BatchEvalPython/PythonUDF nodes and any Velox fallback blocks.
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
print(json.dumps(udf_before_evidence, default=str, indent=2))

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

# ==================================================================================================
# 2️⃣ FIX — Replace Python UDFs with native Spark SQL expressions
# ==================================================================================================

# Native expressions keep execution inside Spark and avoid the Python boundary.
print("✅ Running fixed query with native expressions...\n")

line_total_col = F.coalesce(
    F.col("line.extended_price").cast("double"),
    F.col("line.quantity").cast("double") * F.col("line.unit_price").cast("double"),
    F.lit(0.0),
)
order_day_col = F.regexp_extract("order_date", r"(\d{4}-\d{2}-\d{2})", 1)

with benchmark_op("Python UDF Overhead", "after", spark):
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
    )
    udf_after_pdf = result_udf_after.toPandas()

display(udf_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Confirm the fixed plan no longer has Python UDF nodes or fallback operators.
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

# ============================================================
# 3️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# The baseline collects every raw transaction to the driver before aggregating.
print("🐌 Running baseline query with driver-side collect...\n")

inv = spark.table(table_ref("inventory_transaction")).select(
    "line_id", "part_num", "quantity", "transaction_type"
)
print(f"About to collect {TABLE_METRICS['inventory_transaction']['rows']:,} inventory rows to the driver.")
print("spark.driver.maxResultSize =", spark.conf.get("spark.driver.maxResultSize"))

start = time.time()
with benchmark_op("Driver Collect", "before", spark):
    collected_inventory = inv.collect()
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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3️⃣ DIAGNOSE — Prove the root cause is raw-row transfer to the driver
# =================================================================================================

# Record how many rows crossed to the driver and why that creates OOM risk.
driver_before_evidence = {
    "antiPattern": "Driver-side collect and Python aggregation",
    "sourceRows": TABLE_METRICS["inventory_transaction"]["rows"],
    "collectedRows": len(collected_inventory),
    "resultRows": len(driver_result_rows),
    "driverOomRisk": "Raw rows are transferred to the driver; toPandas has the same risk profile.",
}
print(json.dumps(driver_before_evidence, default=str, indent=2))

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

# ==================================================================================================
# 3️⃣ FIX — Aggregate on executors and collect only the small grouped result
# ==================================================================================================

# Spark computes net inventory by line before the driver receives the display result.
print("✅ Running fixed query with distributed aggregation...\n")

with benchmark_op("Driver Collect", "after", spark):
    result_driver_after = (
        starter_signed_inventory
        .groupBy("line_id")
        .agg(F.sum("signed_quantity").alias("net_quantity"))
        .orderBy(F.desc("net_quantity"))
    )
    driver_after_pdf = result_driver_after.toPandas()

display(driver_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Verify the fixed path collects only the final grouped rows.
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

# ============================================================
# 4️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# The baseline omits the equality key and creates N × M join work.
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

with benchmark_op("Cartesian Join", "before", spark):
    result_cartesian_before = (
        qi.crossJoin(po)
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
        )
        .orderBy(F.desc("joined_rows"))
    )
    cartesian_before_pdf = result_cartesian_before.toPandas()

display(cartesian_before_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 4️⃣ DIAGNOSE — Prove the root cause is a Cartesian or nested-loop join
# =================================================================================================

# Inspect the executed plan and compare expected pair counts with displayed results.
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
print(json.dumps(cartesian_before_evidence, default=str, indent=2))

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

# ==================================================================================================
# 4️⃣ FIX — Add the production_order_id equality join condition
# ==================================================================================================

# The fixed query joins inspections to production orders by the real key.
print("✅ Running fixed query with the correct join predicate...\n")

with benchmark_op("Cartesian Join", "after", spark):
    result_cartesian_after = (
        qi.join(po, join_condition)
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
        )
        .orderBy(F.desc("joined_rows"))
    )
    cartesian_after_pdf = result_cartesian_after.toPandas()

display(cartesian_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Confirm the fixed plan uses a real equi-join and processes matched rows only.
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
print(json.dumps({
    "antiPattern": "Inequality self-join instead of previous-row window",
    "filteredRows": cycle_rows,
    "estimatedPairs": cycle_rows * cycle_rows,
    "planHasCartesianSignal": "CartesianProduct" in bad_cycle_plan or "BroadcastNestedLoopJoin" in bad_cycle_plan,
}, default=str, indent=2))
bad_cycle_join.explain(mode="formatted")

w = Window.partitionBy("machine_id").orderBy("timestamp")
with benchmark_op("Cartesian Window Rewrite", "after", spark):
    fixed_cycle_delta = (
        cycle_events
        .withColumn("prev_cycle_time_ms", F.lag("cycle_time_ms").over(w))
        .withColumn("delta_ms", F.col("cycle_time_ms") - F.col("prev_cycle_time_ms"))
        .filter(F.col("delta_ms").isNotNull())
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
