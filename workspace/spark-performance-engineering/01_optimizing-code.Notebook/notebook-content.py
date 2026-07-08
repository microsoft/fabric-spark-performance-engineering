# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "28f1e957-ea23-49e8-846b-be0d8a67412e",
# META       "default_lakehouse_name": "toy_bricks",
# META       "default_lakehouse_workspace_id": "7fc5eff4-7153-4da9-b909-54981a3ffcdb",
# META       "known_lakehouses": [
# META         {
# META           "id": "28f1e957-ea23-49e8-846b-be0d8a67412e"
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "3cdd45c3-659b-bb60-4877-86d399fb9cb3",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # **Module 1 — Optimizing Code**
# 
# Welcome to the first Fabric Jumpstart Spark performance lab for **Toy Brick Manufacturing**.
# 
# ## **What this module teaches**
# 
# This module teaches you to recognize and fix **code-level** anti-patterns: the table design and cluster are fine, but the query as written is wrong or wasteful. You will also start using the diagnostic toolkit that Modules 2 and 3 reuse: Spark UI, `explain()` and physical plans, Delta metadata from `DESCRIBE DETAIL` / `DESCRIBE HISTORY`, and `inputFiles()`.
# 
# > Litmus test: if the fix is a diff to the transformation logic, it belongs here. Storage fixes are Module 2; execution, AQE, caching, and repartitioning fixes are Module 3.

# MARKDOWN ********************

# ## **Exercise summary**
# 
# | Exercise | Scenario | Outcome |
# |---|---|---|
# | 1 — Predicate pushdown | A daily defect-rate dashboard derives a string day before filtering `manufacturing_event`. | Fewer files read / filter pushed to FileScan; substring disappears from the filter path. |
# | 2 — De-duplicate on the key, not the whole row | A "distinct orders" step calls `dropDuplicates()` with no subset, shuffling every column including the nested `order_lines` array. | FileScan ReadSchema shrinks; only the key columns are shuffled. |
# | 3 — Prune before a window / row_number | A "latest order per customer" runs `row_number()` over the wide `web_order` row, shuffling and sorting every column — including the nested `order_lines` array — through the window. | Only the key columns cross the window Exchange/Sort; `order_lines` is pruned; identical latest-row result. |
# | 4 — One pass, not many | A per-type report loops over each `transaction_type`, filtering and aggregating `inventory_transaction` once per type and unioning the results. | A single `groupBy` scans the table once; N scans collapse to 1; identical per-type totals. |
# | 5 — Cartesian / missing join key | A pass-rate query omits the production-order join key, and a cycle-time variant uses an inequality self-join. | `CartesianProduct` / nested-loop work replaced by equi-join or window logic; runtime and pair counts drop. |
# | 6 — Python UDFs → native expressions | A top-customer query computes line totals/order days with Python UDFs (NEE disabled to expose the JVM boundary). | `BatchEvalPython` removed after rewriting UDFs as native expressions; NEE makes the rewrite optional (see Module 3). |
# | 7 — `withColumn` loop → `withColumns` | Feature engineering adds ~50 derived columns by chaining `.withColumn()`, one per column. | Analyzed plan collapses from ~50 nested `Project` nodes to 1; identical columns and rows. |
# | 8 — Schema inference vs a declared schema | Reading the JSON landing zone with `spark.read.json()` infers types by scanning the files before the query runs. | Declaring a `StructType` schema removes the eager inference scan; same columns, faster read. |
# | 9 — Driver `collect()` / `toPandas()` and driver OOM | An inventory workflow collects raw transactions to the driver and aggregates in Python. Run last so a driver crash cannot abort earlier exercises. | Driver result size shrinks / raw-row collect avoided; no OOM risk from full-result transfer. |


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
from pyspark.testing import assertDataFrameEqual

SOURCE_SCHEMA = "bronze"

# Disable Intelligent Cache to avoid skewing benchmark results
spark.conf.set("spark.synapse.vegas.useCache", "false")
spark.catalog.clearCache()

expected_tables = [
    "manufacturing_event",
    "web_order",
    "inventory_transaction",
    "quality_inspection",
    "production_order",
    "parts",
]
require_tables(expected_tables, SOURCE_SCHEMA)

print("\n=== Delta table metrics from DESCRIBE DETAIL ===")
for table_name in expected_tables:
    show_metrics(table_ref(table_name, SOURCE_SCHEMA), "source")

TABLE_METRICS = {name: table_metrics(name, SOURCE_SCHEMA) for name in expected_tables}
print(json.dumps(TABLE_METRICS, default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## **Exercise 1 — Predicate pushdown**
# 
# **Problem:** The daily defect-rate query transforms the timestamp into a string before filtering. That makes Spark scan more of `manufacturing_event` than the dashboard needs.
# 
# **Why it matters:** Full scans waste I/O and make a small daily dashboard behave like a whole-factory history query.
# 
# **Fix in one line:** Filter with a native timestamp/date expression first, then derive presentation columns.

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
    display(result_predicate_before)

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

# ### 🎯 **Challenge: Check the Spark Plan and Spark UI to diagnose the problem**
# 
# You've seen that the query is slow, and you suspect it's because of the large number of files and the filter not being pushed down. You want to confirm this by checking the Spark Plan and the Spark UI.
# 
# **Your task:** Check the Spark Plan and Spark UI to confirm that the query is doing a full scan of all files and that the filter on `event_day` is not being pushed down to the file scan level.
# 
# > 💡 Hint: **Explain** methods in Spark can help you understand the physical plan and see if filters are being pushed down. Look for **PushedFilters** in the **DefaultDeltaScanTransformer** node of the plan. 
# 
# Try it in the cell below!

# CELL ********************

# Starter: inspect the baseline physical plan and a runnable native-filter sketch.
result_predicate_before.explain(mode="formatted")

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
        .filter(F.col("timestamp").startswith(F.lit(str(latest_day)))) # filter is now the first operation
        .withColumn("event_day", F.to_date("timestamp"))
        .groupBy("event_day", F.col("machine_id"))
        .agg(
            F.count("*").alias("events"),
            F.sum(F.col("defect_detected").cast("int")).alias("defects"),
            (F.sum(F.col("defect_detected").cast("int")) / F.count("*")).alias("defect_rate"),
        )
        .orderBy(F.desc("defect_rate"))
    )
    display(result_predicate_after)

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
    "antiPattern": "Full scan / weak pushdown due to string timestamp transformation",
    "baselineReadFiles": predicate_before_evidence["readFiles"],
    "fixedReadFiles": len(result_predicate_after.inputFiles()),
    "fixedDataFilters": predicate_after_scan["dataFilters"],
    "fixedPushedFilters": predicate_after_scan["pushedFilters"],
    "fixedPlanHasSubstring": "substring" in plan_string(result_predicate_after).lower(),
}
print(json.dumps(predicate_after_evidence, default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ✅ Verify with the query plan that the `DefaultDeltaScanTransformer` node of the new plan has `PushedFilters`. This enables file skipping on read to minimize the amount of data that must be filtered post scan.

# CELL ********************

result_predicate_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# Filtering on the **raw** `timestamp` column let Delta push the predicate into the file scan — notice `PushedFilters` now appears in the `DefaultDeltaScanTransformer` node. With the predicate at the scan, Spark uses each file's min/max statistics to **skip files** that can't contain the target day, so it reads a fraction of the table instead of all of it.
# 
# The baseline derived `event_day` with `substring(...)` *before* filtering. That hides the real column behind an expression the scan can't reason about, so pushdown is lost and Spark reads every file, then throws most rows away. The rule: **filter on native columns first, derive presentation columns after.**
# 
# > 📝 **Note:** File skipping is only as good as your data layout. Module 2 covers the table-side levers (compaction, clustering, stats) that make pushdown even more effective.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 2 — De-duplicate on the key, not the whole row**
# 
# **Problem:** A "distinct orders" step calls `distinct()` or `dropDuplicates` with no column list, so Spark uses every column as the de-duplication key — including the nested `order_lines` array.
# 
# **Why it matters:** `distinct()` compares all column values. `dropDuplicates()` with no subset also reads the full row schema and shuffles wide, nested data even though the business key is just a couple of columns.
# 
# **Fix in one line:** Project only the columns you need or pass the key subset to `dropDuplicates` there's less comparison and aggregation work.

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# Baseline: distinct() must aggregate every column, including order_lines.
orders_wide = spark.table(table_ref("web_order")).selectExpr("web_order.*")

with benchmark_op("Column pruning / projection", "before", spark):
    proj_before_df = orders_wide.distinct()
    proj_before_count = proj_before_df.count()

print("Distinct full rows:", proj_before_count)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 2️⃣ DIAGNOSE — Prove the whole row (including nested arrays) is shuffled to de-duplicate
# =================================================================================================
def get_agg_key_count(node_name: str, df: DataFrame) -> int:
    plan = plan_string(df)
    key_name = "keys" if node_name == "HashAggregate" else "key"
    first_agg = plan.split(f"{node_name}({key_name}=[", 1)[1] 
    keys_text = first_agg.split("], functions=", 1)[0]
    key_count = len([k.strip() for k in keys_text.split(",") if k.strip()])
    return key_count

# The FileScan ReadSchema lists every column and the Exchange carries them all.
proj_before_plan = plan_string(proj_before_df)
print(json.dumps({
    "antiPattern": "dropDuplicates() with no subset uses all columns as the key",
    "baselineColumnsScannedAndShuffled": get_agg_key_count("HashAggregate", proj_before_df)
}, default=str, indent=2))
proj_before_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 **Challenge: Deduplicate on specific keys**
# 
# You only need distinct `customer_id` / `order_date` pairs. Rewrite the de-duplication so Spark reads and shuffles just those columns — either `select(...)` before `dropDuplicates()`, or pass the key subset to `dropDuplicates([...])`. Compare the scanned columns and the plan.

# CELL ********************

# Starter: reduce the columns that reach the shuffle, then de-duplicate.
proj_starter_df = orders_wide.dropDuplicates()  # TODO: project the key columns first
print("Columns shuffled:", len(proj_starter_df.columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 2️⃣ FIX — Project the key columns before de-duplicating
# ==================================================================================================

# Selecting the business key first prunes the scan and shrinks the exchange.
with benchmark_op("Column pruning / projection", "after", spark):
    proj_after_df = orders_wide.dropDuplicates(["customer_id", "order_date"])
    proj_after_count = proj_after_df.count()

print("Distinct key rows:", proj_after_count)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# The fix reads far fewer columns and shuffles a narrow row.
print(json.dumps({
    "antiPattern": "dropDuplicates() over all columns",
    "baselineColumnsShuffled": get_agg_key_count("HashAggregate", proj_before_df),
    "fixedColumnsShuffled": get_agg_key_count("SortAggregate", proj_after_df),
    "baselineDistinctFullRows": proj_before_count,
    "fixedDistinctKeyRows": proj_after_count,
}, default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ✅ Verify that the `SortAggregate` node of the plan uses only 2 columns as keys instead of 11 that the `HashAggregate` used.

# CELL ********************

proj_after_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# `dropDuplicates()` (and `distinct()`) with no column list uses **every** column as the de-duplication key. Spark therefore scans the full row schema and shuffles all of it — including the nested `order_lines` array — through a `HashAggregate`, even though the business key is just `customer_id` + `order_date`.
# 
# Passing the key subset `dropDuplicates(["customer_id", "order_date"])` prunes the scan to those two columns and shrinks the exchange to a narrow row. The plan switches to a `SortAggregate` keyed on 2 columns instead of a `HashAggregate` over 11, and the distinct result is the business-meaningful one.
# 
# > 📝 **Note:** "Read/shuffle only what you need" is the recurring theme of the next two exercises too — projection is one of the cheapest wins in Spark.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 3 — Prune columns before a window / row_number**
# **Problem:** A "latest order per customer" step runs `row_number()` over the wide `web_order` row. The window has to shuffle (Exchange) and sort every column — including the nested `order_lines` array — even though only a few fields are needed.
# **Why it matters:** A window with `partitionBy` / `orderBy` forces an Exchange + Sort. Whatever columns are on the DataFrame ride through that shuffle and sort, so carrying an unused nested array inflates the shuffle and sort spill for nothing.
# **Fix in one line:** Project just the columns the window needs before applying it, so the Exchange and Sort move a narrow row.

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Capture baseline query time
# ============================================================
from pyspark.sql import Window

# Baseline: row_number() over the wide, nested frame — every column rides through the window.
w_latest = Window.partitionBy("customer_id").orderBy(F.col("order_date").desc())
orders_wide = spark.table(table_ref("web_order")).selectExpr("web_order.*")
latest_before = orders_wide.withColumn("rn", F.row_number().over(w_latest)).filter("rn = 1")

# noop sink forces full execution (window shuffle + sort) without pulling rows to the driver.
with benchmark_op("Prune before window", "before", spark):
    latest_before.write.format("noop").mode("overwrite").save()

print("Columns through the window:", len(latest_before.columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3️⃣ DIAGNOSE — Prove the window shuffles and sorts the unused nested array
# =================================================================================================

# The Exchange/Sort feeding the Window carries every column, including order_lines.
before_win_plan = plan_string(latest_before)
print(json.dumps({
    "antiPattern": "row_number() window applied to the full wide row",
    "columnsThroughWindow": len(latest_before.columns),
    "windowShufflesNestedOrderLines": "order_lines" in before_win_plan,
}, default=str, indent=2))
latest_before.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 **Challenge: Project a subset of columns**
# You only need `customer_id`, `order_date`, and `order_total` to pick the latest order. Project those columns before the `row_number()` window so the Exchange and Sort move a narrow row and `order_lines` never enters the shuffle. Confirm the latest-row result is unchanged.

# CELL ********************

# Starter: project the columns the window needs BEFORE applying it.
orders_keys_starter = spark.table(table_ref("web_order")).selectExpr("web_order.*")  # TODO: select customer_id, order_date, order_total
print("Columns that would enter the window:", len(orders_keys_starter.columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3️⃣ FIX — Project the window's columns before applying it
# ==================================================================================================

# Selecting first keeps order_lines out of the Exchange and Sort.
orders_keys = spark.table(table_ref("web_order")).select(
    F.col("web_order.customer_id").alias("customer_id"),
    F.col("web_order.order_date").alias("order_date"),
    F.col("web_order.order_total").alias("order_total"),
)
latest_after = orders_keys.withColumn("rn", F.row_number().over(w_latest)).filter("rn = 1")

with benchmark_op("Prune before window", "after", spark):
    latest_after.write.format("noop").mode("overwrite").save()

print("Columns through the window:", len(latest_after.columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Fewer columns cross the window Exchange/Sort, and the latest order per customer is identical.
after_win_plan = plan_string(latest_after)
before_keys = {r["customer_id"]: (str(r["order_date"]), round(float(r["order_total"] or 0), 2))
               for r in latest_before.select("customer_id", "order_date", "order_total").collect()}
after_keys = {r["customer_id"]: (str(r["order_date"]), round(float(r["order_total"] or 0), 2))
              for r in latest_after.select("customer_id", "order_date", "order_total").collect()}
print(json.dumps({
    "antiPattern": "windowing over unpruned columns",
    "baselineColumnsThroughWindow": len(latest_before.columns),
    "fixedColumnsThroughWindow": len(latest_after.columns),
    "baselineWindowShufflesOrderLines": "order_lines" in before_win_plan,
    "fixedWindowShufflesOrderLines": "order_lines" in after_win_plan,
    "sameLatestPerCustomer": before_keys == after_keys,
}, default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ✅ Verify that `fixedWindowShufflesOrderLines` is `False` and `fixedColumnsThroughWindow` is `4` (the three keys plus `rn`), while `sameLatestPerCustomer` is `True`.

# CELL ********************

latest_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# A window with `partitionBy` / `orderBy` forces an **Exchange (shuffle) + Sort**. Whatever columns are on the DataFrame ride through that shuffle and sort — so the baseline carried the entire wide row, including the nested `order_lines` array, just to pick the latest order per customer.
# 
# Projecting the three needed columns (`customer_id`, `order_date`, `order_total`) **before** applying `row_number()` keeps `order_lines` out of the shuffle and sort entirely. Only 4 columns cross the window (the 3 keys plus `rn`), the sort spills less, and the latest-row result is identical.
# 
# > 📝 **Note:** The same idea applies to joins and aggregations — prune columns as early as possible so every downstream shuffle moves a narrow row.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 4 — One pass, not many (avoid repeated scans)**
# **Problem:** A per-transaction-type report is built by looping over each `transaction_type`, filtering and aggregating `inventory_transaction` once per type, then unioning the results — so the table is scanned once per category.
# 
# **Why it matters:** Spark is excellent at joins and aggregations, but it will *not* merge independent filtered passes into a single read. Each pass re-scans the table, so the work grows with the number of categories instead of staying a single scan.
# 
# **Fix in one line:** Compute every bucket in one `groupBy(transaction_type)` (or conditional aggregation) so the table is scanned once.

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Capture baseline query time
# ============================================================
from functools import reduce

inv4 = spark.table(table_ref("inventory_transaction")).select(
    F.col("transaction_type"),
    F.col("quantity").cast("int").alias("quantity"),
)
txn_types = [r["transaction_type"] for r in inv4.select("transaction_type").distinct().orderBy("transaction_type").collect()]
print("Transaction types:", txn_types)

# Baseline: one filtered aggregation per type, unioned — the table is re-scanned for each type.
parts = [
    inv4.filter(F.col("transaction_type") == t)
        .groupBy("transaction_type")
        .agg(F.count("*").alias("txns"), F.sum("quantity").alias("total_qty"))
    for t in txn_types
]
onepass_before_df = reduce(lambda a, b: a.unionByName(b), parts)

with benchmark_op("One pass, not many", "many scans", spark):
    onepass_before = {r["transaction_type"]: (r["txns"], int(r["total_qty"] or 0)) for r in onepass_before_df.collect()}
print("Buckets returned:", len(onepass_before))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

plan_string(onepass_before_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 4️⃣ DIAGNOSE — Prove the table is scanned once per category
# =================================================================================================

# Count the parquet scans in the physical plan; the union re-reads the table for every type.
def count_table_scans(df):
    plan = plan_string(df).split("Initial Plan")[0]
    for token in ("+- FileScan parquet", "+- Scan parquet", "+- BatchScan"):
        hits = plan.count(token)
        if hits:
            return hits
    return plan.count("Scan ")

before_scans = count_table_scans(onepass_before_df)
print(json.dumps({
    "antiPattern": "filter + aggregate once per category, then union",
    "categories": len(txn_types),
    "tableScansInPlan": before_scans,
}, default=str, indent=2))
onepass_before_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Aggregate every bucket in a single pass
# ==================================================================================================

# One groupBy scans the table once and produces all category totals together.
onepass_after_df = (
    inv4.groupBy("transaction_type")
    .agg(F.count("*").alias("txns"), F.sum("quantity").alias("total_qty"))
)

with benchmark_op("One pass, not many", "one scan", spark):
    onepass_after = {r["transaction_type"]: (r["txns"], int(r["total_qty"] or 0)) for r in onepass_after_df.collect()}
print("Buckets returned:", len(onepass_after))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# One scan instead of N, with identical per-type totals.
after_scans = count_table_scans(onepass_after_df)
print(json.dumps({
    "antiPattern": "repeated filter+aggregate passes unioned",
    "categories": len(txn_types),
    "baselineTableScans": before_scans,
    "fixedTableScans": after_scans,
    "sameResult": onepass_before == onepass_after,
}, default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ✅ Verify that `fixedTableScans` is `1` while `baselineTableScans` equals the number of categories, and `sameResult` is `True`.

# CELL ********************

onepass_after_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# Spark does **not** merge independent filtered passes into a single read. Looping over each `transaction_type` — filter, aggregate, then `unionByName` — produced one `FileScan` of `inventory_transaction` **per category**, so the work grew linearly with the number of types.
# 
# A single `groupBy("transaction_type").agg(...)` scans the table **once** and computes every bucket together. The plan collapses from N scans to 1 (`fixedTableScans` is `1`), and the per-type totals are identical (`sameResult` is `True`).
# 
# > 📝 **Note:** When you need multiple conditional buckets in one pass, `sum(when(cond, x))` / conditional aggregation keeps it a single scan too.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 5 — Cartesian / missing join key**
# 
# **Problem:** A pass-rate query combines `quality_inspection` with `production_order` without the production-order join key. A related cycle-time query uses an inequality self-join when it really needs the previous event per machine.
# 
# **Why it matters:** Missing equality keys create N × M pairs. In production this shows up as massive shuffle, spill, executor loss, or out-of-memory.
# 
# **Fix in one line:** Supply the correct equality join condition; when the intent is "previous row," use a window function instead of an inequality self-join.

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Capture baseline query time
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

with benchmark_op("Avoid Cartesian Join", "before", spark):
    result_cartesian_before = (
        qi.crossJoin(po)
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
        )
        .orderBy(F.desc("joined_rows"))
    )
    display(result_cartesian_before)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 5️⃣ DIAGNOSE — Prove the root cause is a Cartesian or nested-loop join
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

# ### 🎯 **Challenge: Inspect the plan**
# 
# Inspect the plan for `CartesianProduct` or `BroadcastNestedLoopJoin` to identify the root cause of the poor performance.

# CELL ********************

# Starter: inspect the missing-key plan and build the equi-join condition explicitly.
result_cartesian_before # TODO: review the query plan

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# 
# <details>
#   <summary><strong>🔑 Solution:</strong> Click to reveal</summary>
# 
# <br/>
# 
# ```python
# result_cartesian_before.explain(mode="formatted")
# ```
# 
# </details>
# 
# ---

# CELL ********************

# ==================================================================================================
# 5️⃣ FIX — Add the production_order_id equality join condition
# ==================================================================================================

# The fixed query joins inspections to production orders by the real key.
print("✅ Running fixed query with the correct join predicate...\n")

with benchmark_op("AvoidCartesian Join", "after", spark):
    result_cartesian_after = (
        qi.join(po, F.col("qi_production_order_id") == F.col("po_production_order_id"))
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
        )
        .orderBy(F.desc("joined_rows"))
    )
    display(result_cartesian_after)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Confirm the fixed plan uses a real equi-join and processes matched rows only.
cartesian_after_plan = plan_string(result_cartesian_after)
cartesian_after_join = (
    "SortMergeJoin" if "SortMergeJoin" in cartesian_after_plan
    else "BroadcastHashJoin" if "BroadcastHashJoin" in cartesian_after_plan
    else "ShuffledHashJoin" if "ShuffledHashJoin" in cartesian_after_plan
    else "None"
)
print(json.dumps({
    "antiPattern": "Missing join predicate / Cartesian join",
    "baselineJoin": cartesian_before_evidence["executedJoin"],
    "fixedJoin": cartesian_after_join,
    "improvement": "Added equality join on production_order_id instead of crossJoin.",
}, default=str, indent=2))

result_cartesian_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# Without an equality predicate, Spark can only pair **every** left row with **every** right row — N × M pairs — which the plan exposes as `CartesianProduct` or `BroadcastNestedLoopJoin`. On real data this is where you see runaway shuffle, spill, executor loss, and OOM.
# 
# Supplying the real key (`production_order_id`) lets Spark use a proper equi-join (`SortMergeJoin` / `BroadcastHashJoin`) that only processes matched rows, so runtime and pair counts collapse. When the intent is "the previous row," a window function (`lag()`) replaces an inequality self-join for the same reason.
# 
# > 📝 **Note:** Seeing `CartesianProduct` or `BroadcastNestedLoopJoin` in a plan is almost always a bug — check that every join has an equality condition on the right key.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 6 — Python UDFs → native expressions**
# 
# **Problem:** A top-customer spend query computes line totals and order days with scalar Python UDFs.
# 
# **Why it matters:** On the JVM, a scalar Python UDF forces a JVM↔Python boundary (`BatchEvalPython`) and per-row serialization. Rewriting the UDFs as built-in Spark expressions removes that boundary — a pure **code** fix.
# 
# **Fix in one line:** Replace the scalar Python UDFs with native Spark SQL expressions.
# 
# > Note: Microsoft Fabric's Native Execution Engine runs vectorized Python UDFs natively, so this rewrite is often **not** required. To isolate the code-level delta, this exercise temporarily disables NEE so the JVM Python boundary is visible. Module 3 shows the complementary execution-lever fix: leave the UDF code unchanged and simply enable NEE.

# CELL ********************

# ============================================================
# 6️⃣ BENCHMARK — Baseline Python UDFs on the JVM
# ============================================================

# NEE is disabled here only to expose the JVM Python boundary; it is restored at the end.
from pyspark.sql.types import DoubleType

remember_conf("spark.native.enabled")
spark.conf.set("spark.native.enabled", "false")
print("🐌 Running the Python-UDF query on the JVM...\n")


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


orders8 = spark.table(table_ref("web_order")).selectExpr("web_order.*")
exploded_orders8 = orders8.select(
    F.col("customer_id"),
    F.col("order_date"),
    F.explode("order_lines").alias("line"),
)

with benchmark_op("Python UDFs vs Native Functions", "Python UDF", spark):
    udf_before_df = (
        exploded_orders8
        .withColumn("line_total", python_line_total("line.quantity", "line.unit_price", "line.extended_price"))
        .withColumn("order_day", python_extract_day("order_date"))
        .groupBy("customer_id")
        .agg(F.sum("line_total").alias("total_spend"), F.max("order_day").alias("latest_day"), F.count("*").alias("line_count"))
        .orderBy(F.desc("total_spend"))
        .limit(10)
    )
    udf_before_pdf = udf_before_df.toPandas()

display(udf_before_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 6️⃣ DIAGNOSE — Prove the root cause is the JVM Python boundary
# =================================================================================================

# On the JVM the plan shows BatchEvalPython / PythonUDF and NEE fallback blocks.
udf_before_plan = plan_string(udf_before_df)
udf_before_fallbacks = extract_nee_fallbacks(udf_before_plan)
print(json.dumps({
    "neeEnabled": spark.conf.get("spark.native.enabled"),
    "hasBatchEvalPython": "BatchEvalPython" in udf_before_plan or "PythonUDF" in udf_before_plan,
    "neeFallbackBlockCount": udf_before_fallbacks["blockCount"],
    "neeFallbackOperators": udf_before_fallbacks["operators"],
}, default=str, indent=2))
print(udf_before_plan[:1600])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 **Challenge: Confirm BatchEvalPython disappears from the plan**
# 
# Both UDFs were rewritten as native Spark expressions (`coalesce` / arithmetic for the line total and `regexp_extract` (or a date function) for the order day). Re-run and confirm `BatchEvalPython` disappears from the plan.

# CELL ********************

# Starter: preview the native columns that replace the UDFs.
starter_native = exploded_orders8.select(
    "customer_id",
    F.coalesce(
        F.col("line.extended_price").cast("double"),
        F.col("line.quantity").cast("double") * F.col("line.unit_price").cast("double"),
        F.lit(0.0),
    ).alias("line_total"),
    F.regexp_extract("order_date", r"(\d{4}-\d{2}-\d{2})", 1).alias("order_day"),
).limit(5)
starter_native.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 6️⃣ FIX — Native Spark expressions remove the Python boundary (still on the JVM)
# ==================================================================================================

# Native expressions keep execution inside Spark — no JVM↔Python round-trip.
line_total_native = F.coalesce(
    F.col("line.extended_price").cast("double"),
    F.col("line.quantity").cast("double") * F.col("line.unit_price").cast("double"),
    F.lit(0.0),
)
order_day_native = F.regexp_extract("order_date", r"(\d{4}-\d{2}-\d{2})", 1)

with benchmark_op("Python UDFs vs Native Functions", "Native Functions", spark):
    native_after_df = (
        exploded_orders8
        .withColumn("line_total", line_total_native)
        .withColumn("order_day", order_day_native)
        .groupBy("customer_id")
        .agg(F.sum("line_total").alias("total_spend"), F.max("order_day").alias("latest_day"), F.count("*").alias("line_count"))
        .orderBy(F.desc("total_spend"))
        .limit(10)
    )
    native_after_pdf = native_after_df.toPandas()

display(native_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 6️⃣ CHECK-CHANGES — Compare against baseline 
# ============================================================

# Same result, Python boundary gone.
def _spend_map(pdf):
    return {row["customer_id"]: round(float(row["total_spend"] or 0), 4) for _, row in pdf.iterrows()}

native_after_plan = plan_string(native_after_df)
print(json.dumps({
    "antiPattern": "scalar Python UDFs on the JVM",
    "sameBusinessResult": _spend_map(udf_before_pdf) == _spend_map(native_after_pdf),
    "baselineHadBatchEvalPython": "BatchEvalPython" in udf_before_plan or "PythonUDF" in udf_before_plan,
    "fixedHasBatchEvalPython": "BatchEvalPython" in native_after_plan or "PythonUDF" in native_after_plan,
    "note": "NEE (Fabric default) vectorizes Python UDFs, so this rewrite is often unnecessary - see Module 3",
}, default=str, indent=2))
restore_conf("spark.native.enabled")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# On the JVM, a scalar Python UDF forces a **JVM↔Python boundary** — the plan shows `BatchEvalPython` — where every row is serialized to a Python worker and back. That per-row round-trip, plus the loss of whole-stage codegen, is what makes UDFs slow.
# 
# Rewriting the UDFs as native Spark expressions (`coalesce` + arithmetic for the line total, `regexp_extract` for the order day) keeps execution entirely inside Spark. `BatchEvalPython` disappears from the plan and the business result is unchanged — a pure **code** fix.
# 
# > 📝 **Note:** NEE was disabled here only to expose the boundary. Microsoft Fabric's **Native Execution Engine** vectorizes Python UDFs, so this rewrite is often unnecessary — Module 3 shows the complementary execution-lever fix: leave the UDF code as-is and just enable NEE.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 7 — `withColumn` in a loop → `withColumns`**
# 
# **Problem:** Feature-engineering code adds many derived columns by chaining `.withColumn()` in a loop — one call per column.
# 
# **Why it matters:** Each `.withColumn()` adds another nested `Project` to the logical plan and re-resolves the schema on the driver. Dozens of chained calls build a deep plan that is slow to analyze/plan (and can even `StackOverflow`), even though the executed result is identical.
# 
# **Fix in one line:** Add every column in a single `.withColumns({...})` call (Spark 3.3+).

# CELL ********************

# ============================================================
# 7️⃣ BENCHMARK — Build many columns by chaining withColumn in a loop
# ============================================================

# Baseline: one .withColumn() per feature nests a Project per column on the driver.
events_wc = spark.table(table_ref("manufacturing_event")).select(
    F.col("manufacturing_event.machine_id").alias("machine_id"),
    F.col("manufacturing_event.cycle_time_ms").alias("cycle_time_ms"),
)
N_FEATURES = 100

with benchmark_op("withColumn() vs withColumns()", "withColumn()", spark):
    wc_before_df = events_wc
    for i in range(N_FEATURES):
        wc_before_df = wc_before_df.withColumn(f"feat_{i}", F.col("cycle_time_ms") + F.lit(i))
    wc_before_count = wc_before_df.count()

print("Columns:", len(wc_before_df.columns), "| Rows:", wc_before_count)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 7️⃣ DIAGNOSE — The chained calls create a deep, project-heavy analyzed plan
# =================================================================================================

# Count the nested Project nodes the chained withColumn() calls produced.
wc_before_analyzed = wc_before_df._jdf.queryExecution().analyzed().toString()
print(json.dumps({
    "antiPattern": "chained withColumn() — one Project per column",
    "featureColumns": N_FEATURES,
    "projectNodesInAnalyzedPlan": wc_before_analyzed.count("Project"),
}, default=str, indent=2))

print(f"{wc_before_analyzed[:3600]}...")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 **Challenge: Replace `withColumn()` with `withColumns()`**
# 
# Replace the loop of `.withColumn()` calls with a single `.withColumns({...})` that maps each new column name to its expression. The result must match, with far fewer `Project` nodes in the analyzed plan.

# CELL ********************

# Starter: build the {name: expression} mapping, then call withColumns once.
feature_exprs = {f"feat_{i}": F.col("cycle_time_ms") + F.lit(i) for i in range(N_FEATURES)}
print("Mapping size:", len(feature_exprs))

wc_starter_df = events_wc # TODO: use withColumns to add 100 feature columns as a single projection
display(wc_starter_df.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# 
# <details>
#   <summary><strong>🔑 Solution:</strong> Click to reveal</summary>
# 
# <br/>
# 
# ```python
# feature_exprs = {f"feat_{i}": F.col("cycle_time_ms") + F.lit(i) for i in range(N_FEATURES)}
# 
# wc_starter_df = events_wc.withColumns(feature_exprs)
# display(wc_starter_df.limit(5))
# ```
# 
# </details>
# 
# ---

# CELL ********************

# ==================================================================================================
# 7️⃣ FIX — Add all columns in a single withColumns() call
# ==================================================================================================

# One withColumns() adds every column in a single projection.
with benchmark_op("withColumn() vs withColumns()", "withColumns()", spark):
    feature_exprs = {f"feat_{i}": F.col("cycle_time_ms") + F.lit(i) for i in range(N_FEATURES)}
    wc_after_df = events_wc.withColumns(feature_exprs)
    wc_after_count = wc_after_df.count()

print("Columns:", len(wc_after_df.columns), "| Rows:", wc_after_count)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 7️⃣ CHECK-CHANGES — Same columns/rows, a far simpler analyzed plan
# ============================================================

# Identical output, dramatically fewer Project nodes to analyze.
wc_after_analyzed = wc_after_df._jdf.queryExecution().analyzed().toString()
print(json.dumps({
    "antiPattern": "chained withColumn() in a loop",
    "sameColumns": sorted(wc_before_df.columns) == sorted(wc_after_df.columns),
    "sameRowCount": wc_before_count == wc_after_count,
    "baselineProjectNodes": wc_before_analyzed.count("Project"),
    "fixedProjectNodes": wc_after_analyzed.count("Project"),
}, default=str, indent=2))

print(f"{wc_before_analyzed[:3600]}...")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# Each `.withColumn()` call adds another nested `Project` node to the logical plan and re-resolves the schema on the driver. Chaining 100 of them built a plan with ~100 stacked `Project` nodes that is slow to analyze and optimize (and can even `StackOverflow` on deep chains) — all before a single row is processed.
# 
# `.withColumns({...})` (Spark 3.3+) adds every column in a **single** projection. The analyzed plan collapses to one `Project`, and the executed columns and row count are identical. The win here is entirely at **plan-build / analysis** time, not runtime.
# 
# > 📝 **Note:** The same pattern applies to `.withColumnRenamed()` in a loop — build a mapping and rename in one shot instead of chaining.
# 
# ---

# MARKDOWN ********************

# ## **Exercise 8 — Schema inference vs a declared schema**
# 
# **Problem:** Reading the JSON landing zone with `spark.read.json(...)` lets Spark **infer** the schema. To do that it must open and scan the files *before your query runs* — an eager, hidden startup cost that grows with the number of files.
# 
# **Why it matters:** Inference triggers an extra pass over the data every time the pipeline starts. On thousands of small landing-zone files that scan can dominate a job that otherwise reads very little.
# 
# **Fix in one line:** Declare the schema with `StructType` and pass it to `.schema(...)`, so the read skips inference entirely.
# 
# > NOTE: This exercise reads the raw JSON landing zone at `Files/landing/manufacturing_event` produced by the `source_to_bronze` job in setup. If that path is absent, skip this exercise.


# CELL ********************

# ============================================================
# 8️⃣ BENCHMARK — Time schema inference on the landing zone
# ============================================================

# The unoptimized pipeline left many small files in the landing zone.
# NOTE: spark.read.json() triggers inference EAGERLY at construction time,
# so timing the DataFrame construction itself captures the inference scan.
LANDING_TABLE = "product_return"
landing_path = f"Files/landing/{LANDING_TABLE}"

print(f"🐌 Inferring schema from landing zone: {landing_path}\n")
print("   Spark must open files and read metadata to discover columns + types...\n")

with benchmark_op("Avoid Schema Inference in Production", "inferred (file scan)", spark):
    inferred_df = spark.read.option("multiline", "true").json(landing_path)

inferred_df.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 1️⃣ FIX — Declare the schema so the read skips inference
# ==================================================================================================
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, BooleanType, DecimalType

# A production pipeline defines the schema once in code — no file scanning needed.
static_schema = StructType([
    StructField("EventId", StringType()),
    StructField("Timestamp", TimestampType()),
    StructField("MachineId", StringType()),
    StructField("PartNum", StringType()),
    StructField("ColorId", IntegerType()),
    StructField("MoldTemp", DecimalType(5, 1)),
    StructField("InjectionPressure", DecimalType(6, 1)),
    StructField("CycleTimeMs", IntegerType()),
    StructField("DefectDetected", BooleanType()),
    StructField("DefectType", StringType()),
    StructField("BatchId", StringType()),
])

with benchmark_op("Avoid Schema Inference in Production", "static (no scan)", spark):
    static_df = spark.read.schema(static_schema).json(landing_path)

static_df.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# `spark.read.json(path)` with no schema makes Spark **infer** the columns and types — and to do that it must open and scan the files *before your query runs*. That is an eager, hidden startup cost that grows with the number of files, so on thousands of small landing-zone files the inference scan can dominate a job that otherwise reads very little.
# 
# Declaring a `StructType` and passing it to `.schema(...)` removes that inference pass entirely: Spark already knows the columns and types, so it skips straight to reading. Same columns, faster start.
# 
# > 📝 **Note:** A declared schema also protects you from silent type drift — inference can pick different types run-to-run as the data changes, whereas a static schema is deterministic.
# 
# > 📝 **Key takeaway:** define schemas upfront for production pipelines.
# > Inference is convenient for exploration but adds startup latency, especially when scanning thousands of small files.
# ---

# MARKDOWN ********************

# ## **Exercise 9 — Driver `collect()` / `toPandas()` and driver OOM**
# 
# **Problem:** The inventory workflow pulls every transaction to the driver with `collect()` and aggregates in Python. `.toPandas()` has the same raw-data movement risk.
# 
# **Why it matters:** Pulling distributed data into one process can trip task-result transport limits, executor memory while serializing results, or `spark.driver.maxResultSize`.
# 
# **Fix in one line:** Keep the aggregation distributed and only bring the small final result to the driver.

# CELL ********************

# ============================================================
# 9️⃣ BENCHMARK — Capture baseline query time
# ============================================================
from collections import defaultdict

# The baseline collects every raw transaction to the driver before aggregating.
print("🐌 Running baseline query with driver-side collect...\n")

inv = spark.table(table_ref("inventory_transaction")).select(
    "line_id", "part_num", "quantity", "transaction_type"
)
print(f"About to collect {TABLE_METRICS['inventory_transaction']['rows']:,} inventory rows to the driver.")
print("spark.driver.maxResultSize =", spark.conf.get("spark.driver.maxResultSize"))

start = time.time()
with benchmark_op("Driver Collect vs Keeping Data Distributed", "Keeping Data Distributed", spark):
    collected_inventory = inv.collect()

with benchmark_op("Aggregations - Driver vs Distributed", "Driver", spark):
    collected_inventory = inv.collect()
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
# 9️⃣ DIAGNOSE — Prove the root cause is raw-row transfer to the driver
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

# CELL ********************

# ==================================================================================================
# 9️⃣ FIX — Aggregate on executors and collect only the small grouped result
# ==================================================================================================

# Spark computes net inventory by line before the driver receives the display result.
print("✅ Running fixed query with distributed aggregation...\n")

with benchmark_op("Driver Collect vs Keeping Data Distributed", "Keeping Data Distributed", spark):
    inv.write.format("noop").mode("overwrite").save()

with benchmark_op("Aggregations - Driver vs Distributed", "Distributed", spark):
    starter_signed_inventory = inv.withColumn(
        "signed_quantity",
        F.when(
            F.col("transaction_type").isin("CONSUMPTION", "ORDER_PICK", "SCRAP"),
            -F.abs(F.col("quantity").cast("int")),
        ).otherwise(F.col("quantity").cast("int")),
    )

    result_driver_after = (
        starter_signed_inventory
        .groupBy("line_id")
        .agg(F.sum("signed_quantity").alias("net_quantity"))
        .orderBy(F.desc("net_quantity"))
    )
    display(result_driver_after)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 9️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# Verify the fixed path collects only the final grouped rows.
print(json.dumps({
    "antiPattern": "Driver-side collect and Python aggregation",
    "baselineCollectedRows": driver_before_evidence["collectedRows"],
    "fixedRawRowsCollected": 0,
    "fixedResultRowsReturnedToDriver": result_driver_after.count(),
    "improvement": "Aggregation runs on executors; only the small grouped result reaches the driver.",
}, default=str, indent=2))
result_driver_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 *What Just Happened?*
# 
# `collect()` (and `toPandas()`) pulls **every raw row** into the single driver process. That transfer can trip task-result transport limits, exhaust executor memory while serializing results, or blow past `spark.driver.maxResultSize` — and the aggregation then runs single-threaded in Python instead of across the cluster.
# 
# Keeping the aggregation distributed (`groupBy("line_id").agg(sum(...))`) lets executors do the heavy lifting; only the **small grouped result** reaches the driver. No raw rows cross the boundary, so the OOM risk is gone and the result is identical.
# 
# > 📝 **Note:** Reserve `collect()` / `toPandas()` for genuinely small, already-aggregated results. To peek at data, prefer `.show()` / `.limit()` / `display()` which bound how much is pulled back.
# 
# ---

# MARKDOWN ********************

# # 🏆 **Performance Impact by Exercise**
# 
# Execute the below to see the full impact across every exercise.
# 


# CELL ********************

print_benchmark_summary()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## **Summary — Optimizing code**
# 
# You worked through nine code-level Spark anti-patterns, each with the same loop: benchmark the symptom, diagnose it in the physical plan, change one line, and re-check.
# 
# 1. **Predicate pushdown** — filtered on a raw column so the scan prunes files instead of deriving a value first.
# 1. **De-duplicate on the key, not the whole row** — passed the key subset to `dropDuplicates` instead of comparing every column.
# 1. **Prune before a window** — projected the key columns before a `row_number()` window so the Exchange/Sort no longer carries the nested `order_lines` array.
# 1. **One pass, not many** — replaced a per-category filter-and-union loop with a single `groupBy` so the table is scanned once instead of once per category.
# 1. **Cartesian / missing join key** — restored the equi-join key so nested-loop work collapses to a hash join.
# 1. **Python UDFs → native expressions** — replaced `BatchEvalPython` with native expressions (NEE makes the rewrite optional — see Module 3).
# 1. **`withColumn` loop → `withColumns`** — collapsed ~50 chained `Project` nodes into one.
# 1. **Schema inference vs a declared schema** — declared a `StructType` so the read skips the eager file-scan inference pass.
# 1. **Driver `collect()` / OOM** — kept the aggregation distributed and returned only the small result to the driver.
# 
# Carry the same workflow into the next modules: benchmark the symptom, inspect the Spark UI and physical plan, check Delta metadata, change the right lever, and validate the before/after result.

