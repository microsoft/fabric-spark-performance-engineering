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

# # Module 1 — Optimizing Code (Spark SQL track)
# 
# Welcome to the first Fabric Jumpstart Spark performance lab for **Toy Brick Manufacturing**.
# 
# > 🔀 **This is the Spark SQL track.** Every exercise expresses the query with `spark.sql("...")` instead of the DataFrame API. If you prefer the fluent DataFrame API, use `dataframe-notebooks/01_optimizing-code` instead — the concepts, data, and benchmarks are identical.
# 
# ## What this module teaches
# 
# This module teaches you to recognize and fix **code-level** anti-patterns: the table design and cluster are fine, but the query as written is wrong or wasteful. You will also start using the diagnostic toolkit that Modules 2 and 3 reuse: Spark UI, `EXPLAIN` / physical plans, Delta metadata from `DESCRIBE DETAIL` / `DESCRIBE HISTORY`, and `inputFiles()`.
# 
# > Litmus test: if the fix is a diff to the SQL, it belongs here. Storage fixes are Module 2; execution, AQE, caching, and repartitioning fixes are Module 3.

# MARKDOWN ********************

# ## Exercise summary
# 
# | Exercise | Scenario | Expected performance signal |
# |---|---|---|
# | 1 — Predicate pushdown | A daily defect-rate dashboard wraps `timestamp` in `SUBSTRING(...)` before filtering `manufacturing_event`. | Fewer files read / filter pushed to FileScan; the derived expression disappears from the filter path. |
# | 2 — De-duplicate on the key, not the whole row | A "distinct orders" step runs `SELECT DISTINCT *`, shuffling every column including the nested `order_lines` array. | `HashAggregate` keys shrink from every column to just the business key. |
# | 3 — Prune before a window / row_number | A "latest order per customer" runs `ROW_NUMBER()` over `SELECT *`, shuffling and sorting every column — including `order_lines` — through the window. | Only the key columns cross the window Exchange/Sort; `order_lines` is pruned; identical latest-row result. |
# | 4 — One pass, not many | A per-type report `UNION ALL`s one filtered aggregation per `transaction_type`, scanning `inventory_transaction` once per type. | A single `GROUP BY` scans the table once; N scans collapse to 1; identical per-type totals. |
# | 5 — Cartesian / missing join key | A pass-rate query uses `CROSS JOIN` instead of the production-order key. | `CartesianProduct` / nested-loop work replaced by an equi-join; runtime and pair counts drop. |
# | 6 — Python UDFs → native SQL | A top-customer query calls registered Python UDFs (NEE disabled to expose the JVM boundary). | `BatchEvalPython` removed after rewriting the UDFs as native SQL functions; NEE makes the rewrite optional (see Module 3). |
# | 7 — Projection shape (SQL vs DataFrame) | The DataFrame-API `withColumn`-in-a-loop anti-pattern has no SQL analog — SQL expresses every column in one projection. | A 100-column `SELECT` is a single `Project`; nothing to fix in SQL. |
# | 8 — Schema inference vs a declared schema | Creating a temp view over the JSON landing zone lets Spark infer types by scanning files before the query runs. | Declaring the column schema removes the eager inference scan; same columns, faster read. |
# | 9 — Driver `collect()` and driver OOM | An inventory workflow collects raw `spark.sql(...)` rows to the driver and aggregates in Python. Run last so a driver crash cannot abort earlier exercises. | Driver result size shrinks / raw-row collect avoided; no OOM risk from full-result transfer. |


# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Setup: validate sources and capture baseline metrics.
import re
from pyspark.sql.types import DoubleType

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
# ## Exercise 1 — Predicate pushdown
# 
# **Problem:** The daily defect-rate query wraps the timestamp in `SUBSTRING(...)` before filtering. That derived expression can't be pushed into the scan, so Spark reads more of `manufacturing_event` than the dashboard needs.
# 
# **Why it matters:** Full scans waste I/O and make a small daily dashboard behave like a whole-factory history query.
# 
# **Fix in one line:** Filter on the raw `timestamp` column (`LIKE 'day%'`) first, then derive presentation columns.

# CELL ********************

# ============================================================
# 1️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# The baseline filters on SUBSTRING(timestamp, ...), which weakens pushdown.
print("🐌 Running baseline predicate-pushdown query...\n")

latest_day = spark.sql(
    f"SELECT MAX(TO_DATE(timestamp)) AS d FROM {table_ref('manufacturing_event')}"
).collect()[0]["d"]

sql_predicate_before = f"""
    SELECT event_day, machine_id,
           COUNT(*) AS events,
           SUM(CAST(defect_detected AS INT)) AS defects,
           SUM(CAST(defect_detected AS INT)) / COUNT(*) AS defect_rate
    FROM (
        SELECT *, SUBSTRING(timestamp, 1, 10) AS event_day
        FROM {table_ref('manufacturing_event')}
    )
    WHERE event_day = '{latest_day}'
    GROUP BY event_day, machine_id
    ORDER BY defect_rate DESC
"""

with benchmark_op("Predicate Pushdown", "before", spark):
    result_predicate_before = spark.sql(sql_predicate_before)
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
    "antiPattern": "SUBSTRING(timestamp) transformation before filtering",
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

# ### 🎯 Challenge: Check the Spark Plan and Spark UI to diagnose the problem
# 
# You've seen that the query is slow, and you suspect it's because of the large number of files and the filter not being pushed down. You want to confirm this by checking the Spark Plan and the Spark UI.
# 
# **Your task:** Use `EXPLAIN` (or `.explain()`) and the Spark UI to confirm that the query is doing a full scan and that the filter on `event_day` is **not** pushed down to the file scan.
# 
# > 💡 Hint: `EXPLAIN FORMATTED <query>` shows the physical plan. Look for **PushedFilters** in the **DefaultDeltaScanTransformer** node.
# 
# Try it in the cell below!

# CELL ********************

# Starter: inspect the baseline physical plan.
spark.sql(f"EXPLAIN FORMATTED {sql_predicate_before}").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 1️⃣ FIX — Filter on the raw timestamp column before aggregating
# ==================================================================================================

# The fixed query applies the date predicate on the raw column, then derives presentation columns.
print("✅ Running fixed predicate-pushdown query...\n")

sql_predicate_after = f"""
    SELECT TO_DATE(timestamp) AS event_day, machine_id,
           COUNT(*) AS events,
           SUM(CAST(defect_detected AS INT)) AS defects,
           SUM(CAST(defect_detected AS INT)) / COUNT(*) AS defect_rate
    FROM {table_ref('manufacturing_event')}
    WHERE timestamp LIKE '{latest_day}%'   -- filter on the raw column first (pushes down as StartsWith)
    GROUP BY TO_DATE(timestamp), machine_id
    ORDER BY defect_rate DESC
"""

with benchmark_op("Predicate Pushdown", "after", spark):
    result_predicate_after = spark.sql(sql_predicate_after)
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
record_result("1 predicate pushdown", "after", {
    "antiPattern": "Full scan / weak pushdown due to SUBSTRING(timestamp) transformation",
    "baselineReadFiles": predicate_before_evidence["readFiles"],
    "fixedReadFiles": len(result_predicate_after.inputFiles()),
    "fixedDataFilters": predicate_after_scan["dataFilters"],
    "fixedPushedFilters": predicate_after_scan["pushedFilters"],
    "fixedPlanHasSubstring": "substring" in plan_string(result_predicate_after).lower(),
})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ✅ Verify with the query plan that the `DefaultDeltaScanTransformer` node of the new plan has `PushedFilters`. This enables file skipping on read to minimize the amount of data that must be filtered post scan.

# CELL ********************

spark.sql(f"EXPLAIN FORMATTED {sql_predicate_after}").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Filtering on the **raw** `timestamp` column let Delta push the predicate into the file scan — notice `PushedFilters` now appears in the `DefaultDeltaScanTransformer` node. `WHERE timestamp LIKE 'day%'` pushes down as a `StartsWith` filter, so Spark uses each file's min/max statistics to **skip files** that can't contain the target day.
# 
# The baseline wrapped the column in `SUBSTRING(timestamp, 1, 10)` *before* filtering. That hides the real column behind an expression the scan can't reason about, so pushdown is lost and Spark reads every file, then throws most rows away. The rule: **filter on native columns first, derive presentation columns after.**
# 
# > 📝 **Note:** File skipping is only as good as your data layout. Module 2 covers the table-side levers (compaction, clustering, stats) that make pushdown even more effective.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 2 — De-duplicate on the key, not the whole row
# 
# **Problem:** A "distinct orders" step runs `SELECT DISTINCT *`, so Spark uses every column as the de-duplication key — including the nested `order_lines` array.
# 
# **Why it matters:** `DISTINCT *` compares all column values and shuffles the full, nested row schema even though the business key is just a couple of columns.
# 
# **Fix in one line:** `SELECT DISTINCT` only the key columns so there's less comparison and shuffle work.

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# Baseline: DISTINCT * must aggregate every column, including order_lines.
sql_proj_before = f"SELECT DISTINCT * FROM {table_ref('web_order')}"

with benchmark_op("Column pruning / projection", "before", spark):
    proj_before_df = spark.sql(sql_proj_before)
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
print(json.dumps({
    "antiPattern": "SELECT DISTINCT * uses all columns as the key",
    "baselineColumnsScannedAndShuffled": get_agg_key_count("HashAggregate", proj_before_df)
}, default=str, indent=2))
proj_before_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# You only need distinct `customer_id` / `order_date` pairs. Rewrite the query so Spark reads and shuffles just those columns — `SELECT DISTINCT customer_id, order_date`. Compare the scanned columns and the plan.

# CELL ********************

# Starter: reduce the columns that reach the shuffle, then de-duplicate.
sql_proj_starter = f"SELECT DISTINCT * FROM {table_ref('web_order')}"  # TODO: select only the key columns
proj_starter_df = spark.sql(sql_proj_starter)
print("Columns shuffled:", len(proj_starter_df.columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 2️⃣ FIX — Select the key columns before de-duplicating
# ==================================================================================================

# Selecting the business key first prunes the scan and shrinks the exchange.
sql_proj_after = f"SELECT DISTINCT customer_id, order_date FROM {table_ref('web_order')}"

with benchmark_op("Column pruning / projection", "after", spark):
    proj_after_df = spark.sql(sql_proj_after)
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
record_result("2 de-duplicate on key", "after", {
    "antiPattern": "SELECT DISTINCT * over all columns",
    "baselineColumnsShuffled": get_agg_key_count("HashAggregate", proj_before_df),
    "fixedColumnsShuffled": get_agg_key_count("HashAggregate", proj_after_df),
    "baselineDistinctFullRows": proj_before_count,
    "fixedDistinctKeyRows": proj_after_count,
})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ✅ Verify that the `HashAggregate` node of the fixed plan uses only 2 columns as keys instead of every column of `web_order`.

# CELL ********************

proj_after_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# `SELECT DISTINCT *` uses **every** column as the de-duplication key. Spark therefore scans the full row schema and shuffles all of it — including the nested `order_lines` array — through a `HashAggregate`, even though the business key is just `customer_id` + `order_date`.
# 
# Restricting the projection to `SELECT DISTINCT customer_id, order_date` prunes the scan to those two columns and shrinks the exchange to a narrow row, and the distinct result is the business-meaningful one.
# 
# > 📝 **Note:** "Read/shuffle only what you need" is the recurring theme of the next two exercises too — projection is one of the cheapest wins in Spark.
# 
# ---

# MARKDOWN ********************

# ## Exercise 3 — Prune columns before a window / row_number
# 
# **Problem:** A "latest order per customer" step runs `ROW_NUMBER()` over `SELECT *`. The window has to shuffle (Exchange) and sort every column — including the nested `order_lines` array — even though only a few fields are needed.
# 
# **Why it matters:** A window with `PARTITION BY` / `ORDER BY` forces an Exchange + Sort. Whatever columns are in the `SELECT` ride through that shuffle and sort, so carrying an unused nested array inflates the shuffle and sort spill for nothing.
# 
# **Fix in one line:** Select just the columns the window needs *inside* the subquery, so the Exchange and Sort move a narrow row.

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# Baseline: ROW_NUMBER() over the wide, nested row — every column rides through the window.
sql_latest_before = f"""
    SELECT * FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
        FROM {table_ref('web_order')}
    )
    WHERE rn = 1
"""
latest_before = spark.sql(sql_latest_before)

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
    "antiPattern": "ROW_NUMBER() window applied to SELECT * (the full wide row)",
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

# ### 🎯 Challenge
# 
# You only need `customer_id`, `order_date`, and `order_total` to pick the latest order. Select those columns *inside* the subquery before the `ROW_NUMBER()` window so the Exchange and Sort move a narrow row and `order_lines` never enters the shuffle. Confirm the latest-row result is unchanged.

# CELL ********************

# Starter: project the columns the window needs INSIDE the subquery.
sql_latest_starter = f"""
    SELECT * FROM (
        SELECT *,   -- TODO: select only customer_id, order_date, order_total
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
        FROM {table_ref('web_order')}
    )
    WHERE rn = 1
"""
print("Columns that would enter the window:", len(spark.sql(sql_latest_starter).columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3️⃣ FIX — Select the window's columns before applying it
# ==================================================================================================

# Selecting first keeps order_lines out of the Exchange and Sort.
sql_latest_after = f"""
    SELECT customer_id, order_date, order_total, rn FROM (
        SELECT customer_id, order_date, order_total,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
        FROM {table_ref('web_order')}
    )
    WHERE rn = 1
"""
latest_after = spark.sql(sql_latest_after)

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
record_result("3 prune before window", "after", {
    "antiPattern": "windowing over unpruned columns",
    "baselineColumnsThroughWindow": len(latest_before.columns),
    "fixedColumnsThroughWindow": len(latest_after.columns),
    "baselineWindowShufflesOrderLines": "order_lines" in before_win_plan,
    "fixedWindowShufflesOrderLines": "order_lines" in after_win_plan,
    "sameLatestPerCustomer": before_keys == after_keys,
})

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

# ### 💡 What Just Happened?
# 
# A window with `PARTITION BY` / `ORDER BY` forces an **Exchange (shuffle) + Sort**. Whatever columns are in the subquery's `SELECT` ride through that shuffle and sort — so the baseline's `SELECT *` carried the entire wide row, including the nested `order_lines` array, just to pick the latest order per customer.
# 
# Selecting the three needed columns (`customer_id`, `order_date`, `order_total`) **inside** the subquery keeps `order_lines` out of the shuffle and sort entirely. Only 4 columns cross the window (the 3 keys plus `rn`), the sort spills less, and the latest-row result is identical.
# 
# > 📝 **Note:** The same idea applies to joins and aggregations — prune columns as early as possible so every downstream shuffle moves a narrow row.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 4 — One pass, not many (avoid repeated scans)
# 
# **Problem:** A per-transaction-type report is built by `UNION ALL`-ing one filtered aggregation per `transaction_type` — so the table is scanned once per category.
# 
# **Why it matters:** Spark is excellent at aggregations, but it will *not* merge independent filtered passes into a single read. Each `UNION ALL` branch re-scans the table, so the work grows with the number of categories instead of staying a single scan.
# 
# **Fix in one line:** Compute every bucket in one `GROUP BY transaction_type` so the table is scanned once.

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Capture baseline query time
# ============================================================

txn_types = [r["transaction_type"] for r in spark.sql(
    f"SELECT DISTINCT transaction_type FROM {table_ref('inventory_transaction')} ORDER BY transaction_type"
).collect()]
print("Transaction types:", txn_types)

# Baseline: one filtered aggregation per type, UNION ALL'd — the table is re-scanned for each type.
union_branches = [
    f"""SELECT transaction_type, COUNT(*) AS txns, SUM(CAST(quantity AS INT)) AS total_qty
        FROM {table_ref('inventory_transaction')}
        WHERE transaction_type = '{t}' GROUP BY transaction_type"""
    for t in txn_types
]
sql_onepass_before = "\nUNION ALL\n".join(union_branches)

with benchmark_op("One pass, not many", "before", spark):
    onepass_before_df = spark.sql(sql_onepass_before)
    onepass_before = {r["transaction_type"]: (r["txns"], int(r["total_qty"] or 0)) for r in onepass_before_df.collect()}
print("Buckets returned:", len(onepass_before))

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
    "antiPattern": "filter + aggregate once per category, then UNION ALL",
    "categories": len(txn_types),
    "tableScansInPlan": before_scans,
}, default=str, indent=2))
onepass_before_df.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Produce the same per-type totals with a single pass over `inventory_transaction`. Use one `GROUP BY transaction_type` aggregation so the plan contains a single scan, and confirm the totals match the `UNION ALL` version.

# CELL ********************

# Starter: compute every bucket in one pass instead of one query per type.
sql_onepass_starter = f"SELECT * FROM {table_ref('inventory_transaction')}"  # TODO: GROUP BY transaction_type
print("Scans in starter plan:", count_table_scans(spark.sql(sql_onepass_starter)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Aggregate every bucket in a single pass
# ==================================================================================================

# One GROUP BY scans the table once and produces all category totals together.
sql_onepass_after = f"""
    SELECT transaction_type, COUNT(*) AS txns, SUM(CAST(quantity AS INT)) AS total_qty
    FROM {table_ref('inventory_transaction')}
    GROUP BY transaction_type
"""

with benchmark_op("One pass, not many", "after", spark):
    onepass_after_df = spark.sql(sql_onepass_after)
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
record_result("4 one pass, not many", "after", {
    "antiPattern": "repeated filter+aggregate passes UNION ALL'd",
    "categories": len(txn_types),
    "baselineTableScans": before_scans,
    "fixedTableScans": after_scans,
    "sameResult": onepass_before == onepass_after,
})

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

# ### 💡 What Just Happened?
# 
# Spark does **not** merge independent filtered passes into a single read. `UNION ALL`-ing one filtered aggregation per `transaction_type` produced one `FileScan` of `inventory_transaction` **per category**, so the work grew linearly with the number of types.
# 
# A single `GROUP BY transaction_type` scans the table **once** and computes every bucket together. The plan collapses from N scans to 1 (`fixedTableScans` is `1`), and the per-type totals are identical (`sameResult` is `True`).
# 
# > 📝 **Note:** When you need multiple conditional buckets in one pass, `SUM(CASE WHEN cond THEN x END)` keeps it a single scan too.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 5 — Cartesian / missing join key
# 
# **Problem:** A pass-rate query combines `quality_inspection` with `production_order` using a `CROSS JOIN` — it omits the production-order join key.
# 
# **Why it matters:** Missing equality keys create N × M pairs. In production this shows up as massive shuffle, spill, executor loss, or out-of-memory.
# 
# **Fix in one line:** Supply the correct equality join condition (`ON qi.production_order_id = po.production_order_id`).

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Capture baseline query time
# ============================================================

# The baseline uses CROSS JOIN and creates N × M join work.
print("🐌 Running baseline query with missing join key...\n")

estimated_pairs = TABLE_METRICS["quality_inspection"]["rows"] * TABLE_METRICS["production_order"]["rows"]
print(f"Estimated Cartesian pairs: {estimated_pairs:,}")

sql_cartesian_before = f"""
    SELECT po.machine_id,
           COUNT(*) AS joined_rows,
           SUM(qi.pass_count) / SUM(qi.sample_size) AS pass_rate
    FROM {table_ref('quality_inspection')} qi
    CROSS JOIN {table_ref('production_order')} po
    GROUP BY po.machine_id
    ORDER BY joined_rows DESC
"""

with benchmark_op("Cartesian Join", "before", spark):
    result_cartesian_before = spark.sql(sql_cartesian_before)
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
    "antiPattern": "Missing join predicate / CROSS JOIN",
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

# ### 🎯 Challenge
# 
# Inspect the plan for `CartesianProduct` or `BroadcastNestedLoopJoin`. Then rewrite the pass-rate query with the real key, `production_order_id`, so Spark can use an equi-join (`JOIN ... ON qi.production_order_id = po.production_order_id`).

# CELL ********************

# Starter: inspect the missing-key plan, then build the equi-join.
result_cartesian_before.explain(mode="formatted")

starter_preview = spark.sql(f"""
    SELECT qi.production_order_id, po.production_order_id, po.machine_id, qi.pass_count, qi.sample_size
    FROM {table_ref('quality_inspection')} qi
    JOIN {table_ref('production_order')} po ON qi.production_order_id = po.production_order_id
    LIMIT 5
""")
display(starter_preview)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 5️⃣ FIX — Add the production_order_id equality join condition
# ==================================================================================================

# The fixed query joins inspections to production orders by the real key.
print("✅ Running fixed query with the correct join predicate...\n")

sql_cartesian_after = f"""
    SELECT po.machine_id,
           COUNT(*) AS joined_rows,
           SUM(qi.pass_count) / SUM(qi.sample_size) AS pass_rate
    FROM {table_ref('quality_inspection')} qi
    JOIN {table_ref('production_order')} po
      ON qi.production_order_id = po.production_order_id
    GROUP BY po.machine_id
    ORDER BY joined_rows DESC
"""

with benchmark_op("Cartesian Join", "after", spark):
    result_cartesian_after = spark.sql(sql_cartesian_after)
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
record_result("5 cartesian / missing join key", "after", {
    "antiPattern": "Missing join predicate / CROSS JOIN",
    "baselineJoin": cartesian_before_evidence["executedJoin"],
    "fixedJoin": cartesian_after_join,
    "improvement": "Added equality join on production_order_id instead of CROSS JOIN.",
})

result_cartesian_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Without an equality predicate, `CROSS JOIN` pairs **every** left row with **every** right row — N × M pairs — which the plan exposes as `CartesianProduct` or `BroadcastNestedLoopJoin`. On real data this is where you see runaway shuffle, spill, executor loss, and OOM.
# 
# Supplying the real key (`ON qi.production_order_id = po.production_order_id`) lets Spark use a proper equi-join (`SortMergeJoin` / `BroadcastHashJoin`) that only processes matched rows, so runtime and pair counts collapse.
# 
# > 📝 **Note:** Seeing `CartesianProduct` or `BroadcastNestedLoopJoin` in a plan is almost always a bug — check that every join has an `ON` clause with the right equality key.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 6 — Python UDFs → native SQL functions
# 
# **Problem:** A top-customer spend query computes line totals and order days with scalar Python UDFs registered for SQL (`spark.udf.register`).
# 
# **Why it matters:** On the JVM, a scalar Python UDF forces a JVM↔Python boundary (`BatchEvalPython`) and per-row serialization. Rewriting the UDFs as built-in SQL functions removes that boundary — a pure **code** fix.
# 
# **Fix in one line:** Replace the registered Python UDFs with native SQL expressions (`COALESCE` / arithmetic and `REGEXP_EXTRACT`).
# 
# > Note: Microsoft Fabric's Native Execution Engine runs vectorized Python UDFs natively, so this rewrite is often **not** required. To isolate the code-level delta, this exercise temporarily disables NEE so the JVM Python boundary is visible. Module 3 shows the complementary execution-lever fix: leave the UDF code unchanged and simply enable NEE.

# CELL ********************

# ============================================================
# 6️⃣ BENCHMARK — Baseline Python UDFs on the JVM
# ============================================================

# NEE is disabled here only to expose the JVM Python boundary; it is restored at the end.
remember_conf("spark.native.enabled")
spark.conf.set("spark.native.enabled", "false")
print("🐌 Running the Python-UDF query on the JVM...\n")


def python_line_total(quantity, unit_price, extended_price):
    if extended_price is not None:
        return float(extended_price)
    if quantity is None or unit_price is None:
        return 0.0
    return float(quantity) * float(unit_price)


def python_extract_day(timestamp_str):
    if timestamp_str is None:
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(timestamp_str))
    return match.group(1) if match else None


spark.udf.register("python_line_total", python_line_total, DoubleType())
spark.udf.register("python_extract_day", python_extract_day, "string")

sql_udf_before = f"""
    SELECT customer_id,
           SUM(python_line_total(line.quantity, line.unit_price, line.extended_price)) AS total_spend,
           MAX(python_extract_day(order_date)) AS latest_day,
           COUNT(*) AS line_count
    FROM (
        SELECT customer_id, order_date, EXPLODE(order_lines) AS line
        FROM {table_ref('web_order')}
    )
    GROUP BY customer_id
    ORDER BY total_spend DESC
    LIMIT 10
"""

with benchmark_op("Python UDF vs native", "before", spark):
    udf_before_df = spark.sql(sql_udf_before)
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

# ### 🎯 Challenge
# 
# Rewrite both UDFs as native SQL functions — `COALESCE` / arithmetic for the line total and `REGEXP_EXTRACT` for the order day. Re-run and confirm `BatchEvalPython` disappears from the plan.

# CELL ********************

# Starter: preview the native SQL columns that replace the UDFs.
spark.sql(f"""
    SELECT customer_id,
           COALESCE(CAST(line.extended_price AS DOUBLE),
                    CAST(line.quantity AS DOUBLE) * CAST(line.unit_price AS DOUBLE),
                    0.0) AS line_total,
           REGEXP_EXTRACT(order_date, '([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1) AS order_day
    FROM (SELECT customer_id, order_date, EXPLODE(order_lines) AS line FROM {table_ref('web_order')})
    LIMIT 5
""").explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 6️⃣ FIX — Native SQL functions remove the Python boundary (still on the JVM)
# ==================================================================================================

# Native SQL keeps execution inside Spark — no JVM↔Python round-trip.
sql_udf_after = f"""
    SELECT customer_id,
           SUM(COALESCE(CAST(line.extended_price AS DOUBLE),
                        CAST(line.quantity AS DOUBLE) * CAST(line.unit_price AS DOUBLE),
                        0.0)) AS total_spend,
           MAX(REGEXP_EXTRACT(order_date, '([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1)) AS latest_day,
           COUNT(*) AS line_count
    FROM (
        SELECT customer_id, order_date, EXPLODE(order_lines) AS line
        FROM {table_ref('web_order')}
    )
    GROUP BY customer_id
    ORDER BY total_spend DESC
    LIMIT 10
"""

with benchmark_op("Python UDF vs native", "after", spark):
    native_after_df = spark.sql(sql_udf_after)
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
record_result("6 python UDFs -> native SQL", "after", {
    "antiPattern": "scalar Python UDFs on the JVM",
    "sameBusinessResult": _spend_map(udf_before_pdf) == _spend_map(native_after_pdf),
    "baselineHadBatchEvalPython": "BatchEvalPython" in udf_before_plan or "PythonUDF" in udf_before_plan,
    "fixedHasBatchEvalPython": "BatchEvalPython" in native_after_plan or "PythonUDF" in native_after_plan,
    "note": "NEE (Fabric default) vectorizes Python UDFs, so this rewrite is often unnecessary - see Module 3",
})
restore_conf("spark.native.enabled")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# On the JVM, a scalar Python UDF forces a **JVM↔Python boundary** — the plan shows `BatchEvalPython` — where every row is serialized to a Python worker and back. That per-row round-trip, plus the loss of whole-stage codegen, is what makes UDFs slow.
# 
# Rewriting the UDFs as native SQL functions (`COALESCE` + arithmetic for the line total, `REGEXP_EXTRACT` for the order day) keeps execution entirely inside Spark. `BatchEvalPython` disappears from the plan and the business result is unchanged — a pure **code** fix.
# 
# > 📝 **Note:** NEE was disabled here only to expose the boundary. Microsoft Fabric's **Native Execution Engine** (the default) vectorizes Python UDFs, so this rewrite is often unnecessary — Module 3 shows the complementary execution-lever fix: leave the UDF code as-is and just enable NEE.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 7 — Projection shape: why SQL avoids the `withColumn` trap
# 
# **Context:** In the DataFrame API, adding many derived columns by chaining `.withColumn()` in a loop nests one `Project` node per column, building a deep plan that is slow to analyze (see the DataFrame track's Exercise 7).
# 
# **In SQL there is no such anti-pattern.** A `SELECT` lists every derived column in a **single projection**, so the analyzed plan has just one `Project` no matter how many columns you compute. There's nothing to fix — this exercise demonstrates *why* the SQL surface sidesteps the problem.

# CELL ********************

# ============================================================
# 7️⃣ DEMONSTRATE — 100 derived columns are a single projection in SQL
# ============================================================

# Build a SELECT that computes 100 feature columns in one projection.
N_FEATURES = 100
feature_cols = ",\n           ".join(
    f"cycle_time_ms + {i} AS feat_{i}" for i in range(N_FEATURES)
)
sql_features = f"""
    SELECT machine_id, cycle_time_ms,
           {feature_cols}
    FROM {table_ref('manufacturing_event')}
"""
features_df = spark.sql(sql_features)

# The analyzed plan collapses all 100 derived columns into ONE Project node.
features_analyzed = features_df._jdf.queryExecution().analyzed().toString()
print(json.dumps({
    "featureColumns": N_FEATURES,
    "totalColumns": len(features_df.columns),
    "projectNodesInAnalyzedPlan": features_analyzed.count("Project"),
    "lesson": "SQL expresses every column in one projection - the DataFrame withColumn-loop trap does not exist here",
}, default=str, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Each `.withColumn()` call in the DataFrame API adds another nested `Project` and re-resolves the schema on the driver; chaining 100 of them builds a plan with ~100 stacked `Project` nodes that is slow to analyze (and can even `StackOverflow`).
# 
# In SQL you naturally list all 100 derived columns in one `SELECT`, and Spark represents that as a **single `Project`** — as the diagnostic above confirms. The projection-shape problem is specific to the imperative DataFrame builder pattern, so the SQL track has nothing to fix here.
# 
# > 📝 **Note:** If you *do* build SQL dynamically in a loop (string concatenation), assemble the full column list first and issue **one** `spark.sql(...)` — don't run a query per column.
# 
# ---

# MARKDOWN ********************

# ## Exercise 8 — Schema inference vs a declared schema
# 
# **Problem:** Creating a temp view over the JSON landing zone without a schema lets Spark **infer** the columns and types. To do that it must open and scan the files *before your query runs* — an eager, hidden startup cost that grows with the number of files.
# 
# **Why it matters:** Inference triggers an extra pass over the data every time the pipeline starts. On thousands of small landing-zone files that scan can dominate a job that otherwise reads very little.
# 
# **Fix in one line:** Declare the column schema in the `CREATE TEMPORARY VIEW ... (col type, ...)` DDL so the read skips inference entirely.
# 
# > NOTE: This exercise reads the raw JSON landing zone at `Files/landing/product_return` produced by the `source_to_bronze` job in setup. If that path is absent, skip this exercise.

# CELL ********************

# ============================================================
# 8️⃣ BENCHMARK — Time schema inference on the landing zone
# ============================================================

# The unoptimized pipeline left many small files in the landing zone.
# CREATE TEMPORARY VIEW ... USING json (no column list) resolves the data source
# and infers the schema eagerly, so timing the CREATE captures the inference scan.
LANDING_TABLE = "product_return"
landing_path = f"Files/landing/{LANDING_TABLE}"

print(f"🐌 Inferring schema from landing zone: {landing_path}\n")
print("   Spark must open files and read metadata to discover columns + types...\n")

with benchmark_op("Schema Inference", "inferred (file scan)", spark):
    spark.sql(f"""
        CREATE OR REPLACE TEMPORARY VIEW landing_inferred
        USING json
        OPTIONS (path '{landing_path}', multiline 'true')
    """)
    # Force schema resolution to complete inside the timed block.
    spark.sql("SELECT * FROM landing_inferred LIMIT 0").collect()

spark.sql("DESCRIBE landing_inferred").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 8️⃣ FIX — Declare the schema so the read skips inference
# ==================================================================================================

# A production pipeline defines the schema once in DDL — no file scanning needed.
with benchmark_op("Schema Inference", "static (no scan)", spark):
    spark.sql(f"""
        CREATE OR REPLACE TEMPORARY VIEW landing_declared (
            EventId STRING,
            Timestamp TIMESTAMP,
            MachineId STRING,
            PartNum STRING,
            ColorId INT,
            MoldTemp DECIMAL(5,1),
            InjectionPressure DECIMAL(6,1),
            CycleTimeMs INT,
            DefectDetected BOOLEAN,
            DefectType STRING,
            BatchId STRING
        )
        USING json
        OPTIONS (path '{landing_path}', multiline 'true')
    """)
    spark.sql("SELECT * FROM landing_declared LIMIT 0").collect()

spark.sql("DESCRIBE landing_declared").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# `CREATE TEMPORARY VIEW ... USING json` with no column list makes Spark **infer** the columns and types — and to do that it must open and scan the files *before your query runs*. That is an eager, hidden startup cost that grows with the number of files, so on thousands of small landing-zone files the inference scan can dominate a job that otherwise reads very little.
# 
# Declaring the columns in the DDL removes that inference pass entirely: Spark already knows the schema, so it skips straight to reading. Same columns, faster start.
# 
# > 📝 **Note:** A declared schema also protects you from silent type drift — inference can pick different types run-to-run as the data changes, whereas a static schema is deterministic.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 9 — Driver `collect()` and driver OOM
# 
# **Problem:** The inventory workflow pulls every transaction to the driver with `spark.sql(...).collect()` and aggregates in Python.
# 
# **Why it matters:** Pulling distributed data into one process can trip task-result transport limits, executor memory while serializing results, or `spark.driver.maxResultSize`.
# 
# **Fix in one line:** Keep the aggregation in SQL (`GROUP BY`) and only bring the small final result to the driver.

# CELL ********************

# ============================================================
# 9️⃣ BENCHMARK — Capture baseline query time
# ============================================================
from collections import defaultdict

# The baseline collects every raw transaction to the driver before aggregating.
print("🐌 Running baseline query with driver-side collect...\n")

print(f"About to collect {TABLE_METRICS['inventory_transaction']['rows']:,} inventory rows to the driver.")
print("spark.driver.maxResultSize =", spark.conf.get("spark.driver.maxResultSize"))

start = time.time()
with benchmark_op("Driver Collect", "before", spark):
    collected_inventory = spark.sql(
        f"SELECT line_id, part_num, quantity, transaction_type FROM {table_ref('inventory_transaction')}"
    ).collect()

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

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Rewrite the workflow so executors compute the net inventory by line in SQL. The driver should receive only the grouped result, not every raw transaction row. Use a `SUM(CASE WHEN ...)` to sign the quantity, then `GROUP BY line_id`.

# CELL ********************

# ==================================================================================================
# 9️⃣ FIX — Aggregate in SQL and collect only the small grouped result
# ==================================================================================================

# Spark computes net inventory by line before the driver receives the display result.
print("✅ Running fixed query with distributed aggregation...\n")

sql_driver_after = f"""
    SELECT line_id,
           SUM(CASE WHEN transaction_type IN ('CONSUMPTION', 'ORDER_PICK', 'SCRAP')
                    THEN -ABS(CAST(quantity AS INT))
                    ELSE CAST(quantity AS INT) END) AS net_quantity
    FROM {table_ref('inventory_transaction')}
    GROUP BY line_id
    ORDER BY net_quantity DESC
"""

with benchmark_op("Driver Collect", "after", spark):
    spark.sql(f"SELECT line_id, part_num, quantity, transaction_type FROM {table_ref('inventory_transaction')}") \
        .write.format("noop").mode("overwrite").save()

with benchmark_op("Driver Python Aggregation", "after", spark):
    result_driver_after = spark.sql(sql_driver_after)
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
record_result("9 driver collect / OOM", "after", {
    "antiPattern": "Driver-side collect and Python aggregation",
    "baselineCollectedRows": driver_before_evidence["collectedRows"],
    "fixedRawRowsCollected": 0,
    "fixedResultRowsReturnedToDriver": result_driver_after.count(),
    "improvement": "Aggregation runs on executors via SQL GROUP BY; only the small grouped result reaches the driver.",
})
result_driver_after.explain(mode="formatted")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# `spark.sql(...).collect()` (and `.toPandas()`) pulls **every raw row** into the single driver process. That transfer can trip task-result transport limits, exhaust executor memory while serializing results, or blow past `spark.driver.maxResultSize` — and the aggregation then runs single-threaded in Python instead of across the cluster.
# 
# Keeping the aggregation in SQL (`SUM(CASE WHEN ...) ... GROUP BY line_id`) lets executors do the heavy lifting; only the **small grouped result** reaches the driver. No raw rows cross the boundary, so the OOM risk is gone and the result is identical.
# 
# > 📝 **Note:** Reserve `collect()` / `toPandas()` for genuinely small, already-aggregated results. To peek at data, prefer `LIMIT` / `display()` which bound how much is pulled back.
# 
# ---

# MARKDOWN ********************

# ---
# 
# # 🏆 Performance Impact by Exercise
# 
# Execute the below to see the full impact across every exercise.
# 


# CELL ********************

# ============================================================
# SUMMARY — All benchmark results
# ============================================================

print_benchmark_summary()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Summary — Optimizing code (Spark SQL)
# 
# You worked through the same code-level Spark anti-patterns as the DataFrame track, each with the same loop: benchmark the symptom, diagnose it in the physical plan, change the SQL, and re-check.
# 
# 1. **Predicate pushdown** — filtered on the raw column so the scan prunes files instead of wrapping it in `SUBSTRING(...)` first.
# 1. **De-duplicate on the key, not the whole row** — `SELECT DISTINCT` the key columns instead of `SELECT DISTINCT *`.
# 1. **Prune before a window** — selected the key columns before a `ROW_NUMBER()` window so the Exchange/Sort no longer carries the nested `order_lines` array.
# 1. **One pass, not many** — replaced a per-category `UNION ALL` with a single `GROUP BY` so the table is scanned once.
# 1. **Cartesian / missing join key** — replaced `CROSS JOIN` with an equi-join so nested-loop work collapses to a hash/sort-merge join.
# 1. **Python UDFs → native SQL** — replaced `BatchEvalPython` with native SQL functions (NEE makes the rewrite optional — see Module 3).
# 1. **Projection shape** — saw that SQL expresses every column in one projection, so the DataFrame `withColumn`-loop trap doesn't exist here.
# 1. **Schema inference vs a declared schema** — declared the columns in DDL so the read skips the eager file-scan inference pass.
# 1. **Driver `collect()` / OOM** — kept the aggregation in SQL and returned only the small result to the driver.
# 
# Carry the same workflow into the next modules: benchmark the symptom, inspect the Spark UI and physical plan, check Delta metadata, change the right lever, and validate the before/after result.
