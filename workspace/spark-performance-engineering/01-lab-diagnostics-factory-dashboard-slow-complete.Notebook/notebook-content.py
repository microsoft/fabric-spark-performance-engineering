# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "d23a82d0-b0a6-44b8-be37-0ac615f18d2a",
# META       "default_lakehouse_name": "Lego",
# META       "default_lakehouse_workspace_id": "7fc5eff4-7153-4da9-b909-54981a3ffcdb"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Lab 1: Diagnostics - The Factory Dashboard is Slow (Complete Version)
# 
# This notebook follows the six-point Lab Details pattern from `lab-technical-plan.md` for each diagnostic prompt:
# 
# 1. **Context cell** - scenario story and what is broken
# 2. **Setup cell** - loads data and starts timers/job descriptions
# 3. **Problem cell** - intentionally broken/slow code
# 4. **Investigation cell** - completed diagnostic checks (`EXPLAIN`, metrics, counts, file stats)
# 5. **Fix cell** - optimized/read-safe version of the query
# 6. **Validation cell** - compares problem vs. fix metrics and output correctness
# 
# The notebook is intentionally self-contained and complete: every investigation and fix cell is filled in. It uses only existing data from the `Lego` lakehouse and does not write, optimize, vacuum, analyze, insert, update, delete, or merge any lakehouse data.

# CELL ********************

# Global setup used by every lab section
import json
import time
from collections import defaultdict

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

SCHEMA = "bronze"
LAB_RESULTS = []
PROBLEM_OUTPUTS = {}
FIX_OUTPUTS = {}
INVESTIGATIONS = {}


def table_ref(name: str) -> str:
    return f"`{SCHEMA}`.`{name}`"


def set_job(label: str) -> None:
    spark.sparkContext.setJobDescription(f"Lab 1 Complete - {label}")


def run_timed(label: str, fn):
    print(f"START|{label}")
    start = time.time()
    value = fn()
    elapsed = time.time() - start
    print(f"END|{label}|elapsedSeconds={elapsed:.3f}")
    return value, elapsed


def explain_to_string(df) -> str:
    return df._jdf.queryExecution().toString()


def record(prompt: str, phase: str, status: str, evidence: dict):
    row = {"prompt": prompt, "phase": phase, "status": status, "evidence": evidence}
    LAB_RESULTS.append(row)
    print("LAB_COMPLETE_RESULT|" + json.dumps(row, sort_keys=True, default=str))


def table_metrics(name: str) -> dict:
    ref = table_ref(name)
    detail = spark.sql(f"DESCRIBE DETAIL {ref}").collect()[0].asDict()
    num_files = int(detail.get("numFiles") or 0)
    size_bytes = int(detail.get("sizeInBytes") or 0)
    return {
        "table": f"{SCHEMA}.{name}",
        "rows": spark.table(ref).count(),
        "numFiles": num_files,
        "sizeBytes": size_bytes,
        "avgFileMB": (size_bytes / num_files / 1024 / 1024) if num_files else 0,
        "partitions": spark.table(ref).rdd.getNumPartitions(),
    }

print("Spark application ID:", spark.sparkContext.applicationId)
print("Current database:", spark.catalog.currentDatabase())
print("Schema:", SCHEMA)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Table discovery / prerequisite validation
set_job("00 prerequisite discovery")
expected_tables = [
    "manufacturing_event", "web_order", "sets", "themes",
    "inventory_transaction", "quality_inspection", "production_order"
]
available = {r.tableName for r in spark.sql(f"SHOW TABLES IN `{SCHEMA}`").collect()}
missing = [t for t in expected_tables if t not in available]
if missing:
    raise RuntimeError(f"Missing required Lab 1 tables in {SCHEMA}: {missing}")

TABLE_METRICS = {t: table_metrics(t) for t in expected_tables}
for metric in TABLE_METRICS.values():
    print("TABLE_METRIC|" + json.dumps(metric, sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Q1 Context - Daily defect rate by machine
# 
# The dashboard asks for the latest daily defect rate by machine, but the slow version transforms the string timestamp inline before filtering. The investigation should show the query scans the whole `manufacturing_event` table even though it returns one day of results.

# CELL ********************

# Q1 Setup cell: load source and select dashboard day
set_job("Q1 setup")
q1_mfg = spark.table(table_ref("manufacturing_event"))
q1_day = q1_mfg.select(F.max(F.to_date("data.timestamp")).alias("event_day")).collect()[0]["event_day"]
print("Q1 dashboard day:", q1_day)
print("Q1 source metrics:", json.dumps(TABLE_METRICS["manufacturing_event"], sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q1 Problem cell: string timestamp transformation blocks efficient filtering
set_job("Q1 problem full scan")
def q1_problem_action():
    df = (
        q1_mfg
        .withColumn("event_day", F.substring(F.col("data.timestamp"), 1, 10))
        .filter(F.col("event_day") == F.lit(str(q1_day)))
        .groupBy("event_day", F.col("data.machine_id").alias("machine_id"))
        .agg(
            F.count("*").alias("events"),
            F.sum(F.col("data.defect_detected").cast("int")).alias("defects"),
            (F.sum(F.col("data.defect_detected").cast("int")) / F.count("*")).alias("defect_rate"),
        )
        .orderBy(F.desc("defect_rate"), "machine_id")
    )
    rows = df.collect()
    return df, rows
(PROBLEM_OUTPUTS["Q1"], q1_problem_seconds) = run_timed("Q1 problem", q1_problem_action)
q1_problem_df, q1_problem_rows = PROBLEM_OUTPUTS["Q1"]
display(q1_problem_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q1 Investigation cell: completed diagnostic checks
set_job("Q1 investigation")
q1_problem_plan = explain_to_string(q1_problem_df)
q1_filtered_count = q1_mfg.filter(F.to_date("data.timestamp") == F.lit(q1_day)).count()
INVESTIGATIONS["Q1"] = {
    "problemPlanContainsSubstring": "substring" in q1_problem_plan.lower(),
    "sourceRows": TABLE_METRICS["manufacturing_event"]["rows"],
    "filteredRows": q1_filtered_count,
    "sourceFiles": TABLE_METRICS["manufacturing_event"]["numFiles"],
    "scanToOutputRatio": TABLE_METRICS["manufacturing_event"]["rows"] / max(q1_filtered_count, 1),
}
print(q1_problem_plan)
print("Q1_INVESTIGATION|" + json.dumps(INVESTIGATIONS["Q1"], sort_keys=True))
record("Q1", "investigation", "complete", INVESTIGATIONS["Q1"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q1 Fix cell: parse timestamp once, filter with typed date column, then aggregate
set_job("Q1 fix typed date filter")
def q1_fix_action():
    prepared = q1_mfg.select(
        F.to_date("data.timestamp").alias("event_day"),
        F.col("data.machine_id").alias("machine_id"),
        F.col("data.defect_detected").cast("int").alias("is_defect"),
    )
    df = (
        prepared
        .filter(F.col("event_day") == F.lit(q1_day))
        .groupBy("event_day", "machine_id")
        .agg(
            F.count("*").alias("events"),
            F.sum("is_defect").alias("defects"),
            (F.sum("is_defect") / F.count("*")).alias("defect_rate"),
        )
        .orderBy(F.desc("defect_rate"), "machine_id")
    )
    rows = df.collect()
    return df, rows
(FIX_OUTPUTS["Q1"], q1_fix_seconds) = run_timed("Q1 fix", q1_fix_action)
q1_fix_df, q1_fix_rows = FIX_OUTPUTS["Q1"]
display(q1_fix_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q1 Validation cell: fixed query returns same machine/day metrics
q1_problem_set = {(str(r["event_day"]), r["machine_id"], r["events"], r["defects"]) for r in q1_problem_rows}
q1_fix_set = {(str(r["event_day"]), r["machine_id"], r["events"], r["defects"]) for r in q1_fix_rows}
q1_valid = q1_problem_set == q1_fix_set and INVESTIGATIONS["Q1"]["scanToOutputRatio"] > 1
record("Q1", "validation", "passed" if q1_valid else "failed", {
    "sameBusinessResult": q1_problem_set == q1_fix_set,
    "problemSeconds": q1_problem_seconds,
    "fixSeconds": q1_fix_seconds,
    **INVESTIGATIONS["Q1"],
})
assert q1_valid, "Q1 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Q2 Context - Top 10 customers by spend
# 
# The dashboard calculates spend using a regular Python UDF. The investigation should identify Python execution in the plan. The fix replaces the UDF with native Spark SQL expressions.

# CELL ********************

# Q2 Setup cell: explode order lines and load reference dimensions
set_job("Q2 setup")
q2_orders = spark.table(table_ref("web_order"))
q2_lines = q2_orders.select(F.col("data.customer_id").alias("customer_id"), F.explode("data.order_lines").alias("line"))
q2_sets = spark.table(table_ref("sets")).select("set_num", F.col("theme_id").alias("set_theme_id"))
q2_themes = spark.table(table_ref("themes")).select(F.col("id").alias("theme_id"), F.col("name").alias("theme_name"))
print("Q2 order metrics:", json.dumps(TABLE_METRICS["web_order"], sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q2 Problem cell: Python UDF for arithmetic that Spark can do natively
set_job("Q2 problem Python UDF")
@F.udf(DoubleType())
def q2_python_line_total(quantity, unit_price, extended_price):
    if extended_price is not None:
        return float(extended_price)
    if quantity is None or unit_price is None:
        return 0.0
    return float(quantity) * float(unit_price)

def q2_problem_action():
    df = (
        q2_lines
        .withColumn("line_total", q2_python_line_total("line.quantity", "line.unit_price", "line.extended_price"))
        .join(q2_sets, F.col("line.set_num") == F.col("set_num"), "left")
        .join(q2_themes, F.col("set_theme_id") == F.col("theme_id"), "left")
        .groupBy("customer_id")
        .agg(F.sum("line_total").alias("total_spend"), F.count("*").alias("line_count"))
        .orderBy(F.desc("total_spend"), "customer_id")
        .limit(10)
    )
    rows = df.collect()
    return df, rows
(PROBLEM_OUTPUTS["Q2"], q2_problem_seconds) = run_timed("Q2 problem", q2_problem_action)
q2_problem_df, q2_problem_rows = PROBLEM_OUTPUTS["Q2"]
display(q2_problem_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q2 Investigation cell: completed plan inspection for Python UDF symptoms
set_job("Q2 investigation")
q2_problem_plan = explain_to_string(q2_problem_df)
INVESTIGATIONS["Q2"] = {
    "planContainsBatchEvalPython": "BatchEvalPython" in q2_problem_plan,
    "planContainsPythonUDF": "pythonUDF" in q2_problem_plan or "q2_python_line_total" in q2_problem_plan,
    "sourceRows": TABLE_METRICS["web_order"]["rows"],
}
print(q2_problem_plan)
print("Q2_INVESTIGATION|" + json.dumps(INVESTIGATIONS["Q2"], sort_keys=True))
record("Q2", "investigation", "complete", INVESTIGATIONS["Q2"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q2 Fix cell: native expression replaces the Python UDF
set_job("Q2 fix native expression")
def q2_fix_action():
    native_total = F.coalesce(F.col("line.extended_price"), F.col("line.quantity") * F.col("line.unit_price"), F.lit(0.0))
    df = (
        q2_lines
        .withColumn("line_total", native_total)
        .join(q2_sets, F.col("line.set_num") == F.col("set_num"), "left")
        .join(q2_themes, F.col("set_theme_id") == F.col("theme_id"), "left")
        .groupBy("customer_id")
        .agg(F.sum("line_total").alias("total_spend"), F.count("*").alias("line_count"))
        .orderBy(F.desc("total_spend"), "customer_id")
        .limit(10)
    )
    rows = df.collect()
    return df, rows
(FIX_OUTPUTS["Q2"], q2_fix_seconds) = run_timed("Q2 fix", q2_fix_action)
q2_fix_df, q2_fix_rows = FIX_OUTPUTS["Q2"]
display(q2_fix_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q2 Validation cell: same top customers/spend, no Python UDF in fixed plan
q2_fix_plan = explain_to_string(q2_fix_df)
q2_problem_pairs = [(r["customer_id"], round(float(r["total_spend"] or 0), 2), r["line_count"]) for r in q2_problem_rows]
q2_fix_pairs = [(r["customer_id"], round(float(r["total_spend"] or 0), 2), r["line_count"]) for r in q2_fix_rows]
q2_valid = q2_problem_pairs == q2_fix_pairs and "BatchEvalPython" not in q2_fix_plan
record("Q2", "validation", "passed" if q2_valid else "failed", {
    "sameBusinessResult": q2_problem_pairs == q2_fix_pairs,
    "fixedPlanContainsBatchEvalPython": "BatchEvalPython" in q2_fix_plan,
    "problemSeconds": q2_problem_seconds,
    "fixSeconds": q2_fix_seconds,
    **INVESTIGATIONS["Q2"],
})
assert q2_valid, "Q2 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Q3 Context - Inventory levels by plant/line
# 
# The slow dashboard path collects all inventory transactions to the driver and aggregates in Python. The fix performs the same aggregation in Spark.

# CELL ********************

# Q3 Setup cell: load inventory transactions
set_job("Q3 setup")
q3_inv = spark.table(table_ref("inventory_transaction")).select("line_id", "quantity", "transaction_type")
print("Q3 inventory metrics:", json.dumps(TABLE_METRICS["inventory_transaction"], sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q3 Problem cell: collect rows to the driver, then aggregate in Python
set_job("Q3 problem collect")
def q3_problem_action():
    rows = q3_inv.collect()
    totals = defaultdict(int)
    for row in rows:
        qty = int(row["quantity"] or 0)
        if row["transaction_type"] in ("CONSUMPTION", "ORDER_PICK", "SCRAP"):
            qty = -abs(qty)
        totals[row["line_id"] or "UNKNOWN"] += qty
    df = spark.createDataFrame([(k, v) for k, v in totals.items()], ["line_id", "net_quantity"]).orderBy("line_id")
    result_rows = df.collect()
    return df, result_rows, len(rows)
(PROBLEM_OUTPUTS["Q3"], q3_problem_seconds) = run_timed("Q3 problem", q3_problem_action)
q3_problem_df, q3_problem_rows, q3_collected_rows = PROBLEM_OUTPUTS["Q3"]
display(q3_problem_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q3 Investigation cell: completed check for driver collection risk
set_job("Q3 investigation")
INVESTIGATIONS["Q3"] = {
    "antiPattern": "collect() pulls source rows to driver",
    "collectedRows": q3_collected_rows,
    "sourceRows": TABLE_METRICS["inventory_transaction"]["rows"],
    "resultRows": len(q3_problem_rows),
}
print("Q3_INVESTIGATION|" + json.dumps(INVESTIGATIONS["Q3"], sort_keys=True))
record("Q3", "investigation", "complete", INVESTIGATIONS["Q3"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q3 Fix cell: Spark-native aggregation, no source collect
set_job("Q3 fix Spark aggregation")
def q3_fix_action():
    signed_qty = F.when(F.col("transaction_type").isin("CONSUMPTION", "ORDER_PICK", "SCRAP"), -F.abs(F.col("quantity"))).otherwise(F.col("quantity"))
    df = (
        q3_inv
        .withColumn("line_id_safe", F.coalesce(F.col("line_id"), F.lit("UNKNOWN")))
        .withColumn("signed_quantity", signed_qty.cast("long"))
        .groupBy("line_id_safe")
        .agg(F.sum("signed_quantity").alias("net_quantity"))
        .select(F.col("line_id_safe").alias("line_id"), "net_quantity")
        .orderBy("line_id")
    )
    rows = df.collect()
    return df, rows
(FIX_OUTPUTS["Q3"], q3_fix_seconds) = run_timed("Q3 fix", q3_fix_action)
q3_fix_df, q3_fix_rows = FIX_OUTPUTS["Q3"]
display(q3_fix_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q3 Validation cell: Spark fix returns identical inventory totals
q3_problem_map = {r["line_id"]: int(r["net_quantity"] or 0) for r in q3_problem_rows}
q3_fix_map = {r["line_id"]: int(r["net_quantity"] or 0) for r in q3_fix_rows}
q3_valid = q3_problem_map == q3_fix_map and q3_collected_rows == TABLE_METRICS["inventory_transaction"]["rows"]
record("Q3", "validation", "passed" if q3_valid else "failed", {
    "sameBusinessResult": q3_problem_map == q3_fix_map,
    "problemSeconds": q3_problem_seconds,
    "fixSeconds": q3_fix_seconds,
    **INVESTIGATIONS["Q3"],
})
assert q3_valid, "Q3 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Q4 Context - Manufacturing events per shift
# 
# The dashboard loops over machine/day combinations and runs a separate Spark action for each one. The fix computes all requested combinations in one grouped Spark query.

# CELL ********************

# Q4 Setup cell: choose bounded machine/day combinations
set_job("Q4 setup")
q4_mfg = spark.table(table_ref("manufacturing_event"))
q4_machines = [r["machine_id"] for r in q4_mfg.select(F.col("data.machine_id").alias("machine_id")).distinct().orderBy("machine_id").limit(8).collect()]
q4_days = [r["event_day"] for r in q4_mfg.select(F.to_date("data.timestamp").alias("event_day")).distinct().orderBy(F.desc("event_day")).limit(5).collect()]
print("Q4 machines:", q4_machines)
print("Q4 days:", [str(d) for d in q4_days])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q4 Problem cell: repeated table reads/actions in a nested loop
set_job("Q4 problem repeated scans")
def q4_problem_action():
    loop_rows = []
    for machine in q4_machines:
        for day in q4_days:
            count_value = (
                spark.table(table_ref("manufacturing_event"))
                .filter((F.col("data.machine_id") == machine) & (F.to_date("data.timestamp") == F.lit(day)))
                .count()
            )
            loop_rows.append((machine, str(day), count_value))
    df = spark.createDataFrame(loop_rows, ["machine_id", "event_day", "event_count"]).orderBy("machine_id", "event_day")
    rows = df.collect()
    return df, rows
(PROBLEM_OUTPUTS["Q4"], q4_problem_seconds) = run_timed("Q4 problem", q4_problem_action)
q4_problem_df, q4_problem_rows = PROBLEM_OUTPUTS["Q4"]
display(q4_problem_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q4 Investigation cell: completed diagnostic count of repeated Spark actions
set_job("Q4 investigation")
q4_expected_actions = len(q4_machines) * len(q4_days)
INVESTIGATIONS["Q4"] = {
    "machines": len(q4_machines),
    "days": len(q4_days),
    "separateCountActions": q4_expected_actions,
    "antiPattern": "nested loop triggers repeated scans/actions without caching",
}
print("Q4_INVESTIGATION|" + json.dumps(INVESTIGATIONS["Q4"], sort_keys=True))
record("Q4", "investigation", "complete", INVESTIGATIONS["Q4"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q4 Fix cell: one grouped query for all selected machines/days
set_job("Q4 fix grouped query")
def q4_fix_action():
    df = (
        q4_mfg
        .select(F.col("data.machine_id").alias("machine_id"), F.to_date("data.timestamp").alias("event_day"))
        .filter(F.col("machine_id").isin(q4_machines) & F.col("event_day").isin(q4_days))
        .groupBy("machine_id", "event_day")
        .agg(F.count("*").alias("event_count"))
        .withColumn("event_day", F.col("event_day").cast("string"))
        .orderBy("machine_id", "event_day")
    )
    rows = df.collect()
    return df, rows
(FIX_OUTPUTS["Q4"], q4_fix_seconds) = run_timed("Q4 fix", q4_fix_action)
q4_fix_df, q4_fix_rows = FIX_OUTPUTS["Q4"]
display(q4_fix_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q4 Validation cell: one grouped query returns same counts
q4_problem_map = {(r["machine_id"], r["event_day"]): int(r["event_count"]) for r in q4_problem_rows}
q4_fix_map_raw = {(r["machine_id"], r["event_day"]): int(r["event_count"]) for r in q4_fix_rows}
# The grouped fix naturally omits zero-count combinations; normalize them back to zero for a business-equivalent comparison.
q4_fix_map = {key: q4_fix_map_raw.get(key, 0) for key in q4_problem_map}
q4_valid = q4_problem_map == q4_fix_map and INVESTIGATIONS["Q4"]["separateCountActions"] > 1
record("Q4", "validation", "passed" if q4_valid else "failed", {
    "sameBusinessResult": q4_problem_map == q4_fix_map,
    "problemSeconds": q4_problem_seconds,
    "fixSeconds": q4_fix_seconds,
    "zeroCountCombinations": sum(1 for value in q4_problem_map.values() if value == 0),
    **INVESTIGATIONS["Q4"],
})
assert q4_valid, "Q4 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Q5 Context - Quality inspection pass rates
# 
# The slow version omits the join predicate between quality inspections and production orders. The investigation should find a Cartesian/nested-loop plan and a very large estimated joined row count. The fix joins on `production_order_id`.

# CELL ********************

# Q5 Setup cell: load quality and production order tables
set_job("Q5 setup")
q5_qi = spark.table(table_ref("quality_inspection")).select(
    F.col("data.production_order_id").alias("production_order_id"),
    F.col("data.pass_count").alias("pass_count"),
    F.col("data.sample_size").alias("sample_size"),
)
q5_po = spark.table(table_ref("production_order")).select(
    F.col("data.production_order_id").alias("production_order_id"),
    F.col("data.machine_id").alias("machine_id"),
)
q5_quality_rows = TABLE_METRICS["quality_inspection"]["rows"]
q5_order_rows = TABLE_METRICS["production_order"]["rows"]
print("Q5 estimated Cartesian pairs:", q5_quality_rows * q5_order_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q5 Problem cell: explicit cross join to reproduce the missing-predicate issue safely and visibly
set_job("Q5 problem Cartesian join")
def q5_problem_action():
    df = (
        q5_qi.crossJoin(q5_po.withColumnRenamed("production_order_id", "po_production_order_id"))
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
        )
        .orderBy(F.desc("joined_rows"), "machine_id")
    )
    rows = df.collect()
    return df, rows
(PROBLEM_OUTPUTS["Q5"], q5_problem_seconds) = run_timed("Q5 problem", q5_problem_action)
q5_problem_df, q5_problem_rows = PROBLEM_OUTPUTS["Q5"]
display(q5_problem_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q5 Investigation cell: completed plan inspection for Cartesian/nested-loop join
set_job("Q5 investigation")
q5_problem_plan = explain_to_string(q5_problem_df)
INVESTIGATIONS["Q5"] = {
    "estimatedCartesianPairs": q5_quality_rows * q5_order_rows,
    "planContainsCartesianProduct": "CartesianProduct" in q5_problem_plan,
    "planContainsBroadcastNestedLoopJoin": "BroadcastNestedLoopJoin" in q5_problem_plan,
    "qualityRows": q5_quality_rows,
    "productionOrderRows": q5_order_rows,
}
print(q5_problem_plan)
print("Q5_INVESTIGATION|" + json.dumps(INVESTIGATIONS["Q5"], sort_keys=True))
record("Q5", "investigation", "complete", INVESTIGATIONS["Q5"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q5 Fix cell: join on production_order_id before aggregation
set_job("Q5 fix keyed join")
def q5_fix_action():
    df = (
        q5_qi.join(q5_po, "production_order_id", "inner")
        .groupBy("machine_id")
        .agg(
            F.count("*").alias("joined_rows"),
            (F.sum("pass_count") / F.sum("sample_size")).alias("pass_rate"),
        )
        .orderBy(F.desc("joined_rows"), "machine_id")
    )
    rows = df.collect()
    return df, rows
(FIX_OUTPUTS["Q5"], q5_fix_seconds) = run_timed("Q5 fix", q5_fix_action)
q5_fix_df, q5_fix_rows = FIX_OUTPUTS["Q5"]
display(q5_fix_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q5 Validation cell: fixed plan removes Cartesian join and reduces joined rows
q5_fix_plan = explain_to_string(q5_fix_df)
q5_problem_joined = sum(int(r["joined_rows"] or 0) for r in q5_problem_rows)
q5_fix_joined = sum(int(r["joined_rows"] or 0) for r in q5_fix_rows)
q5_valid = q5_fix_joined < q5_problem_joined and "CartesianProduct" not in q5_fix_plan and "BroadcastNestedLoopJoin" not in q5_fix_plan
record("Q5", "validation", "passed" if q5_valid else "failed", {
    "problemJoinedRows": q5_problem_joined,
    "fixJoinedRows": q5_fix_joined,
    "fixedPlanContainsCartesianProduct": "CartesianProduct" in q5_fix_plan,
    "fixedPlanContainsBroadcastNestedLoopJoin": "BroadcastNestedLoopJoin" in q5_fix_plan,
    "problemSeconds": q5_problem_seconds,
    "fixSeconds": q5_fix_seconds,
    **INVESTIGATIONS["Q5"],
})
assert q5_valid, "Q5 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Q6 Context - Monthly revenue trend
# 
# The repo has inconsistent Q6 wording (`web_order` tiny files vs. `manufacturing_event_messy`). This workspace has no `*_messy` tables, so the problem uses existing `bronze.web_order`, whose average Delta file size is tiny. The read-only fix avoids modifying data and reduces repeated scan cost by projecting and caching only the small columns needed for the dashboard calculation.

# CELL ********************

# Q6 Setup cell: load web orders and small-file metrics
set_job("Q6 setup")
q6_orders = spark.table(table_ref("web_order"))
q6_metrics = TABLE_METRICS["web_order"]
print("Q6 web_order metrics:", json.dumps(q6_metrics, sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q6 Problem cell: monthly revenue trend directly scans tiny web_order files
set_job("Q6 problem tiny-file scan")
def q6_problem_action():
    df = (
        q6_orders
        .withColumn("order_month", F.substring(F.col("data.order_date"), 1, 7))
        .groupBy("order_month")
        .agg(F.count("*").alias("orders"), F.sum("data.order_total").alias("revenue"))
        .orderBy("order_month")
    )
    rows = df.collect()
    return df, rows
(PROBLEM_OUTPUTS["Q6"], q6_problem_seconds) = run_timed("Q6 problem", q6_problem_action)
q6_problem_df, q6_problem_rows = PROBLEM_OUTPUTS["Q6"]
display(q6_problem_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q6 Investigation cell: completed small-file diagnostics
set_job("Q6 investigation")
q6_problem_plan = explain_to_string(q6_problem_df)
INVESTIGATIONS["Q6"] = {
    "table": q6_metrics["table"],
    "numFiles": q6_metrics["numFiles"],
    "sizeBytes": q6_metrics["sizeBytes"],
    "avgFileMB": q6_metrics["avgFileMB"],
    "smallFileThresholdMB": 32,
    "smallFileProblemDetected": q6_metrics["avgFileMB"] < 32,
    "missingMessyTables": ["manufacturing_event_messy", "web_order_messy"],
}
print(q6_problem_plan)
print("Q6_INVESTIGATION|" + json.dumps(INVESTIGATIONS["Q6"], sort_keys=True))
record("Q6", "investigation", "complete", INVESTIGATIONS["Q6"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q6 Fix cell: read-only projection cache for the dashboard columns
set_job("Q6 fix projected cache")
def q6_fix_action():
    projected = (
        q6_orders
        .select(F.substring(F.col("data.order_date"), 1, 7).alias("order_month"), F.col("data.order_total").alias("order_total"))
        .cache()
    )
    projected.count()  # materialize the cache; this is in-memory only and does not modify lakehouse data
    df = (
        projected
        .groupBy("order_month")
        .agg(F.count("*").alias("orders"), F.sum("order_total").alias("revenue"))
        .orderBy("order_month")
    )
    rows = df.collect()
    projected.unpersist()
    return df, rows
(FIX_OUTPUTS["Q6"], q6_fix_seconds) = run_timed("Q6 fix", q6_fix_action)
q6_fix_df, q6_fix_rows = FIX_OUTPUTS["Q6"]
display(q6_fix_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Q6 Validation cell: same revenue trend, small-file diagnosis recorded
q6_problem_pairs = [(r["order_month"], r["orders"], round(float(r["revenue"] or 0), 2)) for r in q6_problem_rows]
q6_fix_pairs = [(r["order_month"], r["orders"], round(float(r["revenue"] or 0), 2)) for r in q6_fix_rows]
q6_valid = q6_problem_pairs == q6_fix_pairs and INVESTIGATIONS["Q6"]["smallFileProblemDetected"]
record("Q6", "validation", "passed" if q6_valid else "failed", {
    "sameBusinessResult": q6_problem_pairs == q6_fix_pairs,
    "problemSeconds": q6_problem_seconds,
    "fixSeconds": q6_fix_seconds,
    **INVESTIGATIONS["Q6"],
})
assert q6_valid, "Q6 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Final validation summary
# 
# All six prompts above include context, setup, problem, investigation, fix, and validation. The final cell emits a machine-readable summary for CLI validation.

# CELL ********************

# Final validation summary
set_job("final complete summary")
validation_rows = [r for r in LAB_RESULTS if r["phase"] == "validation"]
failed = [r for r in validation_rows if r["status"] != "passed"]
summary = {
    "sparkApplicationId": spark.sparkContext.applicationId,
    "schema": SCHEMA,
    "validationCount": len(validation_rows),
    "failedValidationCount": len(failed),
    "results": LAB_RESULTS,
}
print("LAB_COMPLETE_FINAL_SUMMARY_START")
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
print("LAB_COMPLETE_FINAL_SUMMARY_END")
assert len(validation_rows) == 6, f"Expected 6 validation rows, got {len(validation_rows)}"
assert not failed, f"Failed validations: {failed}"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
