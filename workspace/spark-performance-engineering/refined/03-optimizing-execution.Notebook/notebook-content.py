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

# # Module 3 — Optimizing Execution
# 
# This module is for Toy Brick Manufacturing Spark workloads where the code is logically correct and the tables are already well-designed, but Spark runs the work sub-optimally because of data distribution or resource choices.
# 
# ## What this module teaches
# 
# - Fix execution with join strategy, AQE, partition sizing, salting, caching, and streaming state choices.
# - Keep transformation logic and results identical while changing how Spark executes the query.
# - Use Spark UI stages, physical plans, task skew, and spill metrics to prove the bottleneck.
# 
# ## Assumed from prior modules
# 
# - Module 1: diagnostic toolkit, including plans, Spark UI, and task metrics.
# - Module 2: tables are already well-designed; do not change layout or source data here.
# 
# **Litmus note:** Module 3 fixes are hints/config/`.cache()`/repartition. Module 1 rewrites inefficient code. Module 2 changes table design.


# MARKDOWN ********************

# ## Exercise summary
# 
# | Exercise | Scenario | Expected performance signal |
# |---|---|---|
# | 1. Join strategies / broadcast | High-volume manufacturing events join to small production-order and parts references. | Results stay identical; `SortMergeJoin` becomes `BroadcastHashJoin` and shuffle/sort overhead is removed for small references. |
# | 2. Skew handling / AQE skew join | One machine dominates the event join key and creates a straggler shuffle partition. | Results stay identical; AQE skew join splits the hot partition and task times rebalance, with manual salting as the fallback. |
# | 3. Shuffle partition sizing (tiny-task storm) | A KPI rollup runs with a large static shuffle-partition count and AQE coalescing off. | Results stay identical; AQE coalesces the near-empty partitions into right-sized tasks, removing the tiny-task storm. |
# | 4. Caching / materialization | Multiple dashboard branches repeatedly read and join the same inventory/order base. | Results stay identical; the shared joined base is materialized once and reused through cache hits instead of repeated source scans. |
# | 5. Streaming state and watermarking | An hourly stateful streaming aggregation runs without an event-time watermark. | Results stay identical; adding a watermark bounds streaming state (`EventTimeWatermark`) so old windows are dropped and memory stays bounded. |
# | 6. Python UDFs / Native Execution Engine (NEE) | A correct top-customer query uses scalar Python UDFs, slow on the JVM. | Results stay identical; enabling NEE runs the same UDF code natively (no `BatchEvalPython`), removing the Python-boundary slowdown without any code change. |


# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Setup: reset the work schema, snapshot execution configs, and validate sources.
from pyspark import StorageLevel
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import broadcast, spark_partition_id

SOURCE_SCHEMA = "bronze"
WORK_SCHEMA = "opt_exec"

for key in [
    "spark.sql.adaptive.enabled",
    "spark.sql.adaptive.skewJoin.enabled",
    "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.autoBroadcastJoinThreshold",
    "spark.sql.shuffle.partitions",
    "spark.sql.streaming.statefulOperator.checkCorrectness.enabled",
]:
    remember_conf(key)

reset_work_schema(WORK_SCHEMA)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.microsoft.delta.parallelSnapshotLoading.enabled", "true")
spark.conf.set("spark.microsoft.delta.snapshot.driverMode.enabled", "true")

required = [
    "manufacturing_event", "production_order", "parts", "web_order",
    "inventory_transaction", "inventory_parts", "inventories", "sets", "themes",
]
require_tables(required, SOURCE_SCHEMA)
SOURCE_METRICS = {t: get_table_metrics(table_ref(t, SOURCE_SCHEMA)) for t in required[:4]}
show_metrics(table_ref("manufacturing_event", SOURCE_SCHEMA), "read-only source sample")
print("Spark application ID:", spark.sparkContext.applicationId)
print("Read-only source schema:", SOURCE_SCHEMA, "| Work schema:", WORK_SCHEMA)
print(json.dumps({"sourceMetricsSample": SOURCE_METRICS}, indent=2, sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 1 — Join strategies / broadcast
# 
# ### Context and problem
# 
# A correct query joins high-volume manufacturing events to small production-order and part reference tables. With automatic broadcast disabled, Spark defaults to sort-merge joins, adding shuffle and sort overhead. Fix only the join strategy; the aggregation stays identical.


# CELL ********************

# ============================================================
# 1️⃣ BENCHMARK — Baseline sort-merge join with broadcast disabled
# ============================================================

# Baseline: correct result, suboptimal sort-merge execution.
set_job("1 baseline sort-merge join")
remember_conf("spark.sql.autoBroadcastJoinThreshold")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
q1_events = spark.table(table_ref("manufacturing_event")).select(
    F.col("manufacturing_event.production_order_id").alias("production_order_id"),
    F.col("manufacturing_event.part_num").alias("part_num"),
    F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"),
    F.col("manufacturing_event.cycle_time_ms").alias("cycle_time_ms"),
)
q1_orders = spark.table(table_ref("production_order")).select(
    F.col("production_order.production_order_id").alias("production_order_id"),
    F.col("production_order.status").alias("status"),
)
q1_parts = spark.table(table_ref("parts")).select("part_num", "part_material", "part_cat_id")
with benchmark_op("Join strategy / broadcast", "before", spark):
    q1_before_df = (q1_events.join(q1_orders, "production_order_id").join(q1_parts, "part_num")
        .groupBy("part_material").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"), F.avg("cycle_time_ms").alias("avg_cycle_ms"))
        .orderBy("part_material"))
    q1_before_rows = q1_before_df.collect()
display(spark.createDataFrame(q1_before_rows))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 1️⃣ DIAGNOSE — Plan proves SortMergeJoin and shuffle stages before broadcast
# =================================================================================================

# Diagnosis: verify sort-merge and use Spark UI > SQL/DataFrame for shuffle stages.
q1_before_plan = explain_string(q1_before_df)
print(q1_before_plan)
print(json.dumps({
    "hasSortMergeJoin": "SortMergeJoin" in q1_before_plan,
    "hasBroadcastHashJoin": "BroadcastHashJoin" in q1_before_plan,
    "sparkUIPointer": "SQL/DataFrame query details and shuffle stages",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Broadcast the small reference tables, or restore automatic broadcast/AQE, then verify `BroadcastHashJoin` appears in the physical plan.


# CELL ********************

# Challenge starter: edit the joins below to use broadcast on small references.
q1_attempt_df = (q1_events.join(q1_orders, "production_order_id").join(q1_parts, "part_num")
    .groupBy("part_material").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"), F.avg("cycle_time_ms").alias("avg_cycle_ms"))
    .orderBy("part_material"))
print(explain_string(q1_attempt_df)[:1200])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 1️⃣ FIX — Add broadcast hints for small references while aggregation logic stays unchanged
# ==================================================================================================

# ✅ Solution: force broadcast hash joins for small dimensions.
set_job("1 solution broadcast join")
with benchmark_op("Join strategy / broadcast", "after", spark):
    q1_after_df = (q1_events.join(broadcast(q1_orders), "production_order_id").join(broadcast(q1_parts), "part_num")
        .groupBy("part_material").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"), F.avg("cycle_time_ms").alias("avg_cycle_ms"))
        .orderBy("part_material"))
    q1_after_rows = q1_after_df.collect()
display(spark.createDataFrame(q1_after_rows))
restore_conf("spark.sql.autoBroadcastJoinThreshold")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 1️⃣ CHECK-CHANGES — Compare against baseline (results identical)
# ============================================================

# Validation: same result, broadcast plan.
q1_after_plan = explain_string(q1_after_df)
q1_before_map = {r["part_material"]: (r["events"], r["defects"], round(float(r["avg_cycle_ms"] or 0), 4)) for r in q1_before_rows}
q1_after_map = {r["part_material"]: (r["events"], r["defects"], round(float(r["avg_cycle_ms"] or 0), 4)) for r in q1_after_rows}
valid = q1_before_map == q1_after_map and "BroadcastHashJoin" in q1_after_plan
record_result("1 join strategy / broadcast", "passed" if valid else "failed", {
    "sameBusinessResult": q1_before_map == q1_after_map,
    "beforeHasSortMergeJoin": "SortMergeJoin" in q1_before_plan,
    "afterHasBroadcastHashJoin": "BroadcastHashJoin" in q1_after_plan,
})
assert valid, "Exercise 1 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 2 — Skew handling / AQE skew join
# 
# ### Context and problem
# 
# A plant throughput rollup joins high-volume manufacturing events to a small machine dimension. One machine ("the busy one") produces orders of magnitude more events than the others — classic data skew on the **join key**. With broadcast disabled the join is a shuffle join, and with AQE skew handling off the hot key lands in a single shuffle partition, so one straggler task dominates the stage. The fix keeps the query identical and lets AQE split the hot partition; manual salting is the fallback when AQE will not trigger.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Setup: materialize a skewed fact and a small machine dimension in the work schema.
set_job("2 setup skewed plant join")

q2_events = spark.table(table_ref("manufacturing_event")).select(
    F.col("manufacturing_event.machine_id").alias("machine_id"),
    F.col("manufacturing_event.event_id").alias("event_id"),
    F.col("manufacturing_event.mold_temp").alias("mold_temp"),
    F.col("manufacturing_event.defect_detected").cast("int").alias("defect_detected"),
)
q2_hot_machine = q2_events.groupBy("machine_id").count().orderBy(F.desc("count")).first()["machine_id"]
print("Hot machine:", q2_hot_machine)

# ~100x extra copies of the hot machine's rows -> one dominant join key.
q2_dup = 100
q2_hot = q2_events.filter(F.col("machine_id") == q2_hot_machine)
q2_skewed = q2_events.unionByName(
    q2_hot.crossJoin(spark.range(q2_dup).withColumnRenamed("id", "_dup")).drop("_dup")
)
spark.sql(f"DROP TABLE IF EXISTS {table_ref('skewed_events', WORK_SCHEMA)}")
q2_skewed.write.mode("overwrite").saveAsTable(table_ref("skewed_events", WORK_SCHEMA))

# One row per machine; a small dimension we deliberately shuffle-join (broadcast disabled below).
q2_machine_dim = (q2_events.select("machine_id").distinct()
    .withColumn("plant", F.concat(F.lit("PLANT-"), F.substring("machine_id", 6, 3))))
spark.sql(f"DROP TABLE IF EXISTS {table_ref('machine_dim', WORK_SCHEMA)}")
q2_machine_dim.write.mode("overwrite").saveAsTable(table_ref("machine_dim", WORK_SCHEMA))

display(spark.sql(f"SELECT machine_id, COUNT(*) AS row_count FROM {table_ref('skewed_events', WORK_SCHEMA)} GROUP BY machine_id ORDER BY row_count DESC LIMIT 10"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Baseline skewed shuffle join with AQE skew handling OFF
# ============================================================

# Baseline: force a SortMergeJoin (broadcast disabled) with skew handling OFF so the hot key becomes a straggler.
set_job("2 baseline skewed join")
remember_conf("spark.sql.adaptive.skewJoin.enabled")
remember_conf("spark.sql.adaptive.coalescePartitions.enabled")
remember_conf("spark.sql.autoBroadcastJoinThreshold")
remember_conf("spark.sql.shuffle.partitions")
remember_conf("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes")
remember_conf("spark.sql.adaptive.advisoryPartitionSizeInBytes")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.shuffle.partitions", "200")


def q2_plant_rollup():
    return (spark.table(table_ref("skewed_events", WORK_SCHEMA)).alias("e")
        .join(spark.table(table_ref("machine_dim", WORK_SCHEMA)).alias("m"), "machine_id")
        .groupBy("plant")
        .agg(F.count("*").alias("events"), F.avg("mold_temp").alias("avg_temp"), F.sum(F.col("defect_detected")).alias("defects"))
        .orderBy("plant"))


with benchmark_op("Skew handling / AQE skew join", "before", spark):
    q2_before_df = q2_plant_rollup()
    q2_before_rows = q2_before_df.collect()

display(spark.createDataFrame(q2_before_rows))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 2️⃣ DIAGNOSE — Join-key partition distribution proves one straggler partition
# =================================================================================================

# Diagnosis: hashing the join key into 200 partitions shows one partition holding the hot machine.
q2_before_plan = explain_string(q2_before_df)
print(json.dumps({
    "hotMachine": q2_hot_machine,
    "aqeSkewJoinEnabled": spark.conf.get("spark.sql.adaptive.skewJoin.enabled"),
    "hasSortMergeJoin": "SortMergeJoin" in q2_before_plan,
    "sparkUIPointer": "Stages > Tasks: sort by Duration; one straggler task dwarfs the rest",
}, indent=2))
spark.table(table_ref("skewed_events", WORK_SCHEMA)).repartition(200, "machine_id").groupBy(spark_partition_id().alias("pid")).count().orderBy(F.desc("count")).show(10, False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Keep the plant rollup exactly as written. Re-enable `spark.sql.adaptive.skewJoin.enabled` and lower `skewedPartitionThresholdInBytes` / `advisoryPartitionSizeInBytes` so AQE splits the hot partition on this small, highly-compressible lab data. (Manual salting — add `pmod(xxhash64(event_id), N)` to the fact and explode the dimension across the same salts — is the fallback when AQE will not trigger.)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: inspect how a deterministic salt would spread the hot machine.
q2_salt_buckets = 16
q2_salt_preview = spark.table(table_ref("skewed_events", WORK_SCHEMA)).withColumn("salt", F.pmod(F.xxhash64("event_id"), F.lit(q2_salt_buckets)))
display(q2_salt_preview.groupBy("machine_id", "salt").count().orderBy(F.desc("count")).limit(12))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 2️⃣ FIX — Enable AQE skew join; the query is unchanged
# ==================================================================================================

# ✅ Solution: turn AQE skew handling on and lower the byte thresholds so it fires on small lab data.
set_job("2 solution AQE skew join")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "4m")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "8m")

with benchmark_op("Skew handling / AQE skew join", "after", spark):
    q2_after_df = q2_plant_rollup()
    q2_after_rows = q2_after_df.collect()

# Drive the plan from this DataFrame's own action so executedPlan holds the final adaptive plan.
q2_after_final_plan = q2_after_df._jdf.queryExecution().executedPlan().toString()
print("AQEShuffleRead skew split present:", "skew" in q2_after_final_plan.lower())
display(spark.createDataFrame(q2_after_rows))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ CHECK-CHANGES — Compare against baseline (results identical)
# ============================================================

# Validation: same plant rollup, skew now handled by AQE.
q2_before_map = {r["plant"]: (int(r["events"]), int(r["defects"]), round(float(r["avg_temp"] or 0), 4)) for r in q2_before_rows}
q2_after_map = {r["plant"]: (int(r["events"]), int(r["defects"]), round(float(r["avg_temp"] or 0), 4)) for r in q2_after_rows}
valid = q2_before_map == q2_after_map and spark.conf.get("spark.sql.adaptive.skewJoin.enabled") == "true"
record_result("2 skew handling / AQE skew join", "passed" if valid else "failed", {
    "sameBusinessResult": q2_before_map == q2_after_map,
    "hotMachine": q2_hot_machine,
    "aqeSkewJoinEnabled": spark.conf.get("spark.sql.adaptive.skewJoin.enabled"),
})
assert valid, "Exercise 2 validation failed"
restore_conf("spark.sql.adaptive.skewJoin.enabled")
restore_conf("spark.sql.adaptive.coalescePartitions.enabled")
restore_conf("spark.sql.autoBroadcastJoinThreshold")
restore_conf("spark.sql.shuffle.partitions")
restore_conf("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes")
restore_conf("spark.sql.adaptive.advisoryPartitionSizeInBytes")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 3 — Shuffle partition sizing (tiny-task storm)
# 
# ### Context and problem
# 
# A plant KPI rollup is correct, but someone set `spark.sql.shuffle.partitions` to a large static value "to be safe" and disabled AQE coalescing. On a small result the aggregation produces thousands of nearly-empty shuffle partitions — each paying task-launch overhead — so most of the wall-clock is scheduling, not compute. The fix keeps the query identical and lets AQE coalesce the tiny partitions into right-sized tasks.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Baseline over-partitioned shuffle with AQE coalesce OFF
# ============================================================

# Baseline: a large static shuffle-partition count with AQE coalescing disabled -> a tiny-task storm.
set_job("3 baseline over-partitioned shuffle")
remember_conf("spark.sql.shuffle.partitions")
remember_conf("spark.sql.adaptive.coalescePartitions.enabled")
spark.conf.set("spark.sql.shuffle.partitions", "1000")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

q3_events = spark.table(table_ref("manufacturing_event")).select(
    F.col("manufacturing_event.machine_id").alias("machine_id"),
    F.col("manufacturing_event.cycle_time_ms").alias("cycle_time_ms"),
    F.col("manufacturing_event.defect_detected").cast("int").alias("defect_detected"),
)


def q3_machine_rollup():
    return (q3_events.groupBy("machine_id")
        .agg(F.count("*").alias("events"), F.avg("cycle_time_ms").alias("avg_cycle_ms"), F.sum(F.col("defect_detected")).alias("defects"))
        .orderBy(F.desc("events"), "machine_id"))


with benchmark_op("Shuffle partition sizing", "before", spark):
    q3_before_df = q3_machine_rollup()
    q3_before_rows = q3_before_df.collect()

display(spark.createDataFrame(q3_before_rows).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3️⃣ DIAGNOSE — Static partition count creates thousands of near-empty tasks
# =================================================================================================

# Diagnosis: confirm the static shuffle-partition count and that AQE coalescing is off.
q3_before_plan = explain_string(q3_before_df)
print(json.dumps({
    "shufflePartitions": spark.conf.get("spark.sql.shuffle.partitions"),
    "aqeCoalesceEnabled": spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"),
    "hasExchange": "Exchange" in q3_before_plan,
    "sparkUIPointer": "Stages: thousands of tasks, most processing 0 rows",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Keep the KPI query exactly as written and change only execution: re-enable `spark.sql.adaptive.coalescePartitions.enabled` so AQE merges the near-empty shuffle partitions into right-sized chunks. No code change is required.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: check the current execution settings before fixing them.
print(json.dumps({
    "shufflePartitions": spark.conf.get("spark.sql.shuffle.partitions"),
    "aqeCoalesceEnabled": spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"),
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3️⃣ FIX — Re-enable AQE coalescing; the query is unchanged
# ==================================================================================================

# ✅ Solution: let AQE coalesce the over-partitioned shuffle into right-sized tasks.
set_job("3 solution AQE coalesce")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

with benchmark_op("Shuffle partition sizing", "after", spark):
    q3_after_df = q3_machine_rollup()
    q3_after_rows = q3_after_df.collect()

display(spark.createDataFrame(q3_after_rows).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ CHECK-CHANGES — Compare against baseline (results identical)
# ============================================================

# Validation: same KPI result with AQE coalescing the tiny partitions.
q3_before_map = {r["machine_id"]: (int(r["events"]), int(r["defects"]), round(float(r["avg_cycle_ms"] or 0), 4)) for r in q3_before_rows}
q3_after_map = {r["machine_id"]: (int(r["events"]), int(r["defects"]), round(float(r["avg_cycle_ms"] or 0), 4)) for r in q3_after_rows}
valid = q3_before_map == q3_after_map and spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"
record_result("3 shuffle partition sizing (tiny-task storm)", "passed" if valid else "failed", {
    "sameBusinessResult": q3_before_map == q3_after_map,
    "staticShufflePartitions": "1000",
    "aqeCoalesceEnabled": spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"),
})
assert valid, "Exercise 3 validation failed"
restore_conf("spark.sql.shuffle.partitions")
restore_conf("spark.sql.adaptive.coalescePartitions.enabled")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 4 — Caching / materialization for repeated reads
# 
# ### Context and problem
# 
# A dashboard builds three rollups — by order status, by part material, and by machine — all from the **same** enriched base: inventory transactions joined to production orders and parts. The baseline recomputes that big-fact scan-and-join once per rollup, so the expensive work runs three times. The fix materializes the shared base once and reuses it; every rollup's filters and aggregations stay identical.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Baseline recomputes the shared enriched base for every rollup
# ============================================================

# Baseline: each dashboard rollup rebuilds the same inventory-order-parts join from scratch.
set_job("4 baseline repeated joins")
q4_orders = spark.table(table_ref("production_order")).select(
    F.col("production_order.production_order_id").alias("reference_id"),
    F.col("production_order.status").alias("order_status"),
    F.col("production_order.machine_id").alias("machine_id"),
)
q4_parts = spark.table(table_ref("parts")).select("part_num", "part_material")


def q4_enriched():
    # The expensive shared base: scan the large fact and join the two dimensions.
    return (spark.table(table_ref("inventory_transaction"))
        .select("transaction_type", "reference_id", "quantity", "part_num")
        .join(q4_orders, "reference_id", "left")
        .join(q4_parts, "part_num", "left")
        .select("transaction_type", "quantity", "order_status", "machine_id", "part_material"))


def q4_by_status(base):
    return base.groupBy("order_status").agg(F.count("*").alias("transactions"), F.sum("quantity").alias("quantity")).orderBy("order_status")


def q4_by_material(base):
    return base.groupBy("part_material").agg(F.count("*").alias("transactions"), F.sum("quantity").alias("quantity")).orderBy("part_material")


def q4_by_machine(base):
    return base.groupBy("machine_id").agg(F.count("*").alias("transactions"), F.sum("quantity").alias("quantity")).orderBy("machine_id")


with benchmark_op("Caching / repeated reads", "before", spark):
    q4_before_status = q4_by_status(q4_enriched()).collect()
    q4_before_material = q4_by_material(q4_enriched()).collect()
    q4_before_machine = q4_by_machine(q4_enriched()).collect()

display(spark.createDataFrame(q4_before_status))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 4️⃣ DIAGNOSE — Repeated FileScan operators prove the shared base is recomputed
# =================================================================================================

# Diagnosis: every rollup re-scans and re-joins the same base; Spark UI Jobs tab shows a job per collect.
q4_before_plans = [
    explain_string(q4_by_status(q4_enriched())),
    explain_string(q4_by_material(q4_enriched())),
    explain_string(q4_by_machine(q4_enriched())),
]
q4_file_scans_before = sum(plan.count("FileScan") for plan in q4_before_plans)
print(json.dumps({
    "antiPattern": "Recompute the same inventory-order-parts join for every rollup",
    "dashboards": ["by order_status", "by part_material", "by machine_id"],
    "rollups": len(q4_before_plans),
    "totalFileScanOperatorsAcrossPlans": q4_file_scans_before,
    "sparkUIPointer": "Jobs tab: every collect re-runs the full scan + join",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Materialize the shared enriched base once, then build all three rollups from the cached DataFrame. The base is reused in full (no per-rollup filter), so caching removes two of the three big-fact scans.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: this enriched join is the cache candidate (reused in full by every rollup).
q4_candidate_base = q4_enriched()
print("Cache and materialize this enriched base, then aggregate it three different ways.")
print(explain_string(q4_candidate_base)[:1200])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Cache and materialize the enriched base once while rollup logic stays unchanged
# ==================================================================================================

# ✅ Solution: materialize the shared base once, then reuse it for all three rollups.
set_job("4 solution cache enriched base")
q4_after_plans = []
with benchmark_op("Caching / repeated reads", "after", spark):
    q4_cached_base = q4_enriched().persist(StorageLevel.MEMORY_AND_DISK)
    q4_cached_base.count()
    q4_status_df = q4_by_status(q4_cached_base)
    q4_material_df = q4_by_material(q4_cached_base)
    q4_machine_df = q4_by_machine(q4_cached_base)
    q4_after_plans = [explain_string(q4_status_df), explain_string(q4_material_df), explain_string(q4_machine_df)]
    q4_after_status = q4_status_df.collect()
    q4_after_material = q4_material_df.collect()
    q4_after_machine = q4_machine_df.collect()

display(spark.createDataFrame(q4_after_status))
q4_cached_base.unpersist()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ CHECK-CHANGES — Compare against baseline (results identical)
# ============================================================

# Validation: every rollup matches the baseline and now reads from the in-memory cache.
def _q4_map(rows, key):
    return {r[key]: (int(r["transactions"]), int(r["quantity"] or 0)) for r in rows}

same_result = (
    _q4_map(q4_before_status, "order_status") == _q4_map(q4_after_status, "order_status")
    and _q4_map(q4_before_material, "part_material") == _q4_map(q4_after_material, "part_material")
    and _q4_map(q4_before_machine, "machine_id") == _q4_map(q4_after_machine, "machine_id")
)
q4_in_memory_scans = sum(plan.count("InMemoryTableScan") for plan in q4_after_plans)
valid = same_result and q4_in_memory_scans > 0
record_result("4 caching / materialization", "passed" if valid else "failed", {
    "sameBusinessResult": same_result,
    "fileScansBefore": q4_file_scans_before,
    "inMemoryScansAfter": q4_in_memory_scans,
    "cachedBase": "inventory_transaction joined to production_order and parts",
})
assert valid, "Exercise 4 validation failed" 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 5 — Streaming state and watermarking
# 
# ### Context and problem
# 
# An hourly machine defect aggregation is a correct **stateful** streaming query, but it has no event-time watermark, so Spark keeps window state forever and memory grows unbounded. Both runs use `trigger(availableNow=True)`, which processes the currently-available data in micro-batches and then stops, so the notebook stays bounded. The fix adds a two-hour watermark so Spark can drop state for windows older than `max_event_time - 2 hours`.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Stateful streaming aggregation WITHOUT a watermark
# ============================================================

# Baseline: stateful windowed aggregation with no watermark -> unbounded state growth.
set_job("5 baseline streaming no watermark")


def q5_stream_source():
    return (spark.readStream.option("maxFilesPerTrigger", 15).table(table_ref("manufacturing_event"))
        .select(
            F.to_timestamp(F.col("manufacturing_event.timestamp")).alias("event_ts"),
            F.col("manufacturing_event.machine_id").alias("machine_id"),
            F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"),
        ))


def q5_state_metrics(q):
    lp = q.lastProgress or {}
    ops = lp.get("stateOperators") or [{}]
    op = ops[0] if ops else {}
    return (op.get("memoryUsedBytes"), op.get("numRowsTotal"), op.get("numRowsDroppedByWatermark"))


q5_before_stream = (q5_stream_source()
    .groupBy(F.window("event_ts", "1 hour"), "machine_id")
    .agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects")))
q5_before_plan = explain_string(q5_before_stream)

q5_before_ckpt = f"Files/{WORK_SCHEMA}/checkpoints/stream_before_{spark.sparkContext.applicationId}"
with benchmark_op("Streaming state / watermark", "before", spark):
    q5_before_query = (q5_before_stream.writeStream
        .trigger(availableNow=True)
        .option("checkpointLocation", q5_before_ckpt)
        .format("memory").queryName("m3_defects_before").outputMode("update").start())
    q5_before_query.awaitTermination()

q5_before_mem, q5_before_rows_total, q5_before_dropped = q5_state_metrics(q5_before_query)
print(f"State memory (no watermark): {(q5_before_mem or 0) / (1024 * 1024):.2f} MB | rowsTotal: {q5_before_rows_total} | droppedByWatermark: {q5_before_dropped}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 5️⃣ DIAGNOSE — No EventTimeWatermark, so window state is never dropped
# =================================================================================================

# Diagnosis: the streaming plan has no watermark bound on the stateful aggregation.
print(json.dumps({
    "isStreaming": q5_before_stream.isStreaming,
    "hasWatermark": "EventTimeWatermark" in q5_before_plan,
    "statefulAggregation": "Aggregate" in q5_before_plan,
    "sparkUIPointer": "Structured Streaming > stateOperators: memoryUsedBytes, numRowsTotal, numRowsDroppedByWatermark",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Keep the same hourly aggregation. Add `.withWatermark("event_ts", "2 hours")` before the `groupBy` so Spark can drop state for windows older than `max_event_time - 2 hours`. Choose a processing-time trigger and a durable checkpoint path for a real deployment.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: document the runtime choices for a real streaming write.
q5_watermark_delay = "2 hours"
q5_trigger_interval = "1 minute"
print(json.dumps({"watermarkDelay": q5_watermark_delay, "triggerIntervalForRealRun": q5_trigger_interval}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 5️⃣ FIX — Add a two-hour event-time watermark; the aggregation is unchanged
# ==================================================================================================

# ✅ Solution: bound state with a watermark so old windows are cleaned up.
set_job("5 solution streaming watermark")

q5_after_stream = (q5_stream_source()
    .withWatermark("event_ts", q5_watermark_delay)
    .groupBy(F.window("event_ts", "1 hour"), "machine_id")
    .agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects")))
q5_after_plan = explain_string(q5_after_stream)

q5_after_ckpt = f"Files/{WORK_SCHEMA}/checkpoints/stream_after_{spark.sparkContext.applicationId}"
with benchmark_op("Streaming state / watermark", "after", spark):
    q5_after_query = (q5_after_stream.writeStream
        .trigger(availableNow=True)
        .option("checkpointLocation", q5_after_ckpt)
        .format("memory").queryName("m3_defects_after").outputMode("update").start())
    q5_after_query.awaitTermination()

q5_after_mem, q5_after_rows_total, q5_after_dropped = q5_state_metrics(q5_after_query)
print(f"State memory (2h watermark): {(q5_after_mem or 0) / (1024 * 1024):.2f} MB | rowsTotal: {q5_after_rows_total} | droppedByWatermark: {q5_after_dropped}")
print(q5_after_plan[:1200])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5️⃣ CHECK-CHANGES — Watermark now bounds streaming state
# ============================================================

# Validation: the fixed streaming plan has an EventTimeWatermark; the baseline did not.
valid = (q5_before_stream.isStreaming and q5_after_stream.isStreaming
    and "EventTimeWatermark" not in q5_before_plan and "EventTimeWatermark" in q5_after_plan)
record_result("5 streaming state / watermark", "passed" if valid else "failed", {
    "beforeHasWatermark": "EventTimeWatermark" in q5_before_plan,
    "afterHasWatermark": "EventTimeWatermark" in q5_after_plan,
    "beforeStateMemoryMB": round((q5_before_mem or 0) / (1024 * 1024), 2),
    "afterStateMemoryMB": round((q5_after_mem or 0) / (1024 * 1024), 2),
    "afterRowsDroppedByWatermark": q5_after_dropped,
})
assert valid, "Exercise 5 validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Exercise 6 — Python UDFs and the Native Execution Engine (NEE)
# 
# ### Context and problem
# 
# A correct top-customer query uses scalar Python UDFs. On the JVM this forces a Python boundary (`BatchEvalPython`) and a large slowdown — the classic "Python UDFs are slow" regression. This is an **execution-lever** fix: the code is fine, so instead of rewriting it (Module 1's code lever), simply enable the Native Execution Engine (NEE) — the Fabric default — and the same UDF code runs natively. NEE is disabled here only to reproduce the regression, then re-enabled to show the boost.

# CELL ********************

# ============================================================
# 6️⃣ BENCHMARK — Same UDF code with NEE disabled (JVM execution)
# ============================================================

# Disable NEE to reproduce the historical Python-UDF regression on the JVM.
from pyspark.sql.types import DoubleType

remember_conf("spark.native.enabled")
spark.conf.set("spark.native.enabled", "false")


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


exploded_orders6 = spark.table(table_ref("web_order")).selectExpr("web_order.*").select(
    F.col("customer_id"),
    F.col("order_date"),
    F.explode("order_lines").alias("line"),
)


def top_customer_spend6():
    return (
        exploded_orders6
        .withColumn("line_total", python_line_total("line.quantity", "line.unit_price", "line.extended_price"))
        .withColumn("order_day", python_extract_day("order_date"))
        .groupBy("customer_id")
        .agg(F.sum("line_total").alias("total_spend"), F.max("order_day").alias("latest_day"), F.count("*").alias("line_count"))
        .orderBy(F.desc("total_spend"))
        .limit(10)
    )


with benchmark_op("Python UDF engine (NEE)", "before", spark):
    q6_before_df = top_customer_spend6()
    q6_before_pdf = q6_before_df.toPandas()

display(q6_before_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 6️⃣ DIAGNOSE — Prove the Python boundary and NEE fallback under JVM execution
# =================================================================================================

# With NEE off, the plan shows BatchEvalPython / PythonUDF and NEE fallback blocks.
q6_before_plan = plan_string(q6_before_df)
q6_before_fallbacks = extract_nee_fallbacks(q6_before_plan)
print(json.dumps({
    "neeEnabled": spark.conf.get("spark.native.enabled"),
    "hasBatchEvalPython": "BatchEvalPython" in q6_before_plan or "PythonUDF" in q6_before_plan,
    "neeFallbackBlockCount": q6_before_fallbacks["blockCount"],
    "neeFallbackOperators": q6_before_fallbacks["operators"],
}, default=str, indent=2))
print(q6_before_plan[:1200])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Enable the Native Execution Engine (`spark.native.enabled=true`, the Fabric default) and re-run the **same** UDF query — no code change. Confirm the query speeds up and the business result is unchanged.

# CELL ********************

# Challenge starter: flip NEE on and re-run the identical query.
print("NEE currently:", spark.conf.get("spark.native.enabled"))
q6_attempt_df = top_customer_spend6()  # TODO: enable NEE before running
print("Attempt has BatchEvalPython:", "BatchEvalPython" in plan_string(q6_attempt_df))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 6️⃣ FIX — Enable NEE (the default); the same UDF code now runs natively
# ==================================================================================================

# No code change — just turn the engine on.
spark.conf.set("spark.native.enabled", "true")

with benchmark_op("Python UDF engine (NEE)", "after", spark):
    q6_after_df = top_customer_spend6()
    q6_after_pdf = q6_after_df.toPandas()

display(q6_after_pdf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 6️⃣ CHECK-CHANGES — Same code and result; the engine provides the speedup
# ============================================================

def _spend_map6(pdf):
    return {row["customer_id"]: round(float(row["total_spend"] or 0), 4) for _, row in pdf.iterrows()}

q6_after_plan = plan_string(q6_after_df)
same_result = _spend_map6(q6_before_pdf) == _spend_map6(q6_after_pdf)
valid = same_result
record_result("6 python UDFs / NEE engine", "passed" if valid else "failed", {
    "lesson": "Enabling NEE runs the same Python-UDF code natively — no rewrite required",
    "sameBusinessResult": same_result,
    "neeOffHadBatchEvalPython": "BatchEvalPython" in q6_before_plan or "PythonUDF" in q6_before_plan,
    "neeOnHasBatchEvalPython": "BatchEvalPython" in q6_after_plan or "PythonUDF" in q6_after_plan,
})
assert valid, "Exercise 6 validation failed"
restore_conf("spark.native.enabled")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## Summary — Optimizing how Spark runs
# 
# You tuned execution without changing source tables or business logic:
# 
# 1. **Join strategy / broadcast** — replaced sort-merge shuffle with broadcast hash joins for small references.
# 2. **Skew handling / AQE skew join** — diagnosed a hot-key straggler and let AQE split the hot partition (salting as the manual fallback).
# 3. **Shuffle partition sizing** — re-enabled AQE coalescing to merge a static over-partitioned shuffle into right-sized tasks.
# 4. **Caching / materialization** — cached the reused prepared branch after expensive joins/aggregation.
# 5. **Streaming state and watermarking** — added an event-time watermark so Spark drops old window state and memory stays bounded.
# 6. **Python UDFs / NEE** — enabled the Native Execution Engine so the same Python-UDF code runs natively, with no rewrite.


# CELL ********************

# Final validation summary and config cleanup
failed = [r for r in results if r["status"] != "passed"]
summary = {"sparkApplicationId": spark.sparkContext.applicationId, "resultCount": len(results), "failedCount": len(failed), "results": results}
print("OPT_EXEC_FINAL_SUMMARY_START")
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
print("OPT_EXEC_FINAL_SUMMARY_END")
for key in list(_ORIGINAL_CONF.keys()):
    restore_conf(key)
assert len(results) == 6, f"Expected 6 exercise validations, got {len(results)}"
assert not failed, f"Failed validations: {failed}"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
