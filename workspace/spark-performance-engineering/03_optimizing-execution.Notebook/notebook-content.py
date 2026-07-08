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
# - Fix execution with join strategy, AQE, partition sizing, salting, caching, and native execution choices.
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
# | 4. Caching / materialization | Three dashboards each recompute the same expensive scan-join-aggregate base. | Results stay identical; the aggregated base is materialized once and reused, so the costly shuffle runs a single time instead of once per dashboard. |
# | 5. Python UDFs / Native Execution Engine (NEE) | A correct top-customer query uses scalar Python UDFs, slow on the JVM. | Results stay identical; enabling NEE runs the same UDF code natively (no `BatchEvalPython`), removing the Python-boundary slowdown without any code change. |


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
# A correct query joins high-volume manufacturing events to small production-order and part reference tables. With automatic broadcast disabled, Spark defaults to sort-merge joins, adding shuffle and sort overhead. The baseline also turns AQE off and runs an untimed warm-up pass, so the measured gap reflects the join strategy on warm data rather than a one-time cold read. Fix only the join strategy; the aggregation stays identical.


# CELL ********************

# ============================================================
# 1️⃣ BENCHMARK — Baseline sort-merge join with broadcast disabled
# ============================================================

# Baseline: correct result, suboptimal sort-merge execution.
set_job("1 baseline sort-merge join")
remember_conf("spark.sql.autoBroadcastJoinThreshold")
remember_conf("spark.sql.adaptive.enabled")
# Force a plain shuffle sort-merge join: disable automatic broadcast AND AQE so
# the measured delta reflects join strategy alone. A warm-up pass primes the
# disk cache / JVM so both timed runs read warm and the gap is reproducible on
# repeated (warm) runs instead of being a one-time cold-vs-warm read artifact.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.enabled", "false")
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

def q1_agg(events, orders, parts):
    return (events.join(orders, "production_order_id").join(parts, "part_num")
        .groupBy("part_material").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"), F.avg("cycle_time_ms").alias("avg_cycle_ms"))
        .orderBy("part_material"))

# Warm-up (untimed): prime disk cache + JVM so the timed run is steady-state.
q1_agg(q1_events, q1_orders, q1_parts).count()

with benchmark_op("Join strategy / broadcast", "before", spark):
    q1_before_df = q1_agg(q1_events, q1_orders, q1_parts)
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
spark.conf.set("spark.sql.adaptive.enabled", "true")

def q1_agg_broadcast(events, orders, parts):
    return (events.join(broadcast(orders), "production_order_id").join(broadcast(parts), "part_num")
        .groupBy("part_material").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"), F.avg("cycle_time_ms").alias("avg_cycle_ms"))
        .orderBy("part_material"))

# Warm-up (untimed) so before/after are measured on equal, warm footing.
q1_agg_broadcast(q1_events, q1_orders, q1_parts).count()

with benchmark_op("Join strategy / broadcast", "after", spark):
    q1_after_df = q1_agg_broadcast(q1_events, q1_orders, q1_parts)
    q1_after_rows = q1_after_df.collect()
display(spark.createDataFrame(q1_after_rows))
restore_conf("spark.sql.autoBroadcastJoinThreshold")
restore_conf("spark.sql.adaptive.enabled")

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

# ### 💡 What Just Happened?
# 
# With automatic broadcast disabled, Spark defaulted to a `SortMergeJoin`: **both** sides are shuffled and sorted on the join key before matching — expensive, and wasteful when one side is tiny. The physical plan confirmed the sort-merge and the extra shuffle stages.
# 
# Wrapping the small reference tables in `broadcast(...)` (or restoring automatic broadcast + AQE) ships those dimensions to every executor so the large fact is joined **in place** — a `BroadcastHashJoin` with no shuffle or sort of the big side. Only the join *strategy* changed; the aggregation and the result are identical.
# 
# > 📝 **Note:** Broadcast is only safe when the reference truly fits in executor memory. Fabric's automatic broadcast (governed by `spark.sql.autoBroadcastJoinThreshold`, and AQE's runtime size estimates) usually picks this for you — this exercise disabled it to make the lever visible.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 2 — Skew handling / AQE skew join
# 
# ### Context and problem
# 
# A plant throughput rollup joins high-volume manufacturing events to a small machine dimension. One machine ("the busy one") produces orders of magnitude more events than the others — classic data skew on the **join key**. With broadcast disabled the join is a shuffle join, and with AQE skew handling off the hot key lands in a single shuffle partition, so one straggler task dominates the stage. The fix keeps the query identical and lets AQE split the hot partition; manual salting is the fallback when AQE will not trigger.

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

# ### 💡 What Just Happened?
# 
# One machine dominated the join key, so hashing the key into 200 shuffle partitions dropped all of the hot machine's rows into a **single** partition. That partition's task became a straggler — the whole stage waited on one core while the rest sat idle. The Spark UI Stages view made this obvious: one task's duration dwarfed the others.
# 
# With `spark.sql.adaptive.skewJoin.enabled` on (and the byte thresholds lowered so it fires on this small, highly-compressible lab data), **AQE detects the oversized partition at runtime and splits it** into several sub-partitions that run in parallel, rebalancing task times. The query text and the result are unchanged.
# 
# > 📝 **Note:** When AQE won't trigger (e.g. the skew is below threshold or the join isn't eligible), **manual salting** is the fallback: add `pmod(xxhash64(key), N)` to the fact and explode the dimension across the same N salts to spread the hot key.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 3 — Shuffle partition sizing (tiny-task storm)
# 
# ### Context and problem
# 
# A plant KPI rollup is correct, but someone set `spark.sql.shuffle.partitions` to a large static value "to be safe" and disabled AQE coalescing. On a small result the aggregation produces thousands of nearly-empty shuffle partitions — each paying task-launch overhead — so most of the wall-clock is scheduling, not compute. The fix keeps the query identical and lets AQE coalesce the tiny partitions into right-sized tasks.

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

# ### 💡 What Just Happened?
# 
# A large static `spark.sql.shuffle.partitions` (1000) with AQE coalescing off forced the small aggregation into **thousands of near-empty shuffle partitions**. Each partition still launches a task, so most of the wall-clock was task-scheduling overhead, not compute — a "tiny-task storm." The Stages view showed thousands of tasks processing 0 rows.
# 
# Re-enabling `spark.sql.adaptive.coalescePartitions.enabled` lets **AQE coalesce** those tiny partitions into right-sized tasks *at runtime*, based on the actual shuffle output rather than a guessed static count. The KPI query and result are identical; only the task count changed.
# 
# > 📝 **Note:** There is no magic fixed number for `shuffle.partitions`. Let AQE size partitions dynamically instead of hard-coding a value "to be safe" — the right count depends on the data volume of each specific shuffle.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 4 — Caching / materialization for repeated reads
# 
# ### Context and problem
# 
# Three dashboards — by order status, by part material, and by machine — are all built from the **same** expensive intermediate: inventory transactions joined to production orders and parts, then aggregated (a full scan of the large fact plus a shuffle). The baseline recomputes that scan-join-shuffle once per dashboard, so the costly shuffle runs three times. The fix materializes the aggregated base once and reuses it; each dashboard then does a cheap additive roll-up with identical results.

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Baseline recomputes the expensive aggregated base for every dashboard
# ============================================================

# Baseline: each dashboard rebuilds the same scan + join + shuffle-aggregate from scratch.
set_job("4 baseline repeated aggregation")
q4_orders = spark.table(table_ref("production_order")).select(
    F.col("production_order.production_order_id").alias("reference_id"),
    F.col("production_order.status").alias("order_status"),
    F.col("production_order.machine_id").alias("machine_id"),
)
q4_parts = spark.table(table_ref("parts")).select("part_num", "part_material")


def q4_base_agg():
    # The expensive shared base: scan the large fact, join two dimensions, then SHUFFLE-aggregate
    # to a compact grain. This is what every dashboard re-derives in the baseline.
    return (spark.table(table_ref("inventory_transaction"))
        .select("transaction_type", "reference_id", "quantity", "part_num")
        .join(q4_orders, "reference_id", "left")
        .join(q4_parts, "part_num", "left")
        .groupBy("machine_id", "part_material", "order_status")
        .agg(F.sum("quantity").alias("quantity"), F.count("*").alias("transactions")))


def q4_by_status(base):
    return base.groupBy("order_status").agg(F.sum("transactions").alias("transactions"), F.sum("quantity").alias("quantity")).orderBy("order_status")


def q4_by_material(base):
    return base.groupBy("part_material").agg(F.sum("transactions").alias("transactions"), F.sum("quantity").alias("quantity")).orderBy("part_material")


def q4_by_machine(base):
    return base.groupBy("machine_id").agg(F.sum("transactions").alias("transactions"), F.sum("quantity").alias("quantity")).orderBy("machine_id")


with benchmark_op("Caching / repeated reads", "before", spark):
    q4_before_status = q4_by_status(q4_base_agg()).collect()
    q4_before_material = q4_by_material(q4_base_agg()).collect()
    q4_before_machine = q4_by_machine(q4_base_agg()).collect()

display(spark.createDataFrame(q4_before_status))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 4️⃣ DIAGNOSE — Repeated FileScan + shuffle prove the aggregated base is recomputed
# =================================================================================================

# Diagnosis: every dashboard re-scans, re-joins, and re-shuffles the same base; one job per collect.
q4_before_plans = [
    explain_string(q4_by_status(q4_base_agg())),
    explain_string(q4_by_material(q4_base_agg())),
    explain_string(q4_by_machine(q4_base_agg())),
]
q4_file_scans_before = sum(plan.count("FileScan") for plan in q4_before_plans)
print(json.dumps({
    "antiPattern": "Recompute the same scan-join-shuffle aggregate for every dashboard",
    "dashboards": ["by order_status", "by part_material", "by machine_id"],
    "rollups": len(q4_before_plans),
    "totalFileScanOperatorsAcrossPlans": q4_file_scans_before,
    "sparkUIPointer": "Jobs tab: every collect re-runs the full scan + join + shuffle",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Materialize the expensive aggregated base once, then build all three dashboards from the cached DataFrame with cheap additive roll-ups (sum the pre-aggregated counts and quantities). Because the cached result is small, caching runs the costly shuffle a single time instead of three.

# CELL ********************

# Challenge starter: this aggregated base is the cache candidate (small output, expensive to build).
q4_candidate_base = q4_base_agg()
print("Cache and materialize this aggregated base, then roll it up three different ways.")
print(explain_string(q4_candidate_base)[:1200])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Cache and materialize the aggregated base once while roll-up logic stays unchanged
# ==================================================================================================

# ✅ Solution: materialize the expensive aggregate once, then reuse it for all three dashboards.
set_job("4 solution cache aggregated base")
q4_after_plans = []
with benchmark_op("Caching / repeated reads", "after", spark):
    q4_cached_base = q4_base_agg().persist(StorageLevel.MEMORY_AND_DISK)
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

# Validation: every dashboard matches the baseline and now reads from the in-memory cache.
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
    "cachedBase": "aggregated inventory_transaction joined to production_order and parts",
})
assert valid, "Exercise 4 validation failed" 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Spark DataFrames are **lazy**: each of the three dashboards triggered its own action, so the shared scan-join-shuffle base was recomputed from scratch every time. The plans showed the same `FileScan` + join + shuffle repeated once per dashboard — the expensive shuffle ran three times.
# 
# `persist(MEMORY_AND_DISK)` followed by a `count()` **materializes** the aggregated base once. The three roll-ups then read from the in-memory cache (`InMemoryTableScan` in their plans) and only do cheap additive sums, so the costly shuffle runs a single time. Results are identical across all three dashboards.
# 
# > 📝 **Note:** Caching pays off only when a result is **reused** and is **small enough** to fit — and you should `unpersist()` when done (as this exercise does) to free the memory. Caching a single-use DataFrame just adds overhead.
# 
# ---

# MARKDOWN ********************

# ---
# 
# ## Exercise 5 — Python UDFs and the Native Execution Engine (NEE)
# 
# ### Context and problem
# 
# A correct top-customer query uses scalar Python UDFs. On the JVM this forces a Python boundary (`BatchEvalPython`) and a large slowdown — the classic "Python UDFs are slow" regression. This is an **execution-lever** fix: the code is fine, so instead of rewriting it (Module 1's code lever), simply enable the Native Execution Engine (NEE) — the Fabric default — and the same UDF code runs natively. NEE is disabled here only to reproduce the regression, then re-enabled to show the boost.

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Same UDF code with NEE disabled (JVM execution)
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
# 5️⃣ DIAGNOSE — Prove the Python boundary and NEE fallback under JVM execution
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
# 5️⃣ FIX — Enable NEE (the default); the same UDF code now runs natively
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
# 5️⃣ CHECK-CHANGES — Same code and result; the engine provides the speedup
# ============================================================

def _spend_map6(pdf):
    return {row["customer_id"]: round(float(row["total_spend"] or 0), 4) for _, row in pdf.iterrows()}

q6_after_plan = plan_string(q6_after_df)
same_result = _spend_map6(q6_before_pdf) == _spend_map6(q6_after_pdf)
valid = same_result
record_result("5 python UDFs / NEE engine", "passed" if valid else "failed", {
    "lesson": "Enabling NEE runs the same Python-UDF code natively — no rewrite required",
    "sameBusinessResult": same_result,
    "neeOffHadBatchEvalPython": "BatchEvalPython" in q6_before_plan or "PythonUDF" in q6_before_plan,
    "neeOnHasBatchEvalPython": "BatchEvalPython" in q6_after_plan or "PythonUDF" in q6_after_plan,
})
assert valid, "Exercise 5 validation failed"
restore_conf("spark.native.enabled")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# The UDF code was perfectly correct — the slowdown came from **where it ran**. With `spark.native.enabled=false`, the scalar Python UDFs executed on the JVM behind a `BatchEvalPython` boundary (per-row serialization to a Python worker), reproducing the classic "Python UDFs are slow" regression.
# 
# Enabling the **Native Execution Engine** (`spark.native.enabled=true`, the Fabric default) runs the *same* UDF code natively/vectorized — no `BatchEvalPython`, no code change — and the business result is unchanged. This is the **execution lever**: flip the engine on rather than rewriting the logic.
# 
# > 📝 **Note:** Contrast this with Module 1's Exercise 6, where the *code* lever rewrote the same UDFs as native expressions. Both fix the Python-boundary cost; here you change execution config, there you change code. On Fabric, NEE usually makes the rewrite optional.
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

print_benchmark_summary()

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
# 4. **Caching / materialization** — materialized an expensive aggregated base once and reused it across dashboards, running the costly shuffle a single time.
# 5. **Python UDFs / NEE** — enabled the Native Execution Engine so the same Python-UDF code runs natively, with no rewrite.

