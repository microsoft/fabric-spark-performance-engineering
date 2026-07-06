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
# | 1. Join strategies / broadcast | High-volume manufacturing events join to small production-order and parts references. | Results stay identical; `SortMergeJoin` becomes `BroadcastHashJoin` and shuffle/sort overhead is removed for small references. |
# | 2. Skew handling / salting / AQE skew join | A hot customer dominates the order join key and creates straggler shuffle tasks. | Results stay identical; hot-key partition pressure is split across salt buckets, AQE skew join is enabled, and task times become more balanced. |
# | 3. Shuffle partitions + executor spill | A KPI rollup is forced through one input partition and 4000 shuffle partitions. | Results stay identical; shuffle partitions are right-sized, AQE coalesces small outputs, and Spark UI spill/tiny-task signals are reduced or eliminated. |
# | 4. Caching / materialization | Multiple dashboard branches repeatedly read and join the same inventory/order base. | Results stay identical; the shared joined base is materialized once and reused through cache hits instead of repeated source scans. |
# | 5. Streaming optimizations | Hourly machine streaming aggregation is stateful without event-time cleanup. | Results stay identical for the batch proxy; streaming state is bounded by `EventTimeWatermark`, with trigger and checkpoint choices documented. |


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


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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
# ## Exercise 2 — Skew handling / salting / AQE skew join
# 
# ### Context and problem
# 
# A customer rollup is correct, but one hot customer dominates the join key. With broadcast disabled and AQE skew handling off, one shuffle partition can become a straggler. Diagnose the skew, then keep the join and aggregation semantics identical while enabling AQE skew support and applying deterministic salting as a manual fallback.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Build a skewed customer-order input from read-only source data.
set_job("2 setup skewed customer join")
q2_orders = spark.table(table_ref("web_order")).select(
    F.col("web_order.customer_id").alias("customer_id"),
    F.col("web_order.order_id").alias("order_id"),
    F.col("web_order.order_total").alias("order_total"),
)
q2_hot_customers = [r["customer_id"] for r in q2_orders.groupBy("customer_id").count().orderBy(F.desc("count")).limit(3).collect()]
q2_hot = q2_orders.filter(F.col("customer_id").isin(q2_hot_customers))
q2_skewed = q2_orders
for _ in range(10):
    q2_skewed = q2_skewed.unionByName(q2_hot)
q2_customer_dim = q2_skewed.select("customer_id").distinct().withColumn(
    "customer_segment", F.concat(F.lit("SEG-"), F.pmod(F.xxhash64("customer_id"), F.lit(8)))
)
print("Hot customers:", q2_hot_customers)
display(q2_skewed.groupBy("customer_id").count().orderBy(F.desc("count")).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Baseline skewed shuffle join with AQE skew handling disabled
# ============================================================

# Baseline: correct skewed join with skew handling and coalesce disabled.
set_job("2 baseline skewed customer join")
remember_conf("spark.sql.adaptive.skewJoin.enabled")
remember_conf("spark.sql.adaptive.coalescePartitions.enabled")
remember_conf("spark.sql.autoBroadcastJoinThreshold")
remember_conf("spark.sql.shuffle.partitions")
remember_conf("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes")
remember_conf("spark.sql.adaptive.advisoryPartitionSizeInBytes")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.shuffle.partitions", "200")
with benchmark_op("Skew handling / salting", "before", spark):
    q2_before_df = (q2_skewed.join(q2_customer_dim, "customer_id")
        .groupBy("customer_id", "customer_segment")
        .agg(F.count("*").alias("orders"), F.sum("order_total").alias("revenue"))
        .orderBy(F.desc("orders"), "customer_id"))
    q2_before_rows = q2_before_df.collect()
display(spark.createDataFrame(q2_before_rows).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 2️⃣ DIAGNOSE — Hot-key counts and partition distribution prove skew-driven stragglers
# =================================================================================================

# Diagnosis: quantify hot-key skew and inspect post-shuffle partition sizes.
q2_stats = q2_skewed.groupBy("customer_id").count().agg(F.max("count").alias("max_rows"), F.expr("percentile_approx(count, 0.5)").alias("median_rows"), F.avg("count").alias("avg_rows")).collect()[0]
q2_skew_ratio = float(q2_stats["max_rows"]) / max(float(q2_stats["median_rows"]), 1.0)
print(json.dumps({
    "hotCustomers": q2_hot_customers,
    "maxRows": int(q2_stats["max_rows"]),
    "medianRows": int(q2_stats["median_rows"]),
    "skewRatio": round(q2_skew_ratio, 2),
    "sparkUIPointer": "Stages > Tasks; sort by Duration and compare shuffle read sizes",
}, indent=2))
q2_skewed.repartition(200, "customer_id").groupBy(spark_partition_id().alias("pid")).count().orderBy(F.desc("count")).show(10, False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Keep the same customer rollup. Enable AQE skew/coalesce settings and use deterministic salting: add `pmod(xxhash64(order_id), N)` to the large side, duplicate the small side across salts, then join on `(customer_id, salt)`.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: inspect how deterministic salt spreads the hot customers.
salt_buckets = 16
q2_starter = q2_skewed.withColumn("salt", F.pmod(F.xxhash64("order_id"), F.lit(salt_buckets)))
display(q2_starter.groupBy("customer_id", "salt").agg(F.count("*").alias("orders")).orderBy(F.desc("orders")).limit(12))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 2️⃣ FIX — Enable AQE skew handling and salt the hot key while rollup logic stays unchanged
# ==================================================================================================

# ✅ Solution: AQE settings plus salted join fallback.
set_job("2 solution salted customer join")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "4m")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "8m")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
q2_dim_salted = q2_customer_dim.withColumn("salt", F.explode(F.sequence(F.lit(0), F.lit(salt_buckets - 1))))
with benchmark_op("Skew handling / salting", "after", spark):
    q2_after_df = (q2_starter.join(q2_dim_salted, ["customer_id", "salt"])
        .groupBy("customer_id", "customer_segment")
        .agg(F.count("*").alias("orders"), F.sum("order_total").alias("revenue"))
        .orderBy(F.desc("orders"), "customer_id"))
    q2_after_rows = q2_after_df.collect()
display(spark.createDataFrame(q2_after_rows).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ CHECK-CHANGES — Compare against baseline (results identical)
# ============================================================

# Validation: salted result equals baseline and skew was real.
q2_before_map = {(r["customer_id"], r["customer_segment"]): (int(r["orders"]), round(float(r["revenue"] or 0), 2)) for r in q2_before_rows}
q2_after_map = {(r["customer_id"], r["customer_segment"]): (int(r["orders"]), round(float(r["revenue"] or 0), 2)) for r in q2_after_rows}
valid = q2_before_map == q2_after_map and q2_skew_ratio > 3
record_result("2 skew handling / salting / AQE", "passed" if valid else "failed", {
    "sameBusinessResult": q2_before_map == q2_after_map,
    "skewRatio": round(q2_skew_ratio, 2),
    "saltBuckets": salt_buckets,
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
# ## Exercise 3 — Shuffle partitions + executor spill
# 
# ### Context and problem
# 
# A KPI rollup is correct, but execution is forced into bad partitioning. One variant creates huge spill-prone tasks; another creates thousands of tiny tasks. Use Spark UI spill columns, task-duration spread, `explain()`, and `spark_partition_id()` to diagnose, then right-size partitions and re-enable AQE coalescing.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Baseline poor partition sizing with spill-prone and tiny-task signals
# ============================================================

# Baseline: correct KPI query with a single input partition and 4000 shuffle partitions.
set_job("3 baseline poor shuffle partitioning")
remember_conf("spark.sql.shuffle.partitions")
remember_conf("spark.sql.adaptive.coalescePartitions.enabled")
spark.conf.set("spark.sql.shuffle.partitions", "4000")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
q3_events = spark.table(table_ref("manufacturing_event")).selectExpr("manufacturing_event.*").select("machine_id", "timestamp", "cycle_time_ms", "defect_detected")
q3_before_input = q3_events.repartition(1)
with benchmark_op("Shuffle partitions + spill", "before", spark):
    q3_before_df = (q3_before_input.groupBy("machine_id")
        .agg(F.count("*").alias("events"), F.avg("cycle_time_ms").alias("avg_cycle_ms"), F.sum(F.col("defect_detected").cast("int")).alias("defects"))
        .orderBy(F.desc("events"), "machine_id"))
    q3_before_rows = q3_before_df.collect()
display(spark.createDataFrame(q3_before_rows).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3️⃣ DIAGNOSE — Plan, partition counts, and Spark UI spill columns prove bad sizing
# =================================================================================================

# Diagnosis: plan, partition distribution, and Spark UI task/spill pointer.
q3_before_plan = explain_string(q3_before_df)
print(q3_before_plan)
print(json.dumps({
    "shufflePartitions": spark.conf.get("spark.sql.shuffle.partitions"),
    "aqeCoalesceEnabled": spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"),
    "hasExchange": "Exchange" in q3_before_plan,
    "sparkUIPointer": "Stages > Tasks: inspect Duration, Spill (Memory), Spill (Disk), and task count",
}, indent=2))
q3_before_input.groupBy(spark_partition_id().alias("pid")).count().orderBy(F.desc("count")).show(10, False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Preserve the KPI result but change only execution: remove the single-partition collapse, choose a reasonable shuffle count, repartition by `machine_id`, and turn AQE coalescing back on.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: inspect distribution after key-based repartitioning.
q3_target_partitions = 96
q3_starter_input = q3_events.repartition(q3_target_partitions, "machine_id")
q3_starter_input.groupBy(spark_partition_id().alias("pid")).count().orderBy(F.desc("count")).show(10, False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3️⃣ FIX — Repartition by machine and right-size shuffles while KPI logic stays unchanged
# ==================================================================================================

# ✅ Solution: right-size shuffle partitions and let AQE coalesce tiny outputs.
set_job("3 solution right-sized shuffle")
spark.conf.set("spark.sql.shuffle.partitions", str(q3_target_partitions))
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
q3_after_input = q3_events.repartition(q3_target_partitions, "machine_id")
with benchmark_op("Shuffle partitions + spill", "after", spark):
    q3_after_df = (q3_after_input.groupBy("machine_id")
        .agg(F.count("*").alias("events"), F.avg("cycle_time_ms").alias("avg_cycle_ms"), F.sum(F.col("defect_detected").cast("int")).alias("defects"))
        .orderBy(F.desc("events"), "machine_id"))
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

# Validation: same KPI result with healthier execution settings.
q3_before_map = {r["machine_id"]: (int(r["events"]), int(r["defects"]), round(float(r["avg_cycle_ms"] or 0), 4)) for r in q3_before_rows}
q3_after_map = {r["machine_id"]: (int(r["events"]), int(r["defects"]), round(float(r["avg_cycle_ms"] or 0), 4)) for r in q3_after_rows}
valid = q3_before_map == q3_after_map and spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"
record_result("3 shuffle partitions + executor spill", "passed" if valid else "failed", {
    "sameBusinessResult": q3_before_map == q3_after_map,
    "beforeShufflePartitions": "4000",
    "afterShufflePartitions": spark.conf.get("spark.sql.shuffle.partitions"),
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
# A dashboard asks for the same inventory transaction base joined to production orders, then branches by transaction type. The baseline repeatedly reads and joins the same source. The fix caches the reused joined base once; filters and aggregations remain identical.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Baseline repeated scans and joins across dashboard branches
# ============================================================

# Baseline: repeated reads and joins across independent actions.
set_job("4 baseline repeated reads")
q4_inv = spark.table(table_ref("inventory_transaction")).select(
    "transaction_type", "reference_id", "quantity", "part_num", "line_id"
)
q4_orders = spark.table(table_ref("production_order")).select(
    F.col("production_order.production_order_id").alias("reference_id"),
    F.col("production_order.status").alias("order_status"),
    F.col("production_order.machine_id").alias("machine_id"),
)
q4_types = [r["transaction_type"] for r in q4_inv.groupBy("transaction_type").count().orderBy(F.desc("count")).limit(3).collect()]
q4_before_rows, q4_before_plans = [], []
with benchmark_op("Caching / repeated reads", "before", spark):
    for tx in q4_types:
        branch = (
            spark.table(table_ref("inventory_transaction"))
            .select("transaction_type", "reference_id", "quantity", "part_num", "line_id")
            .filter(F.col("transaction_type") == tx)
            .join(q4_orders, "reference_id", "left")
            .groupBy("transaction_type", "order_status")
            .agg(F.count("*").alias("transactions"), F.sum("quantity").alias("quantity"))
        )
        q4_before_plans.append(explain_string(branch))
        q4_before_rows.extend(branch.collect())
display(spark.createDataFrame(q4_before_rows))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 4️⃣ DIAGNOSE — Repeated FileScan operators prove the shared base is recomputed
# =================================================================================================

# Diagnosis: repeated file scans; Spark UI Jobs tab shows one job per collect.
q4_file_scans_before = sum(plan.count("FileScan") for plan in q4_before_plans)
print(json.dumps({
    "antiPattern": "Repeated scans of the same inventory/order joined base",
    "transactionTypes": q4_types,
    "actions": len(q4_types),
    "totalFileScanOperatorsAcrossPlans": q4_file_scans_before,
    "sparkUIPointer": "Jobs tab: every collect re-runs source preparation",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Materialize the shared post-join intermediate once. Then branch from the cached DataFrame for the same transaction-type filters and aggregations.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: this joined base is the cache candidate.
q4_candidate_base = q4_inv.join(q4_orders, "reference_id", "left")
print("Cache and materialize this joined base, not the raw source table.")
print(explain_string(q4_candidate_base)[:1200])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Cache and materialize the joined base once while branch logic stays unchanged
# ==================================================================================================

# ✅ Solution: cache and materialize the joined base once, then reuse it for all branches.
set_job("4 solution cache joined base")
q4_after_rows, q4_after_plans = [], []
with benchmark_op("Caching / repeated reads", "after", spark):
    q4_cached_base = q4_candidate_base.persist(StorageLevel.MEMORY_AND_DISK)
    q4_cached_base.count()
    for tx in q4_types:
        branch = (
            q4_cached_base.filter(F.col("transaction_type") == tx)
            .groupBy("transaction_type", "order_status")
            .agg(F.count("*").alias("transactions"), F.sum("quantity").alias("quantity"))
        )
        q4_after_plans.append(explain_string(branch))
        q4_after_rows.extend(branch.collect())
display(spark.createDataFrame(q4_after_rows))
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

# Validation: same branches and cached-scan evidence.
def q4_signature(rows):
    return sorted((r["transaction_type"], r["order_status"], int(r["transactions"]), int(r["quantity"] or 0)) for r in rows)

q4_before_sig = q4_signature(q4_before_rows)
q4_after_sig = q4_signature(q4_after_rows)
q4_in_memory_scans = sum(plan.count("InMemoryTableScan") for plan in q4_after_plans)
valid = q4_before_sig == q4_after_sig and q4_in_memory_scans > 0
record_result("4 caching / materialization", "passed" if valid else "failed", {
    "sameBusinessResult": q4_before_sig == q4_after_sig,
    "fileScansBefore": q4_file_scans_before,
    "inMemoryScansAfter": q4_in_memory_scans,
    "cachedBase": "inventory_transaction joined to production_order",
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
# ## Exercise 5 — Streaming optimizations
# 
# ### Context and problem
# 
# A streaming aggregation over manufacturing events groups by one-hour event-time windows and machine. The baseline is stateful but has no event-time watermark, so state cleanup is unbounded. The optimized plan adds watermarking, a deliberate trigger interval, and state-store settings. This notebook builds plans safely without starting a long-running stream.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Baseline streaming aggregation plan without watermark
# ============================================================

# Baseline: streaming plan without watermark plus a batch proxy benchmark for the same aggregation shape.
set_job("5 baseline streaming plan")
q5_batch = spark.table(table_ref("manufacturing_event")).select(
    F.to_timestamp(F.col("manufacturing_event.timestamp")).alias("event_ts"),
    F.col("manufacturing_event.machine_id").alias("machine_id"),
    F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"),
)
q5_before_batch_df = q5_batch.groupBy(F.window("event_ts", "1 hour"), "machine_id").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"))
with benchmark_op("Streaming aggregation tuning", "before", spark):
    q5_before_sample_df = q5_before_batch_df.orderBy("machine_id", "window").limit(20)
    q5_before_batch_rows = q5_before_sample_df.collect()
q5_stream_before_df = (spark.readStream.table(table_ref("manufacturing_event"))
    .select(F.to_timestamp(F.col("manufacturing_event.timestamp")).alias("event_ts"), F.col("manufacturing_event.machine_id").alias("machine_id"), F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"))
    .groupBy(F.window("event_ts", "1 hour"), "machine_id")
    .agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects")))
q5_before_plan = explain_string(q5_stream_before_df)
print(q5_before_plan)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 5️⃣ DIAGNOSE — Streaming plan proves stateful aggregation has no watermark bound
# =================================================================================================

# Diagnosis: stateful streaming aggregation has no watermark.
print(json.dumps({
    "isStreaming": q5_stream_before_df.isStreaming,
    "hasWatermark": "EventTimeWatermark" in q5_before_plan,
    "statefulAggregation": "Aggregate" in q5_before_plan,
    "sparkUIPointer": "Streaming progress > stateOperators: rowsTotal, memoryUsedBytes, numRowsDroppedByWatermark",
}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Add a two-hour watermark on `event_ts`, choose a processing-time trigger such as `1 minute`, and plan a durable checkpoint path before starting a real `writeStream`.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenge starter: select runtime choices for a real streaming write.
q5_watermark_delay = "2 hours"
q5_trigger_interval = "1 minute"
q5_checkpoint_path = f"Files/{WORK_SCHEMA}/checkpoints/manufacturing_event_hourly"
print(json.dumps({"watermarkDelay": q5_watermark_delay, "triggerInterval": q5_trigger_interval, "checkpointPathForRealRun": q5_checkpoint_path}, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 5️⃣ FIX — Add event-time watermark and runtime settings while aggregation logic stays unchanged
# ==================================================================================================

# ✅ Solution: add watermark and state-aware settings; batch proxy result remains the same.
set_job("5 solution streaming watermark")
spark.conf.set("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "true")
q5_after_batch_df = q5_batch.groupBy(F.window("event_ts", "1 hour"), "machine_id").agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects"))
with benchmark_op("Streaming aggregation tuning", "after", spark):
    q5_after_sample_df = q5_after_batch_df.orderBy("machine_id", "window").limit(20)
    q5_after_batch_rows = q5_after_sample_df.collect()
q5_stream_after_df = (spark.readStream.table(table_ref("manufacturing_event"))
    .select(F.to_timestamp(F.col("manufacturing_event.timestamp")).alias("event_ts"), F.col("manufacturing_event.machine_id").alias("machine_id"), F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"))
    .withWatermark("event_ts", q5_watermark_delay)
    .groupBy(F.window("event_ts", "1 hour"), "machine_id")
    .agg(F.count("*").alias("events"), F.sum("is_defect").alias("defects")))
q5_after_plan = explain_string(q5_stream_after_df)
print(q5_after_plan)
print("For a real stream: .trigger(processingTime=q5_trigger_interval) and .option('checkpointLocation', q5_checkpoint_path).")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5️⃣ CHECK-CHANGES — Compare against baseline (results identical)
# ============================================================

# Validation: streaming plan is bounded by watermark and batch proxy samples match.
q5_before_sig = [(r["window"].start, r["window"].end, r["machine_id"], int(r["events"]), int(r["defects"] or 0)) for r in q5_before_batch_rows]
q5_after_sig = [(r["window"].start, r["window"].end, r["machine_id"], int(r["events"]), int(r["defects"] or 0)) for r in q5_after_batch_rows]
valid = q5_stream_before_df.isStreaming and q5_stream_after_df.isStreaming and "EventTimeWatermark" not in q5_before_plan and "EventTimeWatermark" in q5_after_plan and q5_before_sig == q5_after_sig
record_result("5 streaming optimizations", "passed" if valid else "failed", {
    "beforeHasWatermark": "EventTimeWatermark" in q5_before_plan,
    "afterHasWatermark": "EventTimeWatermark" in q5_after_plan,
    "triggerIntervalRecommendation": q5_trigger_interval,
    "sameBatchProxySample": q5_before_sig == q5_after_sig,
})
assert valid, "Exercise 5 validation failed"
restore_conf("spark.sql.streaming.statefulOperator.checkCorrectness.enabled")

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
# 2. **Skew handling / salting / AQE** — diagnosed hot-key stragglers and split the hot key across salt buckets.
# 3. **Shuffle partitions + executor spill** — right-sized partitions and re-enabled AQE coalescing to avoid huge spilling tasks and tiny-task storms.
# 4. **Caching / materialization** — cached the reused prepared branch after expensive joins/aggregation.
# 5. **Streaming optimizations** — bounded state with watermarking and documented trigger/checkpoint choices.


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Final validation summary and config cleanup
failed = [r for r in results if r["status"] != "passed"]
summary = {"sparkApplicationId": spark.sparkContext.applicationId, "resultCount": len(results), "failedCount": len(failed), "results": results}
print("OPT_EXEC_FINAL_SUMMARY_START")
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
print("OPT_EXEC_FINAL_SUMMARY_END")
for key in list(_ORIGINAL_CONF.keys()):
    restore_conf(key)
assert len(results) == 5, f"Expected 5 exercise validations, got {len(results)}"
assert not failed, f"Failed validations: {failed}"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
