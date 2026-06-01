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
# META       "environmentId": "3cdd45c3-659b-bb60-4877-86d399fb9cb3",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# MARKDOWN ********************

# #  **Module 3: LEGO Supply Chain Optimization Challenge**
#
# Learn how to diagnose and fix **Spark performance issues** in a realistic LEGO supply-chain analytics scenario. You'll work through billion-row joins, skewed keys, array-heavy transformations, shuffle storms, and streaming aggregations-measuring before/after impact with repeatable benchmarks.
#
# **Duration:** 60 minutes | **Level:** 300-400
#
# ---
#
# ### Scenario
#
# The LEGO manufacturing, e-commerce, and planning teams rely on a shared analytics platform to answer questions about **defect rates, customer behavior, Technic-heavy sets, color diversity, and streaming machine telemetry**.
#
# The current implementation works, but it suffers from common Spark anti-patterns:
# - Shuffle-heavy joins on a massive `manufacturing_event` fact table
# - Severe data skew on hot customers in `web_order_skewed`
# - Unnecessary `explode` of large arrays for Technic set detection
# - Join-then-aggregate patterns that create a "shuffle storm" on inventory data
# - Streaming aggregations without watermarks, risking unbounded state growth
#
# **Your mission:** For each sub-scenario (3A-3E), identify the anti-pattern, apply an appropriate optimization (broadcast joins, salting, array functions, pre-aggregation, watermarks), and validate the improvement using the provided benchmarking utilities.
#
# ### Lab Pattern
#
# Every exercise follows the same steps:
#
# | Step | What you do |
# |------|------------|
# | [baseline] **Benchmark** | Run a query and capture the baseline time/metric |
# | [diagnose] **Diagnose** | Inspect table metadata and query plans to prove the root cause |
# | [fix] **Fix** | Apply the optimization (join strategy, salting, array rewrite, pre-aggregation, watermark) |
# | [benchmark] **Re-benchmark** | Run the same test and compare against the baseline |

# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Global setup
import json, time
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql import SparkSession

ORIGINAL_CONF = {}
SCHEMA = "bronze"

def table_ref(name): return f"`{SCHEMA}`.`{name}`"

def explain_to_string(df): return df._jdf.queryExecution().toString()

def table_metrics(name):
    ref=table_ref(name); detail=spark.sql(f"DESCRIBE DETAIL {ref}").collect()[0].asDict(); files=int(detail.get('numFiles') or 0); size=int(detail.get('sizeInBytes') or 0)
    return {"table":f"{SCHEMA}.{name}","rows":spark.table(ref).count(),"partitions":spark.table(ref).rdd.getNumPartitions(),"numFiles":files,"sizeBytes":size}

def remember_conf(key):
    if key not in ORIGINAL_CONF: ORIGINAL_CONF[key]=spark.conf.get(key, None)

def restore_conf(key):
    if key in ORIGINAL_CONF:
        if ORIGINAL_CONF[key] is None: spark.conf.unset(key)
        else: spark.conf.set(key, ORIGINAL_CONF[key])

print("Spark application ID:", spark.sparkContext.applicationId)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Prerequisite discovery
required=["manufacturing_event","production_order","parts","web_order","inventory_transaction","inventory_parts","inventories","sets","themes"]
available={r.tableName for r in spark.sql(f"SHOW TABLES IN `{SCHEMA}`").collect()}
missing=[t for t in required if t not in available]
if missing: raise RuntimeError(f"Missing required Lab 3 tables: {missing}")
TABLE_METRICS={t: table_metrics(t) for t in required}
for metric in TABLE_METRICS.values(): print("TABLE_METRIC|"+json.dumps(metric, sort_keys=True))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3A Context - The Billion-Row Join
#
# In this scenario, the LEGO manufacturing team needs fast insight into **defect rates and cycle times** across massive volumes of production events.
#
# Each row in `manufacturing_event` represents a single machine cycle with a defect flag and cycle time in milliseconds. To understand **where defects are happening and which materials are at risk**, analysts need to join this high-volume fact table with two much smaller reference tables:
#
# - `production_order` - provides order status and other order-level attributes
# - `parts` - provides material and category information for each part
#
# The goal is to compute **per-material defect counts and average cycle times** over billions of events so that operations can pinpoint problematic materials and production lines.
#
# ### Anti-Pattern: Shuffle-Heavy SortMergeJoins on a Huge Fact Table
#
# The baseline approach disables automatic broadcast joins and lets Spark pick `SortMergeJoin` for all joins:
#
# - `spark.sql.autoBroadcastJoinThreshold` is set to `-1`, **preventing Spark from broadcasting small dimension tables**
# - Spark must **shuffle the giant `manufacturing_event` fact table** to perform `SortMergeJoin` with `production_order` and `parts`
# - The physical plan shows `SortMergeJoin` operators and `Exchange hashpartitioning` on the fact table columns
#
# This is an anti-pattern for a classic **star-schema workload** (one very large fact table + small dimensions): it forces expensive shuffles of billions of rows when the small reference tables could be cheaply broadcast to all executors instead. The rest of the lab demonstrates how explicit `broadcast()` hints fix this and convert the plan to `BroadcastHashJoin`.

# CELL ********************

# ============================================================
# 3A SETUP - Prepare source DataFrames
# ============================================================

print("=== Table Metrics ===")
print(f"manufacturing_event: {TABLE_METRICS['manufacturing_event']['rows']:,} rows")
print(f"production_order: {TABLE_METRICS['production_order']['rows']:,} rows")
print(f"parts: {TABLE_METRICS['parts']['rows']:,} rows\n")

# Prepare fact table (large)
q3a_mfg=(
    spark.table(table_ref("manufacturing_event"))
    .select(
        F.col("production_order_id"),
        F.col("part_num"),
        F.col("defect_detected").cast("int").alias("is_defect"),
        F.col("cycle_time_ms")
    )
)
q3a_po = (
    spark.table(table_ref("production_order"))
    .select(
        F.col("production_order_id"),
        F.col("status")
    )
)
q3a_parts = (
    spark.table(table_ref("parts"))
    .select(
        "part_num",
        "part_material",
        "part_cat_id"
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3A PROBLEM - Baseline with broadcast disabled
# ============================================================

print("[baseline] Running baseline query with broadcast joins disabled...\n")

# Disable broadcast joins to force shuffle-heavy SortMergeJoin
remember_conf("spark.sql.autoBroadcastJoinThreshold")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

with benchmark_op("3A Inefficient Join", "No broadcast", spark):
    q3a_problem_df = (
        q3a_mfg.join(q3a_po, "production_order_id")
        .join(q3a_parts, "part_num")
        .groupBy("part_material")
        .agg(
            F.count("*").alias("events"),
            F.sum("is_defect").alias("defects"),
            F.avg("cycle_time_ms").alias("avg_cycle_ms"),
        )
        .orderBy("part_material")
    )
    q3a_problem_rows = q3a_problem_df.toPandas()

display(q3a_problem_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3A INVESTIGATE - Prove the problem is shuffle-heavy SortMergeJoin
# =================================================================================================

# Extract physical plan

q3a_problem_plan = explain_to_string(q3a_problem_df)

has_sort_merge_join = "SortMergeJoin" in q3a_problem_plan
has_broadcast_join = "BroadcastHashJoin" in q3a_problem_plan
has_exchange = "Exchange hashpartitioning" in q3a_problem_plan

print("Evidence from physical plan:")
q3a_problem_df.explain(mode="formatted")

print(f"Contains SortMergeJoin: {has_sort_merge_join}")
print(f"Contains BroadcastHashJoin: {has_broadcast_join}")
print(f"Contains Exchange hashpartitioning: {has_exchange}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3A FIX - Add explicit broadcast hints for dimension tables
# ==================================================================================================

print("[fixed] Running fixed query with explicit broadcast hints...\n")

with benchmark_op("3A Inefficient Join", "Broadcast hint", spark):

    q3a_fix_df = (# Restore original config
        q3a_mfg
        .join(broadcast(q3a_po),"production_order_id")
        .join(broadcast(q3a_parts),"part_num")
        .groupBy("part_material")
        .agg(
            F.count("*").alias("events"),
            F.sum("is_defect").alias("defects"),
            F.avg("cycle_time_ms").alias("avg_cycle_ms")
        ).orderBy("part_material")
    )
    q3a_fix_rows = q3a_fix_df.toPandas()

display(q3a_fix_rows)

restore_conf("spark.sql.autoBroadcastJoinThreshold")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3A CHECK-CHANGES - Compare metrics
# ============================================================

# Extract plan from fixed query
q3a_fix_plan = explain_to_string(q3a_fix_df)

# Check join strategy
fix_has_broadcast = "BroadcastHashJoin" in q3a_fix_plan
fix_has_sort_merge = "SortMergeJoin" in q3a_fix_plan
fix_has_exchange = "Exchange hashpartitioning" in q3a_fix_plan

print("Evidence from physical plan:")
q3a_fix_df.explain(mode="formatted")

print(f"Contains SortMergeJoin: {fix_has_sort_merge}")
print(f"Contains BroadcastHashJoin: {fix_has_broadcast}")
print(f"Contains Exchange hashpartitioning: {fix_has_exchange}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Tip: What Just Happened? (3A: Broadcast Join)
#
# We re-enabled broadcast joins by adding explicit `broadcast()` hints to small dimension tables (`production_order` and `parts`). This eliminated expensive shuffle operations on the large fact table (`manufacturing_event`).
#
# **Before the fix:**
# - Spark disabled broadcast joins (config set to `-1`)
# - Used `SortMergeJoin` requiring full shuffle of manufacturing events (billions of rows)
#
# **After the fix:**
# - Applied `broadcast()` hints to small tables (thousands of rows)
# - Used `BroadcastHashJoin` - dimension tables broadcast to all executors
#
# ---
#
# **Plan: Query Plan Comparison** (optional inspection)
#
# **Problem Query Plan** (SortMergeJoin with shuffle)
#
# Run this to see the baseline plan:
# ```python
# print(q3a_problem_plan)
# ```
#
# Look for `SortMergeJoin` and `Exchange hashpartitioning` nodes showing the shuffle overhead.
#
# **Fixed Query Plan** (BroadcastHashJoin)
#
# Run this to see the optimized plan:
# ```python
# print(explain_to_string(q3a_fix_df))
# ```
#
# Look for `BroadcastHashJoin` and `BroadcastExchange` nodes - no shuffle of fact table.
#
# ---
#
# > Note: **Key Takeaway:** For star-schema joins (large fact table + small dimensions), explicitly broadcast small tables (< 10MB) to avoid shuffling the fact table. Use `broadcast()` hints when auto-broadcast is disabled or threshold is too conservative.

# MARKDOWN ********************

# ## 3B Context - The Skewed Customer
#
# In this scenario, the LEGO e-commerce team wants to understand **which customers generate the most orders and revenue** so they can target loyalty campaigns, special offers, and capacity planning.
#
# The source data is `web_order`, where each row represents a single online order with customer, total value, and order ID. To simulate real-world behavior where a few customers are extremely active (VIP buyers, resellers, bots, etc.), we derive a synthetic table `web_order_skewed`:
#
# - Start from `web_order`
# - Identify the **top 5 most active customers** by order count
# - Duplicate their orders multiple times to create **severely skewed keys**
#
# The business question is: _"Who are our top customers by order volume and revenue?"_
#
# ### Anti-Pattern: Aggregation on Severely Skewed Keys
#
# The baseline query joins `web_order_skewed` with `customer` and then aggregates directly by customer:
#
# - A few hot customers now own **a massive fraction of the rows**
# - Grouping by `customer.name` sends **most of the data for those keys to a single or very few partitions**
# - Some tasks finish quickly, while the task(s) handling hot customers become **stragglers**
#
# This is a classic **data skew** anti-pattern:
#
# - Parallelism is limited by the most skewed keys
# - One or a handful of partitions become huge, causing long-running tasks and potential OOM
#
# The rest of the lab shows how to mitigate this using **manual salting**, join hints, and **Adaptive Query Execution (AQE)** so that work is spread evenly across executors while preserving correct results.

# CELL ********************

# ============================================================
# 3B SETUP - Create skewed customer dataset
# ============================================================

print("=== Table Metrics ===")

print(f"web_order: {TABLE_METRICS['web_order']['rows']:,} rows\n")

q3b_base=(
    spark.table(table_ref("web_order"))
    .select(
        F.col("web_order.customer_id").alias("customer_id"),
        F.col("web_order.order_total").alias("order_total"),
        F.col("web_order.order_id").alias("order_id")
    )
)
q3b_top=[r["customer_id"] for r in q3b_base.groupBy("customer_id").count().orderBy(F.desc("count")).limit(5).collect()]

q3b_hot=q3b_base.filter(F.col("customer_id").isin(q3b_top))
q3b_skewed=q3b_base

for _ in range(10): q3b_skewed=q3b_skewed.unionByName(q3b_hot)

q3b_skewed.write.mode("overwrite").saveAsTable(table_ref("web_order_skewed"))

remember_conf("spark.sql.adaptive.enabled")
remember_conf("spark.sql.adaptive.skewedJoin.enabled")
remember_conf("spark.sql.adaptive.coalescePartitions.enabled")
remember_conf("spark.sql.adaptive.advisoryPartitionSizeInBytes")
spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.adaptive.skewedJoin.enabled", "false")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3B PROBLEM - Baseline aggregation with skewed keys
# ============================================================

print("[baseline] Running aggregation on skewed customer data...\n")

with benchmark_op("Skewed Join", "baseline", spark):
    q3b_problem_df = (
        spark.table(table_ref("web_order_skewed"))
        .join(spark.table(table_ref("customer")), "customer_id")
        .groupBy("customer.name")
        .agg(F.count("*").alias("orders"),F.sum("order_total").alias("revenue"))
        .orderBy(F.desc("orders"),"name")
    )

    q3b_problem_rows = q3b_problem_df.limit(20).toPandas()

display(q3b_problem_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3B INVESTIGATE - Measure data skew ratio
# =================================================================================================

# Plot distribution of orders per customer to demonstrate skew
import matplotlib.pyplot as plt
from pyspark.sql import Window
orders_per_customer = (
    spark.table(table_ref("web_order_skewed"))
    .groupBy("customer_id")
    .agg(F.count("*").alias("orders"))
    .withColumn("ratio_to_avg", F.col("orders") / F.avg("orders").over(Window.partitionBy()))
    .orderBy(F.desc("orders"))
    .toPandas()
)
plt.figure(figsize=(10,6))
plt.hist(orders_per_customer["orders"], bins=50, log=True)
plt.title("Distribution of Orders per Customer (Log Scale)")
plt.xlabel("Number of Orders")
plt.ylabel("Number of Customers")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3B FIX - Apply salting to distribute skewed keys
# ==================================================================================================

print("[fixed] Running aggregation with key salting (16 buckets)...\n")

# Too large => too much overhead; too small => doesn't solve skew. 16 is a common starting point for salting.
# Iterate through experimentation to find the right balance for your data and cluster!
skewFactor=16

with benchmark_op("Skewed Join", "Manual Salting", spark):
    salted_df = (
        spark.range(0, skewFactor, 1).toDF("salt")
    )

    partitions = int(865 / 128)

    salted_web_order = (
        spark.table(table_ref("web_order_skewed"))
        .repartition(partitions)
        .crossJoin(salted_df)
        .withColumn("salted_customer_id", F.concat(F.col("customer_id"), F.lit("_"), F.col("salt")))
        .drop("salt")
    )

    salted_customer = (
        spark.table(table_ref("customer"))
        .withColumn("salt", (F.lit(skewFactor)*F.rand()).cast("int"))  # Add a dummy salt column for the cross join
        .withColumn("salted_customer_id", F.concat(F.col("customer_id"), F.lit("_"), F.col("salt")))
        .drop("salt")
    )

    q3b_fix_salted_df = (
    salted_web_order
        .join(salted_customer, "salted_customer_id")
        .groupBy("customer.name")
        .agg(F.count("*").alias("orders"),F.sum("order_total").alias("revenue"))
        .orderBy(F.desc("orders"),"name")
    )

    q3b_fix_salted_rows = q3b_fix_salted_df.limit(20).toPandas()

display(q3b_fix_salted_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3B FIX - Apply Join hints
# ==================================================================================================

print("[fixed] Running aggregation with skew hint optimization...\n")

with benchmark_op("Skewed Join", "Hint", spark):
    q3b_fix_hint_df = (
        spark.table(table_ref("web_order_skewed"))
        .hint("skew", "customer_id")
        .join(spark.table(table_ref("customer")), "customer_id")
        .groupBy("customer.name")
        .agg(F.count("*").alias("orders"),F.sum("order_total").alias("revenue"))
        .orderBy(F.desc("orders"),"name")
    )

    q3b_fix_hint_rows = q3b_fix_hint_df.limit(20).toPandas()

display(q3b_fix_hint_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3B FIX - Apply AQE skew join optimization
# ==================================================================================================

print("[fixed] Running aggregation with AQE skew join optimization...\n")

with benchmark_op("Skewed Join", "AQE", spark):
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewedJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
    q3b_fix_aqe_df = (
        spark.table(table_ref("web_order_skewed"))
        .join(spark.table(table_ref("customer")), "customer_id")
        .groupBy("customer.name")
        .agg(F.count("*").alias("orders"),F.sum("order_total").alias("revenue"))
        .orderBy(F.desc("orders"),"name")
    )

    q3b_fix_aqe_rows = q3b_fix_aqe_df.limit(20).toPandas()

display(q3b_fix_aqe_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

restore_conf("spark.sql.adaptive.enabled")
restore_conf("spark.sql.adaptive.skewedJoin.enabled")
restore_conf("spark.sql.adaptive.coalescePartitions.enabled")
restore_conf("spark.sql.adaptive.advisoryPartitionSizeInBytes")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Tip: What Just Happened? (3B: Skewed Join)
#
# We applied **key salting** to distribute skewed customer aggregations evenly across executors. The fix added a salt column for partial aggregation, then combined results.
#
# **Before the fix:**
# - Top 5 customers heavily duplicated (9x duplication = 81x skew ratio)
# - Single executor processed all rows for hot customers -> straggler tasks
# - Query took {q3b_problem_seconds:.1f} seconds
#
# **After the fix:**
# - Added salt column: `F.pmod(F.xxhash64("order_id"), F.lit(16))` (16 buckets)
# - Partial aggregation: `groupBy("customer_id", "salt")` distributed hot keys
# - Final aggregation: `groupBy("customer_id")` combined partial results
# - Query took {q3b_fix_seconds:.1f} seconds
#
# **Speedup:** {(q3b_problem_seconds / q3b_fix_seconds):.1f}x faster
#
# ---
#
# **Plan: Skew Analysis:**
# - Max partition rows: {INVESTIGATIONS["3B"]["maxCustomerRows"]:,}
# - Median partition rows: {INVESTIGATIONS["3B"]["medianCustomerRows"]:,}
# - **Skew ratio:** {INVESTIGATIONS["3B"]["skewRatio"]:.1f}x (ratio > 3 indicates severe skew)
#
# ---
#
# **Salting Pattern:**
# ```python
# # 1. Add salt column to distribute hot keys
# salted = df.withColumn("salt", F.pmod(F.xxhash64("order_id"), F.lit(16)))
#
# # 2. Partial aggregation with salt (distributes work across 16 tasks per customer)
# partial = salted.groupBy("customer_id", "salt").agg(
#     F.count("*").alias("orders"),
#     F.sum("order_total").alias("revenue")
# )
#
# # 3. Final aggregation removes salt (combines partial results)
# final = partial.groupBy("customer_id").agg(
#     F.sum("orders").alias("orders"),
#     F.sum("revenue").alias("revenue")
# )
# ```
#
# ---
#
# > Note: **Key Takeaway:** When aggregating on skewed keys (e.g., hot customers, popular products), add a salt column to split each key into multiple partitions. Choose salt cardinality (16-32) based on skew severity. Always verify results match exactly.

# MARKDOWN ********************

# ## 3C Context - Technic-Heavy LEGO Sets
#
# In this scenario, the LEGO planning team wants to identify **sets that are Technic-heavy** so they can prioritize them for marketing, packaging, and inventory decisions.
#
# We already have a denormalized dataset `q3c_sets_with_parts` where each LEGO set has an **array of part structs** (part number, color, quantity). The goal is to **find all sets that contain at least 10 distinct Technic parts**.
#
# Business rules:
# - A **Technic part** is any part whose **category name contains `"Technic"`**.
# - A set is considered **Technic-heavy** if it has **at least 10 distinct Technic part numbers**.
#
# The anti-pattern explodes the parts array for every set and only then joins to part/category tables to find Technic parts. The fix joins and filters first, then explodes only the relevant Technic parts.

# CELL ********************

# ============================================================
# 3C SETUP - Materialize set inventory with array of parts
# ============================================================
remember_conf("spark.sql.autoBroadcastJoinThreshold")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

print("=== Table Metrics ===")
print(f"inventory_parts: {TABLE_METRICS['inventory_parts']['rows']:,} rows")
print(f"inventories: {TABLE_METRICS['inventories']['rows']:,} rows")
print(f"sets: {TABLE_METRICS['sets']['rows']:,} rows")
print(f"parts: {TABLE_METRICS['parts']['rows']:,} rows\n")

# Build a base view that includes part category on each inventory line
# inventory_parts: inventory_id, part_num, color_id, quantity
# parts: part_num, part_cat_id
q3c_inventory_with_cat = (
    spark.table(table_ref("inventory_parts")).alias("ip")
    .join(
        spark.table(table_ref("parts")).alias("p"),
        F.col("ip.part_num") == F.col("p.part_num"),
    )
    .select(
        F.col("ip.inventory_id").alias("inventory_id"),
        F.col("ip.part_num").alias("part_num"),
        F.col("p.part_cat_id").alias("part_cat_id"),
        F.col("ip.quantity").alias("quantity"),
    )
)

# Link inventories to sets
q3c_inventory_set = (
    q3c_inventory_with_cat.alias("ipc")
    .join(
        spark.table(table_ref("inventories")).alias("inv"),
        F.col("ipc.inventory_id") == F.col("inv.id"),
    )
    .select(
        F.col("inv.set_num").alias("set_num"),
        F.col("ipc.part_num").alias("part_num"),
        F.col("ipc.part_cat_id").alias("part_cat_id"),
        F.col("ipc.quantity").alias("quantity"),
    )
)

# Create a denormalized view: each set with an array of part structures
# Each struct contains part_num, part_cat_id, quantity
q3c_set_parts_array = (
    q3c_inventory_set
    .groupBy("set_num")
    .agg(
        F.collect_list(
            F.struct(
                F.col("part_num").alias("part_num"),
                F.col("part_cat_id").alias("part_cat_id"),
                F.col("quantity").alias("quantity"),
            )
        ).alias("parts_array")
    )
)

# Join with sets to get set details
q3c_sets_with_parts = (
    spark.table(table_ref("sets")).alias("s")
    .join(q3c_set_parts_array.alias("spa"), "set_num")
    .select(
        F.col("s.set_num").alias("set_num"),
        F.col("s.name").alias("set_name"),
        F.col("s.theme_id").alias("theme_id"),  # kept for completeness, not used in this scenario
        F.col("spa.parts_array").alias("parts_array"),
    )
)

# Materialize as a managed table so subsequent steps read from it
q3c_sets_with_parts.write.mode("overwrite").saveAsTable(table_ref("q3c_sets_with_parts"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3C PROBLEM - Explode array, then join to find Technic parts
# ============================================================

print("[baseline] Running query with explode-first pattern (many intermediate rows)...\n")

# Always read from the materialized table
q3c_sets_with_parts_df = spark.table(table_ref("q3c_sets_with_parts"))

# Part categories dimension
part_categories = spark.table(table_ref("part_categories")).select(
    F.col("id").alias("part_cat_id"),
    F.col("name").alias("category_name"),
)

with benchmark_op("3C Technic Sets", "Explode-Join-Filter", spark):
    q3c_problem_df = (
        q3c_sets_with_parts_df
        # Step 1: Explode the parts array for ALL sets (creates MANY rows)
        .select(
            "set_num",
            "set_name",
            F.explode("parts_array").alias("part_struct"),
        )
        .select(
            "set_num",
            "set_name",
            F.col("part_struct.part_num").alias("part_num"),
            F.col("part_struct.part_cat_id").alias("part_cat_id"),
            F.col("part_struct.quantity").alias("quantity"),
        )
        # Step 2: Join exploded rows with part_categories via part_cat_id
        .join(part_categories, "part_cat_id")
        # Step 3: Keep only Technic parts based on category name
        .filter(F.lower(F.col("category_name")).contains("technic"))
        # Step 4: Aggregate Technic parts per set
        .groupBy("set_num", "set_name")
        .agg(
            F.countDistinct("part_num").alias("technic_unique_parts"),
            F.sum("quantity").alias("technic_quantity"),
        )
        # Step 5: Keep only Technic-heavy sets (>= 10 distinct Technic parts)
        .filter(F.col("technic_unique_parts") >= F.lit(10))
        .orderBy(F.desc("technic_unique_parts"), F.desc("technic_quantity"), "set_num")
    )

    q3c_problem_rows = q3c_problem_df.limit(20).toPandas()

display(q3c_problem_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3C INVESTIGATE - Quantify explosion overhead for Technic detection
# =================================================================================================



q3c_sets_with_parts_df = spark.table(table_ref("q3c_sets_with_parts"))

part_categories = spark.table(table_ref("part_categories")).select(
    F.col("id").alias("part_cat_id"),
    F.col("name").alias("category_name"),
)

# Count intermediate rows after explode (ALL parts from ALL sets)
exploded_rows = (
    q3c_sets_with_parts_df
    .select(F.explode("parts_array").alias("part_struct"))
    .count()
)

# Count rows that actually correspond to Technic parts after join on part_cat_id
filtered_rows = (
    q3c_sets_with_parts_df
    .select(F.explode("parts_array").alias("part_struct"))
    .select(
        F.col("part_struct.part_cat_id").alias("part_cat_id"),
    )
    .join(part_categories, "part_cat_id")
    .filter(F.lower(F.col("category_name")).contains("technic"))
    .count()
)

explosion_ratio = exploded_rows / filtered_rows if filtered_rows > 0 else 0
waste_percentage = ((exploded_rows - filtered_rows) / exploded_rows * 100) if exploded_rows > 0 else 0

details = {
    "antiPattern": "Explode array, then join to identify Technic parts via part_cat_id",
    "rowsAfterExplode": exploded_rows,
    "rowsAfterFilter": filtered_rows,
    "wastedRows": exploded_rows - filtered_rows,
    "wastePercentage": f"{waste_percentage:.1f}%",
    "explosionRatio": f"{explosion_ratio:.1f}x",
    "impact": (
        f"Exploded {exploded_rows:,} parts from all sets, "
        f"but only {filtered_rows:,} belong to Technic categories - "
        f"{waste_percentage:.1f}% wasted"
    ),
}

print(json.dumps(details, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3C FIX - Filter Technic categories and aggregate via array functions (no explode)
# ==================================================================================================

print("[fixed] Running query with category-filtered array and array aggregations (no explode)...\n")

# Read from materialized table
q3c_sets_with_parts_df = spark.table(table_ref("q3c_sets_with_parts"))

# Identify Technic categories once
technic_categories = (
    spark.table(table_ref("part_categories"))
    .select(F.col("id").alias("part_cat_id"), F.col("name").alias("category_name"))
    .filter(F.lower(F.col("category_name")).contains("technic"))
    .select("part_cat_id")
    .distinct()
)


with benchmark_op("3C Technic Sets", "Category-ArrayFilter-NoExplode", spark):

    # Collect Technic category IDs for array_filter optimization
    q3c_technic_part_cat_ids = [r["part_cat_id"] for r in technic_categories.collect()]

    # Build a literal array of Technic category IDs for use inside array_filter
    technic_cat_ids_lit = (
        F.array(*[F.lit(c) for c in q3c_technic_part_cat_ids])
        if q3c_technic_part_cat_ids
        else F.array()
    )

    q3c_fix_df = (
        q3c_sets_with_parts_df
        # Step 1: Filter the ARRAY using Technic category membership (BEFORE any row expansion)
        .withColumn(
            "technic_parts",
            F.filter(
                F.col("parts_array"),
                lambda part: F.array_contains(technic_cat_ids_lit, part.part_cat_id),
            ),
        )
        # Skip sets with no Technic parts
        .filter(F.size("technic_parts") > 0)
        # Step 2: Derive metrics directly from the technic_parts array
        .withColumn(
            "technic_unique_parts",
            F.size(
                F.array_distinct(
                    F.transform("technic_parts", lambda p: p.part_num)
                )
            ),
        )
        .withColumn(
            "technic_quantity",
            F.aggregate(
                "technic_parts",
                # Use BIGINT accumulator to match quantity type and avoid INT/BIGINT mismatch
                F.lit(0).cast("long"),
                lambda acc, p: acc + p.quantity.cast("long"),
            ),
        )
        # Step 3: Keep only Technic-heavy sets (>= 10 distinct Technic parts)
        .filter(F.col("technic_unique_parts") >= F.lit(10))
        .select("set_num", "set_name", "technic_unique_parts", "technic_quantity")
        .orderBy(F.desc("technic_unique_parts"), F.desc("technic_quantity"), "set_num")
    )

    q3c_fix_rows = q3c_fix_df.limit(20).toPandas()

display(q3c_fix_rows)
restore_conf("spark.sql.autoBroadcastJoinThreshold")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3C CHECK-CHANGES - Verify correctness and compare
# ============================================================

# Extract physical plans
q3c_problem_plan = explain_to_string(q3c_problem_df)
q3c_fix_plan = explain_to_string(q3c_fix_df)

# Check for explode operations in plans
problem_has_explode = "Generate" in q3c_problem_plan or "explode" in q3c_problem_plan.lower()
fix_has_explode = "Generate" in q3c_fix_plan or "explode" in q3c_fix_plan.lower()

print("=== Plan Analysis ===")
print(f"Problem plan has explode: {problem_has_explode}")
print(f"Fix plan has explode: {fix_has_explode}")
print(f"Fix plan has filter: {'filter' in q3c_fix_plan.lower()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Tip: What Just Happened? (3C: Technic-Heavy Sets Without Unnecessary Explode)
#
# We refactored the scenario to answer a concrete business question:
#
# > **Business use case:** _"Find all LEGO sets that are **Technic-heavy**, i.e., sets that contain at least **10 distinct Technic parts** (parts whose category name contains `Technic`)."_
#
# We still use `q3c_sets_with_parts`, where each set has an array of parts, but we changed **how** we detect Technic-heavy sets.
#
# **Before the fix (anti-pattern):**
# - Exploded `parts_array` for **all sets and all parts** -> created {INVESTIGATIONS["3C"]["rowsAfterExplode"]:,} intermediate rows.
# - Joined every exploded row to part/category tables to find Technic parts.
# - Only {INVESTIGATIONS["3C"]["rowsAfterFilter"]:,} rows actually belonged to Technic categories.
# - **Wasted processing:** {INVESTIGATIONS["3C"]["wastePercentage"]} of exploded rows were discarded.
#
# **After the fix (optimized):**
# - Precomputed the list of **Technic part numbers** from `parts` and `part_categories` where the category name contains `"Technic"`.
# - Used `array_filter()` on `parts_array` to keep only elements whose `part_num` is in the Technic list.
# - Exploded only the filtered `technic_parts` array.
# - Aggregated per set and kept only sets with **>= 10 distinct Technic parts**.
#
# ---
#
# **Pattern comparison:**
#
# ```python
# # [anti-pattern] Anti-pattern: Explode, then join, then filter Technic parts
# technic_sets_problem = (
#     q3c_sets_with_parts
#       .select("set_num", "set_name", explode("parts_array").alias("part_struct"))
#       .select("set_num", "set_name",
#               col("part_struct.part_num").alias("part_num"),
#               col("part_struct.quantity").alias("quantity"))
#       .join(q3c_technic_parts, "part_num")
#       .groupBy("set_num", "set_name")
#       .agg(
#           countDistinct("part_num").alias("technic_unique_parts"),
#           sum("quantity").alias("technic_quantity"),
#       )
#       .filter(col("technic_unique_parts") >= 10)
# )
#
# # [fixed] Optimized: Filter array using Technic part list, then explode
# technic_part_nums_lit = array(*[lit(p) for p in q3c_technic_part_nums])
# technic_sets_fix = (
#     q3c_sets_with_parts
#       .withColumn(
#           "technic_parts",
#           array_filter(
#               col("parts_array"),
#               lambda part: array_contains(technic_part_nums_lit, part.part_num)
#           ),
#       )
#       .filter(size("technic_parts") > 0)
#       .select("set_num", "set_name", explode("technic_parts").alias("part_struct"))
#       .groupBy("set_num", "set_name")
#       .agg(
#           countDistinct("part_struct.part_num").alias("technic_unique_parts"),
#           sum("part_struct.quantity").alias("technic_quantity"),
#       )
#       .filter(col("technic_unique_parts") >= 10)
# )
# ```
#
# ---
#
# **Execution flow comparison:**
#
# - **Problem:** _explode all parts -> join to Technic dimension -> filter Technic parts -> aggregate._
# - **Fix:** _precompute Technic part list -> filter arrays in-place -> explode only Technic parts -> aggregate._
#
# This drastically reduces the number of rows that need to be exploded and joined, while answering the same business question.
#
# > Note: **Key Takeaway:** When you need to find sets/items that satisfy a condition based on a **subset of array elements** (e.g., Technic parts), avoid exploding everything. Instead, derive the qualifying keys (Technic part numbers), use `array_filter()` to keep only matching elements, and **then** explode and aggregate.

# MARKDOWN ********************

# ## 3D Context - The Shuffle Storm
#
# In this scenario, the LEGO planning and design teams want to understand **color diversity and piece counts per set**.
#
# They use several tables:
#
# - `inventory_parts` - each row is a specific part/color/quantity in an inventory
# - `inventories` - maps inventories to LEGO set numbers
# - `sets` - provides set name and theme ID
# - `themes` - provides the theme name (e.g., City, Technic, Star Wars)
#
# The business questions include:
#
# - _"Which sets use the **widest variety of colors**?"_
# - _"Which sets have the **largest total number of pieces**?"_
# - _"How does this vary by theme for marketing and supply chain decisions?"_
#
# Answering this requires computing **distinct colors** and **total pieces per set**, then ranking sets by these metrics.
#
# ### Anti-Pattern: Join-Then-Aggregate on a Large Fact Table
#
# The baseline query takes the most straightforward approach:
#
# 1. Start from the large `inventory_parts` table
# 2. Join to `inventories`, then `sets`, then `themes`
# 3. Only after all joins, perform a `groupBy` to compute `countDistinct(color_id)` and `sum(quantity)`
#
# This creates a **shuffle storm**:
#
# - The big `inventory_parts` fact table is fully joined and shuffled before any reduction
# - `countDistinct` and `sum` run over a large, fully-joined dataset
# - The physical plan shows heavy `Exchange` operators with many shuffle partitions
#
# This is an anti-pattern for aggregation-heavy workloads: **large joins happen before obvious pre-aggregation opportunities**, causing unnecessary shuffles of millions of rows. The remainder of the lab demonstrates how to **pre-aggregate and de-duplicate** (colors and piece counts) before joining, significantly reducing shuffle volume and runtime.

# CELL ********************

# ============================================================
# 3D SETUP - Prepare LEGO inventory DataFrames
# ============================================================

print("=== Table Metrics ===")
print(f"inventory_parts: {TABLE_METRICS['inventory_parts']['rows']:,} rows")
print(f"inventories: {TABLE_METRICS['inventories']['rows']:,} rows")
print(f"sets: {TABLE_METRICS['sets']['rows']:,} rows")
print(f"themes: {TABLE_METRICS['themes']['rows']:,} rows\n")

q3d_ip=spark.table(table_ref("inventory_parts")).select("inventory_id","part_num","color_id","quantity")
q3d_inv=spark.table(table_ref("inventories")).select(F.col("id").alias("inventory_id"),"set_num")
q3d_sets=spark.table(table_ref("sets")).select("set_num",F.col("name").alias("set_name"),"theme_id")
q3d_themes=spark.table(table_ref("themes")).select(F.col("id").alias("theme_id"),F.col("name").alias("theme_name"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3D PROBLEM - Join large table before aggregation
# ============================================================

print("[baseline] Running with join-then-aggregate (large shuffle)...\n")

with benchmark_op("Shuffle Storm", "before", spark):
    q3d_problem_df=(
        q3d_ip
        .join(q3d_inv,"inventory_id")
        .join(q3d_sets,"set_num")
        .join(q3d_themes,"theme_id","left")
        .groupBy("set_num","set_name","theme_name")
        .agg(F.countDistinct("color_id").alias("distinct_colors"),F.sum("quantity").alias("total_pieces"))
        .orderBy(F.desc("distinct_colors"),F.desc("total_pieces"),"set_num")
        .limit(20)
    )
    q3d_problem_rows=q3d_problem_df.toPandas()

display(q3d_problem_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3D INVESTIGATE - Confirm large shuffle with high partition count
# =================================================================================================


q3d_problem_plan=explain_to_string(q3d_problem_df)
INVESTIGATIONS["3D"]={"inventoryPartRows":TABLE_METRICS["inventory_parts"]["rows"],"problemHasExchange":"Exchange" in q3d_problem_plan,"shufflePartitions":spark.conf.get("spark.sql.shuffle.partitions")}
print(q3d_problem_plan); record("3D","investigation","complete",INVESTIGATIONS["3D"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3D FIX - Pre-aggregate before joins, reduce shuffle partitions
# ==================================================================================================

print("[fixed] Running with pre-aggregation (reduced shuffle)...\n")

with benchmark_op("Shuffle Storm", "after", spark):# Restore config

    # Step 1: Pre-aggregate colors (de-duplicate)
    distinct_colors = q3d_ip.select("inventory_id", "color_id").distinct()

    # Step 2: Pre-aggregate pieces (sum)    
    pieces_sum = q3d_ip.groupBy("inventory_id").agg(F.sum("quantity").alias("inventory_pieces"))

    # Step 3: Join small aggregates to dimensions (colors branch)
    colors_agg = (    
        distinct_colors
            .join(q3d_inv,"inventory_id")
            .join(q3d_sets,"set_num")
            .join(q3d_themes,"theme_id","left")
            .groupBy("set_num","set_name","theme_name")
            .agg(F.countDistinct("color_id").alias("distinct_colors"))
    )

    # Step 4: Join small aggregates to dimensions (pieces branch)
    pieces_agg = (
        pieces_sum
            .join(q3d_inv,"inventory_id")
            .groupBy("set_num")
            .agg(F.sum("inventory_pieces").alias("total_pieces"))
    )

    # Step 5: Join pre-aggregated branches together (small join)
    q3d_fix_df = (
        colors_agg
        .join(pieces_agg, "set_num")
        .select("set_num", "set_name", "theme_name", "distinct_colors", "total_pieces")
        .orderBy(F.desc("distinct_colors"), F.desc("total_pieces"), "set_num")
        .limit(20)
    )

    q3d_fix_rows = q3d_fix_df.toPandas()

display(q3d_fix_rows)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3D CHECK-CHANGES - Verify shuffle reduction
# ============================================================

q3d_fix_plan = explain_to_string(q3d_fix_df)
fix_has_exchange = "Exchange" in q3d_fix_plan
fix_has_exchange

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Tip: What Just Happened? (3D: Pre-Aggregation to Reduce Shuffle)
#
# We pre-aggregated and de-duplicated the large `inventory_parts` table **before** joining to downstream tables. This dramatically reduced shuffle data volume.
#
# **Before the fix:**
# - Joined full `inventory_parts` ({INVESTIGATIONS["3D"]["inventoryPartRows"]:,} rows) to `inventories`, `sets`, `themes`
# - Applied `countDistinct("color_id")` after all joins -> shuffle entire joined dataset
# - Shuffle partitions: {INVESTIGATIONS["3D"]["shufflePartitions"]} (default, too high for small result)
# - Query took {q3d_problem_seconds:.1f} seconds
#
# **After the fix:**
# - **Pre-aggregated colors:** `select("inventory_id", "color_id").distinct()` reduced to unique pairs
# - **Pre-aggregated pieces:** `groupBy("inventory_id").agg(F.sum("quantity"))` 
# - Joined smaller aggregated DataFrames to dimensions
# - Reduced shuffle partitions to 32 (appropriate for result size)
# - Query took {q3d_fix_seconds:.1f} seconds
#
# **Speedup:** {(q3d_problem_seconds / q3d_fix_seconds):.1f}x faster
#
# ---
#
# **Optimization Strategy:**
#
# 1. **Push-down aggregation:** Reduce data volume early in the pipeline
# 2. **Separate aggregations:** Split `countDistinct` and `sum` into separate branches
# 3. **Join smaller DataFrames:** Reduced shuffle from GBs to MBs
# 4. **Tune shuffle partitions:** Match partition count to data volume
#
# ---
#
# **Pre-Aggregation Pattern:**
# ```python
# # Instead of:
# result = large_table.join(dim1).join(dim2).groupBy(...).agg(
#     F.countDistinct("col_a"),
#     F.sum("col_b")
# )
#
# # Do this:
# # 1. Pre-aggregate each metric separately
# distinct_a = large_table.select("key", "col_a").distinct()
# sum_b = large_table.groupBy("key").agg(F.sum("col_b"))
#
# # 2. Join small aggregates to dimensions
# agg1 = distinct_a.join(dim1).join(dim2).groupBy(...).agg(F.countDistinct("col_a"))
# agg2 = sum_b.join(dim1).groupBy(...).agg(F.sum("col_b"))
#
# # 3. Combine results
# result = agg1.join(agg2, "key")
# ```
#
# ---
#
# > Note: **Key Takeaway:** When joining large tables followed by aggregation, pre-aggregate or de-duplicate **before** joining to minimize shuffle data. Separate independent aggregations into parallel branches, then combine results.

# MARKDOWN ********************

# ## 3E Context - The Streaming Pipeline
#
# The baseline creates a stateful streaming aggregation without watermarking. The fix adds a two-hour event-time watermark and validates the streaming plan.

# CELL ********************

# ============================================================
# 3E SETUP - Prepare batch data for streaming
# ============================================================

print("=== Table Metrics ===")

print(f"manufacturing_event: {TABLE_METRICS['manufacturing_event']['numFiles']:,} files\n {TABLE_METRICS['manufacturing_event']['rows']:,} rows\n")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3E PROBLEM - Streaming aggregation without watermark
# ============================================================


print("[baseline] Creating streaming query WITHOUT watermark (unbounded state growth)...\n")

# Read stream
stream = (
    spark.readStream.option("maxFilesPerTrigger",15).table(table_ref("manufacturing_event"))
    .select(
        F.to_timestamp("manufacturing_event.timestamp").alias("event_ts"),
        F.col("manufacturing_event.machine_id").alias("machine_id"),
        F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"),
    )
)

# Stateful aggregation WITHOUT watermark
agg = (
    stream.groupBy(F.window("event_ts", "1 hour"), "machine_id")
    .agg(
        F.count("*").alias("events"),
        F.sum("is_defect").alias("defects")
    )
)

query = (
        agg.writeStream.trigger(availableNow=True)
        .option("checkpointLocation", "Files/tmp/manufacturing_event_checkpoint/before")
        .format("memory")
        .queryName("defect_counts")
        .outputMode("update")
)

streaming_query = query.start()
streaming_query.awaitTermination()   

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =================================================================================================
# 3E INVESTIGATE - Confirm missing watermark in stateful aggregation
# =================================================================================================
last_progress = streaming_query.lastProgress
# Check memoryUsedBytes and numRowsDroppedByWatermark in the stateOperators = last_progress["stateOperators"][0]
memory_used_bytes = last_progress["stateOperators"][0].get("memoryUsedBytes", None)
num_rows_dropped_by_watermark = last_progress["stateOperators"][0].get("numRowsDroppedByWatermark", None)

print(f"Memory used by state operator: {memory_used_bytes / (1024 * 1024):.2f} MB")
print(f"Number of rows dropped by watermark: {num_rows_dropped_by_watermark}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3E FIX - Add event-time watermark for state cleanup
# ==================================================================================================

print("[fixed] Creating streaming query WITH 2-hour watermark (bounded state)...\n")

# Read stream
stream = (
    spark.readStream.option("maxFilesPerTrigger",15).table(table_ref("manufacturing_event"))
    .select(
        F.to_timestamp("manufacturing_event.timestamp").alias("event_ts"),
        F.col("manufacturing_event.machine_id").alias("machine_id"),
        F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"),
    )
)

# Stateful aggregation WITH watermark
agg = (
    stream
    .withWatermark("event_ts", "2 hours")    
    .groupBy(F.window("event_ts", "1 hour"), "machine_id")
    .agg(
        F.count("*").alias("events"),
        F.sum("is_defect").alias("defects")
    )
)

query = (
        agg.writeStream.trigger(availableNow=True)
        .option("checkpointLocation", "Files/tmp/manufacturing_event_checkpoint/after")
        .format("memory")
        .queryName("defect_counts")
        .outputMode("update")
)

streaming_query = query.start()
streaming_query.awaitTermination()  

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3E CHECK-CHANGES - Verify watermark in plan
# ============================================================

last_progress = streaming_query.lastProgress
# Check memoryUsedBytes and numRowsDroppedByWatermark in the stateOperators = last_progress["stateOperators"][0]
memory_used_bytes = last_progress["stateOperators"][0].get("memoryUsedBytes", None)
num_rows_dropped_by_watermark = last_progress["stateOperators"][0].get("numRowsDroppedByWatermark", None)

print(f"Memory used by state operator: {memory_used_bytes / (1024 * 1024):.2f} MB")
print(f"Number of rows dropped by watermark: {num_rows_dropped_by_watermark}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Tip: What Just Happened? (3E: Streaming Watermark)
#
# We added an **event-time watermark** to the streaming query to enable Spark to drop old state for time-windowed aggregations. Without watermark, state grows unbounded.
#
# **Before the fix:**
# - Stateful streaming aggregation with `window("event_ts", "1 hour")`
# - **No watermark** -> Spark retains all window state forever
# - Risk: state grows indefinitely -> OOM in long-running streams
# - Problem plan has NO `EventTimeWatermark` operator
#
# **After the fix:**
# - Added `.withWatermark("event_ts", "2 hours")` before windowing
# - Spark can drop state for windows older than: `max_event_time - 2 hours`
# - Fixed plan has `EventTimeWatermark` operator
# - Enables late-data tolerance while bounding state size
#
# ---
#
# **Watermark Mechanics:**
#
# ```python
# # Without watermark (unbounded state growth)
# stream.groupBy(F.window("event_ts", "1 hour"), "key").agg(...)
#
# # With watermark (bounded state)
# stream \
#     .withWatermark("event_ts", "2 hours") \  # Drop state older than this
#     .groupBy(F.window("event_ts", "1 hour"), "key") \
#     .agg(...)
# ```
#
# **Watermark Trade-off:**
# - **Smaller watermark** (e.g., 10 minutes): Faster state cleanup, but drops late data
# - **Larger watermark** (e.g., 24 hours): Handles very late data, but higher memory usage
# - **Rule of thumb:** Set watermark to 2-3x expected max late-arrival time
#
# ---
#
# **Streaming Patterns Requiring Watermark:**
# 1. Time-windowed aggregations (tumbling, sliding, session windows)
# 2. Stream-stream joins with time bounds
# 3. Deduplication with time constraints
# 4. Any stateful operation where old state should be pruned
#
# ---
#
# > Note: **Key Takeaway:** Always add `.withWatermark()` for time-windowed streaming aggregations. Choose watermark duration based on expected late-data arrival tolerance vs. memory constraints. Verify `EventTimeWatermark` appears in the streaming plan.
