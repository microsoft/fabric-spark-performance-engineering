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

# # 🧱 **Module 4: Advanced Debugging — Capstone**
# ## "Break Spark on Purpose"
# 
# **Duration:** 45 minutes | **Level:** 400
# 
# ---
# 
# ### Scenario
# 
# It's Monday morning and the LEGO data platform is on fire. Multiple pipelines are failing
# or running far slower than normal. You're on call. Work through the incident queue below —
# each ticket is a real failure pattern you'll see in production.
# 
# For each incident you will:
# 
# 1. **Break it** — run the bad pipeline and watch Spark misbehave (errors, spill, stragglers)
# 2. **Read the evidence** — exception message, physical plan, Spark UI metrics, partition stats
# 3. **Fix it** — apply the right remediation
# 4. **Validate** — re-run and confirm the fix
# 
# ### The Incident Queue
# 
# | # | Severity | Incident |
# |---|----------|----------|
# | 1 | 🔴 | Driver OOM / `maxResultSize` exceeded — analyst's `toPandas()` on a fact table |
# | 2 | 🔴 | Cartesian explosion — self-join with no useful predicate |
# | 3 | 🟠 | Single-partition spill — `Window().orderBy()` with no `partitionBy` |
# | 4 | 🟠 | Skew straggler — one task dwarfs all others; AQE skew join vs. salting |
# | 5 | 🟡 | Shuffle partition explosion *(optional)* — thousands of tiny tasks |
# 
# > The setup cell deliberately tightens a few session configs so the failures surface within a
# > lab time-slot. The tear-down cell at the end restores them — don't skip it.


# CELL ********************

%run _benchmark_utils


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# SETUP — Aggressive config that makes failures reproducible at lab scale
# ============================================================
import time
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import broadcast, spark_partition_id

SCHEMA = "bronze"
CAPSTONE_SCHEMA = "capstone_v2"
FACT = "manufacturing_event"

# Snapshot the original session conf so we can restore at the end
_ORIGINAL_CONF = {
    "spark.driver.maxResultSize":               spark.conf.get("spark.driver.maxResultSize", "1g"),
    "spark.sql.adaptive.enabled":               spark.conf.get("spark.sql.adaptive.enabled", "true"),
    "spark.sql.adaptive.skewJoin.enabled":      spark.conf.get("spark.sql.adaptive.skewJoin.enabled", "true"),
    "spark.sql.adaptive.coalescePartitions.enabled": spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled", "true"),
    "spark.sql.autoBroadcastJoinThreshold":     spark.conf.get("spark.sql.autoBroadcastJoinThreshold", "10MB"),
    "spark.sql.shuffle.partitions":             spark.conf.get("spark.sql.shuffle.partitions", "200"),
}

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CAPSTONE_SCHEMA}")
spark.sql(f"DROP TABLE IF EXISTS {CAPSTONE_SCHEMA}.{FACT}")
spark.sql(f"CREATE TABLE {CAPSTONE_SCHEMA}.{FACT} SHALLOW CLONE {SCHEMA}.{FACT}")

m = get_table_metrics(f"{CAPSTONE_SCHEMA}.{FACT}")
row_count = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}").count()
print(f"✅ Cloned {CAPSTONE_SCHEMA}.{FACT}: {row_count:,} rows | "
      f"{m['num_files']} files | {m['size_mb']:.1f} MB")
print(f"   Original conf snapshot taken — final cell restores it.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🔴 INC-1: Driver OOM — "Analyst's Notebook Hangs Then Crashes"
# 
# ### 📋 On-Call Ticket
# 
# | Field | Details |
# |-------|---------|
# | **Severity** | 🔴 Critical — analyst session crashes; nothing salvageable |
# | **Symptom** | A data scientist runs `df.toPandas()` on `manufacturing_event` to "do some quick EDA in pandas". The cell hangs for minutes, then throws `SparkException: Total size of serialized results … is bigger than spark.driver.maxResultSize`. |
# | **Suspected Pipeline** | Ad-hoc EDA notebook |
# | **Context** | The table is "only" 350 MB on disk. The analyst insists "that fits in pandas easily". |
# 
# ### 🔍 Why this happens
# 
# `toPandas()` (and `.collect()`) **pulls every row to the driver process**. Three multipliers
# turn 350 MB on disk into multi-GB on the driver:
# 
# 1. **Parquet decompression** — Snappy/ZSTD typically 3–5× expansion
# 2. **JVM object overhead** — `InternalRow → Row` materializes Java objects
# 3. **Arrow / pickle round-trip** to the Python interpreter
# 
# Spark caps how much data a driver will accept via `spark.driver.maxResultSize` (default 1 GB) — when the result exceeds that cap, Spark throws rather than letting the driver OOM.


# CELL ********************

# Cap driver result size so this fails in seconds rather than minutes.
# (Real-world drivers OOM the JVM entirely; the cap is the safe, deterministic equivalent.)
spark.conf.set("spark.driver.maxResultSize", "256m")
print(f"   spark.driver.maxResultSize = {spark.conf.get('spark.driver.maxResultSize')}")

# 💀 THE BROKEN CELL — analyst grabs everything to drive in pandas
df = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")

try:
    pdf = df.toPandas()                # <-- pulls all 8 M rows to driver
    print(f"   pdf shape = {pdf.shape}")
except Exception as e:
    msg = str(e).splitlines()[0]
    print(f"❌ FAILED as expected:\n   {msg[:400]}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🧠 Diagnosis Questions
# 
# 1. Which **process** ran out of memory — driver or executor?
# 2. The on-disk size is 350 MB. Why did the serialized result exceed 256 MB so easily?
# 3. Is `.collect()` any safer than `.toPandas()`? (Hint: same code path, same problem.)
# 4. What's the right tool for "I want a few thousand rows for pandas"?
# 
# <details>
# <summary>💡 Hint</summary>
# 
# The failure is on the **driver**, not an executor. The fix is never "give the driver more RAM" —
# that's a band-aid that pushes the cliff further out. The fix is to **stop pulling all data to a
# single process**. Pre-aggregate, sample, or write to a Delta table and read selectively.
# 
# </details>


# MARKDOWN ********************

# ### 🎯 Challenge: Get the same insight without pulling 8 M rows
# 
# The analyst wanted "defect rate per machine per hour". They don't need 8 M raw rows — they need
# ~`machines × hours` aggregated rows. Rewrite the workflow so the driver receives a tiny result.
# 
# > 💡 The fix is a single `groupBy` before `toPandas()`.


# CELL ********************

# YOUR CODE HERE — aggregate first, then convert
# pdf = (
#     df.groupBy(...)
#       .agg(...)
#       .toPandas()
# )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary><strong>🔑 Solution</strong></summary>
# 
# ```python
# pdf = (
#     df.withColumn("hour", F.date_trunc("hour", "timestamp"))
#       .groupBy("machine_id", "hour")
#       .agg(
#           F.count("*").alias("events"),
#           F.sum(F.col("defect_detected").cast("int")).alias("defects"),
#       )
#       .withColumn("defect_rate", F.col("defects") / F.col("events"))
#       .toPandas()
# )
# print(pdf.shape)   # ~thousands of rows, not millions
# ```
# 
# **Rule of thumb:** if the result you want has more than ~100 000 rows, don't `toPandas()`.
# Write to a Delta table and read selectively, or use `.limit(n)` first.
# </details>


# CELL ********************

# ✅ INC-1 FIX — pre-aggregate, then convert
pdf = (
    spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
         .withColumn("hour", F.date_trunc("hour", "timestamp"))
         .groupBy("machine_id", "hour")
         .agg(
             F.count("*").alias("events"),
             F.sum(F.col("defect_detected").cast("int")).alias("defects"),
         )
         .withColumn("defect_rate", F.round(F.col("defects") / F.col("events"), 4))
         .toPandas()
)
print(f"✅ Result fits the driver: {pdf.shape[0]:,} rows × {pdf.shape[1]} cols")
pdf.head(10)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🔴 INC-2: Cartesian Explosion — "Cycle-Time Variance Pipeline"
# 
# ### 📋 On-Call Ticket
# 
# | Field | Details |
# |-------|---------|
# | **Severity** | 🔴 Critical — job runs for hours, then fails with executor lost / out-of-memory |
# | **Symptom** | A new data engineer wrote a "compare every event to every other event from the same machine" query to compute cycle-time variance. The job spills hundreds of GB before crashing. |
# | **Suspected Pipeline** | `manufacturing_event` self-join |
# | **Context** | They forgot a join predicate. Spark treats it as a cross join. |
# 
# ### 🔍 Why this happens
# 
# 8 M × 8 M = **64 × 10¹² output rows** in the worst case. Even with a `machine_id` filter, a
# self-join on a high-cardinality table produces an `n²` blowup per machine. AQE can't save you
# from a quadratic plan — it can only re-shuffle results.


# CELL ********************

# 💀 THE BROKEN PIPELINE — self-join, "scoped" by machine_id only.
# The author thinks the machine_id equality makes this safe. It doesn't:
# inside each machine partition, this is still a cross join (n² per machine).

me = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
a = me.alias("a")
b = me.alias("b")

# 👇 The author intended "compare each event to the PREVIOUS one on the same machine"
#    but wrote it as a plain equi-join on machine_id. Result: cartesian per machine.
bad_self_join = (
    a.join(b, F.col("a.machine_id") == F.col("b.machine_id"))
     .select(
         F.col("a.machine_id"),
         (F.col("a.cycle_time_ms") - F.col("b.cycle_time_ms")).alias("delta_ms"),
     )
)

print("📋 Physical plan — look for `CartesianProduct` or a join with no useful predicate:")
bad_self_join.explain()

# Don't .count() this — it would actually run the cartesian. Just measure the row blow-up
# for a SINGLE machine to make the math vivid.
sample_machine = me.select("machine_id").first()["machine_id"]
n = me.filter(F.col("machine_id") == sample_machine).count()
print(f"\n💥 For ONE machine ({sample_machine}) the self-join produces "
      f"{n:,} × {n:,} = {n*n:,} rows.")
print(f"   Across all machines that's roughly {n*n/1e9:.1f} BILLION rows per machine — "
      f"and there are dozens of machines.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🧠 Diagnosis Checklist
# 
# | Evidence | Where to look | What you'll see |
# |----------|---------------|------------------|
# | Plan node | `explain()` output | `BroadcastNestedLoopJoin` / `SortMergeJoin` with no equi-predicate beyond a high-cardinality key |
# | Output cardinality | `EXPLAIN COST` or manual `count × count` | Per-key squared |
# | Spark UI | Stages tab | Spill (memory) and Spill (disk) in **GB**, single stage running forever |
# | Failure mode | Executor logs | `OutOfMemoryError` or `ExecutorLostFailure` |
# 
# ### 🎯 Challenge: Replace the cartesian with a proper "previous event" comparison
# 
# The analyst really wanted **lag-1 delta per machine, ordered by timestamp**. That's a window
# function, not a self-join. A window function on a 350 MB table runs in seconds.


# CELL ********************

# YOUR CODE HERE
# w = Window.partitionBy(...).orderBy(...)
# fixed = me.withColumn("prev_cycle_time", F.lag(...).over(w)) \
#           .withColumn("delta_ms", ...)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary><strong>🔑 Solution</strong></summary>
# 
# ```python
# w = Window.partitionBy("machine_id").orderBy("timestamp")
# fixed = (
#     me.withColumn("prev_cycle_time", F.lag("cycle_time_ms").over(w))
#       .withColumn("delta_ms", F.col("cycle_time_ms") - F.col("prev_cycle_time"))
#       .filter(F.col("delta_ms").isNotNull())
# )
# ```
# 
# **Why this works:** linear-time (`O(n log n)` for the sort) instead of quadratic. The window
# also stays within each machine_id partition, so memory pressure is bounded.
# </details>


# CELL ********************

# ✅ INC-2 FIX — window function instead of self-join
w = Window.partitionBy("machine_id").orderBy("timestamp")
fixed = (
    me.withColumn("prev_cycle_time", F.lag("cycle_time_ms").over(w))
      .withColumn("delta_ms", F.col("cycle_time_ms") - F.col("prev_cycle_time"))
      .filter(F.col("delta_ms").isNotNull())
)

print("📋 Plan — Window node, no CartesianProduct:")
fixed.explain()

with benchmark_op("INC-2: cycle-time variance", "fixed (window)", spark):
    fixed.count()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🟠 INC-3: Single-Partition Spill — "Global Event Ranking"
# 
# ### 📋 On-Call Ticket
# 
# | Field | Details |
# |-------|---------|
# | **Severity** | 🟠 High — job runs but spills heavily and takes 20× longer than necessary |
# | **Symptom** | A `ROW_NUMBER()` over the entire `manufacturing_event` table to assign a global event sequence. The Spark UI shows hundreds of MB to GBs of spill (memory and disk). |
# | **Suspected Pipeline** | Event sequencing for downstream change-data-feed |
# | **Context** | The window has `ORDER BY timestamp` but no `PARTITION BY`. |
# 
# ### 🔍 Why this is the most common "silent" performance bug
# 
# A window function with `ORDER BY` but **no `PARTITION BY` collapses all data into a single
# partition** on a single executor, regardless of cluster size. Spark even warns you in the log:
# 
# ```
# WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single
# partition, this can cause serious performance degradation.
# ```


# CELL ********************

# 💀 THE BROKEN PIPELINE — global ordering, no partitionBy
me = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")

w_global = Window.orderBy("timestamp")  # <-- NO partitionBy()

global_seq = me.withColumn("global_event_seq", F.row_number().over(w_global))

print("📋 Plan — note `SinglePartition` exchange feeding the Window:")
global_seq.explain()

# Run it — you WILL see spill in the Spark UI Stages tab for this stage.
# (Open Spark UI ▸ Stages ▸ this stage ▸ look at the "Spill (Memory)" and "Spill (Disk)" columns.)
with benchmark_op("INC-3: window ordering", "before (no partitionBy)", spark):
    global_seq.count()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🧠 Diagnosis Checklist
# 
# 1. Run `global_seq.explain()` → look for `Exchange SinglePartition` followed by `Window`.
# 2. Open Spark UI → Stages tab → find the stage with **1 task** doing all the work.
# 3. That task's **Spill (Memory)** and **Spill (Disk)** columns will show hundreds of MB+.
# 4. Note: this is **not** an OOM — Spark spills gracefully. The cost is wall-clock time.
# 
# > 💡 The "all data in one partition" trap also shows up in `df.repartition(1)` and
# > `df.coalesce(1)`. If you ever write to a single output file, you've made the same trade.
# 
# ### 🎯 Challenge: Keep the global ordering semantics but parallelize
# 
# There are two correct fixes depending on what you actually need:
# 
# - **(a)** If "global" was a mistake and partitioning by `machine_id` is fine, do that.
# - **(b)** If you truly need a single monotonic ID across the whole table, use
#   `F.monotonically_increasing_id()` (no shuffle, no sort) — it's not gap-free but it IS
#   globally unique and unordered-ish per partition.


# CELL ********************

# YOUR CODE HERE — try option (a): partition by machine_id
# w_partitioned = Window.partitionBy(...).orderBy(...)
# parallel_seq = me.withColumn("event_seq", ...)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary><strong>🔑 Solution</strong></summary>
# 
# ```python
# # Option (a): per-machine sequence — parallelizes across executors
# w_partitioned = Window.partitionBy("machine_id").orderBy("timestamp")
# parallel_seq = me.withColumn("event_seq", F.row_number().over(w_partitioned))
# 
# # Option (b): truly global unique id, no shuffle, no sort
# unique_id = me.withColumn("event_id_64", F.monotonically_increasing_id())
# ```
# 
# After the fix the Spark UI shows the Window stage with N tasks (instead of 1) and **zero spill**.
# </details>


# CELL ********************

# ✅ INC-3 FIX (a) — partitioned window
w_partitioned = Window.partitionBy("machine_id").orderBy("timestamp")
parallel_seq = me.withColumn("event_seq", F.row_number().over(w_partitioned))

print("📋 Plan — Exchange now uses `hashpartitioning(machine_id, …)`, multiple tasks:")
parallel_seq.explain()

with benchmark_op("INC-3: window ordering", "after (partitionBy machine_id)", spark):
    parallel_seq.count()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🟠 INC-4: Skew Straggler — "Plant Throughput Report"
# 
# ### 📋 On-Call Ticket
# 
# | Field | Details |
# |-------|---------|
# | **Severity** | 🟠 High — one task takes 90% of the stage time |
# | **Symptom** | Aggregating `manufacturing_event` by `machine_id` should have ~90 even partitions. Instead, the stage finishes 89 tasks in seconds and one task runs for several minutes. |
# | **Suspected Pipeline** | Daily plant throughput rollup |
# | **Context** | One machine ("the busy one") produces 10× more events than the others — classic data skew. |
# 
# ### 🔍 Why skew hurts
# 
# A shuffle hashes rows by the key. If one key is dominant, one shuffle partition gets most of
# the data and one task does most of the work — your stage finishes only when that task does.


# CELL ********************

# Build the skewed view: pick the most-common machine_id, append 9× extra copies of its rows.
me = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
hot_machine = (
    me.groupBy("machine_id").count().orderBy(F.desc("count")).first()["machine_id"]
)
print(f"   Hot machine = {hot_machine}")

hot = me.filter(F.col("machine_id") == hot_machine)
skewed = me.unionByName(
    hot.crossJoin(spark.range(9).withColumnRenamed("id", "_dup")).drop("_dup")
)
skewed.createOrReplaceTempView("skewed_events")

# Show the skew profile
print("\n📊 Row counts per machine (top 10):")
spark.sql("""
    SELECT machine_id, COUNT(*) AS row_count
    FROM skewed_events GROUP BY machine_id
    ORDER BY row_count DESC LIMIT 10
""").show(truncate=False)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 💀 THE BROKEN PIPELINE — groupBy on the skewed key with AQE skew handling OFF
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")        # disable skew join
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")  # no coalesce either
spark.conf.set("spark.sql.shuffle.partitions", "200")

agg = (
    spark.table("skewed_events")
         .groupBy("machine_id")
         .agg(
             F.count("*").alias("events"),
             F.avg("mold_temp").alias("avg_temp"),
             F.sum(F.col("defect_detected").cast("int")).alias("defects"),
         )
)

print("📋 Plan (AQE skew handling OFF):")
agg.explain()

with benchmark_op("INC-4: skewed groupBy", "before (AQE skew OFF)", spark):
    agg.count()

# Inspect partition size distribution to PROVE the skew exists
print("\n🔬 Partition size distribution AFTER shuffle:")
spark.table("skewed_events").repartition(200, "machine_id") \
    .groupBy(spark_partition_id().alias("pid")).count() \
    .orderBy(F.desc("count")).show(10, False)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🧠 Diagnosis Checklist
# 
# 1. **Spark UI → Stages → Tasks** — sort by Duration descending. One task dwarfs the others.
# 2. **`spark_partition_id()` trick** — `df.groupBy(spark_partition_id()).count()` shows the
#    uneven distribution directly.
# 3. **`explain()`** — without AQE skew join, the plan is plain `HashAggregate` over `Exchange`.
# 
# ### 🎯 Challenge: Two ways to fix skew
# 
# - **(a)** Let AQE handle it: enable `spark.sql.adaptive.skewJoin.enabled` and
#   `spark.sql.adaptive.coalescePartitions.enabled`. AQE detects oversized shuffle blocks and
#   splits them at runtime.
# - **(b)** Salt the key: add a random suffix (`machine_id || rand % N`) so the hot key
#   fans out across N partitions, then aggregate twice. AQE is preferred when possible.


# CELL ********************

# ✅ INC-4 FIX (a) — turn AQE skew handling back on
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

agg_aqe = (
    spark.table("skewed_events")
         .groupBy("machine_id")
         .agg(
             F.count("*").alias("events"),
             F.avg("mold_temp").alias("avg_temp"),
             F.sum(F.col("defect_detected").cast("int")).alias("defects"),
         )
)

with benchmark_op("INC-4: skewed groupBy", "after (AQE skew ON)", spark):
    agg_aqe.count()
print("\n💡 In the Spark UI, the slow stage now shows split skewed partitions.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🟡 INC-5 (Optional): Shuffle Partition Explosion — "Tiny Task Storm"
# 
# ### 📋 On-Call Ticket
# 
# | Field | Details |
# |-------|---------|
# | **Severity** | 🟡 Medium — query runs but most of the wall-clock is task scheduling |
# | **Symptom** | A simple `groupBy(plant)` aggregation takes 60 s. The actual compute is < 1 s; Spark spends the rest scheduling thousands of empty tasks. |
# | **Suspected Pipeline** | Plant-level KPI rollup |
# | **Context** | Somebody set `spark.sql.shuffle.partitions = 4000` "to be safe" and disabled AQE coalesce. |
# 
# ### 🔍 Why this matters
# 
# `spark.sql.shuffle.partitions` is a *static* default. On a small result set it produces
# thousands of nearly-empty tasks — each one paying task-launch overhead (~50–100 ms in Fabric).
# This is the **opposite** of the skew problem: too many tiny pieces instead of one huge piece.


# CELL ********************

# 💀 THE BROKEN PIPELINE — fixed 4000 shuffle partitions, AQE coalesce disabled
spark.conf.set("spark.sql.shuffle.partitions", "4000")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

bad = (
    spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
         .groupBy("machine_id")
         .agg(F.count("*").alias("events"))
)

with benchmark_op("INC-5: tiny task storm", "before (4000 partitions, no coalesce)", spark):
    bad.count()
print("   ⚠️  Spark UI ▸ Stages: 4000 tasks, most processing 0 rows.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Let AQE pick the right number
# 
# Re-enable `spark.sql.adaptive.coalescePartitions.enabled`. AQE will look at the actual shuffle
# output sizes and merge tiny partitions into ~`spark.sql.adaptive.advisoryPartitionSizeInBytes`
# sized chunks (default 64 MB). No code change required.


# CELL ********************

# ✅ INC-5 FIX — re-enable AQE coalesce
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

good = (
    spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
         .groupBy("machine_id")
         .agg(F.count("*").alias("events"))
)

with benchmark_op("INC-5: tiny task storm", "after (AQE coalesce ON)", spark):
    good.count()
print("   ✅ Spark UI ▸ Stages: tens of tasks instead of thousands.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🏆 Capstone Summary
# 
# Five failure modes, five fixes:
# 
# | # | Failure Pattern | Diagnostic Tool | Fix |
# |---|------------------|-----------------|-----|
# | 1 | Driver result-size overflow | `SparkException` message, `spark.driver.maxResultSize` | Pre-aggregate before `toPandas()` / `collect()` |
# | 2 | Cartesian / `O(n²)` self-join | `explain()` → `BroadcastNestedLoopJoin` | Replace with window function |
# | 3 | Single-partition window | `explain()` → `Exchange SinglePartition`; Spark UI spill metrics | Add `partitionBy()` or use `monotonically_increasing_id()` |
# | 4 | Data skew | Task duration variance; `spark_partition_id()` histogram | Enable AQE skew join, or salt the key |
# | 5 | Shuffle partition explosion | Stage with thousands of empty tasks | Enable AQE coalesce |
# 
# ### The Three Pillars of Spark Performance Pain
# 
# Every incident in this capstone — and most you'll meet in production — is a variation of:
# 
# 1. **I/O amplification** (INC-1 driver pull, INC-5 task storm)
# 2. **Memory pressure** (INC-2 cartesian, INC-3 single-partition spill)
# 3. **Scheduling / parallelism imbalance** (INC-4 skew, INC-5 over-partitioning)
# 
# The diagnostic flow is always the same: **read the plan, read the Spark UI metrics, identify
# which pillar, apply the targeted fix.**


# CELL ********************

# 🧹 Restore original session conf so subsequent notebooks aren't affected
for k, v in _ORIGINAL_CONF.items():
    spark.conf.set(k, v)
print("✅ Session conf restored to pre-capstone state.")
print("   Tear-down note: capstone_v2 schema and views are left in place for re-runs.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
