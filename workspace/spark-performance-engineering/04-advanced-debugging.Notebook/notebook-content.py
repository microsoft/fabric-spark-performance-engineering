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

# # 🧱 **Module 4: Advanced Debugging**
# ## Failure Patterns in Distributed Jobs
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
# | 2 | 🔴 | Cartesian explosion — self-join on an inequality with no equi-key |
# | 3 | 🟠 | Skew straggler — one task dwarfs all others; AQE skew join auto-splits the hot partition |
# | 4 | 🟡 | Shuffle partition explosion *(optional)* — thousands of tiny tasks |
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
CAPSTONE_SCHEMA = "capstone"
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
# | **Symptom** | A data scientist runs `df.toPandas()` on `manufacturing_event` to "do some quick EDA in pandas". The cell hangs for minutes, then the session dies with a memory / result-size error. |
# | **Suspected Pipeline** | Ad-hoc EDA notebook |
# | **Context** | The table is "only" ~200 MB on disk. The analyst insists "that fits in pandas easily". |
# 
# ### 🔍 Why this happens
# 
# `toPandas()` (and `.collect()`) **pull every row back to the driver process**. Moving bytes from
# the executors into a single Python process means crossing **three independent guardrails** — and
# in this incident we trip them one at a time:
# 
# | Layer | Guardrail | Default | What trips it |
# |-------|-----------|---------|----------------|
# | **1. Task result transport** | `spark.rpc.message.maxSize` | **128 MB** | A *single task* tries to ship a result block larger than the RPC frame allows |
# | **2. Executor memory** | executor heap / container limit | cluster-dependent | Serializing a huge result on the executor before it's sent blows the heap → container killed |
# | **3. Driver intake** | `spark.driver.maxResultSize` | **4 GB** | The *total* size of all task results the driver is asked to accept exceeds the cap |
# 
# Three multipliers turn a modest on-disk size into multi-GB in flight:
# 
# 1. **Parquet decompression** — Snappy/ZSTD typically 3–5× expansion
# 2. **JVM object overhead** — `InternalRow → Row` materializes Java objects
# 3. **Arrow / pickle round-trip** to the Python interpreter
# 
# > 🎯 The goal here isn't to "make it fit" — it's to recognise *which* guardrail fired and see that
# > **all three are symptoms of one root cause: pulling distributed data into a single process.**
# > The fix is never to raise the limit; it's to stop collecting.


# CELL ********************

print(f"   spark.rpc.message.maxSize = {spark.conf.get('spark.rpc.message.maxSize')}")

# 💀 GUARDRAIL #1 — task result transport (spark.rpc.message.maxSize, default 128 MB).
# Each row gets a 256 MB string. The table is tiny and reads as a SINGLE partition, so ONE
# task tries to ship >128 MB back to the driver in one result block and is rejected by the
# RPC layer — before any OOM even has a chance to happen.

df = spark.table("lego.bronze.colors").selectExpr("id", "repeat('x', 1024 * 1024 * 256) AS payload_256mb")

pdf = df.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Check the number of partitions for the DataFrame `df`
# 
# _Tip: you can convert a DataFrame to an `rdd` and then use the `getNumPartitions()` method to check the partition count_

# CELL ********************


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
# df.rdd.getNumPartitions()
# ```
# 
# 
# Note: `getNumPartitions()` tells you how many Spark partitions the DataFrame currently has at that point in the logical/physical plan.
# 
# </details>


# MARKDOWN ********************

# ## Can we fix by adding parallelism?
# 
# The DataFrame only has 1 partition and the massive result set for a single task resulted in exceeding the rpc transport layer max size of 128 MB. Maybe breaking this into smaller tasks could help? 

# CELL ********************

# 💀 Executor OOM + fails entire session because executor and driver run on the same JVM in single node mode.

df = spark.table("lego.bronze.colors").selectExpr("id", "repeat('x', 1024 * 1024 * 256) AS payload_256mb").repartition(8)

pdf = df.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Parallelizing into 8 partitions got us *past* guardrail #1 — each task's result block is now small
# enough for the RPC layer — but it crashed the whole session instead.
# 
# In the Spark Monitoring page for this session you'll see **`Spark_System_Executor_ExitCode143BadNode`**.
# Exit code **143** = the process received **SIGTERM (128 + 15)**: the cluster manager killed the
# container because it blew its memory limit (**guardrail #2**). Each task now materialises ~256 MB of
# payload *and* the executor must hold those rows in memory while serializing them to send to the
# driver. In single-node mode the **executor and driver share one JVM**, so when the executor dies the
# driver — and the entire session — goes down with it.
# 
# > 💡 Adding parallelism didn't reduce how much data we're pulling — it just moved the bottleneck from
# > the *transport layer* to *executor memory*, so a different guardrail caught it.

# MARKDOWN ********************

# ## Can we fix by making the message payload ~ 10x smaller?

# CELL ********************

print(f"   spark.driver.maxResultSize = {spark.conf.get('spark.driver.maxResultSize')}")

# 💀 THE BROKEN CELL — evaluates a 25MB payload across 275 records resulting in ~ 10 GiB of data.
# The size isn't the problem, it's that the DataFrame is being converted to Pandas which triggers collection of data to the driver.
df = spark.sql("""
SELECT
    id,
    repeat('x', 1024 * 1024 * 25) AS payload_25mb
FROM lego.bronze.colors
""")

pdf = df.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# A different error again. This time individual tasks are fine and no executor OOMs — we hit
# **guardrail #3: `spark.driver.maxResultSize`**. With 25 MB per row across the colors table, the
# *aggregate* result Spark is asked to deliver to the driver (~ 10 GB) exceeds the 4 GB cap, so Spark
# **fails the job proactively** rather than letting the driver OOM.
# 
# That's the key insight: a driver OOM kills **every** running job regardless of cluster size, so Spark
# guards the driver's intake *before* memory is actually exhausted. The data is perfectly fine to work
# with **as a distributed DataFrame** — the only problem is funnelling all of it into one process via
# `toPandas()`.
# 
# > 🧭 Three different errors, one root cause. Raising any of these limits just pushes the cliff further
# > out. The real fix is to **not collect everything** — `LIMIT`, aggregate, or sample first.

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


# CELL ********************

# ✅ INC-1 FIX — if you must work with Pandas or processes which only run on the driver, ensure to use `LIMIT`
# or aggregate results to prevent out of memory errors.
df = spark.sql("""
SELECT
    id,
    repeat('x', 1024 * 1024 * 25) AS payload_25mb
FROM lego.bronze.colors
""").limit(1)

pdf = df.toPandas()

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
# | **Symptom** | A new data engineer wrote a query to compute cycle-time variance by comparing each event to the *previous* event on the same machine. The job spills hundreds of GB before crashing. |
# | **Suspected Pipeline** | `manufacturing_event` self-join |
# | **Context** | To grab "earlier" events they joined the table to itself on an **inequality** (`b.timestamp < a.timestamp`) and dropped the equality key. |
# 
# ### 🔍 Why this happens
# 
# The engineer *meant* "for each event, look at the immediately preceding event." Instead they wrote a
# self-join whose only predicate is an **inequality** — and that single choice is fatal for two reasons:
# 
# 1. **Wrong semantics** — an inequality matches *every* earlier event, not just the previous one.
# 2. **Catastrophic physical plan** — hash joins and sort-merge joins both need an **equality** (equi-)
#    key to partition rows into matching buckets. A pure inequality (theta) join gives Spark no key to
#    hash on, so the *only* operator it can choose is a **`CartesianProduct`**: it pairs **every row with
#    every row** and applies the predicate afterwards as a filter.
# 
# That's `3.2M × 3.2M ≈ 10¹³` (ten **trillion**) comparison rows from a table that fits in memory. AQE
# can't save you here — it can re-shuffle and coalesce, but it cannot turn an `O(n²)` plan into anything
# sub-quadratic.


# CELL ********************

# 💀 THE BROKEN PIPELINE — self-join on an inequality, no equi-key.
# The author wanted "the previous event on the same machine" but wrote an inequality
# (b.timestamp < a.timestamp). With NO equality predicate Spark cannot hash/sort-merge —
# the only plan it can produce is a CartesianProduct (every row paired with every row).

me = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
a = me.alias("a")
b = me.alias("b")

bad_self_join = (
    a.join(b, F.col("b.timestamp") < F.col("a.timestamp"))
     .select(
         F.col("a.machine_id"),
         (F.col("a.cycle_time_ms") - F.col("b.cycle_time_ms")).alias("delta_ms"),
     )
)

print("📋 Physical plan — look for `CartesianProduct` (a join with no equi-key):")
bad_self_join.explain()

# Don't .count() this — it would actually run the cartesian. Just make the math vivid.
n = me.count()
print(f"\n💥 The inequality self-join pairs every row with every row: "
      f"{n:,} × {n:,} = {n*n:,} rows.")
print(f"   That's ~{n*n/1e12:.1f} TRILLION comparison rows from a {n:,}-row table — "
      f"and AQE cannot make a quadratic plan sub-quadratic.")

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
# | Plan node | `explain()` output | **`CartesianProduct`** (or `BroadcastNestedLoopJoin` if one side is broadcastable) — i.e. a join with no equi-predicate |
# | Output cardinality | `EXPLAIN COST` or manual `count × count` | `n²` — every row paired with every row |
# | Spark UI | Stages tab | Spill (memory) and Spill (disk) in **GB**, a single stage running effectively forever |
# | Failure mode | Executor logs | `OutOfMemoryError` or `ExecutorLostFailure` |
# 
# > ⚠️ `CartesianProduct` (and `BroadcastNestedLoopJoin`) over a large table are red flags. They're only
# > ever correct for genuinely tiny inputs — seeing one usually means a missing or non-equality join
# > predicate.
# 
# ### 🎯 Challenge: Replace the cartesian with a proper "previous event" comparison
# 
# The engineer really wanted **lag-1 delta per machine, ordered by timestamp**. That's a window function
# (`lag`), not a self-join — and it fixes *both* the semantics and the performance. A window over a
# ~200 MB table runs in seconds.


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
# ## 🟠 INC-3: Skew Straggler — "Plant Throughput Report"
# 
# ### 📋 On-Call Ticket
# 
# | Field | Details |
# |-------|---------|
# | **Severity** | 🟠 High — one task takes 90%+ of the stage time |
# | **Symptom** | A join of `manufacturing_event` to a machine dimension feeds a plant rollup. The shuffle stage finishes 199 tasks in seconds and one task runs for minutes. |
# | **Suspected Pipeline** | Daily plant throughput rollup |
# | **Context** | One machine ("the busy one") produces orders of magnitude more events than the others — classic data skew on the **join key**. |
# 
# ### 🔍 Why skew hurts — and how AQE fixes it
# 
# A shuffle join hashes both sides by the join key so matching rows land in the same partition. If one
# key dominates, **one shuffle partition gets most of the rows and one task does most of the work** — the
# stage only finishes when that straggler does.
# 
# **Adaptive Query Execution (AQE) can fix this automatically.** With
# `spark.sql.adaptive.skewJoin.enabled` on, AQE inspects the *actual* shuffle-block sizes after the map
# stage, detects partitions far larger than the median, and **splits each oversized partition into several
# smaller sub-partitions** that run as parallel tasks. No code change, no salting required.
# 
# > Skew handling only applies to **joins** (`SortMergeJoin` / `ShuffledHashJoin`), so we disable
# > broadcast below to force a real shuffle join AQE can optimize. (A small dimension would normally just
# > broadcast and sidestep skew entirely.)


# CELL ********************

# Build a skewed join: one hot machine_id dominates the fact, joined to a machine dimension.
# We MATERIALIZE both sides as tables so the join plan is clean (no data-gen / CartesianProduct
# noise) and the skew lives purely in the join-key shuffle.
me = spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
hot_machine = me.groupBy("machine_id").count().orderBy(F.desc("count")).first()["machine_id"]
print(f"   Hot machine = {hot_machine}")

DUP = 100  # ~100x extra copies of the hot machine's rows -> one dominant join key
hot = me.filter(F.col("machine_id") == hot_machine)
skewed = me.unionByName(
    hot.crossJoin(spark.range(DUP).withColumnRenamed("id", "_dup")).drop("_dup")
)
spark.sql(f"DROP TABLE IF EXISTS {CAPSTONE_SCHEMA}.skewed_events")
skewed.write.mode("overwrite").saveAsTable(f"{CAPSTONE_SCHEMA}.skewed_events")

# Machine dimension: one row per machine. Materialized; we force a shuffle join below by
# disabling broadcast so AQE skew handling has something to optimize.
machine_dim = (
    me.select("machine_id").distinct()
      .withColumn("plant", F.concat(F.lit("PLANT-"), F.substring("machine_id", 6, 3)))
)
spark.sql(f"DROP TABLE IF EXISTS {CAPSTONE_SCHEMA}.machine_dim")
machine_dim.write.mode("overwrite").saveAsTable(f"{CAPSTONE_SCHEMA}.machine_dim")

print("\n📊 Row counts per machine on the join key (top 10):")
spark.sql(f"""
    SELECT machine_id, COUNT(*) AS row_count
    FROM {CAPSTONE_SCHEMA}.skewed_events GROUP BY machine_id
    ORDER BY row_count DESC LIMIT 10
""").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 💀 THE BROKEN PIPELINE — skewed join with AQE skew handling OFF.
# Force a SortMergeJoin (disable broadcast) so the skew lands in a single shuffle partition.
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")        # <-- skew handling OFF
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")          # force SortMergeJoin
spark.conf.set("spark.sql.shuffle.partitions", "200")

joined = (
    spark.table(f"{CAPSTONE_SCHEMA}.skewed_events").alias("e")
         .join(spark.table(f"{CAPSTONE_SCHEMA}.machine_dim").alias("m"), "machine_id")
         .groupBy("plant")
         .agg(
             F.count("*").alias("events"),
             F.avg("mold_temp").alias("avg_temp"),
             F.sum(F.col("defect_detected").cast("int")).alias("defects"),
         )
)

print("📋 Plan (AQE skew handling OFF) — plain SortMergeJoin over a skewed Exchange:")
joined.explain()

with benchmark_op("INC-4: skewed join", "before (AQE skewJoin OFF)", spark):
    joined.collect()

# PROVE the skew: hash the join key into 200 partitions and look at the distribution.
# One partition holds the entire hot machine — that's the single straggler task.
print("\n🔬 Shuffle-partition distribution on the join key (one giant partition = the straggler):")
spark.table(f"{CAPSTONE_SCHEMA}.skewed_events").repartition(200, "machine_id") \
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
# 1. **Spark UI → Stages → Tasks** — sort by Duration descending. One task dwarfs all the others.
# 2. **`spark_partition_id()` trick** — hashing the join key shows one partition holding the entire hot
#    machine while the rest are tiny.
# 3. **`explain()`** — with skew handling off, the plan is a plain `SortMergeJoin` over the skewed
#    `Exchange`; nothing splits the hot partition.
# 
# ### Fix Skew, OPTION 1: Let AQE auto-fix the skew
# 
# Turn `spark.sql.adaptive.skewJoin.enabled` back on (and keep `coalescePartitions` on). AQE will:
# 
# - measure the real shuffle-block **bytes** after the map stage,
# - flag the hot partition (bigger than `skewedPartitionFactor` × the median **and** above
#   `skewedPartitionThresholdInBytes`),
# - **split that one partition into several sub-partitions** so multiple tasks share the hot key.
# 
# Two gotchas this incident exposes — and the fix cell handles both:
# 
# - **Lower the byte threshold for lab scale.** The 256 MB default never fires on this compressible
#   data, so we set `skewedPartitionThresholdInBytes` (and `advisoryPartitionSizeInBytes`) small. The
#   query itself is unchanged.
# - **Read the *final* plan correctly.** Print the plan from an action on the DataFrame itself
#   (`.collect()`), not a separate `noop` write — otherwise `executedPlan` still reports
#   `isFinalPlan=false` and you never see the split. In the final plan, look for **`AQEShuffleRead`**
#   nodes annotated **`skewed`** feeding the `SortMergeJoin`.
# 
# > 🧂 *Salting* (adding a random suffix to the hot key, then aggregating twice) achieves the same thing
# > manually — but with AQE skew join you rarely need to. Let the engine do it.


# CELL ********************

# ✅ INC-4 FIX — turn AQE skew-join handling ON (no change to the query itself)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")         # <-- skew handling ON
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")          # keep it a SortMergeJoin
spark.conf.set("spark.sql.shuffle.partitions", "200")

# The 256 MB default skew threshold is far too high to trigger on this small, highly-compressible
# lab data (the hot key is a constant string), so we lower it. In production the defaults are fine.
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "4m")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "8m")

joined_aqe = (
    spark.table(f"{CAPSTONE_SCHEMA}.skewed_events").alias("e")
         .join(spark.table(f"{CAPSTONE_SCHEMA}.machine_dim").alias("m"), "machine_id")
         .groupBy("plant")
         .agg(
             F.count("*").alias("events"),
             F.avg("mold_temp").alias("avg_temp"),
             F.sum(F.col("defect_detected").cast("int")).alias("defects"),
         )
)

# IMPORTANT: drive the action on THIS DataFrame (collect) so its queryExecution holds the final
# adaptive plan. A separate noop write would leave executedPlan reporting isFinalPlan=false.
with benchmark_op("INC-4: skewed join", "after (AQE skewJoin ON)", spark):
    joined_aqe.collect()

print("\n📋 FINAL adaptive plan (isFinalPlan=true) — look for `AQEShuffleRead` nodes marked")
print("   `skewed` feeding the SortMergeJoin (the hot partition split into parallel tasks):\n")
print(joined_aqe._jdf.queryExecution().executedPlan().toString())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Fix Skew, OPTION 2: Manual Salting for Skewed Joins
# 
# When one join key has most of the rows, Spark can send too much data to a single shuffle partition. AQE can often fix this automatically, but if AQE does not trigger, **salting** is a manual workaround.
# 
# Salting adds a small artificial `salt` value to the large/skewed table, spreading the hot key across multiple join keys. The smaller table is duplicated across the same salt values, then the join uses both columns:
# 
# `machine_id + salt`
# 
# This helps Spark split one overloaded task into many smaller parallel tasks.
# 
# **Trade-off:** the small side of the join is duplicated by the number of salt buckets, so this works best when the dimension table is small.

# CELL ********************

from pyspark.sql import functions as F

# ✅ INC-4 FIX — manual salting for skewed join
# Instead of relying on AQE skew-join handling, we split the hot key across multiple salted join keys.

spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")        # <-- skew handling OFF for this demo
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")          # keep it a SortMergeJoin
spark.conf.set("spark.sql.shuffle.partitions", "200")

SALT_BUCKETS = 16

events_salted = (
    spark.table(f"{CAPSTONE_SCHEMA}.skewed_events")
         .withColumn(
             "salt",
             # Deterministically spread rows for each machine_id across salt buckets.
             # If event_id exists and is high-cardinality, it is a good salting input.
             F.pmod(F.xxhash64(F.col("event_id")), F.lit(SALT_BUCKETS))
         )
         .alias("e")
)

machine_dim_salted = (
    spark.table(f"{CAPSTONE_SCHEMA}.machine_dim")
         # Duplicate each dimension row once per salt bucket so it can match every salted fact row.
         .withColumn("salt", F.explode(F.sequence(F.lit(0), F.lit(SALT_BUCKETS - 1))))
         .alias("m")
)

joined_salted = (
    events_salted
        .join(machine_dim_salted, ["machine_id", "salt"])
        .groupBy("plant")
        .agg(
            F.count("*").alias("events"),
            F.avg("mold_temp").alias("avg_temp"),
            F.sum(F.col("defect_detected").cast("int")).alias("defects"),
        )
)

with benchmark_op("INC-4: skewed join", f"after (manual salting, {SALT_BUCKETS} buckets)", spark):
    joined_salted.collect()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ## 🟡 INC-4: Shuffle Partition Explosion — "Tiny Task Storm"
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
# thousands of nearly-empty tasks — each one paying task-launch overhead (~50–100 ms).
# This is the **opposite** of the skew problem: too many tiny pieces instead of one huge piece.


# CELL ********************

# 💀 THE BROKEN PIPELINE — fixed 4000 shuffle partitions, AQE coalesce disabled
spark.conf.set("spark.sql.shuffle.partitions", "1000")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

bad = (
    spark.table(f"{CAPSTONE_SCHEMA}.{FACT}")
         .groupBy("machine_id")
         .agg(F.count("*").alias("events"))
)

with benchmark_op("INC-5: tiny task storm", "before (1000 partitions, no coalesce)", spark):
    bad.count()
print("   ⚠️  Spark UI ▸ Stages: 1000 tasks, most processing 0 rows.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Let AQE pick the right number
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
# | 1 | Driver result-size overflow | `SparkException` / RPC / `maxResultSize` message | Pre-aggregate, `LIMIT`, or sample before `toPandas()` / `collect()` |
# | 2 | Cartesian / inequality self-join | `explain()` → `CartesianProduct` (no equi-key) | Replace with a `lag()` window function |
# | 3 | Data skew on a join key | Task duration variance; `spark_partition_id()` histogram | Enable AQE skew join — it auto-splits the hot partition |
# | 4 | Shuffle partition explosion | Stage with thousands of empty tasks | Enable AQE coalesce |
# 
# ### The Three Pillars of Spark Performance Pain
# 
# Every incident in this lab — and most you'll meet in production — is a variation of:
# 
# 1. **I/O amplification** (INC-1 driver pull, INC-5 task storm)
# 2. **Memory pressure** (INC-2 cartesian)
# 3. **Scheduling / parallelism imbalance** (INC-3 skew, INC-4 over-partitioning)
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
