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

# # 🧱 **Module 2: Delta Table Design & Optimization**
# 
# Learn how to identify common Delta table performance problems, apply the right optimization, and validate the impact with before-and-after benchmarks.
# 
# **Duration:** 45 minutes | **Level:** 300–400
# 
# ---
# 
# ### Scenario
# 
# The LEGO manufacturing analytics team ran their initial data pipeline with **every Spark and Delta optimization disabled** — no auto-compaction, no optimize-write, no adaptive query execution, no V-Order, no deletion vectors. The result? Thousands of tiny files, full table scans on every query, and painfully slow DML operations.
# 
# **Your mission:** Fix each table live, one optimization at a time, and measure the impact.
# 
# ### Lab Pattern
# 
# Every exercise follows the same steps:
# 
# | Step | What you do |
# |------|------------|
# | 🐌 **Benchmark** | Run a query and capture the baseline time/metric |
# | 🔍 **Diagnose** | Inspect table metadata to prove the root cause |
# | 🔧 **Fix** | Apply the optimization |
# | 🚀 **Re-benchmark** | Run the same test and compare against the baseline |
# 
# ### Exercises
# 
# | # | Optimization | Table | What's broken |
# |---|-------------|-------|----------------|
# | 1 | `OPTIMIZE` (compaction) | `manufacturing_event` | Thousands of tiny files from streaming |
# | 2 | Optimize Write | `inventory_transaction` | Every commit adds many small files instead of one |
# | 3 | Liquid Clustering | `inventory_transaction` | Selective queries scan every file |
# | 4 | Deletion Vectors | `web_order_line` | DELETE/UPDATE rewrites entire files |
# | 5 | Data Skipping Stats | `wide_order_analysis` | Stats only cover first 32 cols — filter column is invisible |
# 
# ---


# MARKDOWN ********************

# ---
# 
# ## OPTIONAL - Run `source_to_bronze_optimized`
# 
# The Spark Job Definition `source_to_bronze` that was trigged in `00-getting-started` intentionally disables best practice configurations to highlight the impact of suboptimal table layout and compaction strategies. Run the `source_to_bronze_optimized` Spark Job Definition to see the impact on table layout health after the completion of the exercises in this notebook.
# 
# ### 🎯 How to trigger it
# 
# 1. Go to the [source_to_bronze_optimized](https://app.powerbi.com/groups/$workspaceId/sparkjobdefinitions/$sparkJobDefinitionId?experience=fabric-developer) Spark Job Definition
# 1. Click **Run** (top header ribbon). _Note: only click **Run** once. It takes ~5 seconds to show up as active in the run history._
# 
# Don't wait for it to succeed. Move on to the rest of the lab exercises while it runs.

# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# SETUP — Helpers
# ============================================================
import time
from pyspark.sql import DataFrame, functions as F
from delta.tables import DeltaTable

ORIG_SCHEMA = "bronze"        # original unoptimized tables (never modified)
FAST_SCHEMA = "bronze_fast"       # shallow clones we'll fix during the lab

# Disable Intelligent Cache to prevent impacting benchmark comparisons
spark.conf.set('spark.synapse.vegas.useCache', False)

if "_BenchmarkProxy" not in globals():
    raise NotImplementedError("_benchmark_utils was not run! Run the prior cell!")

print("\u2705 Setup complete — helpers loaded")
print(f"   Schemas: orig={ORIG_SCHEMA}, lab={FAST_SCHEMA}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# SETUP — Shallow clone tables into lab schema
# ============================================================
# We clone the unoptimized tables so the originals stay untouched.
# This lets you re-run the lab without re-seeding data.
# Shallow clones reference the same Parquet files — instant, zero copy cost.

LAB_TABLES = ['colors', 'customer', 'inventories', 'inventory_parts', 'inventory_sets', 'inventory_transaction', 'manufacturing_event', 'mold', 'part_categories', 'parts', 'product_return', 'production_line', 'production_order', 'quality_inspection', 'set_price_history', 'sets', 'themes', 'web_order', 'web_order_line']

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FAST_SCHEMA}")

for table in LAB_TABLES:
    spark.sql(f"DROP TABLE IF EXISTS {FAST_SCHEMA}.{table}")
    spark.sql(f"""
        CREATE TABLE {FAST_SCHEMA}.{table}
        SHALLOW CLONE {ORIG_SCHEMA}.{table}
    """)
    print(f"  ✅ {FAST_SCHEMA}.{table} ← shallow clone of {ORIG_SCHEMA}.{table}")

# All exercises operate on FAST_SCHEMA
print(f"\n📌 All exercises will use schema: {FAST_SCHEMA}")
print(f"   Original tables in '{ORIG_SCHEMA}' are preserved for re-runs.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🔍 Before We Tune: Let's See the Data
# 
# Before diving into configuration, let's peek at what we're working with. This is **real LEGO catalog data** — actual sets and themes that are combined with synthetic manufacturing and sales events.

# CELL ********************

# What LEGO sets are people ordering?
print("🧱 Top 10 Most-Ordered LEGO Sets\n")
display(spark.sql(f"""
    SELECT s.name AS set_name, s.set_num, t.name AS theme, s.year, s.num_parts,
           COUNT(*) AS times_ordered, ROUND(SUM(wol.extended_price), 2) AS total_revenue
    FROM {ORIG_SCHEMA}.web_order_line wol
    JOIN {ORIG_SCHEMA}.sets s ON wol.set_num = s.set_num
    LEFT JOIN {ORIG_SCHEMA}.themes t ON s.theme_id = t.id
    GROUP BY s.name, s.set_num, t.name, s.year, s.num_parts
    ORDER BY times_ordered DESC
    LIMIT 10
"""))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# What's happening on the factory floor?
print("\n🏭 Manufacturing Defect Breakdown\n")
display(spark.sql(f"""
    SELECT defect_type, COUNT(*) AS defect_count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
    FROM {ORIG_SCHEMA}.manufacturing_event
    WHERE defect_detected = true
    GROUP BY defect_type
    ORDER BY defect_count DESC
"""))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Warm-up: The Cost of Schema Inference
# 
# Before we even run queries, notice how long it takes to simply **discover the schema** from the landing zone. With thousands of small files, Spark must open many of them to infer column types — this is a hidden startup cost every time the pipeline restarts.
# 
# **Production best practice:** Define schemas statically so your pipeline starts instantly — no file scanning needed.
# 
# ---

# CELL ********************

# ============================================================
# WARM-UP — Time schema inference on the landing zone
# ============================================================

# The unoptimized pipeline left thousands of small files in the landing zone.
# Schema inference must scan these files to discover the schema.
#
# NOTE: spark.read.json() triggers inference EAGERLY at creation time,
# so we must time the DataFrame construction itself — not an action.

LANDING_TABLE = "manufacturing_event"
landing_path = f"Files/landing/{LANDING_TABLE}"

print(f"🐌 Inferring schema from landing zone: {landing_path}\n")
print("   Spark must open files and read metadata to discover columns + types...\n")

with benchmark_op("Schema Inference", "inferred (file scan)", spark):
    inferred_df = spark.read.option("multiline", "true").json(landing_path)

inferred_df.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# WARM-UP — Compare: static schema definition (instant)
# ============================================================
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, BooleanType, DecimalType

# A production pipeline defines the schema once in code — no file scanning needed
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

with benchmark_op("Schema Inference", "static (no scan)", spark):
    static_df = spark.read.schema(static_schema).json(landing_path)

static_df.printSchema()

print("\n📝 Takeaway: production pipelines define schemas upfront.")
print("   Inference is convenient for exploration but adds startup latency,")
print("   especially when scanning thousands of small files.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 1: Fix the Small Files Problem
# 
# **Table:** `inventory_transaction` — high-frequency order lines from the web.
# 
# **What's wrong:** The unoptimized pipeline wrote data via Spark Structured Streaming with no auto-compaction and no optimize-write. Every micro-batch created a new tiny Parquet file. The result: thousands of files, each just a few KB.
# 
# **Why it matters:**
# - Each file requires a separate Spark task → scheduling overhead dominates actual compute
# - File metadata (Parquet footer, Delta stats) is disproportionately large vs. data
# - The Delta transaction log bloats with thousands of `AddFile` entries
# 
# **Fix:** [`OPTIMIZE`](https://learn.microsoft.com/en-us/fabric/data-engineering/table-compaction?tabs=sparksql#optimize-command) — compacts small files into ~128 MB target files.
# 
# ---

# CELL ********************

# ============================================================
# 1️⃣ BENCHMARK — Capture baseline query time
# ============================================================

print("🐌 Running baseline query on web_order_line...\n")

with benchmark_op("OPTIMIZE (compaction)", "before", spark):
    order_query = spark.sql(f"""
        SELECT
            set_num,
            SUM(quantity) AS total_quantity,
            SUM(extended_price) AS total_revenue,
            AVG(unit_price) AS avg_price
        FROM {FAST_SCHEMA}.web_order_line GROUP BY set_num
        """).collect()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 1️⃣ DIAGNOSE — Prove the root cause is small files
# ============================================================

print("🔍 Table diagnostics:\n")
metrics_1_before = show_metrics(f"{FAST_SCHEMA}.web_order_line", "before")

if metrics_1_before["avg_file_kb"] < 1024:
    ratio = round(131072 / max(metrics_1_before["avg_file_kb"], 1))
    print(f"\n   ⚠️  Average file is {metrics_1_before['avg_file_kb']:.0f} KB — optimal target is ~128 MB (131,072 KB)")
    print(f"   ⚠️  Files are {ratio:,}× smaller than they should be")

print("\n📋 DESCRIBE DETAIL:")
display(spark.sql(f"DESCRIBE DETAIL {FAST_SCHEMA}.web_order_line").select("format", "numFiles", "sizeInBytes"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Check the table file count and average file size
# 
# You've seen the problem before, thousands of tiny files that makes performance regress. Can you confirm it?
# 
# **Your task:** check the table file count and average file size of `manufacturing_event` in the `{FAST_SCHEMA}`.
# 
# > 💡 Hint: Describe the details of the table...
# 
# Try it in the cell below!

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
#   <summary><strong>🔑 Solution:</strong> Click to reveal</summary>
# 
# <br/>
# 
# ```python
# df = spark.sql(f"DESCRIBE DETAIL {FAST_SCHEMA}.manufacturing_event")
# display(df)
# ```
# 
# **What to check:**
# - sizeInBytes: the aggregate size of all files in the current snapshot, in bytes.
# - numFiles: the count of files in the current snapshot.
# 
# Files are considered _minimally healthy_ in size if they exceed 64 MB.
# 
# </details>
# 
# ---

# CELL ********************

# ============================================================
# 1️⃣ FIX — Run OPTIMIZE to compact small files
# ============================================================

print("🔧 Running OPTIMIZE {FAST_SCHEMA}.web_order_line...\n")
start = time.time()
result = spark.sql(f"OPTIMIZE {FAST_SCHEMA}.web_order_line")
print(f"   Completed in {round(time.time() - start, 1)}s")
display(result)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 1️⃣ RE-BENCHMARK — Same query, compare against baseline
# ============================================================

print("🚀 Running same query after OPTIMIZE...\n")

with benchmark_op("OPTIMIZE (compaction)", "after", spark):
    order_query = spark.sql(f"""
        SELECT
            set_num,
            SUM(quantity) AS total_quantity,
            SUM(extended_price) AS total_revenue,
            AVG(unit_price) AS avg_price
        FROM {FAST_SCHEMA}.web_order_line GROUP BY set_num
        """).collect()

metrics_1_after = show_metrics(f"{FAST_SCHEMA}.web_order_line", "after")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# `OPTIMIZE` rewrote all those tiny files into a smaller number of properly-sized files (~128 MB each). This is a **one-time reactive compaction** — it doesn't prevent small files from appearing again on the next write.
# 
# > 📝 **Note:** `OPTIMIZE` creates new files but keeps old files for time travel. Run `VACUUM` to reclaim storage (default retention: 7 days).
# 
# ---

# MARKDOWN ********************

# ---
# 
# # Exercise 2: Optimize Write
# 
# **Table:** `inventory_transaction` — part movements across production lines
# 
# **What's wrong:** Look at the Delta transaction log: every streaming commit added many small files when the data could have fit in one. Without **optimize write** (bin-packing at write time), each micro-batch creates a file-per-partition, regardless of how small the data is.
# 
# **Why it matters:**
# - `OPTIMIZE` (Exercise 1) is **reactive** — you run it *after* the damage
# - Optimize write is **proactive** — it bin-packs data *during* the write
# - Without it, you're in an endless cycle: write small files → run OPTIMIZE → write small files again
# 
# **Approach:**
# 1. Analyze the Delta log to see the file-per-commit pattern from the original pipeline
# 2. Replicate the problem with a test write
# 3. Enable optimize write and repeat — see the difference
# 
# ---

# CELL ********************

# ============================================================
# 2️⃣ DIAGNOSE — Analyze the Delta log: files added over commits
# ============================================================
from pyspark.sql.window import Window
print("🔍 Analyzing Delta transaction log for inventory_transaction...\n")

history = spark.sql(f"DESCRIBE HISTORY {ORIG_SCHEMA}.inventory_transaction")

w = Window.orderBy("version").rowsBetween(Window.unboundedPreceding, Window.currentRow)

commit_stats = (
    history
    .select(
        F.col("version"),
        F.col("timestamp"),
        F.col("operation"),
        F.col("operationMetrics.numOutputRows").cast("long").alias("rows_written"),
        F.coalesce(
            F.col("operationMetrics.numFiles").cast("int"),
            F.col("operationMetrics.numAddedFiles").cast("int")
        ).alias("files_added"),
        F.lit(1).alias("files_if_ow_enabled")
    )
    .filter("files_added IS NOT NULL AND files_added > 0")
    .withColumn("iteration", F.row_number().over(Window.orderBy("version")))
    .withColumn("compaction_group", F.floor((F.col("iteration") - 1) / 50))
    .orderBy("version")
)

w_compaction_group = (
    Window
    .partitionBy("compaction_group")
    .orderBy("version")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

commit_stats_agg = (
    commit_stats
    .withColumn(
        "cumulative_files",
        F.sum("files_added").over(w)
    )
    .withColumn(
        "simulated_files_if_ow_enabled",
        F.sum("files_if_ow_enabled").over(w)
    )
    .withColumn(
        "simulated_files_if_ow_and_ac_enabled",
        F.sum("files_if_ow_enabled").over(w_compaction_group)
    )
    .withColumn(
        "cuml_rows_written",
        F.sum(F.coalesce("rows_written", F.lit(0))).over(w)
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Visualize file growth over time
# Chart the cumulative files added over time vs. the cumulative files added if Optimize Write were enabled to bin-pack small partitions of data before writing.
# 1. Select **New chart**
# 1. Create a **Line Chart**
# 1. Add **version** to the _X-axis_
# 1. Add the following fields to the _Y-axis_:
#     - **cumulative_files_added**
#     - **simulated_files_if_ow_enabled**
#     - **simulated_files_if_ow_and_ac_enabled**
# 
# Without running `OPTIMIZE`, this table will have 2-3x the number of files causing unnecessary slowness for queries and update operations.

# CELL ********************

display(commit_stats_agg)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Replicate: write WITHOUT optimize write
# ============================================================

# Start from a clean compacted state
spark.sql(f"OPTIMIZE {FAST_SCHEMA}.inventory_transaction")
files_before = get_table_metrics(f"{FAST_SCHEMA}.inventory_transaction")["num_files"]

# Simulate a streaming micro-batch by appending data
# Use repartition to mimic how streaming creates multiple partitions
BATCH_ROWS = 5000

print(f"🐌 Appending {BATCH_ROWS:,} rows WITHOUT optimize write (repartitioned to 8 files)...\n")

batch = spark.table(f"{FAST_SCHEMA}.inventory_transaction").limit(BATCH_ROWS).repartition(8)
batch.write.format("delta").mode("append").saveAsTable(f"{FAST_SCHEMA}.inventory_transaction")

files_after_bad = get_table_metrics(f"{FAST_SCHEMA}.inventory_transaction")["num_files"]
new_files_without_ow = files_after_bad - files_before
print(f"   Files before append: {files_before:,}")
print(f"   Files after append:  {files_after_bad:,}")
print(f"   ⚠️  New files created: {new_files_without_ow}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Mitigate Future Small Files
# 
# You've fixed the existing small files with OPTIMIZE. But new writes will create small files again unless you change the table's write behavior.
# 
# **Your task:** Set the `delta.autoOptimize.optimizeWrite` and `delta.autoOptimize.autoCompact` table properties to `'true'` on `manufacturing_event`.
# 
# > 💡 Hint: Use `ALTER TABLE ... SET TBLPROPERTIES`

# CELL ********************

# YOUR CODE HERE
# Enable optimize write on inventory_transaction
# spark.sql(f"ALTER TABLE ...")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
#   <summary><strong>🔑 Solution:</strong> Click to reveal</summary>
# 
# <br/>
# 
# ```python
# spark.sql(f"""
#     ALTER TABLE {FAST_SCHEMA}.inventory_transaction
#     SET TBLPROPERTIES (
#       'delta.autoOptimize.optimizeWrite' = 'true',
#       'delta.autoOptimize.autoCompact' = 'true'
#     )
# """)
# ```
# 
# **Optimize Write vs Auto Compact:**
# - **Optimize Write** — coalesces small partitions at write time (prevents small files)
# - **Auto Compact** — runs a mini-OPTIMIZE after each write (fixes small files after the fact)
# - Best practice: enable both for most tables
# 
# </details>
# 
# ---

# CELL ********************

# =======================================================================
# 2️⃣ RE-BENCHMARK — Same append, now with optimize write set on the table
# =======================================================================

# Clean baseline
spark.sql(f"OPTIMIZE {FAST_SCHEMA}.inventory_transaction")
files_before_good = get_table_metrics(f"{FAST_SCHEMA}.inventory_transaction")["num_files"]

print(f"🚀 Appending {BATCH_ROWS:,} rows WITH optimize write (same repartition to 8)...\n")

batch = spark.table(f"{FAST_SCHEMA}.inventory_transaction").limit(BATCH_ROWS).repartition(8)
batch.write.format("delta").mode("append").saveAsTable(f"{FAST_SCHEMA}.inventory_transaction")

files_after_good = get_table_metrics(f"{FAST_SCHEMA}.inventory_transaction")["num_files"]
new_files_with_ow = files_after_good - files_before_good

print(f"   Files before append: {files_before_good:,}")
print(f"   Files after append:  {files_after_good:,}")
print(f"   New files created:   {new_files_with_ow}")

print(f"\n{'=' * 60}")
print(f"  Exercise 2: Optimize Write")
print(f"{'=' * 60}")
print(f"  {'Metric':<30} {'Without OW':>12} {'With OW':>12}")
print(f"  {'-' * 54}")
print(f"  {'New files per append':<30} {new_files_without_ow:>12} {new_files_with_ow:>12}")
print(f"  {'Overhead vs. ideal (1 file)':<30} {new_files_without_ow:>11}x {'1x':>12}")
print(f"{'=' * 60}")

benchmarks["Exercise 2: Optimize Write"] = {
    "before_s": new_files_without_ow, "after_s": new_files_with_ow,
    "speedup": round(new_files_without_ow / max(new_files_with_ow, 1), 1),
    "before_files": new_files_without_ow, "after_files": new_files_with_ow,
    "note": "files created per append"
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Without optimize write, Spark wrote one file per output partition — even though the data was tiny. With optimize write enabled, Spark **bin-packs** the data at write time, coalescing small partitions into properly-sized files.
# 
# | Setting | Behavior |
# |---------|----------|
# | `optimizeWrite = false` | 1 file per partition → 8 files for 5,000 rows |
# | `optimizeWrite = true` | Bin-packs into ~1 file (data fits in one ~128 MB target) |
# | `autoCompact = true` | Triggers synchronous compaction after writes exceed small file threshold (defaults to 50) |
# 
# > 📝 **Optimize write is one of the most important settings for preventing small files.** It mitigates small file issue pre-write — no post-hoc `OPTIMIZE` needed.
# ---

# MARKDOWN ********************

# ---
# 
# # Exercise 3: Liquid Clustering
# 
# **Table:** `manufacturing_event`
# 
# **What's wrong:** Even after compaction, data files contain a random mix of all values. When you filter by `part_num` or `transaction_type`, Spark must scan **every file** because the min/max file-level statistics overlap across all files — nothing can be skipped.
# 
# **Why it matters:**
# - A query for one specific part number scans 100% of the data
# - No files can be pruned via Delta data skipping
# - Gets worse as the table grows
# 
# **Fix:** [Liquid clustering](https://learn.microsoft.com/en-us/fabric/data-engineering/liquid-clustering?tabs=sparksql) — co-locates related rows in the same files so Delta can skip irrelevant files entirely.
# 
# ---

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Run a selective query (scans everything)
# ============================================================

# Compact first so file layout is clean. Use a 1m targetFileSize to ensure enough files to demonstrate file skipping 
spark.sql(f"ALTER TABLE {FAST_SCHEMA}.manufacturing_event SET TBLPROPERTIES ('delta.targetFileSize' = '1m')")
spark.sql(f"OPTIMIZE {FAST_SCHEMA}.manufacturing_event")
metrics_3_before = show_metrics(f"{FAST_SCHEMA}.manufacturing_event", "before clustering")

print(f"\n🐌 Running selective query: WHERE color_id = 8...\n")

query = spark.sql(f"""
    SELECT part_num, COUNT(*) AS cnt, SUM(cycle_time_ms) AS total_cycle_time_ms
    FROM {FAST_SCHEMA}.manufacturing_event
    WHERE color_id = 8
    GROUP BY part_num
""")
with benchmark_op("Liquid Clustering", "before (no clustering)", spark):
    query.collect()

print(f"\n🔍 Files scanned in query: {len(query.inputFiles())}\n")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ FIX — Apply liquid clustering and re-optimize
# ============================================================

print("🔧 Applying liquid clustering: CLUSTER BY (color_id)...\n")
spark.sql(f"ALTER TABLE {FAST_SCHEMA}.manufacturing_event CLUSTER BY (color_id)")
print("✅ Clustering columns set\n")

print("Running OPTIMIZE to rewrite files with clustering layout...")
start = time.time()
optimize_metrics_df = spark.sql(f"OPTIMIZE {FAST_SCHEMA}.manufacturing_event FULL")
print(f"   Completed in {round(time.time() - start, 1)}s")

metrics_3_after = show_metrics(f"{FAST_SCHEMA}.manufacturing_event", "after clustering + OPTIMIZE")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Check Liquid Clustering Quality Metrics
# 
# `OPTIMIZE` metrics in Fabric Spark Runtime 2.0 contains a `clusteringQuality` struct. Query the struct to see the clustering quality.
# 
# **Your task:**
# 
# 1. Drill into the `optimize_metrics_df` DataFrame to inspect the `skippingEffectiveness` of the clustering column(s).
# 
# > 💡 Hint: Use `EXPLODE` and `*` expand

# CELL ********************

# Extract clustering quality metrics from optimize_metrics DataFrame


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
#   <summary><strong>🔑 Solution:</strong> Click to reveal</summary>
# 
# <br/>
# 
# ```python
# metrics_parsed_df = optimize_metrics_df \
#     .selectExpr("explode(metrics.clusteringQuality) as clustering_quality") \
#     .select("clustering_quality.*")
#     
# display(metrics_parsed_df)
# ```
# 
# </details>
# 
# ---

# CELL ********************

# ============================================================
# 3️⃣ RE-BENCHMARK — Same query, now with clustering
# ============================================================

print(f"🚀 Running same query: WHERE color_id = 8...\n")

query = spark.sql(f"""
    SELECT transaction_type, COUNT(*) AS cnt, SUM(quantity) AS total_qty
    FROM {FAST_SCHEMA}.inventory_transaction
    WHERE color_id = 8
    GROUP BY transaction_type
""")
with benchmark_op("Liquid Clustering", "after (clustered)", spark):
    query.collect()

metrics_3_after = show_metrics(f"{FAST_SCHEMA}.inventory_transaction", "after clustering")

print(f"\n🔍 Files scanned in query: {len(query.inputFiles())}\n")

print("\n💡 Check Spark UI → SQL tab → Scan node for 'number of files read' —")
print("   with healthy clustering (see skippingEffectiveness in the prior cell), most files should be skipped for selective filters.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Liquid clustering **physically re-organizes data** so rows with similar `PartNum` and `TransactionType` values end up in the same files. Delta's min/max file statistics can then immediately skip files that don't contain matching values.
# 
# | Before clustering | After clustering |
# |---|---|
# | Every file has a random mix of all part numbers | Files contain contiguous ranges of part numbers |
# | Scanning `WHERE PartNum = 'X'` reads ALL files | Same query prunes most files via data skipping |
# 
# > 🆚 **Liquid clustering vs. partitioning:** Partitioning creates directory-level splits — great for low-cardinality columns, but creates a small files nightmare with high cardinality. Liquid clustering achieves similar data skipping **without** the small files trap, and it's incremental — new data is automatically clustered when `autoCompact` is enabled.
# 
# ---

# MARKDOWN ********************

# ---
# 
# # Exercise 4: Deletion Vectors
# 
# **Table:** `web_order_line` — order line items for LEGO set purchases
# 
# **Columns:** `OrderId`, `LineNumber`, `SetNum`, `PartNum`, `ItemName`, `Quantity`, `UnitPrice`, `ExtendedPrice`
# 
# **What's wrong:** Without deletion vectors, every `DELETE` or `UPDATE` must **rewrite the entire Parquet files** that contain affected rows. Even deleting a single row triggers a full file rewrite.
# 
# **Why it matters:**
# - A `DELETE` touching 100 rows across 10 files rewrites all 10 files completely
# - Write amplification is enormous — you rewrite GBs to remove KBs
# - `MERGE` operations (common in ETL) suffer the same penalty
# 
# **Fix:** [Deletion vectors](https://learn.microsoft.com/fabric/data-engineering/delta-optimization-and-v-order?tabs=sparksql#deletion-vectors) — instead of rewriting files, Delta writes a small sidecar file that marks which rows are logically deleted. The original data files stay untouched.
# 
# ---

# CELL ********************

# 🧱 Let's peek at the rows we're about to delete
print("🗑️ Sample rows that will be deleted (defect_detected = true)\n")
display(spark.sql(f"""
    SELECT set_num, COUNT(*) as cnt
    FROM {FAST_SCHEMA}.web_order_line
    WHERE set_num IS NOT NULL
    GROUP BY set_num
    ORDER BY cnt ASC
    LIMIT 2
"""))

total_count = spark.table(f"{FAST_SCHEMA}.web_order_line").count()
print(f"\n📊 2 order lines out of {total_count:,} total ({2/total_count*100:.5f}%)")
print("   Next: we'll DELETE these rows and compare with vs. without Deletion Vectors")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ SETUP — Compact the table and pick two test conditions
# ============================================================

print("🔧 Preparing web_order_line (OPTIMIZE to clean baseline)...\n")
spark.sql(f"OPTIMIZE {FAST_SCHEMA}.web_order_line")
spark.sql(f"ALTER TABLE {FAST_SCHEMA}.web_order_line UNSET TBLPROPERTIES ('delta.enableDeletionVectors')")
metrics_4_before = show_metrics(f"{FAST_SCHEMA}.web_order_line", "baseline")

# Pick two different SetNums for before/after DELETE comparison
set_nums = spark.sql(f"""
    SELECT set_num, COUNT(*) as cnt
    FROM {FAST_SCHEMA}.web_order_line
    WHERE set_num IS NOT NULL
    GROUP BY set_num
    ORDER BY cnt ASC
    LIMIT 2
""").collect()

set_a, count_a = set_nums[0][0], set_nums[0][1]
set_b, count_b = set_nums[1][0], set_nums[1][1]
print(f"\n   Will DELETE set_num='{set_a}' ({count_a:,} rows) without DVs")
print(f"   Will DELETE set_num='{set_b}' ({count_b:,} rows) with DVs")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — DELETE without deletion vectors (rewrites files)
# ============================================================

print(f"🐌 DELETE WHERE set_num = '{set_a}' (without deletion vectors)...\n")

with benchmark_op("Deletion Vectors", "without DVs (file rewrite)", spark):
    spark.sql(f"DELETE FROM {FAST_SCHEMA}.web_order_line WHERE set_num = '{set_a}'")

# Check the commit to see how many files were rewritten
history = spark.sql(f"DESCRIBE HISTORY {FAST_SCHEMA}.web_order_line LIMIT 1").select(
    "version", "operation", "operationMetrics"
).collect()[0]
op_metrics = history["operationMetrics"]
files_rewritten_before = int(op_metrics.get("numRemovedFiles", 0))

print(f"   Files rewritten: {files_rewritten_before}")
print(f"   ⚠️  Entire files rewritten just to remove {count_a:,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ FIX — Enable deletion vectors
# ============================================================

print("🔧 Enabling deletion vectors...\n")
spark.sql(f"""
    ALTER TABLE {FAST_SCHEMA}.web_order_line SET TBLPROPERTIES (
        'delta.enableDeletionVectors' = 'true'
    )
""")

props = spark.sql(f"SHOW TBLPROPERTIES {FAST_SCHEMA}.web_order_line").filter("key LIKE '%deletionVector%'")
display(props)
print("✅ Deletion vectors enabled")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ RE-BENCHMARK — DELETE with deletion vectors (no file rewrite)
# ============================================================

print(f"🚀 DELETE WHERE SetNum = '{set_b}' (with deletion vectors)...\n")

with benchmark_op("Deletion Vectors", "with DVs (sidecar only)", spark):
    spark.sql(f"DELETE FROM {FAST_SCHEMA}.web_order_line WHERE set_num = '{set_b}'")

history = spark.sql(f"DESCRIBE HISTORY {FAST_SCHEMA}.web_order_line LIMIT 1").select(
    "version", "operation", "operationMetrics"
).collect()[0]
op_metrics = history["operationMetrics"]
files_rewritten_after = int(op_metrics.get("numRemovedFiles", 0))

print(f"   Files rewritten: {files_rewritten_after}")
print(f"\n💡 With DVs, Delta wrote a tiny sidecar file marking deleted rows.")
print(f"   The original data files were NOT rewritten — massive write amplification savings.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 5: Data Skipping Stats on Wide Tables
# 
# **Table:** `production_analysis` — a denormalized join of manufacturing events, production lines, molds, colors, parts, and part categories.
# 
# **What's wrong:** Delta collects min/max file statistics for only the **first 32 columns** by default (`delta.dataSkippingNumIndexedCols = 32`). When a wide table has a filter column beyond position 32, Delta has no stats for it — and will **block you from clustering** on that column with `DELTA_CLUSTERING_COLUMN_MISSING_STATS`.
# 
# **Why it matters:**
# - Denormalized / wide tables are common in analytics lakehouses
# - Without stats, data skipping is blind — every file is scanned
# - Clustering requires stats on the clustering column — you can't even enable it
# - No warning until you try — queries silently scan everything
# 
# **Exercise flow:**
# 1. Create a wide table and observe full-table scans via `input_file_name()`
# 2. Try to enable clustering → hit the `DELTA_CLUSTERING_COLUMN_MISSING_STATS` error
# 3. Unblock: extend stats coverage, rewrite files, then cluster
# 4. Measure the improvement
# 
# ---


# CELL ********************

# ============================================================
# 5⃣ SETUP — Create a wide denormalized table
# ============================================================

# Join LEGO manufacturing tables into a single wide table (35+ columns).
# manufacturing_event is the high-frequency IoT fact table — the biggest in the lakehouse.
# The target filter column (part_num) will land PAST position 32.

spark.conf.set('spark.synapse.vegas.useCache', False)

wide_df = spark.sql(f"""
    SELECT
        -- manufacturing_event columns (1-10)
        me.event_id,
        me.timestamp             AS event_timestamp,
        me.machine_id,
        me.color_id,
        me.mold_temp,
        me.injection_pressure,
        me.cycle_time_ms,
        me.defect_detected,
        me.defect_type,
        me.batch_id,

        -- production_line columns (11-15)
        pl.plant,
        pl.cell_type,
        pl.capacity              AS line_capacity,
        pl.installed_year,
        pl.last_maintenance,

        -- mold columns (16-24)
        m.mold_id,
        m.cavity_count,
        m.material               AS mold_material,
        m.max_shots,
        m.current_shots,
        m.status                 AS mold_status,
        m.commission_date,
        m.last_resurfaced,
        m.assigned_line_id,

        -- colors columns (25-31)
        c.name                   AS color_name,
        c.rgb,
        c.is_trans,
        c.num_parts              AS color_num_parts,
        c.num_sets               AS color_num_sets,
        c.y_1                     AS color_first_year,
        c.y_2                     AS color_last_year,

        -- parts columns (32-36) — PAST THE 32-COLUMN BOUNDARY
        p.part_cat_id,
        me.part_num,
        p.name                   AS part_name,
        p.part_material,
        pc.name                  AS part_category_name

    FROM {FAST_SCHEMA}.manufacturing_event me
    JOIN {FAST_SCHEMA}.production_line pl  ON me.machine_id = pl.line_id
    LEFT JOIN (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY assigned_line_id, part_num
            ORDER BY CASE status WHEN 'active' THEN 1 ELSE 2 END, current_shots DESC
        ) AS rn
        FROM {FAST_SCHEMA}.mold
    ) m ON m.assigned_line_id = me.machine_id AND m.part_num = me.part_num AND m.rn = 1
    JOIN {FAST_SCHEMA}.colors c            ON me.color_id = c.id
    JOIN {FAST_SCHEMA}.parts p             ON me.part_num = p.part_num
    JOIN {FAST_SCHEMA}.part_categories pc  ON p.part_cat_id = pc.id
""")

num_cols = len(wide_df.columns)
print(f"\U0001f4d0 Wide table has {num_cols} columns")

# Write as an UNCLUSTERED Delta table (no clustering yet!)
WIDE_TABLE = "production_analysis"

spark.sql(f"DROP TABLE IF EXISTS {FAST_SCHEMA}.{WIDE_TABLE}")

# write table, repartition to force many files (don't do this!)
wide_df.repartition(100).write \
    .option("delta.targetFileSize", "1m") \
    .saveAsTable(f"{FAST_SCHEMA}.{WIDE_TABLE}")

show_metrics(f"{FAST_SCHEMA}.{WIDE_TABLE}", "unclustered")

# Show column positions
print(f"\n\U0001f4cb Column positions:")
for i, col in enumerate(wide_df.columns, 1):
    marker = " \U0001f448 FILTER COLUMN (past position 32!)" if col == "part_num" else ""
    boundary = " \u2500\u2500 stats boundary \u2500\u2500" if i == 32 else ""
    if i <= 3 or i >= 30 or col == "part_num":
        print(f"   {i:>3}. {col}{boundary}{marker}")
    elif i == 4:
        print(f"       ...")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 🧱 Let's explore the wide table — look at the LEGO manufacturing data!
print("🏭 Sample rows from the wide denormalized table\n")
display(spark.sql(f"""
    SELECT part_num, part_name, plant, color_name,
           ROUND(mold_temp, 1) AS mold_temp, defect_detected, batch_id
    FROM {FAST_SCHEMA}.production_analysis
    ORDER BY event_timestamp DESC
    LIMIT 15
"""))

# Show part distribution
print("\n🏷️ Top parts by manufacturing event volume\n")
display(spark.sql(f"""
    SELECT part_num, part_name, COUNT(*) AS events,
           ROUND(AVG(mold_temp), 1) AS avg_temp,
           SUM(CASE WHEN defect_detected THEN 1 ELSE 0 END) AS defects
    FROM {FAST_SCHEMA}.production_analysis
    GROUP BY part_num, part_name
    ORDER BY events DESC
    LIMIT 10
"""))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5⃣ BENCHMARK — Query filtering on a column past position 32
# ============================================================

# Pick a frequently produced part to filter on
sample_part = spark.sql(f"""
    SELECT part_num, COUNT(*) AS cnt
    FROM {FAST_SCHEMA}.{WIDE_TABLE}
    GROUP BY part_num
    ORDER BY cnt DESC LIMIT 1
""").collect()[0]["part_num"]

print(f"\U0001f50d Filtering on part_num = '{sample_part}'")
print(f"   part_num is at column position {wide_df.columns.index('part_num') + 1} (past the 32-col stats window)\n")

# How many files does the query scan?
total_files = len(spark.table(f"{FAST_SCHEMA}.{WIDE_TABLE}").inputFiles())
query_no_stats_unclustered = spark.sql(f"""
        SELECT *
        FROM {FAST_SCHEMA}.{WIDE_TABLE}
        WHERE part_num = '{sample_part}'
    """)
filtered_files = len(query_no_stats_unclustered.inputFiles())

print(f"\U0001f4c1 File scan analysis:")
print(f"   Total files in table:       {total_files}")
print(f"   Files read by filtered query: {filtered_files}")
print(f"   Files pruned:               {total_files - filtered_files}")
print(f"   Pruning ratio:              {((total_files - filtered_files) / max(total_files, 1)) * 100:.0f}%")

if filtered_files >= total_files:
    print(f"\n   \u26a0\ufe0f  No files pruned! Every file was scanned despite filtering on part_num.")
    print(f"   Without stats on part_num, Delta cannot skip any files.")

# Baseline timing
with benchmark_op("Data Skipping Stats", "before (no stats past col 32)", spark):
    display(query_no_stats_unclustered)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5⃣ DIAGNOSE — Try to enable clustering on theme_name
# ============================================================

# You might think: "just cluster on theme_name to co-locate the data!"
# But clustering REQUIRES stats on the clustering column.
# Since theme_name is at position 35 (past the 32-col stats boundary),
# Delta will REJECT the clustering request.


print("\U0001f9ea Attempting: ALTER TABLE ... CLUSTER BY (part_num)\n")

try:
    spark.sql(f"ALTER TABLE {FAST_SCHEMA}.{WIDE_TABLE} CLUSTER BY (part_num)")
    print("\u2705 Clustering enabled (stats already exist for part_num)")
except Exception as e:
    error_msg = str(e)
    print(f"\u274c BLOCKED! Spark error:\n")
    # Extract the key error message
    if "DELTA_CLUSTERING_COLUMN_MISSING_STATS" in error_msg:
        print(f"   DELTA_CLUSTERING_COLUMN_MISSING_STATS")
        print(f"   Clustering column 'part_num' isn't enabled for statistics collection.")
        print(f"\n\U0001f4a1 Why? Delta only collects min/max stats for the first 32 columns.")
        print(f"   part_num is at position {wide_df.columns.index('part_num') + 1} \u2014 outside the stats window.")
        print(f"   Without stats, clustering can't determine how to organize files.")
    else:
        print(f"   {error_msg[:500]}")

print(f"\n\U0001f4ca Current data skipping configuration:")
props = spark.sql(f"SHOW TBLPROPERTIES {FAST_SCHEMA}.{WIDE_TABLE}").collect()
skip_props = {r['key']: r['value'] for r in props if 'skip' in r['key'].lower() or 'indexed' in r['key'].lower()}
if skip_props:
    for k, v in skip_props.items():
        print(f"   {k} = {v}")
else:
    print(f"   delta.dataSkippingNumIndexedCols = 32  (default \u2014 not explicitly set)")

print(f"\n\U0001f4cb Table has {len(wide_df.columns)} columns")
print(f"   Stats collected for:  columns 1-32")
print(f"   Stats MISSING for:   columns 33-{len(wide_df.columns)}")
print(f"   part_num position: {wide_df.columns.index('part_num') + 1} \u2190 not enabled for stats\u2192 clustering blocked")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge: Unblock Clustering on `part_num`
# 
# You saw the `DELTA_CLUSTERING_COLUMN_MISSING_STATS` error. Stats are only collected for the first 32 columns, and `part_num` is at position 33.
# 
# **Your task:** Fix it in 3 steps:
# 1. Apply one of the 3 methods to customize columns indexed with stats:
#     - `delta.dataSkippingNumIndexedCols` to `-1` (collect stats on ALL columns)
#     - `delta.dataSkippingStatsColumns` to `part_num` (collect stats on ALL columns)
#     - Move `part_num` within the first 32 columns (`ALTER TABLE ... ALTER COLUMN part_num [AFTER ... | FIRST]`)
# 
# 2. Run `OPTIMIZE FULL` to rewrite files with complete stats
# 3. Enable `CLUSTER BY (part_num)`
# 
# > 💡 You'll need three `spark.sql()` calls with ALTER TABLE, OPTIMIZE FULL, and ALTER TABLE again.

# CELL ********************

WIDE_TABLE = "production_analysis"

# YOUR CODE HERE
# Step 1: Extend stats coverage to all columns
# spark.sql(f"ALTER TABLE {FAST_SCHEMA}.{WIDE_TABLE} SET TBLPROPERTIES ...")

# Step 2: Now enable clustering
# spark.sql(f"ALTER TABLE ...")

# Step 3: Rewrite files with complete stats
# spark.sql(f"OPTIMIZE ... FULL")
# Step 1: Tell Delta to collect stats on ALL columns

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
#   <summary><strong>🔑 Solution:</strong> Click to reveal</summary>
# 
# <br/>
# 
# ```python
# WIDE_TABLE = "production_analysis"
# 
# # Step 1: Tell Delta to collect stats on ALL columns
# spark.sql(f"ALTER TABLE {FAST_SCHEMA}.{WIDE_TABLE} ALTER COLUMN part_num FIRST")
# 
# # Step 2: Enable clustering
# spark.sql(f"ALTER TABLE {FAST_SCHEMA}.{WIDE_TABLE} CLUSTER BY (part_num)")
# 
# # Step 3: Rewrite existing files to include the new stats
# spark.sql(f"OPTIMIZE {FAST_SCHEMA}.{WIDE_TABLE} FULL")
# ```
# 
# **Why the order matters:**
# 1. Changing the property only affects _future_ file writes
# 2. `OPTIMIZE FULL` rewrites _all_ existing files with the new stats config
# 3. Only THEN can clustering be enabled — it requires stats on the clustering column
# 
# </details>
# 
# ---

# CELL ********************

# ============================================================
# 5⃣ RE-BENCHMARK — Same query, now with stats + clustering
# ============================================================

WIDE_TABLE = "production_analysis"

print(f"\U0001f680 Running same query: WHERE part_num = '{sample_part}'\n")

# Check file pruning again
total_files = len(spark.table(f"{FAST_SCHEMA}.{WIDE_TABLE}").inputFiles())
query_w_stats_clustered = spark.sql(f"""
        SELECT *
        FROM {FAST_SCHEMA}.{WIDE_TABLE}
        WHERE part_num = '{sample_part}'
    """)
filtered_files = len(query_w_stats_clustered.inputFiles())

print(f"\U0001f4c1 File scan analysis (after fix):")
print(f"   Total files in table:       {total_files}")
print(f"   Files read by filtered query: {filtered_files}")
print(f"   Files pruned:               {total_files - filtered_files}")
print(f"   Pruning ratio:              {((total_files - filtered_files) / max(total_files, 1)) * 100:.0f}%")

if filtered_files < total_files:
    print(f"\n   \u2705 Data skipping is working! Most files were pruned.")

# Timing comparison
with benchmark_op("Data Skipping Stats", "after (stats + clustering)", spark):
    display(query_w_stats_clustered)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 💡 What Just Happened?
# 
# Delta Lake collects **min/max statistics** per column per file. These stats power **data skipping** — the ability to prune entire files from a scan when the filter value falls outside a file’s min/max range.
# 
# By default, stats are only collected for the **first 32 columns** (`delta.dataSkippingNumIndexedCols = 32`). This creates two problems for wide tables:
# 
# 1. **Queries silently scan everything** — no stats means no file pruning, even with good data layout
# 2. **Clustering is blocked** — `DELTA_CLUSTERING_COLUMN_MISSING_STATS` prevents you from clustering on columns without stats
# 
# | Scenario | Stats? | Clustering? | Files pruned | Scan cost |
# |----------|--------|-------------|--------------|-----------|
# | `theme_name` at position 35, default settings | ❌ No | ❌ Blocked | 0 | Full scan |
# | After `dataSkippingNumIndexedCols = -1` + OPTIMIZE FULL + CLUSTER BY | ✅ Yes | ✅ Enabled | Most files | Minimal scan |
# 
# **Three ways to fix it:**
# 
# | Fix | Table property | When to use |
# |-----|---------------|-------------|
# | Reorder columns | N/A — schema change | Best if you control the schema; put high-cardinality filter columns first |
# | Index all columns | `delta.dataSkippingNumIndexedCols = -1` | Easy fix; small write overhead per column |
# | Index specific columns | `delta.dataSkippingStatsColumns = 'col1,col2'` | Surgical; only pay stats cost for columns you filter on |
# 
# > ⚠️ **Order matters:** You must (1) extend stats coverage, (2) enable clustering, and THEN (3) `OPTIMIZE FULL` to rewrite files with new stats, Clustering requires columns be _enabled_ for stats collection.
# 
# ---


# MARKDOWN ********************

# ---
# 
# # 🏆 Summary Dashboard
# 
# All five optimizations applied. Here's the full impact across every exercise.
# 
# ---


# CELL ********************

# ============================================================
# SUMMARY — All benchmark results
# ============================================================

print("=" * 62)
print("  🏆  PERFORMANCE IMPACT SUMMARY")
print("=" * 62)

for scenario, states in benchmarks.items():
    if isinstance(states, dict):
        baseline_key = next(iter(states))
        baseline_ms = states[baseline_key]
        best_ms = min(states.values())
        W = 58
        print(f"\n  \u250c{'\u2500' * W}\u2510")
        title = f"\033[1m{scenario}\033[0m"
        title_pad = W - 2 - len(scenario)
        print(f"  \u2502  {title}{' ' * title_pad}\u2502")
        print(f"  \u251c{'\u2500' * W}\u2524")
        print(f"  \u2502  {'State':<28}{'Time (ms)':>12}{'Factor':>14}  \u2502")
        print(f"  \u251c{'\u2500' * W}\u2524")
        for s, ms in states.items():
            ratio = baseline_ms / max(ms, 0.001)
            if s == baseline_key:
                visible_tag = "baseline"
                tag = visible_tag
            elif ms <= best_ms:
                visible_tag = f"{ratio:.1f}x faster"
                tag = f"\033[1;32m{visible_tag}\033[0m"
            else:
                visible_tag = f"{ratio:.1f}x"
                tag = f"\033[1;34m{visible_tag}\033[0m"
            pad = 14 - len(visible_tag)
            print(f"  \u2502  {s:<28}{ms:>12.2f}{' ' * pad}{tag}  \u2502")
        print(f"  \u2514{'\u2500' * W}\u2518")

print(f"\n{'=' * 62}")
print("""
KEY TAKEAWAYS
──────────────
1. OPTIMIZE compacts small files — but it's REACTIVE (run after the damage)
2. Optimize Write bin-packs at write time — PROACTIVE, prevents small files at source
3. Liquid clustering enables data skipping — selective queries skip irrelevant files
4. Deletion vectors eliminate write amplification — DELETEs don't rewrite files
5. Data skipping stats must cover filter columns — wide tables need explicit config
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# ### 📝 What the Optimized Pipeline Configures Automatically
# 
# The `seed_lego_delta_tables` Spark Job Definition uses ArcFlow's `SparkConfigurator` to set these best-practice configs at session startup:
# 
# | Setting | Value | What it does |
# |---------|-------|-------------|
# | `autoCompact.enabled` | `true` | Compacts small files after writes |
# | `optimizeWrite.enabled` | `true` | Bin-packs data during writes |
# | `targetFileSize.adaptive.enabled` | `true` | Adapts target file size to workload |
# | `enableDeletionVectors` | `true` | Marks deleted rows instead of rewriting files |
# | `optimize.fast.enabled` | `true` | Faster OPTIMIZE via incremental compaction |
# | `parquet.compression.codec` | `zstd` | Better compression ratio than snappy |
# | `sql.adaptive.enabled` | `true` | Adaptive Query Execution (AQE) |
# | `native.enabled` | `true` | Native Execution Engine |
# 
# > 💡 **The best optimization is the one you never have to run manually.**
# 
# ---
