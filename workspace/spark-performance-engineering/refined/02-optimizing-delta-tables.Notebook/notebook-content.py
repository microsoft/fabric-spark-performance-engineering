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

# # Module 2 — Optimizing Delta Tables
# 
# ## What this module teaches
# 
# This module focuses on **physical/table-layer fixes**: the query code is already reasonable, but the *table layout / Delta config* is the problem. These changes are the highest-leverage lever in the Spark tuning hierarchy because one fix can speed up every downstream query.
# 
# ## Assumed from prior modules
# 
# Module 1 gave you the diagnostic toolkit: Spark UI, `explain`, `DESCRIBE DETAIL`, `DESCRIBE HISTORY`, and `inputFiles()`. Module 1 taught you to *diagnose* tiny files; here you *fix* them.
# 
# ## Litmus test
# 
# If the remedy is `OPTIMIZE`, clustering, schema/data-type cleanup, partition design, deletion vectors, or Delta table properties, it belongs here. If the remedy changes query code, joins, caching, or AQE, it belongs in a later module.
# 
# | Exercise | Scenario | Expected performance signal |
# |---|---|---|
# | 1. OPTIMIZE / compaction | Compact deliberately tiny `manufacturing_event_tiny` files. | File count drops from many to few, average file size increases, same aggregate scan runs faster. |
# | 2. Optimize Write | Append the same `inventory_transaction` batch before and after table Optimize Write / Auto Compact properties. | No small-file accumulation after repeated writes; new files created per append stays low. |
# | 3. Liquid clustering + data-skipping stats | Cluster `inventory_transaction_clustered` by `color_id` and collect stats for selective filters. | Files skipped via clustering, fewer input files read, less data scanned for the same `color_id` filter. |
# | 4. Deletion vectors | Delete small row sets from matched `web_order_line` tables with DVs off vs on. | DELETE avoids full-file rewrite where supported; history shows fewer removed / rewritten files. |
# | 5. Data-type optimization | Rewrite `inventory_transaction` with numeric `color_id` stored as `int` instead of `string`. | Table size decreases after right-sizing types; numeric stats support cleaner filtered scans. |
# | 6. Partitioning strategy | Compare no partitioning, high-cardinality `part_num`, and date partitioning. | Avoid over-partitioning; balanced file sizes and input-file counts align with time-range and point filters. |
# | 7. Delta storage-regression audit | Use history and properties to find an append that regressed `inventory_transaction_audit` layout. | Isolate the commit/version that regressed file layout via `DESCRIBE HISTORY`, then verify file count and pruning recover. |

# CELL ********************

%run _benchmark_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# SETUP — Imports, reset work schema, and performance configs
# ============================================================
import time
from pyspark.sql import functions as F, Window
from delta.tables import DeltaTable

SOURCE_SCHEMA = "bronze"
WORK_SCHEMA = "opt_tables"

if "_BenchmarkProxy" not in globals():
    raise RuntimeError("_benchmark_utils did not load. Run the first cell before continuing.")

for key in [
    "spark.sql.adaptive.enabled",
    "spark.sql.shuffle.partitions",
    "spark.databricks.delta.optimizeWrite.enabled",
    "spark.databricks.delta.autoCompact.enabled",
]:
    remember_conf(key)

spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "64")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "false")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "false")

reset_work_schema(WORK_SCHEMA)
print(f"   Source tables remain untouched in schema: {SOURCE_SCHEMA}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# SETUP — Create independent exercise tables from bronze
# ============================================================
TABLES_TO_COPY = [
    "colors", "customer", "inventory_transaction", "manufacturing_event",
    "mold", "part_categories", "parts", "production_line",
    "set_price_history", "sets", "themes", "web_order", "web_order_line"
]

for table in TABLES_TO_COPY:
    target = f"{WORK_SCHEMA}.{table}"
    source = f"{SOURCE_SCHEMA}.{table}"
    spark.sql(f"DROP TABLE IF EXISTS {target}")
    spark.table(source).write.format("delta").mode("overwrite").saveAsTable(target)
    show_metrics(target, "copied")

print("\n✅ Module copies are ready. All exercises mutate opt_tables only.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 1 — OPTIMIZE / compaction (the tiny-files fix)
# 
# **Table:** `manufacturing_event` — high-frequency Toy Brick Manufacturing telemetry.
# 
# **Problem:** Streaming or many small batch writes can leave hundreds or thousands of tiny files. Spark spends more time listing files, reading Parquet footers, and scheduling scan tasks than processing data.
# 
# **Fix:** Run `OPTIMIZE` to compact small files into fewer, larger Delta files.

# CELL ********************

# ============================================================
# 1️⃣ BENCHMARK — Create a tiny-file baseline and time the aggregate scan
# ============================================================
EX1_TABLE = f"{WORK_SCHEMA}.manufacturing_event_tiny"

spark.sql(f"DROP TABLE IF EXISTS {EX1_TABLE}")
(
    spark.table(f"{WORK_SCHEMA}.manufacturing_event")
         .repartition(96)
         .write.format("delta")
         .mode("overwrite")
         .option("delta.targetFileSize", "512k")
         .saveAsTable(EX1_TABLE)
)

metrics_1_before = show_metrics(EX1_TABLE, "before OPTIMIZE")

# NOTE: The terminal collect() stays inside the timed block.
with benchmark_op("Ex1 OPTIMIZE compaction", "before", spark):
    spark.sql(f"""
        SELECT part_num, SUM(cycle_time_ms) AS total_cycle_time
        FROM {EX1_TABLE}
        GROUP BY part_num
    """).collect()

# =================================================================================================
# 1️⃣ DIAGNOSE — DESCRIBE DETAIL proves the baseline has many tiny files
# =================================================================================================

# NOTE: numFiles and sizeInBytes are the physical layout evidence for compaction.
display(spark.sql(f"DESCRIBE DETAIL {EX1_TABLE}").select("format", "numFiles", "sizeInBytes"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Compact `manufacturing_event_tiny` with `OPTIMIZE`, then inspect `DESCRIBE DETAIL` again. A healthy result should have fewer files and a larger average file size.

# CELL ********************

# Challenge starter — run this after you try the OPTIMIZE command above.
print(f"Target table: {EX1_TABLE}")
show_metrics(EX1_TABLE, "current")
# spark.sql(f"OPTIMIZE {EX1_TABLE}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 1️⃣ FIX — Apply OPTIMIZE to compact tiny Delta files
# ==================================================================================================

# NOTE: OPTIMIZE is the table-layer fix; the query code is unchanged.
with benchmark_op("Ex1 OPTIMIZE operation", "OPTIMIZE", spark):
    optimize_result_1 = spark.sql(f"OPTIMIZE {EX1_TABLE}")

display(optimize_result_1)
metrics_1_after = show_metrics(EX1_TABLE, "after OPTIMIZE")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 1️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: Run the same aggregate after compaction to compare query timing and layout.
with benchmark_op("Ex1 OPTIMIZE compaction", "after", spark):
    spark.sql(f"""
        SELECT part_num, SUM(cycle_time_ms) AS total_cycle_time
        FROM {EX1_TABLE}
        GROUP BY part_num
    """).collect()

print(f"Files before: {metrics_1_before['num_files']:,}")
print(f"Files after:  {metrics_1_after['num_files']:,}")
print(f"Average file before: {metrics_1_before['avg_file_kb']:,.1f} KB")
print(f"Average file after:  {metrics_1_after['avg_file_kb']:,.1f} KB")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 2 — Optimize Write
# 
# **Table:** `inventory_transaction` — part movements across production lines.
# 
# **Problem:** Without Optimize Write, a small append often creates one file per output partition. That recreates the tiny-files problem immediately after compaction.
# 
# **Fix:** Enable Optimize Write and Auto Compact on the table so new writes are bin-packed at write time and compacted after write when needed.

# CELL ********************

# ============================================================
# 2️⃣ BENCHMARK — Append a small batch with Optimize Write disabled
# ============================================================

# NOTE: This creates a controlled small-file baseline for the append pattern.
EX2_TABLE = f"{WORK_SCHEMA}.inventory_transaction_ow"
BATCH_ROWS = 5000

spark.sql(f"DROP TABLE IF EXISTS {EX2_TABLE}")
spark.table(f"{WORK_SCHEMA}.inventory_transaction").write.format("delta").mode("overwrite").saveAsTable(EX2_TABLE)
spark.sql(f"OPTIMIZE {EX2_TABLE}")
spark.sql(f"""
    ALTER TABLE {EX2_TABLE} SET TBLPROPERTIES (
      'delta.autoOptimize.optimizeWrite' = 'false',
      'delta.autoOptimize.autoCompact' = 'false'
    )
""")

files_before_bad = get_table_metrics(EX2_TABLE)["num_files"]
bad_batch = spark.table(f"{WORK_SCHEMA}.inventory_transaction").limit(BATCH_ROWS).repartition(8)

with benchmark_op("Ex2 Optimize Write append", "without optimizeWrite", spark):
    bad_batch.write.format("delta").mode("append").saveAsTable(EX2_TABLE)

files_after_bad = get_table_metrics(EX2_TABLE)["num_files"]
new_files_without_ow = files_after_bad - files_before_bad

# =================================================================================================
# 2️⃣ DIAGNOSE — File deltas prove the append created small-file accumulation
# =================================================================================================

# NOTE: A high new_files_without_ow value means each append can undo prior compaction.
print(f"Files before append: {files_before_bad:,}")
print(f"Files after append:  {files_after_bad:,}")
print(f"New files created without Optimize Write: {new_files_without_ow:,}")
show_metrics(EX2_TABLE, "after bad append")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Enable `delta.autoOptimize.optimizeWrite` and `delta.autoOptimize.autoCompact` on `inventory_transaction_ow`, then repeat the same append. The exercise and all measurements use `inventory_transaction` consistently.

# CELL ********************

# Challenge starter — inspect the properties before changing them.
display(spark.sql(f"SHOW TBLPROPERTIES {EX2_TABLE}").filter("key LIKE 'delta.autoOptimize.%'"))
# spark.sql(f"ALTER TABLE {EX2_TABLE} SET TBLPROPERTIES (...)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 2️⃣ FIX — Enable Optimize Write and Auto Compact table properties
# ==================================================================================================

# NOTE: These Delta properties change future write layout without changing readers.
spark.sql(f"OPTIMIZE {EX2_TABLE}")
spark.sql(f"""
    ALTER TABLE {EX2_TABLE} SET TBLPROPERTIES (
      'delta.autoOptimize.optimizeWrite' = 'true',
      'delta.autoOptimize.autoCompact' = 'true'
    )
""")

display(spark.sql(f"SHOW TBLPROPERTIES {EX2_TABLE}").filter("key LIKE 'delta.autoOptimize.%'"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 2️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: Repeat the same append shape so only the table properties changed.
files_before_good = get_table_metrics(EX2_TABLE)["num_files"]
good_batch = spark.table(f"{WORK_SCHEMA}.inventory_transaction").limit(BATCH_ROWS).repartition(8)

with benchmark_op("Ex2 Optimize Write append", "with optimizeWrite", spark):
    good_batch.write.format("delta").mode("append").saveAsTable(EX2_TABLE)

files_after_good = get_table_metrics(EX2_TABLE)["num_files"]
new_files_with_ow = files_after_good - files_before_good

print(f"Files before append: {files_before_good:,}")
print(f"Files after append:  {files_after_good:,}")
print(f"New files without Optimize Write: {new_files_without_ow:,}")
print(f"New files with Optimize Write:    {new_files_with_ow:,}")
show_metrics(EX2_TABLE, "after optimized append")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 3 — Liquid clustering + data-skipping stats
# 
# **Table:** `inventory_transaction`.
# 
# **Problem:** A compacted table can still scan too many files if every file contains a random mix of filter values. Data skipping only works when file-level min/max statistics can prove a file cannot contain the requested value.
# 
# **Fix:** Collect data-skipping stats for frequently filtered columns, cluster by those columns, and verify the clustering / skipping metrics.

# CELL ********************

# ============================================================
# 3️⃣ BENCHMARK — Time a selective scan before clustering
# ============================================================

# NOTE: The table is compacted but values are not colocated for data skipping yet.
EX3_TABLE = f"{WORK_SCHEMA}.inventory_transaction_clustered"

spark.sql(f"DROP TABLE IF EXISTS {EX3_TABLE}")
(
    spark.table(f"{WORK_SCHEMA}.inventory_transaction")
         .repartition(96)
         .write.format("delta")
         .mode("overwrite")
         .option("delta.targetFileSize", "1m")
         .saveAsTable(EX3_TABLE)
)
spark.sql(f"OPTIMIZE {EX3_TABLE}")
metrics_3_before = show_metrics(EX3_TABLE, "before clustering")

sample_color = spark.sql(f"""
    SELECT color_id, COUNT(*) AS cnt
    FROM {EX3_TABLE}
    WHERE color_id IS NOT NULL
    GROUP BY color_id
    ORDER BY cnt DESC
    LIMIT 1
""").collect()[0]["color_id"]

query_3_before = spark.sql(f"""
    SELECT transaction_type, COUNT(*) AS cnt, SUM(quantity) AS total_qty
    FROM {EX3_TABLE}
    WHERE color_id = {sample_color}
    GROUP BY transaction_type
""")

with benchmark_op("Ex3 Liquid clustering", "before", spark):
    query_3_before.collect()

# =================================================================================================
# 3️⃣ DIAGNOSE — Input-file count proves clustering has not yet helped file pruning
# =================================================================================================

# NOTE: The same filter should touch fewer files after clustering and stats collection.
files_3_before = len(query_3_before.inputFiles())
print(f"Files referenced before clustering: {files_3_before:,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Enable data-skipping stats for the filtered columns, apply liquid clustering with `CLUSTER BY (color_id)`, run `OPTIMIZE ... FULL`, and inspect the `clusteringQuality` / `skippingEffectiveness` metrics returned by `OPTIMIZE`.

# CELL ********************

# Challenge starter — confirm stats coverage before clustering.
print(f"Cluster table: {EX3_TABLE}")
print("Suggested stats / clustering column: color_id")
display(spark.sql(f"SHOW TBLPROPERTIES {EX3_TABLE}").filter("key LIKE 'delta.dataSkipping%'") )
# spark.sql(f"ALTER TABLE {EX3_TABLE} SET TBLPROPERTIES ('delta.dataSkippingStatsColumns' = 'color_id,transaction_type,quantity')")
# spark.sql(f"ALTER TABLE {EX3_TABLE} CLUSTER BY (color_id)")
# optimize_metrics_df = spark.sql(f"OPTIMIZE {EX3_TABLE} FULL")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 3️⃣ FIX — Enable data-skipping stats, liquid cluster by color_id, and OPTIMIZE FULL
# ==================================================================================================

# NOTE: Stats define what Delta can skip; clustering colocates similar values at rest.
spark.sql(f"""
    ALTER TABLE {EX3_TABLE} SET TBLPROPERTIES (
      'delta.dataSkippingStatsColumns' = 'color_id,transaction_type,quantity'
    )
""")
spark.sql(f"ALTER TABLE {EX3_TABLE} CLUSTER BY (color_id)")
with benchmark_op("Ex3 Liquid clustering operation", "OPTIMIZE FULL", spark):
    optimize_metrics_df = spark.sql(f"OPTIMIZE {EX3_TABLE} FULL")

display(spark.sql(f"SHOW TBLPROPERTIES {EX3_TABLE}").filter("key LIKE 'delta.dataSkipping%'"))
display(optimize_metrics_df)

try:
    clustering_quality_df = (
        optimize_metrics_df
        .selectExpr("explode(metrics.clusteringQuality) AS clustering_quality")
        .select("clustering_quality.*")
    )
    display(clustering_quality_df)
except Exception as exc:
    print("Clustering quality struct was not available in this runtime output.")
    print(str(exc)[:300])

metrics_3_after = show_metrics(EX3_TABLE, "after clustering")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: Run the identical filter after clustering to compare input files and timing.
query_3_after = spark.sql(f"""
    SELECT transaction_type, COUNT(*) AS cnt, SUM(quantity) AS total_qty
    FROM {EX3_TABLE}
    WHERE color_id = {sample_color}
    GROUP BY transaction_type
""")

with benchmark_op("Ex3 Liquid clustering", "after", spark):
    query_3_after.collect()

files_3_after = len(query_3_after.inputFiles())
print(f"Filter value: color_id = {sample_color}")
print(f"Files referenced before clustering: {files_3_before:,}")
print(f"Files referenced after clustering:  {files_3_after:,}")
print("Open the Spark UI SQL scan node to confirm file pruning and data-skipping statistics.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 4 — Deletion vectors
# 
# **Table:** `web_order_line` — Toy Brick order line items.
# 
# **Problem:** Traditional Delta `DELETE`, `UPDATE`, and `MERGE` operations rewrite the data files that contain changed rows. A tiny delete can rewrite much larger Parquet files.
# 
# **Fix:** Enable deletion vectors so Delta records row-level delete markers instead of rewriting entire files for supported DML operations.

# CELL ********************

# ============================================================
# 4️⃣ BENCHMARK — Prepare matched tables for delete timing with DVs off and on
# ============================================================

# NOTE: Both tables start from the same data and are optimized before DML.
EX4_NO_DV = f"{WORK_SCHEMA}.web_order_line_no_dv"
EX4_WITH_DV = f"{WORK_SCHEMA}.web_order_line_with_dv"

for table in [EX4_NO_DV, EX4_WITH_DV]:
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.table(f"{WORK_SCHEMA}.web_order_line").write.format("delta").mode("overwrite").saveAsTable(table)
    spark.sql(f"OPTIMIZE {table}")

spark.sql(f"ALTER TABLE {EX4_NO_DV} SET TBLPROPERTIES ('delta.enableDeletionVectors' = 'false')")
spark.sql(f"ALTER TABLE {EX4_WITH_DV} SET TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')")

candidate_sets = spark.sql(f"""
    SELECT set_num, COUNT(*) AS cnt
    FROM {EX4_NO_DV}
    WHERE set_num IS NOT NULL
    GROUP BY set_num
    HAVING COUNT(*) > 0
    ORDER BY cnt ASC
    LIMIT 2
""").collect()

set_a, count_a = candidate_sets[0]["set_num"], candidate_sets[0]["cnt"]
set_b, count_b = candidate_sets[1]["set_num"], candidate_sets[1]["cnt"]

# =================================================================================================
# 4️⃣ DIAGNOSE — Table properties and history metrics will prove the physical DML pattern
# =================================================================================================

# NOTE: Compare operationMetrics, especially removed files, not just elapsed time.
show_metrics(EX4_NO_DV, "without DVs")
show_metrics(EX4_WITH_DV, "with DVs")
print(f"Delete target without DVs: set_num={set_a}, rows={count_a:,}")
print(f"Delete target with DVs:    set_num={set_b}, rows={count_b:,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Run a delete against each table, then compare the latest `DESCRIBE HISTORY` `operationMetrics`. Focus on removed/copied files and row counts, not just elapsed time.

# CELL ********************

# Challenge starter — inspect DML metrics after your delete.
# spark.sql(f"DELETE FROM {EX4_WITH_DV} WHERE set_num = '{set_b}'")
# display(spark.sql(f"DESCRIBE HISTORY {EX4_WITH_DV} LIMIT 1").select("operation", "operationMetrics"))
display(spark.sql(f"SHOW TBLPROPERTIES {EX4_WITH_DV}").filter("key LIKE '%DeletionVector%' OR key LIKE '%deletionVector%'"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 4️⃣ FIX — Enable deletion-vector behavior and execute comparable DELETE operations
# ==================================================================================================

# NOTE: The timed DELETE statements are inside the benchmark blocks.
with benchmark_op("Ex4 Deletion vectors", "without DVs", spark):
    spark.sql(f"DELETE FROM {EX4_NO_DV} WHERE set_num = '{set_a}'")

history_no_dv = spark.sql(f"DESCRIBE HISTORY {EX4_NO_DV} LIMIT 1").select("version", "operation", "operationMetrics").collect()[0]
metrics_no_dv = history_no_dv["operationMetrics"]
display(spark.sql(f"DESCRIBE HISTORY {EX4_NO_DV} LIMIT 1").select("version", "operation", "operationMetrics"))
print(f"Without DVs removed files: {metrics_no_dv.get('numRemovedFiles', 'n/a')}")

with benchmark_op("Ex4 Deletion vectors", "with DVs", spark):
    spark.sql(f"DELETE FROM {EX4_WITH_DV} WHERE set_num = '{set_b}'")

history_with_dv = spark.sql(f"DESCRIBE HISTORY {EX4_WITH_DV} LIMIT 1").select("version", "operation", "operationMetrics").collect()[0]
metrics_with_dv = history_with_dv["operationMetrics"]
display(spark.sql(f"DESCRIBE HISTORY {EX4_WITH_DV} LIMIT 1").select("version", "operation", "operationMetrics"))
print(f"With DVs removed files: {metrics_with_dv.get('numRemovedFiles', 'n/a')}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 4️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: Validate logical row removal and compare the resulting table metrics.
remaining_no_dv = spark.table(EX4_NO_DV).filter(F.col("set_num") == set_a).count()
remaining_with_dv = spark.table(EX4_WITH_DV).filter(F.col("set_num") == set_b).count()

print(f"Rows remaining for no-DV delete target: {remaining_no_dv:,}")
print(f"Rows remaining for DV delete target:    {remaining_with_dv:,}")
show_metrics(EX4_NO_DV, "after delete without DVs")
show_metrics(EX4_WITH_DV, "after delete with DVs")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 5 — Data-type optimization
# 
# **Table:** `inventory_transaction` derivative.
# 
# **Problem:** A numeric ID stored as `string` takes more space, can be slower to compare, and may prevent efficient statistics. This often happens when ingestion infers types from CSV/JSON or preserves every field as text.
# 
# **Fix:** Store numeric IDs as numeric columns. Here `color_id` is intentionally written as `string`, then corrected to `int`.

# CELL ********************

# ============================================================
# 5️⃣ BENCHMARK — Time a filtered scan with color_id stored as string
# ============================================================

# NOTE: This intentionally stores a numeric identifier as text to show storage overhead.
EX5_BAD = f"{WORK_SCHEMA}.inventory_transaction_color_string"
EX5_GOOD = f"{WORK_SCHEMA}.inventory_transaction_color_int"

for table in [EX5_BAD, EX5_GOOD]:
    spark.sql(f"DROP TABLE IF EXISTS {table}")

bad_type_df = (
    spark.table(f"{WORK_SCHEMA}.inventory_transaction")
         .withColumn("color_id", F.col("color_id").cast("string"))
)

bad_type_df.repartition(64).write.format("delta").mode("overwrite").saveAsTable(EX5_BAD)
spark.sql(f"OPTIMIZE {EX5_BAD}")
metrics_5_before = show_metrics(EX5_BAD, "color_id as string")

sample_color_text = spark.sql(f"""
    SELECT color_id, COUNT(*) AS cnt
    FROM {EX5_BAD}
    WHERE color_id IS NOT NULL
    GROUP BY color_id
    ORDER BY cnt DESC
    LIMIT 1
""").collect()[0]["color_id"]

with benchmark_op("Ex5 Data type", "string color_id", spark):
    spark.sql(f"""
        SELECT color_id, SUM(quantity) AS total_qty
        FROM {EX5_BAD}
        WHERE color_id = '{sample_color_text}'
        GROUP BY color_id
    """).collect()

# =================================================================================================
# 5️⃣ DIAGNOSE — DESCRIBE DETAIL and schema prove string IDs use unnecessary storage
# =================================================================================================

# NOTE: sizeInBytes is the baseline to compare after right-sizing color_id.
display(spark.sql(f"DESCRIBE DETAIL {EX5_BAD}").select("numFiles", "sizeInBytes"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Create a corrected Delta table with `color_id` cast to `int`, then compare `DESCRIBE DETAIL` size and the same filtered scan. Keep the logical query the same; change the data at rest.

# CELL ********************

# Challenge starter — inspect the current schema and target conversion.
spark.table(EX5_BAD).printSchema()
print(f"Convert color_id from string to int in: {EX5_GOOD}")
# corrected_df = spark.table(EX5_BAD).withColumn("color_id", F.col("color_id").cast("int"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 5️⃣ FIX — Rewrite color_id as an int and optimize the corrected table
# ==================================================================================================

# NOTE: This changes schema at rest while preserving the logical query.
corrected_df = spark.table(EX5_BAD).withColumn("color_id", F.col("color_id").cast("int"))
corrected_df.repartition(64).write.format("delta").mode("overwrite").saveAsTable(EX5_GOOD)
spark.sql(f"OPTIMIZE {EX5_GOOD}")
metrics_5_after = show_metrics(EX5_GOOD, "color_id as int")
spark.table(EX5_GOOD).printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: Use the same filter value, cast to int, to compare type-correct storage.
sample_color_int = int(sample_color_text)

with benchmark_op("Ex5 Data type", "int color_id", spark):
    spark.sql(f"""
        SELECT color_id, SUM(quantity) AS total_qty
        FROM {EX5_GOOD}
        WHERE color_id = {sample_color_int}
        GROUP BY color_id
    """).collect()

print(f"Size with string color_id: {metrics_5_before['size_mb']:.2f} MB")
print(f"Size with int color_id:    {metrics_5_after['size_mb']:.2f} MB")
print("Numeric columns also produce numeric min/max statistics for safer filtering and skipping.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 6 — Partitioning strategy
# 
# **Table:** `manufacturing_event` derivatives.
# 
# **Problem:** Partitioning is powerful only when it matches common filters and keeps each partition large enough. High-cardinality partitions often create a small-files trap; no partitioning may scan more files for time-range queries.
# 
# **Fix:** Compare no partitioning, high-cardinality partitioning, and date partitioning against time-range and point-query workloads.

# CELL ********************

# ============================================================
# 6️⃣ BENCHMARK — Build three partition layouts from the same rows
# ============================================================

# NOTE: The benchmark matrix later uses identical filters across all three layouts.
EX6_NONE = f"{WORK_SCHEMA}.mfg_no_partition"
EX6_HIGH = f"{WORK_SCHEMA}.mfg_partition_part_num"
EX6_DATE = f"{WORK_SCHEMA}.mfg_partition_event_date"

for table in [EX6_NONE, EX6_HIGH, EX6_DATE]:
    spark.sql(f"DROP TABLE IF EXISTS {table}")

partition_df = (
    spark.table(f"{WORK_SCHEMA}.manufacturing_event")
         .withColumn("event_date", F.to_date("timestamp"))
)

partition_df.repartition(64).write.format("delta").mode("overwrite").saveAsTable(EX6_NONE)
partition_df.write.format("delta").mode("overwrite").partitionBy("part_num").saveAsTable(EX6_HIGH)
partition_df.write.format("delta").mode("overwrite").partitionBy("event_date").saveAsTable(EX6_DATE)

# =================================================================================================
# 6️⃣ DIAGNOSE — File counts and partition columns prove which layouts risk over-partitioning
# =================================================================================================

# NOTE: Compare num_files and avg_file_kb before judging any query timing.
metrics_6_none = show_metrics(EX6_NONE, "no partition")
metrics_6_high = show_metrics(EX6_HIGH, "partitionBy part_num")
metrics_6_date = show_metrics(EX6_DATE, "partitionBy event_date")

sample_date = spark.sql(f"""
    SELECT event_date, COUNT(*) AS cnt
    FROM {EX6_NONE}
    WHERE event_date IS NOT NULL
    GROUP BY event_date
    ORDER BY cnt DESC
    LIMIT 1
""").collect()[0]["event_date"]

sample_part = spark.sql(f"""
    SELECT part_num, COUNT(*) AS cnt
    FROM {EX6_NONE}
    WHERE part_num IS NOT NULL
    GROUP BY part_num
    ORDER BY cnt DESC
    LIMIT 1
""").collect()[0]["part_num"]

print(f"Time-range filter date: {sample_date}")
print(f"Point-query part_num:   {sample_part}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Benchmark the same time-range query and point query against all three layouts. Which layout helps, and which layout creates too many files?

# CELL ********************

# Challenge starter — edit TABLE_UNDER_TEST to explore each layout.
TABLE_UNDER_TEST = EX6_DATE
print(f"Testing: {TABLE_UNDER_TEST}")
display(spark.sql(f"DESCRIBE DETAIL {TABLE_UNDER_TEST}").select("numFiles", "partitionColumns", "sizeInBytes"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 6️⃣ FIX — Compare no partition, high-cardinality partition, and date partition layouts
# ==================================================================================================

# NOTE: Partitioning is the table-layer design choice under test.
partition_results = []

for label, table in [
    ("none", EX6_NONE),
    ("high-cardinality part_num", EX6_HIGH),
    ("date", EX6_DATE),
]:
    time_query = spark.sql(f"""
        SELECT machine_id, COUNT(*) AS events
        FROM {table}
        WHERE event_date = DATE '{sample_date}'
        GROUP BY machine_id
    """)
    with benchmark_op("Ex6 Partition time-range", label, spark):
        time_query.collect()

    point_query = spark.sql(f"""
        SELECT color_id, COUNT(*) AS events
        FROM {table}
        WHERE part_num = '{sample_part}'
        GROUP BY color_id
    """)
    with benchmark_op("Ex6 Partition point", label, spark):
        point_query.collect()

    metrics = get_table_metrics(table)
    partition_results.append((label, table, metrics["num_files"], metrics["avg_file_kb"], len(time_query.inputFiles()), len(point_query.inputFiles())))

partition_summary_df = spark.createDataFrame(
    partition_results,
    ["layout", "table", "num_files", "avg_file_kb", "time_query_input_files", "point_query_input_files"]
)
display(partition_summary_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 6️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: The summary ties query input-file counts back to physical partition layout.
print("Partitioning takeaways:")
print("1. Date partitioning should help date filters when each date partition has enough data.")
print("2. High-cardinality partitioning can explode file counts and hurt broad scans.")
print("3. No partitioning is often fine for small/medium tables when clustering and stats handle pruning.")

display(spark.sql(f"DESCRIBE DETAIL {EX6_HIGH}").select("partitionColumns", "numFiles", "sizeInBytes"))
display(spark.sql(f"DESCRIBE DETAIL {EX6_DATE}").select("partitionColumns", "numFiles", "sizeInBytes"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Exercise 7 — Delta storage-regression audit
# 
# **Table:** `inventory_transaction` audit copy.
# 
# **Scenario:** A healthy table regressed after a batch append. Optimize Write and Auto Compact were disabled, then a repartitioned append created many small files.
# 
# **Fix:** Use `DESCRIBE HISTORY` and table properties to identify the regression, restore the properties, run `OPTIMIZE`, and verify file pruning/scan behavior.

# CELL ********************

# ============================================================
# 7️⃣ BENCHMARK — Capture healthy scan timing, then introduce a small-file regression
# ============================================================

# NOTE: Start from a healthy optimized and clustered table before simulating drift.
EX7_TABLE = f"{WORK_SCHEMA}.inventory_transaction_audit"

spark.sql(f"DROP TABLE IF EXISTS {EX7_TABLE}")
spark.table(f"{WORK_SCHEMA}.inventory_transaction").write.format("delta").mode("overwrite").saveAsTable(EX7_TABLE)
spark.sql(f"""
    ALTER TABLE {EX7_TABLE} SET TBLPROPERTIES (
      'delta.autoOptimize.optimizeWrite' = 'true',
      'delta.autoOptimize.autoCompact' = 'true',
      'delta.targetFileSize' = '1m'
    )
""")
spark.sql(f"ALTER TABLE {EX7_TABLE} CLUSTER BY (color_id)")
spark.sql(f"OPTIMIZE {EX7_TABLE} FULL")
metrics_7_healthy = show_metrics(EX7_TABLE, "healthy baseline")

sample_audit_color = spark.sql(f"""
    SELECT color_id, COUNT(*) AS cnt
    FROM {EX7_TABLE}
    WHERE color_id IS NOT NULL
    GROUP BY color_id
    ORDER BY cnt DESC
    LIMIT 1
""").collect()[0]["color_id"]

healthy_query = spark.sql(f"SELECT * FROM {EX7_TABLE} WHERE color_id = {sample_audit_color}")
with benchmark_op("Ex7 Storage regression", "healthy", spark):
    healthy_query.count()
healthy_input_files = len(healthy_query.inputFiles())

spark.sql(f"""
    ALTER TABLE {EX7_TABLE} SET TBLPROPERTIES (
      'delta.autoOptimize.optimizeWrite' = 'false',
      'delta.autoOptimize.autoCompact' = 'false'
    )
""")

regression_batch = spark.table(f"{WORK_SCHEMA}.inventory_transaction").limit(20000).repartition(48)
with benchmark_op("Ex7 Regression append", "bad append", spark):
    regression_batch.write.format("delta").mode("append").saveAsTable(EX7_TABLE)

# =================================================================================================
# 7️⃣ DIAGNOSE — DESCRIBE HISTORY and properties identify the layout regression
# =================================================================================================

# NOTE: The bad append and disabled properties are the incident evidence to audit.
metrics_7_regressed = show_metrics(EX7_TABLE, "after regression")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 🎯 Challenge
# 
# Audit the table. Use `DESCRIBE HISTORY` to find the append that added many files, and `SHOW TBLPROPERTIES` to spot property drift. Then restore the properties and compact the table.

# CELL ********************

# Challenge starter — begin the incident review here.
display(spark.sql(f"""
    DESCRIBE HISTORY {EX7_TABLE}
""").select("version", "timestamp", "operation", "operationParameters", "operationMetrics").limit(10))

display(spark.sql(f"SHOW TBLPROPERTIES {EX7_TABLE}").filter("key LIKE 'delta.autoOptimize.%' OR key = 'delta.targetFileSize'"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==================================================================================================
# 7️⃣ FIX — Restore optimization properties and run OPTIMIZE FULL
# ==================================================================================================

# NOTE: History isolates the offending operation; properties and OPTIMIZE remediate it.
history_df = spark.sql(f"DESCRIBE HISTORY {EX7_TABLE}")

audit_df = (
    history_df
    .select(
        "version", "timestamp", "operation", "operationParameters",
        F.col("operationMetrics").getItem("numOutputRows").cast("long").alias("rows_written"),
        F.coalesce(
            F.col("operationMetrics").getItem("numFiles").cast("long"),
            F.col("operationMetrics").getItem("numAddedFiles").cast("long")
        ).alias("files_added")
    )
    .orderBy(F.col("version").desc())
)
display(audit_df)

print("Current optimization properties before remediation:")
display(spark.sql(f"SHOW TBLPROPERTIES {EX7_TABLE}").filter("key LIKE 'delta.autoOptimize.%' OR key = 'delta.targetFileSize'"))

spark.sql(f"""
    ALTER TABLE {EX7_TABLE} SET TBLPROPERTIES (
      'delta.autoOptimize.optimizeWrite' = 'true',
      'delta.autoOptimize.autoCompact' = 'true',
      'delta.targetFileSize' = '1m'
    )
""")

with benchmark_op("Ex7 Remediation", "OPTIMIZE FULL", spark):
    remediation_metrics_df = spark.sql(f"OPTIMIZE {EX7_TABLE} FULL")

display(remediation_metrics_df)
metrics_7_remediated = show_metrics(EX7_TABLE, "after remediation")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 7️⃣ CHECK-CHANGES — Compare against baseline
# ============================================================

# NOTE: Re-run the same color_id filter to verify pruning and file layout recovered.
remediated_query = spark.sql(f"SELECT * FROM {EX7_TABLE} WHERE color_id = {sample_audit_color}")
with benchmark_op("Ex7 Storage regression", "remediated", spark):
    remediated_query.count()
remediated_input_files = len(remediated_query.inputFiles())

print(f"Healthy input files for color_id={sample_audit_color}: {healthy_input_files:,}")
print(f"Remediated input files for same filter:      {remediated_input_files:,}")
print(f"Files after regression:  {metrics_7_regressed['num_files']:,}")
print(f"Files after remediation: {metrics_7_remediated['num_files']:,}")

print("Optimization properties after remediation:")
display(spark.sql(f"SHOW TBLPROPERTIES {EX7_TABLE}").filter("key LIKE 'delta.autoOptimize.%' OR key = 'delta.targetFileSize'"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# 
# # Summary — Optimizing Delta Tables
# 
# You fixed performance by changing the data at rest:
# 
# 1. **OPTIMIZE / compaction** reduced tiny-file overhead.
# 2. **Optimize Write** prevented new small files during appends.
# 3. **Liquid clustering + data-skipping stats** colocated filtered values so Delta could prune files.
# 4. **Deletion vectors** reduced write amplification for row-level DML.
# 5. **Data-type optimization** stored numeric IDs as numeric values for smaller, cleaner scans.
# 6. **Partitioning strategy** showed that partitioning helps only when cardinality and filters align.
# 7. **Storage-regression audit** used Delta history and properties to find drift, restore settings, and compact again.
# 
# The litmus test: every fix changed table layout, schema, partitioning, or Delta properties — not the query logic.

# CELL ********************

# Optional tear-down for session-level configs changed by this notebook
for key in [
    "spark.sql.adaptive.enabled",
    "spark.sql.shuffle.partitions",
    "spark.databricks.delta.optimizeWrite.enabled",
    "spark.databricks.delta.autoCompact.enabled",
]:
    restore_conf(key)

print("✅ Session-level Spark configs restored.")
print(f"Notebook tables remain in {WORK_SCHEMA} for inspection and re-runs.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
