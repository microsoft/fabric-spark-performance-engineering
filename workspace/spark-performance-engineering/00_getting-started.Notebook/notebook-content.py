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

# # 🚀 **Getting Started — Fabric Spark Performance Workshop**
# 
# Welcome! This notebook is your launchpad. Run through it once before Module 1 to confirm your
# environment is ready, seed the bronze Delta tables, and get oriented to the materials.
# 
# **Time required:** ~10–15 minutes (most of it is waiting for the seeding job to finish)
# 
# ---
# 
# ## What you'll do here
# 
# 1. ✅ Verify prerequisites (workspace, attached lakehouse, Fabric Environment)
# 2. 🧱 Trigger the `source_to_bronze` Spark Job Definition to build the bronze layer
# 3. 🔍 Validate the bronze tables landed correctly
# 4. 🗺️ Get oriented to the data model and the three workshop modules


# MARKDOWN ********************

# ---
# 
# ## 🗺️ The Toy Brick Lakehouse — Data Model
# 
# Throughout the workshop we use a synthetic **Toy Brick Manufacturing & Sales** lakehouse. The
# schema has fact tables (manufacturing events, orders, inventory transactions) at the center and
# dimensions (parts, colors, themes, sets) on the periphery.
# 
# > 💡 The biggest fact table is `manufacturing_event` (~8 M rows / ~350 MB). Most performance
# > exercises target it because it's the only table large enough to make Spark break a sweat at
# > lab scale on a single node.


# MARKDOWN ********************

# ---
# 
# ## 1️⃣ Seed the Bronze Layer — Run `source_to_bronze`
# 
# The workshop's source data does not live in the lakehouse yet. The first thing you'll do is run
# a **Spark Job Definition** that:
# 
# 1. Generates synthetic toy-brick landing data (mixed JSON + Parquet) into `Files/landing/…`
# 2. Ingests every table into bronze Delta tables under the `bronze` schema
# 
# ### 🎯 How to trigger it
# 
# 1. Go to the [source_to_bronze](https://app.powerbi.com/groups/$workspaceId/sparkjobdefinitions/$sparkJobDefinitionId?experience=fabric-developer) Spark Job Definition
# 1. Click **Run** (top header ribbon). _Note: only click **Run** once. It takes ~5 seconds to show up as active in the run history._
# 
# > 🔁 **Re-run anytime.** The job is incremental — run it multiple times to generate larger lab data. By default it runs for 30 minutes and then stops.
# 
# Move on to Module 1 once the Spark Job Definition run shows as **Succeeded**. It will run for approximately 30 minutes to generate tables which enough commits that can be used to illustrate real-world performance challenges.


# MARKDOWN ********************

# ---
# 
# ## 2️⃣ Workshop Modules — Where to Go Next
# 
# There are **three** hands-on modules, organized by the *layer you change to fix a problem* —
# the tuning hierarchy: **code → data → execution**. The diagnostic toolkit is introduced in
# Module 1 and reused throughout.
# 
# | Module | Notebook | Fix lever | What you'll learn |
# |--------|----------|-----------|--------------------|
# | **1 — Optimizing Code** | `01_optimizing-code` | Rewrite the query | Reading the Spark UI / query plans / Delta metadata, then fixing code-level anti-patterns: predicate pushdown, Python UDFs vs native / NEE, driver `collect()`/`toPandas()` & OOM, cartesian / missing join keys |
# | **2 — Optimizing Tables** | `02_optimizing-tables` | Change the data at rest | OPTIMIZE / compaction, Optimize Write, liquid clustering & data-skipping stats, deletion vectors, data types, partitioning strategy, storage-regression auditing (`DESCRIBE HISTORY`) |
# | **3 — Optimizing Execution** | `03_optimizing-execution` | Tune how Spark runs it | Join strategies & broadcast, AQE & skew / salting, shuffle-partition sizing & spill, caching / materialization, streaming |
# 
# **How Modules 1 and 3 differ:** Module 1 is when the *query is written badly* (fix = edit the
# code). Module 3 is when the code and tables are fine but Spark *executes* it sub-optimally
# (fix = a hint / config / `.cache()` / repartition, with the logic unchanged). Module 2 is when
# the *table layout* is the problem (fix = `OPTIMIZE` / clustering / schema / partitioning).
# 
# ### Helper / shared notebook
# 
# | Notebook | What it provides |
# |----------|------------------|
# | `_benchmark_utils` | `df.benchmark(scenario, state)` / `benchmark_op(scenario, state, spark)` timers and `get_table_metrics()` / `show_metrics()` helpers used across modules. Don't run it standalone — each module `%run`s it. |

