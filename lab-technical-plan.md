# Technical Lab Plan — Fabric Jumpstart Packaging

## Overview

This document is the technical blueprint for packaging the **Advanced Spark Performance Engineering Workshop** as a [Fabric Jumpstart](https://github.com/microsoft/fabric-jumpstart). The Jumpstart will be self-contained with zero external dependencies — attendees run `jumpstart.install("spark-performance-workshop")` and get everything: notebooks, lakehouse, environment, and data generation.

---

## Jumpstart Identity

| Field | Value |
|-------|-------|
| `logical_id` | `spark-performance-workshop` |
| `name` | Spark Performance Workshop |
| `type` | Tutorial |
| `difficulty` | Advanced |
| `workload_tags` | Data Engineering |
| `scenario_tags` | Performance, Optimization |
| `entry_point` | `workshop_start.Notebook` |
| `minutes_to_deploy` | 5 |
| `minutes_to_complete_jumpstart` | 480 (full day) |

---

## Source Repository Structure

The Jumpstart source repo follows Fabric Jumpstart conventions. All deployable items live under a top-level folder matching the `logical_id`. Item names use `snake_case` (no spaces).

```
spark-performance-workshop/
├── workspace/
│   ├── parameter.yml                          # Jumpstart parameters
│   └── spark-performance-workshop/            # Top-level folder = logical_id
│       ├── lego_lakehouse.Lakehouse           # Target lakehouse for all data
│       ├── workshop_env.Environment           # Spark environment (LegoGen + deps)
│       │
│       │── workshop_start.Notebook            # Entry point — setup + instructions
│       │── generate_data.Notebook             # LegoGen data generator
│       │── generate_perf_scenarios.Notebook    # Post-gen: creates perf problem scenarios
│       │
│       │── lab_1_diagnostics.Notebook          # Lab 1: diagnose 6 broken queries
│       │── lab_2_tuning.Notebook               # Lab 2: table optimization exercises
│       │── lab_3_optimization.Notebook         # Lab 3: join/cache/shuffle/streaming
│       │── lab_4_debugging.Notebook            # Lab 4: capstone incident queue
│       │
│       │── lab_1_solutions.Notebook            # Solutions (instructor can share after)
│       │── lab_2_solutions.Notebook
│       │── lab_3_solutions.Notebook
│       │── lab_4_solutions.Notebook
│       │
│       └── utils.Notebook                     # Shared helpers (setJobDescription, timing, etc.)
├── README.md
└── .gitignore
```

### `parameter.yml`

```yaml
find_replace:
  - find: "{{LAKEHOUSE_NAME}}"
    replace: "lego_lakehouse"
  - find: "{{LAKEHOUSE_ID}}"
    replace: ""  # Resolved at deploy time by fabric-cicd
```

---

## Jumpstart YAML Definition

```yaml
id: <next_available_id>
logical_id: spark-performance-workshop
name: Spark Performance Workshop
description: >-
  An intensive hands-on workshop for mastering Spark performance engineering.
  Covers execution architecture, diagnostics, tuning, optimization patterns,
  and advanced debugging using LEGO manufacturing and sales data.
date_added: <publish_date>
workload_tags:
  - Data Engineering
scenario_tags:
  - Performance
  - Optimization
type: Tutorial
source:
  repo_url: https://github.com/mwc360/fabric-spark-performance-workshop.git
  repo_ref: v1.0.0
  workspace_path: workspace/
items_in_scope:
  - Lakehouse
  - Notebook
  - Environment
entry_point: workshop_start.Notebook
test_suite: ""
owner_email: <owner_email>
minutes_to_deploy: 5
minutes_to_complete_jumpstart: 480
video_url: ""
difficulty: Advanced
last_updated: "<date>"
mermaid_diagram: |
  graph LR
    START[workshop_start]:::Notebook --> GEN[generate_data]:::Notebook
    GEN --> SCEN[generate_perf_scenarios]:::Notebook
    SCEN --> L1[lab_1_diagnostics]:::Notebook
    L1 --> L2[lab_2_tuning]:::Notebook
    L2 --> L3[lab_3_optimization]:::Notebook
    L3 --> L4[lab_4_debugging]:::Notebook
    LH[lego_lakehouse]:::Lakehouse <-.-> GEN
    LH <-.-> L1
    LH <-.-> L2
    LH <-.-> L3
    LH <-.-> L4
    ENV[workshop_env]:::Environment -.-> GEN
```

---

## Data Strategy

### Zero-Dependency Data Generation

Per Jumpstart standards, data must be self-contained. We use LegoGen (from the `lakegen` PyPI package) to generate all data at install time — no external data sources, no pre-bundled large files.

### `generate_data.Notebook`

This is the first notebook the user runs (linked from `workshop_start`). It generates the base LEGO dataset into `lego_lakehouse`.

```python
# Cell 1 — Install lakegen
%pip install lakegen --quiet

# Cell 2 — Generate base dataset
from lakegen.generators.lego import LegoDataGen

gen = LegoDataGen(
    target_folder_uri=f"abfss://{workspace_id}@onelake.dfs.fabric.microsoft.com/{lakehouse_id}/Files/lego_raw",
    output_type_map={
        "manufacturing_event": "parquet",
        "web_order": "parquet",
        "web_order_line": "parquet",
        "customer": "parquet",
        "production_order": "parquet",
        "inventory_transaction": "parquet",
        "quality_inspection": "parquet",
        "product_return": "parquet",
        "set_price_history": "parquet",
        "colors": "parquet",
        "parts": "parquet",
        "sets": "parquet",
        "themes": "parquet",
        "part_categories": "parquet",
        "inventories": "parquet",
        "inventory_parts": "parquet",
        "inventory_sets": "parquet",
        "production_line": "parquet",
        "mold": "parquet",
    },
    concurrenct_threads=4,
    max_events_per_second=5000,
)
gen.run(duration_seconds=600)  # ~10 min, generates 5-10M mfg events
```

### `generate_perf_scenarios.Notebook`

Runs after data generation. Creates the specific performance problem scenarios needed by the labs. This is where we **intentionally degrade** the data to set up triage exercises.

```python
# ============================================================
# SCENARIO 1: Small files problem
# Re-ingest manufacturing_event as thousands of tiny files
# Used by: Lab 1 Q6, Lab 2A, Lab 4 INC-2
# ============================================================
from lakegen.generators.lego import LegoDataGen

gen_small = LegoDataGen(
    target_folder_uri=f".../{lakehouse_id}/Files/lego_small_files",
    output_type_map={
        "manufacturing_event": "json",
        "web_order": "json",
    },
    concurrenct_threads=8,
    buffer_write_by_seconds=0,  # Flush every iteration
    max_events_per_second=2000,
)
gen_small.run(duration_seconds=120)

# Load as unmanaged Delta tables with no optimization
spark.read.json(f".../{lakehouse_id}/Files/lego_small_files/manufacturing_event") \
    .write.format("delta") \
    .mode("overwrite") \
    .save(f".../{lakehouse_id}/Tables/manufacturing_event_messy")
```

```python
# ============================================================
# SCENARIO 2: Skewed customer data
# Duplicate orders for 5 "bulk buyer" customers
# Used by: Lab 3B
# ============================================================
from pyspark.sql import functions as F

orders = spark.read.format("delta").load(f".../Tables/web_order")
order_lines = spark.read.format("delta").load(f".../Tables/web_order_line")

# Pick 5 random customers
top_customers = orders.groupBy("CustomerId").count() \
    .orderBy(F.desc("count")).limit(5) \
    .select("CustomerId").collect()
top_ids = [r.CustomerId for r in top_customers]

# Generate 50x duplicate orders for these customers
skewed_extra = orders.filter(F.col("CustomerId").isin(top_ids))
for _ in range(50):
    skewed_extra = skewed_extra.withColumn("OrderId", F.expr("uuid()"))
    orders = orders.union(skewed_extra.limit(1000))

orders.write.format("delta").mode("overwrite") \
    .save(f".../Tables/web_order_skewed")
```

```python
# ============================================================
# SCENARIO 3: Stale statistics / large table for broadcast OOM
# Write production_order as a large table (500K+ rows)
# Used by: Lab 2D, Lab 4 INC-1
# ============================================================
prod_orders = spark.read.format("delta").load(f".../Tables/production_order")

# Explode to 500K+ rows by duplicating with new IDs
large_po = prod_orders
for _ in range(3):
    large_po = large_po.union(
        prod_orders.withColumn("ProductionOrderId", F.expr("uuid()"))
    )

large_po.write.format("delta").mode("overwrite") \
    .save(f".../Tables/production_order_large")

# Do NOT run ANALYZE TABLE — stale stats are the point
```

```python
# ============================================================
# SCENARIO 4: Non-optimized table (no clustering, default everything)
# Copy web_order with no optimization for Lab 2 exercises
# Used by: Lab 2B, Lab 2E
# ============================================================
spark.read.format("delta").load(f".../Tables/web_order") \
    .write.format("delta") \
    .mode("overwrite") \
    .option("delta.dataSkippingNumIndexedCols", 0) \
    .save(f".../Tables/web_order_unoptimized")
```

---

## Notebook Design — Lab Details

All lab notebooks follow the same structure:
1. **Context cell** (markdown): scenario story, what's broken, what to investigate
2. **Setup cell**: loads data, sets job descriptions, starts timers
3. **Problem cell(s)**: the broken/slow code — attendees run this and observe Spark UI
4. **Investigation cells** (empty or scaffolded): space for attendees to use `EXPLAIN`, `getNumPartitions()`, Spark UI, etc.
5. **Fix cell** (empty): attendees write their optimized version
6. **Validation cell**: compares before/after metrics (duration, shuffle bytes, spill, task count)

### Lab 1: Diagnostics — "The Factory Dashboard is Slow"

Six intentionally bad queries. Attendees diagnose via Spark UI only (no fixes yet).

| # | Query Description | Performance Anti-Pattern | Key Spark UI Signal |
|---|---|---|---|
| Q1 | Daily defect rate by machine | Full table scan — no partition pruning, string timestamps prevent pushdown | SQL tab: scan rows ≫ output rows |
| Q2 | Top 10 customers by spend | Python UDF for aggregation instead of built-in `sum()` | NEE fallback; ArrowBatch serialization overhead; long task durations |
| Q3 | Inventory levels by plant | `.collect()` on 2M rows → Python processing → new DataFrame | Single-executor GC pressure; driver memory spike; Stage 1 has 1 task |
| Q4 | Mfg events per shift per machine | Loop: one query per machine per day, no caching | Jobs tab: 600 separate jobs; repeated identical scans |
| Q5 | QC pass rates | Cross-join (missing join key between `quality_inspection` and `production_order`) | Shuffle write in GB; spill to disk; possible OOM |
| Q6 | Monthly revenue trend | Correct logic but reads from `manufacturing_event_messy` (10K tiny files) | 10,000 tasks in scan stage; each < 1 MB; scheduling overhead dominates |

### Lab 2: Configuration & Tuning — "Optimize the LEGO Warehouse"

Hands-on table optimization. Each exercise has a measurable before/after metric.

| # | Exercise | Table | Task | Metric |
|---|---|---|---|---|
| 2A | Fix small files | `manufacturing_event_messy` | Run `OPTIMIZE`; compare file count and query time | File count: 10K → ~50; query 5-10× faster |
| 2B | Add clustering | `web_order_unoptimized` | Apply liquid clustering on `(OrderDate, CustomerId)`; re-run date + customer filter | Files skipped shown in scan metrics |
| 2C | Optimize data types | `inventory_parts` | Cast string columns to int; measure storage and scan | Smaller files; faster scans |
| 2D | Refresh statistics | `production_order_large` | `EXPLAIN COST` before and after `ANALYZE TABLE`; observe plan change | Optimizer switches from sort-merge to broadcast join |
| 2E | Partition strategy | `manufacturing_event` | Test partitioning by `MachineId` vs `date(Timestamp)` vs none | Measure query times for time-range vs machine-specific queries |

### Lab 3: Optimization Patterns — "Supply Chain Performance Challenge"

Each exercise targets a specific Spark optimization technique.

| # | Scenario | Tables | Optimization |
|---|---|---|---|
| 3A | Billion-row join | `manufacturing_event` + `production_order` + `parts` | Broadcast hints for small reference tables; verify broadcast hash join in plan |
| 3B | Skewed customer aggregation | `web_order_skewed` | Identify straggler tasks; enable AQE skew optimization; salt join keys as alternative |
| 3C | Repeated reads | `inventory_transaction` | Cache after first read; measure I/O reduction across multiple filtered queries |
| 3D | Shuffle storm | `inventory_parts` → `inventories` → `sets` → `themes` | Reorder joins smallest-first; pre-aggregate; tune `shuffle.partitions` |
| 3E | Streaming pipeline | `manufacturing_event` (streaming) | Configure triggers, watermarks, state store; monitor streaming UI |

**Bonus:** Return to Lab 1 and fix all 6 broken queries using patterns learned in Modules 2-3.

### Lab 4: Advanced Debugging — "The Factory Outage" (Capstone)

Incident triage simulation. Attendees work in pairs through an escalating queue.
**Structure:** 3 mandatory incidents (~12 min each) + 2 optional stretch incidents for fast finishers.

Each incident uses an **on-call ticket** format: symptom, SLA impact, cells to inspect,
required evidence, optional hint, and expected remediation.

**Mandatory Incidents:**

| Priority | Incident | Symptoms | Root Cause | Diagnosis | Fix |
|---|---|---|---|---|---|
| 🔴 | OOM Crash — Daily Quality Report | Executor OOM joining `quality_inspection` with `production_order` | Explicit `broadcast(production_order)` but table grew to 500K rows | **Simulated** — pre-captured Spark UI + safe scaled-down validation. Read `explain()`, check `DESCRIBE DETAIL` for table size. | Remove `broadcast()` hint; let AQE choose sort-merge join |
| 🟠 | 10× Slowdown — Customer 360 Pipeline | Multi-table join takes 45 min instead of 5 min | Two compounding issues: small files on `web_order` + stale stats causing wrong join strategy for `product_return` | `DESCRIBE DETAIL` (file count), `DESCRIBE HISTORY` (bad write), `EXPLAIN COST` (join strategy), `inputFiles()` (scan count). Must find BOTH causes. | `OPTIMIZE web_order`; `ANALYZE TABLE product_return COMPUTE STATISTICS` |
| 🟡 | Mysterious Spill — Inventory Reconciliation | Window function spills 200 GB to disk | 5 popular parts have millions of txns → huge skewed partitions in window function | Task duration variance in Spark UI, `groupBy(spark_partition_id()).count()` to quantify skew | Repartition with salting for hot keys; or incremental aggregation |

**Optional Stretch Incidents:**

| Priority | Incident | Symptoms | Root Cause | Diagnosis | Fix |
|---|---|---|---|---|---|
| 🟡 | Partition Explosion — Parts Demand Forecast | Tasks take 2 sec to schedule, 0.1 sec to run | `repartition(PartNum, ColorId)` creates 50K+ partitions on high-cardinality composite key | `df.rdd.getNumPartitions()`, scheduler delay vs execution time in Spark UI | Use `coalesce()` or `repartition(200)` |
| 🟢 | Delta Storage Regression — Overnight Perf Drop | Query 3× slower after yesterday's deployment | Bad batch append disabled auto-compaction and clustering, creating uncompacted small files | `DESCRIBE HISTORY`, `DESCRIBE DETAIL`, `inputFiles()`, check table properties | Restore table properties, run `OPTIMIZE`, verify data skipping |

---

## Environment Configuration

### `workshop_env.Environment`

The Spark environment must include:
- `lakegen` (PyPI) — for data generation
- Spark config defaults that make sense for a workshop (not production):
  ```
  spark.sql.adaptive.enabled = true
  spark.sql.adaptive.coalescePartitions.enabled = true
  ```

The environment should NOT pre-configure optimizations that attendees are meant to discover (e.g., don't set `spark.sql.autoBroadcastJoinThreshold` to a high value — let them figure that out).

---

## `workshop_start.Notebook` — Entry Point

This is the Jumpstart entry point. It must be self-documenting per Jumpstart standards.

### Content Outline

```markdown
# 🧱 Advanced Spark Performance Engineering Workshop

## Welcome!
This Jumpstart deploys a full-day hands-on workshop for mastering Spark
performance engineering in Microsoft Fabric. You'll diagnose, tune, and
optimize Spark workloads using realistic LEGO manufacturing and sales data.

## What's Included
- **4 Lab Notebooks** covering diagnostics, tuning, optimization, and debugging
- **Solution Notebooks** for each lab
- **Data Generator** using LegoGen (LEGO manufacturing & sales simulator)
- **Pre-built Performance Scenarios** (small files, skewed data, bad joins, etc.)

## Getting Started

### Step 1: Generate Data
Open and run the `generate_data` notebook. This takes ~10 minutes and
populates the `lego_lakehouse` with LEGO manufacturing and sales data.

[Open generate_data →](link_to_notebook)

### Step 2: Create Performance Scenarios
Open and run `generate_perf_scenarios`. This creates the intentionally
degraded tables used in the lab exercises.

[Open generate_perf_scenarios →](link_to_notebook)

### Step 3: Work Through the Labs
Complete the labs in order:
1. [Lab 1: Diagnostics](link) — Diagnose 6 broken queries using Spark UI
2. [Lab 2: Tuning](link) — Optimize table physical design
3. [Lab 3: Optimization](link) — Master join strategies, caching, and AQE
4. [Lab 4: Debugging Capstone](link) — Triage a simulated production outage

## Data Model
The workshop uses a LEGO manufacturing and sales data model with 20 tables
spanning web orders, production, inventory, and quality inspection. Key tables:

| Table | ~Rows | Description |
|-------|-------|-------------|
| manufacturing_event | 5-10M | Injection molding IoT telemetry |
| web_order / web_order_line | 500K / 2M | E-commerce orders |
| inventory_transaction | 1-2M | Event-sourced inventory ledger |
| production_order | 100K-200K | Manufacturing work orders |
| inventory_parts | 1.5M | Real LEGO part-color-quantity data |
| sets / parts / colors | 27K / 62K / 275 | Real Rebrickable catalog data |
```

---

## `utils.Notebook` — Shared Helpers

Reusable utilities imported by lab notebooks:

```python
# Timing decorator for before/after comparisons
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    """Context manager that prints elapsed time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"⏱ {label}: {elapsed:.2f}s")

# Job description helper
def set_job_desc(sc, description: str):
    """Sets the Spark job description visible in Spark UI."""
    sc.setJobDescription(description)

# Partition inspector
def inspect_partitions(df, label: str = ""):
    """Prints partition count and approximate row distribution."""
    num_parts = df.rdd.getNumPartitions()
    print(f"📊 {label} — {num_parts} partitions")
    return num_parts

# File count for Delta tables
def count_files(spark, table_path: str):
    """Returns the number of data files in a Delta table."""
    detail = spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]
    return detail.numFiles

# Before/after comparison
def compare_metrics(label, before, after, unit="s"):
    """Prints a formatted before/after comparison."""
    improvement = ((before - after) / before) * 100 if before > 0 else 0
    print(f"📈 {label}: {before:.2f}{unit} → {after:.2f}{unit} ({improvement:.1f}% improvement)")
```

---

## Deployment Flow

```
jumpstart.install("spark-performance-workshop")
        │
        ▼
   ┌─────────────────────────┐
   │  fabric-cicd deploys:   │
   │  • lego_lakehouse       │
   │  • workshop_env         │
   │  • 10 notebooks         │
   └─────────────────────────┘
        │
        ▼
   User opens workshop_start.Notebook
        │
        ▼
   User runs generate_data.Notebook          (~10 min)
        │     └─ LegoGen creates 20 tables in lego_lakehouse
        ▼
   User runs generate_perf_scenarios.Notebook (~5 min)
        │     └─ Creates degraded scenario tables:
        │        manufacturing_event_messy (small files)
        │        web_order_skewed (data skew)
        │        web_order_unoptimized (no clustering)
        │        production_order_large (stale stats)
        ▼
   Labs 1-4 (self-paced or instructor-led)
```

---

## Workshop Delivery Modes

### Mode A: Instructor-Led (Full Day)

- Instructor presents theory (Modules 0-4 lecture portions)
- Attendees work through labs in sync with the group
- Instructor shares solution notebooks after each lab
- Use the agenda from `workshop-plan.md`

### Mode B: Self-Paced (Jumpstart Only)

- Attendee installs the Jumpstart and follows `workshop_start`
- Each lab notebook includes enough markdown context to be self-guided
- Solution notebooks are available immediately (honor system)
- Estimated completion: 4-6 hours without lecture content

### Mode C: Subset / Modular

- Individual labs can be run standalone
- `generate_data` and `generate_perf_scenarios` are prerequisites for all labs
- Labs 1-4 are sequential in narrative but technically independent

---

## Capacity Planning (50 Attendees)

| Consideration | Recommendation |
|---|---|
| Fabric capacity | F64 or higher; 50 concurrent Spark sessions need substantial CU |
| Workspace strategy | Shared read-only lakehouse + individual workspaces for experiments |
| Compute | Medium Spark pools (8 vCores, 56 GB per executor) |
| Concurrent sessions | Stagger data generation; labs read existing data (lower CU) |
| Fallback | If capacity constrained, pair attendees (25 sessions instead of 50) |

---

## Testing Checklist

Before publishing the Jumpstart:

- [ ] `jumpstart.install("spark-performance-workshop")` deploys cleanly to an empty workspace
- [ ] `generate_data` completes in ≤ 15 minutes on Medium capacity
- [ ] `generate_perf_scenarios` completes in ≤ 10 minutes
- [ ] Each lab notebook runs without import errors or missing table references
- [ ] Lab 1 queries are observably slow (Spark UI shows clear anti-patterns)
- [ ] Lab 2 exercises show measurable improvement after optimization
- [ ] Lab 3 optimizations produce verifiable Spark UI improvements
- [ ] Lab 4 incidents are reproducible (OOM, spill, streaming lag)
- [ ] Solution notebooks produce correct results
- [ ] All notebooks work on both Fabric Spark 3.5 (Runtime 1.3) and Spark 4.0
- [ ] No hardcoded workspace IDs or lakehouse IDs (use parameters)
- [ ] Notebook markdown is self-documenting (no external docs needed)
