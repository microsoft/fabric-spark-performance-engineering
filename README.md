<img src="https://github.com/microsoft/fabric-analytics-roadshow-lab/blob/main/assets/images/spark/analytics.png?raw=true"
     width="80"
     align="left"
     style="margin-right:0px; padding-top:20px;" />

<h1 style="border-bottom: none; padding-bottom: 0; margin-bottom: 0;">
  Fabric Spark Performance Engineering Tutorial
</h1>

## Overview

This **Fabric Jumpstart** is a self-guided, hands-on tutorial that teaches you how to find and fix Apache Spark performance problems on Microsoft Fabric. Instead of abstract theory, you work through real, sub-optimal workloads on a synthetic **Toy Brick Manufacturing & Sales** lakehouse — seeded for you during setup — and measure the impact of each fix with before-and-after benchmarks.

The lab is organized around the **tuning hierarchy** — *code → data → execution* — so you learn to identify which layer a problem lives in and reach for the right lever:

| Module | Theme | Fix lever | Examples |
|--------|-------|-----------|----------|
| **1 — Optimizing Code** | The query is written inefficiently | Edit the transformation logic | Predicate pushdown, column pruning, avoiding repeated scans, replacing Python UDFs, cartesian joins, driver OOM |
| **2 — Delta Table Design & Optimization** | The table layout is suboptimal | `OPTIMIZE`, clustering, deletion vectors, stats, schema | Small-file compaction, optimize-write, Liquid Clustering, data-skipping stats |
| **3 — Optimizing Execution** | Code and tables are fine, execution isn't | Join hints, AQE, partition sizing, caching | Broadcast joins, skew handling, shuffle partition sizing, materialization, Native Execution Engine |

### How each exercise works

Every exercise follows the same repeatable loop, so you build a durable performance-tuning workflow rather than memorizing one-off fixes:

| Step | What you do |
|------|-------------|
| 🐌 **Benchmark** | Run the workload and capture a baseline time/metric |
| 🔍 **Diagnose** | Inspect the plan, Spark UI, or Delta metadata to prove the root cause |
| 🔧 **Fix** | Apply the change (as a hands-on challenge, with an inline solution) |
| 🚀 **Re-benchmark** | Re-run and compare against the baseline |
| 💡 **What Just Happened?** | Read a short explanation of *why* the fix worked |

### What's included

- **`00_getting-started`** — verifies prerequisites, seeds the bronze Delta tables, and orients you to the data model.
- **Three module notebooks** (`01_optimizing-code`, `02_optimizing-tables`, `03_optimizing-execution`) — each self-contained with challenge → solution flow, roughly **45 minutes** each.
- **Two learning tracks for all three modules** — the same concepts, data, and benchmarks are offered as a **DataFrame API** track (`dataframe-lab/`) and a **Spark SQL** track (`sql-lab/`, using SparkSQL expressed within `spark.sql(...)`). Pick whichever API you use day-to-day; the exercise numbering is identical so you can switch between them. Spark SQL notebooks are suffixed `_sql`.
- Shared benchmarking utilities and the sample lakehouse, environments, and Spark Job Definitions that build the data.

## Install the Lab in Fabric (Notebook, 2 cells)
**Step 1: Create a Fabric Notebook**

In your Fabric workspace, create a new Notebook (PySpark or Python runtime).

**Step 2: Run installer**

Copy, paste, and the run the below in a Notebook cell.

```python
%pip install fabric-jumpstart --quiet
```

```python
import fabric_jumpstart as jumpstart
jumpstart.install("spark-performance-engineering")
```

> [!NOTE]
>
> It will take approximately 8 minutes to install all lab content into your workspace. You will see a success message once done.
>
> After installation the first step is to run the `source_to_bronze` Spark Job Definition that generates Delta tables with hundreds of transactions to simulate real-world data challenges. This runs for 30 minutes. Make sure to allow this to complete before starting the interactive modules.