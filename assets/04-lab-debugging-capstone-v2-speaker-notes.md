# Module 4 — Speaker Notes

**Lab:** `04-lab-debugging-capstone-v2.ipynb` — *"Break Spark on Purpose"*
**Duration:** 45 min hands-on (after 15 min lecture)
**Level:** 400
**Audience expectation:** They've completed Modules 0–3 and have a baseline mental model of partitions, shuffle, AQE, and Delta layout.

---

## How this capstone is different from a normal "fix the pipeline" lab

Most performance war stories require terabytes to reproduce. We have ~350 MB of
`manufacturing_event` (~8 M rows). Instead of pretending, every incident in this notebook is a
failure pattern that **actually triggers live** on a single Fabric Spark session at this scale.

Tell attendees up front:

> "I'm not going to show you screenshots and say 'imagine the OOM'. We're going to make Spark
> fail in front of you, read the real error, then fix it. That's why I had to crank a couple of
> session configs down — to bring the failure modes into a 45-minute window."

This framing matters because the configs in the setup cell (capped `maxResultSize`, disabled
AQE knobs) **look wrong**. They are wrong on purpose. Call it out so nobody copy-pastes the
setup cell into prod.

---

## Pacing target (45 min)

| Block | Time | Notes |
|-------|------|-------|
| Setup + INC-1 (driver OOM) | 8 min | Highest "wow", lowest risk — start here. |
| INC-2 (cartesian) | 8 min | Don't run the broken cell — just `explain()` it. |
| INC-3 (single-partition spill) | 10 min | Most-frequently-seen bug in production. Linger here. |
| INC-4 (skew) | 10 min | This is the one that benefits most from a Spark UI walkthrough. |
| INC-5 (optional task storm) | 5 min if time, else skip | Fast-finisher material. |
| Wrap / restore conf | 4 min | Don't skip the restore cell — protects next exercise. |

If you fall behind: drop INC-5 entirely, then trim INC-4 to a single "before" run.

---

## Setup cell (cell 2) — what to say while it runs

- "We're using `SHALLOW CLONE` because we don't want to mutate `bronze.manufacturing_event` —
  every other notebook reads it."
- The `_ORIGINAL_CONF` snapshot pattern is reusable: any time you do destructive conf changes in
  a shared session, snapshot first, restore at the end. Point this out as a takeaway in itself.
- Wait for the "✅ Cloned …" print before moving on. Confirm everyone sees ~8 M rows and ~350 MB.

---

## 🔴 INC-1 — Driver OOM (cells 3–9)

**Headline:** *"350 MB on disk → multi-GB in your driver."*

### The "aha" moment
The cell **will** throw `SparkException: Total size of serialized results … is bigger than
spark.driver.maxResultSize`. This is the single most common surprise for analysts moving from
pandas to Spark. Pause here.

### Talking points
1. **Driver vs. executor OOM is the first triage question.** Wrong answer → wrong fix.
   - Driver OOM: result collection (`collect`, `toPandas`, `show(n)` with huge `n`), broadcast
     of an oversized table, query-plan blow-up on the driver.
   - Executor OOM: data partition too big, skew, large UDF state, cartesian join.
2. **The 350 MB → 2+ GB expansion math:** Parquet is heavily compressed (Snappy/ZSTD ≈ 3–5×) +
   Java object overhead in `Row` materialization + Arrow/pickle round-trip to Python. Call out
   each multiplier so nobody walks away thinking "Spark is buggy".
3. **`collect()` and `toPandas()` are the same code path.** Switching from one to the other does
   not help.
4. **Why we capped `maxResultSize` to 256 MB.** Default is 1 GB. We capped it so the failure
   happens deterministically in seconds, not after a long hang. Real production drivers OOM the
   JVM entirely — much messier to demo.

### Common audience questions
- *"Can I just raise `maxResultSize`?"* → Yes, and you'll buy yourself ~10× headroom before
  hitting the same wall, but each subsequent crash is harder to recover from because now the
  JVM dies instead of throwing a clean exception. The fix is to **not pull all data**.
- *"What about `df.toPandas()` with Arrow enabled?"* → Arrow makes the transfer faster and a bit
  more memory-efficient, but the cap is still the same and the analyst's intent (work with all
  the data in pandas) still doesn't fit.
- *"Why not just `.limit(1000).toPandas()`?"* → Sometimes that's right (sampling for plots). For
  aggregations, the solution shown — pre-aggregate in Spark, then convert — is correct.

### Solution cell (cell 9) talking point
Show the shape print: `(machines × hours, 4)` is on the order of thousands, not millions. That's
the "right size for pandas". Recommend: **if your result is > ~100 K rows, don't `toPandas()`.**

---

## 🔴 INC-2 — Cartesian Explosion (cells 10–15)

**Headline:** *"AQE can't save you from a quadratic plan."*

### Important demo discipline
**Do not run the broken cell to `.count()`.** It's a cartesian — it will spill GBs and may not
finish in your slot. The cell as written only calls `explain()` and then computes `n²` arithmetic
on a single machine. Stick to that. Verify the cell does NOT have `bad_self_join.count()` before
you run it live.

### Walking through the plan
In the `explain()` output, point out:
- The lack of any equi-predicate beyond `machine_id` (a high-cardinality key).
- The estimated row count blowing up (if Spark prints stats).
- The keywords `BroadcastNestedLoopJoin` or a SortMergeJoin with no usable predicate. Either
  signals "this is going to be quadratic per group".

### Talking points
1. The author's intent — *"compare each event to the previous one on the same machine"* — is a
   classic **windowing problem**, not a join problem. New engineers reach for joins because
   joins feel intuitive; windows feel like SQL trivia.
2. **AQE skew handling does nothing here.** AQE optimizes the shape of shuffles; it can't rewrite
   `O(n²)` into `O(n log n)`. This is a query-design problem, not a runtime problem.
3. **The window fix is `O(n log n)`** for the sort. On 8 M rows it runs in a few seconds.

### Common audience questions
- *"What if I really do need pairwise comparisons across all rows?"* → That's an honest answer
  of "use the right primitive": locality-sensitive hashing, time-bucketing, or pre-aggregation.
  If you truly need n², you need a different system (or a much smaller n).
- *"Would broadcast hint help?"* → No — broadcasting one side of an 8 M-row self-join makes the
  other executors copy the whole table. Still quadratic, just over the wire too.

### Solution cell (cell 15)
Run `fixed.count()` — should finish in seconds. Contrast with the projected `n²` row count from
the broken cell ("we'd have produced X billion rows").

---

## 🟠 INC-3 — Single-Partition Spill (cells 16–21)

**This is the most important incident in the entire capstone.** It's the failure mode I see
most often in real customer code reviews. Linger here.

### The setup
`Window.orderBy("timestamp")` with no `partitionBy()`. Spark *literally warns you in the log*:
```
WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single
partition, this can cause serious performance degradation.
```
Point at this warning if it shows up in driver logs.

### Live Spark UI walkthrough
After running cell 17, **switch to the Spark UI** in Fabric:
1. **Stages tab.** Find the new stage. It will have **1 task** (or a few — but one will dominate).
2. Click into the stage → **Tasks table**.
3. Scroll right to the **Spill (Memory)** and **Spill (Disk)** columns. You'll see hundreds of
   MB to GB of spill on that one task.
4. Compare task durations: the slow one dwarfs everything else.

This is the moment to say: *"In production this looks like a job that runs but is mysteriously
slow. Nobody gets paged. It silently eats your SLA budget for years until someone profiles it."*

### Talking points
1. The Spark plan contains `Exchange SinglePartition` → that's the smoking gun. Always check for
   it when you see unexpected spill.
2. **The same trap shows up in `df.repartition(1)` and `df.coalesce(1)`.** Anyone who has ever
   written "one CSV at the end of a pipeline" has done this.
3. **Two correct fixes**, depending on intent:
   - **(a)** If "global" was sloppy: `partitionBy("machine_id")` parallelizes. The sequence
     restarts per machine — usually that's what you actually wanted.
   - **(b)** If you genuinely need a globally unique ID: `F.monotonically_increasing_id()` is
     shuffle-free and sort-free. **Gotcha:** the IDs are not gap-free and they're only
     monotonically increasing **within a partition**, not globally ordered. State this clearly.

### Common audience questions
- *"What if I need globally-ordered sequence numbers?"* → You're asking for a serial primary key
  on a distributed system. The honest answer is: write to Delta and use a Delta IDENTITY column,
  or accept that this is a fundamentally non-distributed operation and budget for the cost.
- *"Why doesn't AQE coalesce save us here?"* → AQE coalesces *after* the shuffle. The shuffle
  itself collapsed to one partition because that's what the operator requires.

### Solution cell (cell 21)
Re-run and have attendees re-open Spark UI. Same stage now shows N tasks (matching shuffle
partitions), each with **zero spill**.

---

## 🟠 INC-4 — Skew (cells 22–26)

**Headline:** *"99 tasks finish in seconds, 1 task runs for minutes — that's skew."*

### Why we inject skew
The bronze data is too uniform to demo skew at lab scale. Cell 23 picks the busiest machine and
appends 9 extra copies of its rows. This is an honest engineering trick — call it out:

> "I'm faking skew here because our generator data is too uniform. In production you get this
> for free — one customer, one region, one product, one date partition will always be the hot
> one."

### Reading the partition profile
The `spark_partition_id()` trick in cell 24:
```python
df.repartition(200, "machine_id")
  .groupBy(spark_partition_id()).count()
  .orderBy(F.desc("count"))
```
This is a **reusable diagnostic** worth highlighting on its own. If you want to know whether
your data is skewed, this is the one-liner.

### Talking points
1. **AQE skew handling vs. salting.** AQE detects oversized shuffle blocks at runtime and splits
   them. It's automatic and should be the default. Salting (manual `key || rand % N`) is the
   fallback for cases AQE can't handle (e.g., skew you only see after a UDF).
2. **The "before" cell deliberately disables AQE skew join and coalesce.** Real-world skew shows
   up either when AQE is off, when the skew is on the wrong side of a join AQE doesn't optimize,
   or when block sizes don't trigger the heuristic.
3. The right knobs:
   - `spark.sql.adaptive.enabled = true` (default)
   - `spark.sql.adaptive.skewJoin.enabled = true` (default)
   - `spark.sql.adaptive.coalescePartitions.enabled = true` (default)
   - `spark.sql.adaptive.skewJoin.skewedPartitionFactor` (default 5)
   - `spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes` (default 256 MB)

### Common audience questions
- *"AQE is supposed to handle this — why doesn't it?"* → It does, when enabled. The bug we're
  demoing is "someone disabled it" (very common in older code bases that predate AQE-on-by-default).
- *"Does AQE skew handling work for aggregations or only joins?"* → The official name is
  "skew join". For aggregations, AQE's coalesce partition logic helps with the trailing tasks,
  but heavy aggregation skew is best handled with salting + two-stage aggregation.
- *"What if I have 5 hot keys, not 1?"* → AQE handles each independently. The pathological case
  is when the hot key drives partition size past `skewedPartitionThresholdInBytes` — then it
  splits.

### Spark UI walkthrough
After the broken cell: Stages → Tasks → sort by Duration descending. One task dominates. After
the fix: AQE will report split partitions in the Stage detail view (look for "skewedPartitions"
in the AQE plan output).

---

## 🟡 INC-5 — Tiny Task Storm (cells 27–30) — OPTIONAL

Only run if you have ≥5 minutes left. If you're tight, skip and mention it in the wrap-up.

### Headline
*"Too many tiny tasks is the mirror image of too few giant tasks. Both starve your throughput."*

### Talking points
1. `spark.sql.shuffle.partitions = 4000` is a real anti-pattern — usually set by someone who hit
   one large shuffle once and decided "more = safer". The opposite is true for everything else.
2. Task launch overhead in Fabric is ~50–100 ms. 4000 tasks × 100 ms = 6 minutes of scheduling
   for ~1 second of compute. The wall-clock you see is almost entirely overhead.
3. AQE's `coalescePartitions` solves this automatically by merging tiny post-shuffle partitions
   to approximately `spark.sql.adaptive.advisoryPartitionSizeInBytes` (default 64 MB).

### Common questions
- *"Can I just lower `spark.sql.shuffle.partitions` instead?"* → Yes, that works for static
  workloads. AQE is better because it adapts to actual data sizes per shuffle.
- *"What's the right static value?"* → Rough rule: 2–4× total executor cores. But really, leave
  AQE on and don't think about it.

---

## Wrap-up (cells 31–32)

### The Three Pillars callback
Tie back to Module 0's framing: every failure was either **I/O amplification**, **memory
pressure**, or **scheduling/parallelism imbalance**. Same three pillars. The diagnostic flow is
always: read the plan → read the Spark UI → identify the pillar → apply the targeted fix.

### Restore conf cell (cell 32)
**Do not skip this.** Without it, the destructive session conf bleeds into whatever the
attendee runs next. Walk through what `_ORIGINAL_CONF` captured and confirm the restore prints.

### Debrief prompts (last 5 min)
Ask the room:
1. *"Which incident felt most familiar from your own work?"* — usually INC-3 or INC-4.
2. *"Which Spark UI metric did you find most useful?"* — surfaces who's already comfortable in
   the UI vs. who needs more rep.
3. *"What's one thing you'll check first the next time a Spark job is slow?"* — gets them to
   commit to a habit.

---

## Pre-flight checklist (do this before class)

- [ ] Run the whole notebook end-to-end on a fresh Spark session and confirm every cell behaves
      as described (especially INC-1 raises the exception, INC-3 shows visible spill in the UI).
- [ ] Open Spark UI tabs you'll demo (Stages, SQL) ahead of time so switching is fast.
- [ ] Verify `bronze.manufacturing_event` exists in the workspace and is ~350 MB / ~8 M rows.
- [ ] Confirm `_benchmark_utils` notebook is in the same workspace (cell 1 `%run` will fail
      otherwise).
- [ ] If you plan to skip INC-5, decide in advance — don't decide live.

## Failure recovery (if a demo goes sideways)

- **INC-1 doesn't fail with the exception:** the cluster's default `maxResultSize` is unusually
  low/high. Verify cell 4's print shows `256m`. If the conf set didn't stick, restart the session.
- **INC-2 self-join cell hangs:** confirm you didn't accidentally call `bad_self_join.count()`.
  The notebook as shipped does NOT — just `explain()` plus a `count` on a single machine slice.
- **INC-3 doesn't show spill in the UI:** the cluster has too much memory for the data to spill.
  Either raise the dataset size (re-run setup with a UNION duplicate) or temporarily cap
  executor memory via `spark.executor.memory`.
- **INC-4 skew isn't visible:** the hot-machine multiplier (9×) may be too low for the cluster's
  task split heuristics. Bump to 19× by editing the `spark.range(9)` in cell 23.
- **AQE keeps "fixing" the broken cells:** double-check the broken-cell `spark.conf.set` calls
  actually ran. Some Fabric session configurations override per-session conf — if so, set the
  conf at the workspace level temporarily.

---

*End of speaker notes — Module 4.*
