# Refined workshop content

This folder contains the **refined, publication-ready** version of the Spark Performance
Engineering workshop, restructured for a self-guided **Fabric Jumpstart**. It supersedes the
original notebooks in the parent folder (which are kept for reference). **Install only the
`refined/` set** — do not install both, or notebook names (e.g. `_benchmark_utils`) will collide.

## What changed

- **Four labs consolidated into three modules**, organized by the *fix lever* (the layer you
  change to fix a problem — the tuning hierarchy: **code → data → execution**). The standalone
  `04-advanced-debugging` lab mostly re-taught skew / shuffle / cartesian / OOM already covered
  elsewhere; its unique pieces were redistributed so nothing is lost.
- **One self-contained notebook per module** (challenge → inline solution). The upstream cores
  are a mix of self-guided (`01`, `02`) and answer-key (`03`, `04`) styles; the refined set
  normalizes all three modules to a single challenge → solution flow.
- **De-branded** to a generic "Toy Brick Manufacturing" domain in all learner-facing text.
- **Cleanup**: removed dev leftovers (hardcoded workspace name / placeholder GUIDs / dead
  code / typos), fixed a copy/paste bug and finished incomplete exercises in the tables lab,
  and de-duplicated the shared benchmark utilities.

## Old → new mapping

| Old | New | Notes |
|-----|-----|-------|
| `00-getting-started` | `refined/00-getting-started` | 3-module table, de-branded |
| `01-lab-diagnostics-factory-dashboard-slow` (Q1 pushdown, Q2 UDF, Q3 collect, Q5 cartesian) | `refined/01-optimizing-code` | Code-fix anti-patterns only; carries the diagnostic toolkit |
| `04-advanced-debugging` — INC-1 OOM, INC-2 cartesian | `refined/01-optimizing-code` | Folded in as code-fix content (OOM + cartesian→`lag()` rewrite) |
| `02-delta-table-design-and-optimization` | `refined/02-optimizing-delta-tables` | Ex3 blank cell fixed, Ex2 prose fixed; data-types + partitioning added |
| `04-advanced-debugging` — delta storage regression (`DESCRIBE HISTORY`) | `refined/02-optimizing-delta-tables` | Storage-regression audit exercise |
| `03-lab-optimization-supply-chain-complete` (answer-key) | `refined/03-optimizing-execution` | Converted answer-key → challenge/solution; switched to `_benchmark_utils` |
| `04-advanced-debugging` — INC-3 skew, INC-4 shuffle | `refined/03-optimizing-execution` | Execution-side remediation |
| `utils/_benchmark_utils` | `refined/utils/_benchmark_utils` | Cleaned (dedup imports / print helpers) |

## Module scope (no overlap)

- **01-optimizing-code** — *the query is written badly.* Fix = edit the code. Also introduces
  the diagnostic toolkit (Spark UI, `explain`, `DESCRIBE DETAIL/HISTORY`, `inputFiles()`, NEE
  fallback) reused by later modules.
- **02-optimizing-delta-tables** — *the table layout is wrong.* Fix = `OPTIMIZE` / clustering /
  schema / partitioning.
- **03-optimizing-execution** — *code and tables are fine, execution isn't.* Fix = join hints /
  AQE / partitions / `.cache()`, with the logic unchanged.

## Note on de-branding scope

De-branding is limited to **notebook content**. The backend data generator (LakeGen), its
environment/wheel, the Spark Job Definition imports, and the lakehouse *item* names are
unchanged — so an internal `lego` identifier may still be visible in the Fabric UI. This is
intentional and out of scope for the notebook refinement.
