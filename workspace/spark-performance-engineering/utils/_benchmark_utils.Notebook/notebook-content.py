# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "environment": {
# META       "environmentId": "99FB9CB3-86D3-4877-BB60-659B3CDD45C3",
# META       "workspaceId": "7fc5eff4-7153-4da9-b909-54981a3ffcdb"
# META     }
# META   }
# META }

# CELL ********************

import time
from pyspark.sql import DataFrame

benchmarks = {}  # reset on cell re-run

def get_table_metrics(table_name):
    """Return file count, total size, and avg file size for a Delta table."""
    detail = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()[0]
    num_files = detail["numFiles"]
    size_bytes = detail["sizeInBytes"]
    avg_kb = round(size_bytes / num_files / 1024, 1) if num_files > 0 else 0
    return {"num_files": num_files, "size_mb": round(size_bytes / 1048576, 2), "avg_file_kb": avg_kb}


def show_metrics(table_name, label=""):
    m = get_table_metrics(table_name)
    tag = f" ({label})" if label else ""
    print(f"   {table_name}{tag}:  {m['num_files']:>6,} files  |  {m['size_mb']:>8.1f} MB  |  avg {m['avg_file_kb']:>8.1f} KB/file")
    return m


def print_scenario(scenario):
    states = benchmarks[scenario]
    
    if len(states) > 1:
        baseline_key = next(iter(states))
        baseline_ms = states[baseline_key]
        best_ms = min(states.values())
        W = 58  # inner width
        HR = '─' * W
        print()
        print(f"  \u250c{HR}\u2510")
        title = f"\033[1m{scenario}\033[0m"
        title_pad = W - 2 - len(scenario)
        print(f"  \u2502  {title}{' ' * title_pad}\u2502")
        print(f"  \u251c{HR}\u2524")
        print(f"  \u2502  {'State':<28}{'Time (ms)':>12}{'Factor':>14}  \u2502")
        print(f"  \u251c{HR}\u2524")
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
        print(f"  \u2514{HR}\u2518")


def print_all_scenarios():
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
    


import time
from pyspark.sql import DataFrame


class _BenchmarkProxy:
    """
    DataFrame proxy that times the next terminal action.
    """

    TERMINAL_ACTIONS = {
        "collect", "count", "show", "head", "first",
        "take", "toPandas", "printSchema", "display",
    }

    def __init__(self, df: DataFrame, scenario: str, state: str):
        self._df = df
        self._scenario = scenario
        self._state = state

    def _record(self, elapsed_ms: float):
        benchmarks.setdefault(self._scenario, {})[self._state] = elapsed_ms
        states = benchmarks[self._scenario]

        print(f"⏱️  {self._scenario} [{self._state}]: {elapsed_ms:.2f} ms")

        if len(states) > 1:
            print_scenario(self._scenario)

    def __getattr__(self, name):
        attr = getattr(self._df, name)

        if name in self.TERMINAL_ACTIONS and callable(attr):
            def timed(*args, **kwargs):
                spark = self._df.sparkSession
                spark.catalog.clearCache()
                spark.sparkContext.setJobDescription(
                    f"{self._scenario} [{self._state}]"
                )

                start = time.time()
                try:
                    result = attr(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.time() - start) * 1000
                    spark.sparkContext.setJobDescription(None)
                    self._record(elapsed_ms)

            return timed

        if callable(attr):
            def passthrough(*args, **kwargs):
                result = attr(*args, **kwargs)

                if isinstance(result, DataFrame):
                    return _BenchmarkProxy(
                        result,
                        self._scenario,
                        self._state
                    )

                return result

            return passthrough

        return attr


class BenchmarkTimer:
    """
    General-purpose benchmark helper for timing any Spark operation.
    """

    def __init__(self, scenario: str, state: str, spark=None):
        self.scenario = scenario
        self.state = state
        self.spark = spark
        self.elapsed_ms = None

    def _record(self, elapsed_ms: float):
        benchmarks.setdefault(self.scenario, {})[self.state] = elapsed_ms
        states = benchmarks[self.scenario]

        print(f"⏱️  {self.scenario} [{self.state}]: {elapsed_ms:.2f} ms")

        if len(states) > 1:
            print_scenario(self.scenario)

    def __enter__(self):
        if self.spark is not None:
            self.spark.catalog.clearCache()
            self.spark.sparkContext.setJobDescription(
                f"{self.scenario} [{self.state}]"
            )

        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_ms = (time.time() - self._start) * 1000

        if self.spark is not None:
            self.spark.sparkContext.setJobDescription(None)

        self._record(self.elapsed_ms)

        # Do not swallow exceptions
        return False


def benchmark_op(scenario: str, state: str, spark=None):
    """
    Time any operation using a context manager.

    Examples:
        with benchmark_op("Write Benchmark", "before", spark):
            df.write.mode("overwrite").format("delta").save(path)

        with benchmark_op("Optimize", "after", spark):
            spark.sql(f"OPTIMIZE {table_name}")
    """
    return BenchmarkTimer(scenario, state, spark)


# ---------------------------------------------------------------------------
# Shared lab helpers — imported by every module via `%run _benchmark_utils`
# so modules do not redefine their own niche functions.
# ---------------------------------------------------------------------------
import json
import re

results = []          # qualitative validation records (see record_result)
_ORIGINAL_CONF = {}   # snapshot for remember_conf / restore_conf


def table_ref(name: str, schema: str = "bronze") -> str:
    """Backtick-quoted `schema`.`table` reference."""
    return f"`{schema}`.`{name}`"


def reset_work_schema(schema: str) -> None:
    """Drop and recreate an isolated work schema (idempotent)."""
    spark.sql(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    spark.sql(f"CREATE SCHEMA {schema}")
    print(f"✅ Work schema reset: {schema}")


def require_tables(expected, schema: str = "bronze"):
    """Raise if any expected source table is missing; return the available set."""
    available = {row.tableName for row in spark.sql(f"SHOW TABLES IN `{schema}`").collect()}
    missing = [t for t in expected if t not in available]
    if missing:
        raise RuntimeError(f"Missing required tables in schema {schema}: {missing}")
    return available


def remember_conf(key: str) -> None:
    """Snapshot a Spark conf value once so it can be restored later."""
    if key not in _ORIGINAL_CONF:
        _ORIGINAL_CONF[key] = spark.conf.get(key, None)


def restore_conf(key: str) -> None:
    """Restore a Spark conf value captured by remember_conf."""
    if key in _ORIGINAL_CONF:
        value = _ORIGINAL_CONF[key]
        if value is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, value)


def set_job(label: str) -> None:
    """Set the Spark job description shown in the Spark UI."""
    spark.sparkContext.setJobDescription(label)


def plan_string(df: DataFrame) -> str:
    """Executed physical plan as a string."""
    return df._jdf.queryExecution().executedPlan().toString()


def explain_string(df: DataFrame) -> str:
    """Full queryExecution string (parsed / analyzed / optimized / physical)."""
    return df._jdf.queryExecution().toString()


def scan_filters(df: DataFrame) -> dict:
    """Extract DataFilters / PushedFilters from the last FileScan node."""
    file_scans = [node for node in plan_string(df).split("\n") if "FileScan" in node]
    last_file_scan = file_scans[-1].strip() if file_scans else ""
    data_filters = re.search(r"DataFilters: \[(.*?)\]", last_file_scan)
    pushed_filters = re.search(r"PushedFilters: \[(.*?)\]", last_file_scan)
    return {
        "fileScan": last_file_scan,
        "dataFilters": data_filters.group(1) if data_filters else "",
        "pushedFilters": pushed_filters.group(1) if pushed_filters else "",
    }


# Native Execution Engine (NEE) fallback analysis.
_nee_block_pattern = re.compile(
    r"(?ms)^\s*\+-\s*RowToVeloxColumnar\b[^\n]*\n(?P<block>.*?)^\s*\+-\s*VeloxColumnarToRow\b"
)
_nee_op_pattern = re.compile(
    r"(?m)^\s*\+-\s*(?:\^\(\d+\)\s*)?(?P<op>[A-Za-z][A-Za-z0-9]*)\b"
)


def extract_nee_fallbacks(plan: str) -> dict:
    """Count Velox->row fallback blocks and operators in a physical plan string."""
    fallback_blocks = []
    fallback_operations = []
    for match in _nee_block_pattern.finditer(plan):
        block_text = match.group("block")
        block_lines = [line.strip() for line in block_text.split("\n") if line.strip()]
        block_ops = _nee_op_pattern.findall(block_text)
        fallback_blocks.append({"operations": block_lines, "operatorNames": block_ops})
        fallback_operations.extend(block_ops)
    return {
        "blockCount": len(fallback_blocks),
        "operatorCount": len(fallback_operations),
        "operators": fallback_operations,
        "blocks": fallback_blocks,
    }


def table_metrics(name: str, schema: str = "bronze") -> dict:
    """Rich Delta metrics: rows, files, size, avg file size, format, partitions."""
    ref = table_ref(name, schema)
    detail_metrics = get_table_metrics(ref)
    detail = spark.sql(f"DESCRIBE DETAIL {ref}").collect()[0].asDict()
    return {
        "table": f"{schema}.{name}",
        "rows": spark.table(ref).count(),
        "numFiles": int(detail.get("numFiles") or detail_metrics.get("num_files") or 0),
        "sizeMB": float(detail_metrics.get("size_mb") or 0),
        "avgFileKB": float(detail_metrics.get("avg_file_kb") or 0),
        "format": detail.get("format"),
        "partitions": spark.table(ref).rdd.getNumPartitions(),
    }


def recent_history(name: str, schema: str = "bronze", limit: int = 3) -> list:
    """Recent DESCRIBE HISTORY rows (version, timestamp, operation)."""
    return [
        row.asDict()
        for row in spark.sql(f"DESCRIBE HISTORY {table_ref(name, schema)}")
        .select("version", "timestamp", "operation")
        .limit(limit)
        .collect()
    ]


def record_result(exercise: str, phase: str, evidence: dict) -> dict:
    """
    Append a qualitative validation record and print it.

    Use in CHECK-CHANGES / re-benchmark steps to capture before/after evidence.
    DIAGNOSE steps should only print (do not record).
    """
    row = {"exercise": exercise, "phase": phase, "evidence": evidence}
    results.append(row)
    print("RESULT\n" + json.dumps(row, default=str, indent=2))
    return row


def _benchmark(self: DataFrame, scenario: str, state: str):
    """
    Start a timed benchmark. Chain operations, then call a terminal action.

    Examples:
        df.benchmark("Small Files", "before").count()
        df.benchmark("Small Files", "after").count()
        df.benchmark("Clustering", "off").filter(...).collect()
    """
    return _BenchmarkProxy(self, scenario, state)


DataFrame.benchmark = _benchmark

def print_benchmark_summary():
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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
