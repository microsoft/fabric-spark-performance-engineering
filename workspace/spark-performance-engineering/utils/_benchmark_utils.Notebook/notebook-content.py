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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
