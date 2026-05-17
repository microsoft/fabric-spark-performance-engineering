# Abstract

Take your Spark expertise to the next level with a systematic approach to performance engineering that transforms how you build, tune, and debug production workloads. This intensive workshop moves beyond the basics to help attendees develop expert-level skills in performance optimization and troubleshooting complex production issues. This workshop will deep dive into the following topics, all in the context of hands-on labs that surround streaming Lego manufacturing and sales data.

1. Execution Architecture: Understanding Spark's query planning, Fabric's Native Execution Engine, Delta Lake internals, and distributed execution patterns that inform all tuning decisions.
1. Performance Diagnostics: Reading Spark UI like an expert, interpreting metrics and logs, identifying bottlenecks, and establishing performance baselines.
1. Systematic Tuning Methodology: A hierarchy-based approach from table features and physical design through session configurations.
1. Optimization Patterns: DataFrame transformations, join strategies, caching use cases, resource allocation, Adaptive Query Execution, and streaming optimizations. 
1. Advanced Debugging: Diagnosing OOMs, data skew, spill issues, and storage problems with proven troubleshooting tips, tricks, and best practices.
1. Platform Context: While focused on Spark in Microsoft Fabric, the core concepts apply universally across all Spark platforms.
1. Prerequisites: Spark fundamentals including DataFrames/SQL, basic understanding of distributed systems, and experience building data pipelines. Attendees must have existing experience using Spark.
1. Outcome: A systematic toolkit for optimizing any Spark workload, debugging production issues efficiently, and designing high-performance data solutions that scale.



# Module Ideas
Module	Objective	Requirements / What will be covered	Reference Links	
0: Foundational Theory	(mostly theory, with small instructor demos to show theory in action)	
Spark execution architecture w/ and w/o NEE (parser, analyzer, optimizer, planner, executor)
Thinking distributed: why it matters and core concepts like maximizing task parallelism / optimizing data partitions
Thinking lazy: why it matters, etc.
Optimizing I/O -> why file size and row group size matters
Storage: Delta internals
Statistics: different types and why they matter
	Performance Tuning - Spark 4.0.0 Documentation	
1: Diagnostics	Attendees should leave knowing how to debug performance problems, jobs failures, and storage issues or inefficiencies.	
Reading Spark UI, logs and metricsIdentifying what is running
Jobs v. Stages v. Tasks
Executor metrics
Knowing when NEE is leveraged and reason for fallbacks
Coding practices to set yourself up for success, i.e: setJobDescription()
Top performance anti-patterns and how to identify them
Resource allocation and scheduling
		
2: Configuration & Tuning		
Engine selection (NEE enablement) 
Table Feature configuration (tune file size, maximize file skipping, minimize write amplification) 
Table maintenance (OPTIMIZE, VACUUM, PURGE) 
Physical design (partitioning, clustering, data types, compression codec)
Statistics (DESCRIBE EXTENDED, EXPLAIN COST, ANALYZE)
	
FabCon25SparkWorkshop/module-4-tuning-optimizing-scaling/Lab 4 - Performance Tuning, Optimizing & Scaling.ipynb at main · voidfunction/FabCon25SparkWorkshop
	
3: Optimization	Attendees should leave being comfortable with tuning workloads per desired SLAs and designing solutions with scalability in mind.	
Resource allocation (compute sizing, parallelism - both partitions and I/O) 
Workload optimization (skew mitigation, caching strategies, repartitioning, AQE tuning) 
Code patterns (efficient joins, schema handling, using observations)
Advanced DataFrame patterns (broadcast hints, bucketing strategies) 
Orchestration & pipelining (dependency management, incremental processing, streaming, state management)
	
FabCon25SparkWorkshop/module-4-tuning-optimizing-scaling/Lab 4 - Performance Tuning, Optimizing & Scaling.ipynb at main · voidfunction/FabCon25SparkWorkshop
	
4: Debugging	Attendees should leave knowing how to debug performance problems, jobs failures, and storage issues or inefficiencies.	
Potential partition issues: getNumPartitions
OOM
Spills
Data skew
NEE fallbacks
Outdated statistics (extended stats and file stats)
Identifying perf bottlenecks
Limitations (i.e. log size) and how to work around constraints
Monitoring streaming jobs
Debugging tables and potential storage issuesinputFiles() -> get files that would be read
DESCRIBE DETAIL / HISTORY -> debug commits / what happened
	
databricks-academy/spark-ui-simulator: Apache Spark UI simulator with 30+ educational experiments, pre-recorded job data, and source code examples for learning performance optimization concepts
	
