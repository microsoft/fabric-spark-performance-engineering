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
# META       "environmentId": "",
# META       "workspaceId": ""
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Lab 3: LEGO Supply Chain Optimization Challenge
# 
# Each exercise follows the six Lab Details principles: context, setup, problem, investigation, fix, and validation. The notebook uses existing `Lego` lakehouse data. Missing scenario-specific tables, such as `web_order_skewed`, are derived in memory only and are not written to the lakehouse.

# CELL ********************

# Global setup
import json, time
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

SCHEMA = "bronze"
LAB3_RESULTS, PROBLEM_OUTPUTS, FIX_OUTPUTS, INVESTIGATIONS, ORIGINAL_CONF = [], {}, {}, {}, {}

def table_ref(name): return f"`{SCHEMA}`.`{name}`"
def set_job(label): spark.sparkContext.setJobDescription(f"Lab 3 Complete - {label}")
def run_timed(label, fn):
    print(f"START|{label}"); start=time.time(); value=fn(); elapsed=time.time()-start; print(f"END|{label}|elapsedSeconds={elapsed:.3f}"); return value, elapsed
def explain_to_string(df): return df._jdf.queryExecution().toString()
def record(prompt, phase, status, evidence):
    row={"prompt":prompt,"phase":phase,"status":status,"evidence":evidence}; LAB3_RESULTS.append(row); print("LAB3_COMPLETE_RESULT|"+json.dumps(row, sort_keys=True, default=str))
def table_metrics(name):
    ref=table_ref(name); detail=spark.sql(f"DESCRIBE DETAIL {ref}").collect()[0].asDict(); files=int(detail.get('numFiles') or 0); size=int(detail.get('sizeInBytes') or 0)
    return {"table":f"{SCHEMA}.{name}","rows":spark.table(ref).count(),"partitions":spark.table(ref).rdd.getNumPartitions(),"numFiles":files,"sizeBytes":size}
def remember_conf(key):
    if key not in ORIGINAL_CONF: ORIGINAL_CONF[key]=spark.conf.get(key, None)
def restore_conf(key):
    if key in ORIGINAL_CONF:
        if ORIGINAL_CONF[key] is None: spark.conf.unset(key)
        else: spark.conf.set(key, ORIGINAL_CONF[key])
print("Spark application ID:", spark.sparkContext.applicationId)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Prerequisite discovery
set_job("00 prerequisite discovery")
required=["manufacturing_event","production_order","parts","web_order","inventory_transaction","inventory_parts","inventories","sets","themes"]
available={r.tableName for r in spark.sql(f"SHOW TABLES IN `{SCHEMA}`").collect()}
missing=[t for t in required if t not in available]
if missing: raise RuntimeError(f"Missing required Lab 3 tables: {missing}")
TABLE_METRICS={t: table_metrics(t) for t in required}
for metric in TABLE_METRICS.values(): print("TABLE_METRIC|"+json.dumps(metric, sort_keys=True))
print("web_order_skewed exists:", "web_order_skewed" in available)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3A Context - The Billion-Row Join
# 
# Join high-volume manufacturing events to production orders and parts. The problem disables automatic broadcast; the fix broadcasts reference tables and validates `BroadcastHashJoin`.

# CELL ********************

# 3A Setup cell
set_job("3A setup")
q3a_mfg = (
    spark.table(table_ref("manufacturing_event"))
    .select(
        F.col("manufacturing_event.production_order_id").alias("production_order_id"),
        F.col("manufacturing_event.part_num").alias("part_num"),
        F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"),
        F.col("manufacturing_event.cycle_time_ms").alias("cycle_time_ms"),
    )
)
q3a_po = (
    spark.table(table_ref("production_order"))
    .select(
        F.col("production_order.production_order_id").alias("production_order_id"),
        F.col("production_order.status").alias("status"),
    )
)
q3a_parts = spark.table(table_ref("parts")).select(
    "part_num", "part_material", "part_cat_id"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3A Problem cell
set_job("3A problem no broadcast")
remember_conf("spark.sql.autoBroadcastJoinThreshold")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")


def q3a_problem_action():
    df = (
        q3a_mfg.join(q3a_po, "production_order_id")
        .join(q3a_parts, "part_num")
        .groupBy("part_material")
        .agg(
            F.count("*").alias("events"),
            F.sum("is_defect").alias("defects"),
            F.avg("cycle_time_ms").alias("avg_cycle_ms"),
        )
        .orderBy("part_material")
    )
    return df, df.collect()


(PROBLEM_OUTPUTS["3A"], q3a_problem_seconds) = run_timed(
    "3A problem", q3a_problem_action
)
q3a_problem_df, q3a_problem_rows = PROBLEM_OUTPUTS["3A"]
display(q3a_problem_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3A Investigation cell
set_job("3A investigation"); q3a_problem_plan=explain_to_string(q3a_problem_df)
INVESTIGATIONS["3A"]={"problemHasSortMergeJoin":"SortMergeJoin" in q3a_problem_plan,"problemHasBroadcastHashJoin":"BroadcastHashJoin" in q3a_problem_plan,"factRows":TABLE_METRICS["manufacturing_event"]["rows"],"partsRows":TABLE_METRICS["parts"]["rows"]}
print(q3a_problem_plan); record("3A","investigation","complete",INVESTIGATIONS["3A"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3A Fix cell
set_job("3A fix broadcast")
def q3a_fix_action():
    df=q3a_mfg.join(broadcast(q3a_po),"production_order_id").join(broadcast(q3a_parts),"part_num").groupBy("part_material").agg(F.count("*").alias("events"),F.sum("is_defect").alias("defects"),F.avg("cycle_time_ms").alias("avg_cycle_ms")).orderBy("part_material")
    return df, df.collect()
(FIX_OUTPUTS["3A"], q3a_fix_seconds)=run_timed("3A fix", q3a_fix_action); q3a_fix_df,q3a_fix_rows=FIX_OUTPUTS["3A"]; display(q3a_fix_df); restore_conf("spark.sql.autoBroadcastJoinThreshold")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3A Validation cell
fix_plan=explain_to_string(q3a_fix_df)
pm={r["part_material"]:(r["events"],r["defects"],round(float(r["avg_cycle_ms"] or 0),4)) for r in q3a_problem_rows}; fm={r["part_material"]:(r["events"],r["defects"],round(float(r["avg_cycle_ms"] or 0),4)) for r in q3a_fix_rows}
valid=pm==fm and "BroadcastHashJoin" in fix_plan
record("3A","validation","passed" if valid else "failed",{"sameBusinessResult":pm==fm,"fixHasBroadcastHashJoin":"BroadcastHashJoin" in fix_plan,"problemSeconds":q3a_problem_seconds,"fixSeconds":q3a_fix_seconds,**INVESTIGATIONS["3A"]}); assert valid, "3A validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3B Context - The Skewed Customer
# 
# `web_order_skewed` is absent, so skew is derived in memory from existing orders by duplicating top customers. The fix salts keys before aggregation.

# CELL ********************

# 3B Setup cell
set_job("3B setup")
q3b_base=spark.table(table_ref("web_order")).select(F.col("web_order.customer_id").alias("customer_id"),F.col("web_order.order_total").alias("order_total"),F.col("web_order.order_id").alias("order_id"))
q3b_top=[r["customer_id"] for r in q3b_base.groupBy("customer_id").count().orderBy(F.desc("count")).limit(5).collect()]
q3b_hot=q3b_base.filter(F.col("customer_id").isin(q3b_top)); q3b_skewed=q3b_base
for _ in range(8): q3b_skewed=q3b_skewed.unionByName(q3b_hot)
print("3B in-memory skewed rows:", q3b_skewed.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3B Problem cell
set_job("3B problem skewed aggregation")
def q3b_problem_action():
    df=q3b_skewed.groupBy("customer_id").agg(F.count("*").alias("orders"),F.sum("order_total").alias("revenue")).orderBy(F.desc("orders"),"customer_id")
    return df, df.limit(20).collect()
(PROBLEM_OUTPUTS["3B"], q3b_problem_seconds)=run_timed("3B problem", q3b_problem_action); q3b_problem_df,q3b_problem_rows=PROBLEM_OUTPUTS["3B"]; display(q3b_problem_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3B Investigation cell
set_job("3B investigation")
counts=q3b_skewed.groupBy("customer_id").count(); s=counts.agg(F.max("count").alias("max"),F.expr("percentile_approx(count,0.5)").alias("median"),F.avg("count").alias("avg")).collect()[0]; skew=float(s["max"])/max(float(s["median"]),1.0)
INVESTIGATIONS["3B"]={"maxCustomerRows":int(s["max"]),"medianCustomerRows":int(s["median"]),"avgCustomerRows":float(s["avg"]),"skewRatio":skew,"derivedInMemoryBecauseWebOrderSkewedMissing":True}
record("3B","investigation","complete",INVESTIGATIONS["3B"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3B Fix cell
set_job("3B fix salted aggregation")
def q3b_fix_action():
    salted=q3b_skewed.withColumn("salt",F.pmod(F.xxhash64("order_id"),F.lit(16)))
    partial=salted.groupBy("customer_id","salt").agg(F.count("*").alias("orders"),F.sum("order_total").alias("revenue"))
    df=partial.groupBy("customer_id").agg(F.sum("orders").alias("orders"),F.sum("revenue").alias("revenue")).orderBy(F.desc("orders"),"customer_id")
    return df, df.limit(20).collect()
(FIX_OUTPUTS["3B"], q3b_fix_seconds)=run_timed("3B fix", q3b_fix_action); q3b_fix_df,q3b_fix_rows=FIX_OUTPUTS["3B"]; display(q3b_fix_df.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3B Validation cell
pm={r["customer_id"]:(int(r["orders"]),round(float(r["revenue"] or 0),2)) for r in q3b_problem_df.collect()}; fm={r["customer_id"]:(int(r["orders"]),round(float(r["revenue"] or 0),2)) for r in q3b_fix_df.collect()}
valid=pm==fm and INVESTIGATIONS["3B"]["skewRatio"]>3
record("3B","validation","passed" if valid else "failed",{"sameBusinessResult":pm==fm,"problemSeconds":q3b_problem_seconds,"fixSeconds":q3b_fix_seconds,**INVESTIGATIONS["3B"]}); assert valid, "3B validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3C Context - The Repeated Read
# 
# The baseline repeats the same inventory source read and production-order join for multiple transaction types. The fix caches the joined base once.

# CELL ********************

# 3C Setup cell
set_job("3C setup")
q3c_inv=spark.table(table_ref("inventory_transaction")).select("transaction_type","reference_id","quantity","part_num","line_id")
q3c_po=spark.table(table_ref("production_order")).select(F.col("production_order.production_order_id").alias("reference_id"),F.col("production_order.status").alias("order_status"),F.col("production_order.machine_id").alias("machine_id"))
q3c_types=[r["transaction_type"] for r in q3c_inv.groupBy("transaction_type").count().orderBy(F.desc("count")).limit(2).collect()]
print("3C transaction types:", q3c_types)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3C Problem cell
set_job("3C problem repeated reads")
def q3c_problem_action():
    out=[]
    for tx in q3c_types:
        df=spark.table(table_ref("inventory_transaction")).select("transaction_type","reference_id","quantity","part_num","line_id").filter(F.col("transaction_type")==tx).join(q3c_po,"reference_id","left").groupBy("transaction_type","order_status").agg(F.count("*").alias("transactions"),F.sum("quantity").alias("quantity"))
        out.extend(df.collect())
    return out
(PROBLEM_OUTPUTS["3C"], q3c_problem_seconds)=run_timed("3C problem", q3c_problem_action); q3c_problem_rows=PROBLEM_OUTPUTS["3C"]; display(spark.createDataFrame(q3c_problem_rows))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3C Investigation cell
INVESTIGATIONS["3C"]={"sourceRows":TABLE_METRICS["inventory_transaction"]["rows"],"repeatedScans":len(q3c_types),"transactionTypes":q3c_types}
record("3C","investigation","complete",INVESTIGATIONS["3C"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3C Fix cell
set_job("3C fix cache joined base")
def q3c_fix_action():
    joined=q3c_inv.join(q3c_po,"reference_id","left").cache(); joined.count(); out=[]
    for tx in q3c_types: out.extend(joined.filter(F.col("transaction_type")==tx).groupBy("transaction_type","order_status").agg(F.count("*").alias("transactions"),F.sum("quantity").alias("quantity")).collect())
    joined.unpersist(); return out
(FIX_OUTPUTS["3C"], q3c_fix_seconds)=run_timed("3C fix", q3c_fix_action); q3c_fix_rows=FIX_OUTPUTS["3C"]; display(spark.createDataFrame(q3c_fix_rows))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3C Validation cell
ps={(r["transaction_type"],r["order_status"],int(r["transactions"]),int(r["quantity"] or 0)) for r in q3c_problem_rows}; fs={(r["transaction_type"],r["order_status"],int(r["transactions"]),int(r["quantity"] or 0)) for r in q3c_fix_rows}
valid=ps==fs and INVESTIGATIONS["3C"]["repeatedScans"]>1
record("3C","validation","passed" if valid else "failed",{"sameBusinessResult":ps==fs,"problemSeconds":q3c_problem_seconds,"fixSeconds":q3c_fix_seconds,**INVESTIGATIONS["3C"]}); assert valid, "3C validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3D Context - The Shuffle Storm
# 
# The baseline joins `inventory_parts` before reducing it. The fix de-duplicates color usage first and tunes shuffle partitions for the lab run.

# CELL ********************

# 3D Setup cell
set_job("3D setup")
q3d_ip=spark.table(table_ref("inventory_parts")).select("inventory_id","part_num","color_id","quantity")
q3d_inv=spark.table(table_ref("inventories")).select(F.col("id").alias("inventory_id"),"set_num")
q3d_sets=spark.table(table_ref("sets")).select("set_num",F.col("name").alias("set_name"),"theme_id")
q3d_themes=spark.table(table_ref("themes")).select(F.col("id").alias("theme_id"),F.col("name").alias("theme_name"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3D Problem cell
set_job("3D problem shuffle storm")
def q3d_problem_action():
    df=q3d_ip.join(q3d_inv,"inventory_id").join(q3d_sets,"set_num").join(q3d_themes,"theme_id","left").groupBy("set_num","set_name","theme_name").agg(F.countDistinct("color_id").alias("distinct_colors"),F.sum("quantity").alias("total_pieces")).orderBy(F.desc("distinct_colors"),F.desc("total_pieces"),"set_num").limit(20)
    return df, df.collect()
(PROBLEM_OUTPUTS["3D"], q3d_problem_seconds)=run_timed("3D problem", q3d_problem_action); q3d_problem_df,q3d_problem_rows=PROBLEM_OUTPUTS["3D"]; display(q3d_problem_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3D Investigation cell
q3d_problem_plan=explain_to_string(q3d_problem_df)
INVESTIGATIONS["3D"]={"inventoryPartRows":TABLE_METRICS["inventory_parts"]["rows"],"problemHasExchange":"Exchange" in q3d_problem_plan,"shufflePartitions":spark.conf.get("spark.sql.shuffle.partitions")}
print(q3d_problem_plan); record("3D","investigation","complete",INVESTIGATIONS["3D"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3D Fix cell
set_job("3D fix pre-aggregate colors"); remember_conf("spark.sql.shuffle.partitions"); spark.conf.set("spark.sql.shuffle.partitions","32")
def q3d_fix_action():
    dc=q3d_ip.select("inventory_id","color_id").distinct(); pcs=q3d_ip.groupBy("inventory_id").agg(F.sum("quantity").alias("inventory_pieces"))
    colors=dc.join(q3d_inv,"inventory_id").join(q3d_sets,"set_num").join(q3d_themes,"theme_id","left").groupBy("set_num","set_name","theme_name").agg(F.countDistinct("color_id").alias("distinct_colors"))
    pieces=pcs.join(q3d_inv,"inventory_id").groupBy("set_num").agg(F.sum("inventory_pieces").alias("total_pieces"))
    df=colors.join(pieces,"set_num").orderBy(F.desc("distinct_colors"),F.desc("total_pieces"),"set_num").limit(20)
    return df, df.collect()
(FIX_OUTPUTS["3D"], q3d_fix_seconds)=run_timed("3D fix", q3d_fix_action); q3d_fix_df,q3d_fix_rows=FIX_OUTPUTS["3D"]; display(q3d_fix_df); restore_conf("spark.sql.shuffle.partitions")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3D Validation cell
pp=[(r["set_num"],int(r["distinct_colors"]),int(r["total_pieces"] or 0)) for r in q3d_problem_rows]; fp=[(r["set_num"],int(r["distinct_colors"]),int(r["total_pieces"] or 0)) for r in q3d_fix_rows]
valid=pp==fp and INVESTIGATIONS["3D"]["problemHasExchange"]
record("3D","validation","passed" if valid else "failed",{"sameTop20Result":pp==fp,"problemSeconds":q3d_problem_seconds,"fixSeconds":q3d_fix_seconds,**INVESTIGATIONS["3D"]}); assert valid, "3D validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3E Context - The Streaming Pipeline
# 
# The baseline creates a stateful streaming aggregation without watermarking. The fix adds a two-hour event-time watermark and validates the streaming plan.

# CELL ********************

# 3E Setup cell
set_job("3E setup")
q3e_batch=spark.table(table_ref("manufacturing_event")).select(F.to_timestamp("manufacturing_event.timestamp").alias("event_ts"),F.col("manufacturing_event.machine_id").alias("machine_id"),F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"))
q3e_event_count=q3e_batch.count(); print("3E manufacturing events:", q3e_event_count)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3E Problem cell
set_job("3E problem streaming no watermark")
def q3e_problem_action():
    s=spark.readStream.table(table_ref("manufacturing_event")).select(F.to_timestamp("manufacturing_event.timestamp").alias("event_ts"),F.col("manufacturing_event.machine_id").alias("machine_id"),F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"))
    agg=s.groupBy(F.window("event_ts","1 hour"),"machine_id").agg(F.count("*").alias("events"),F.sum("is_defect").alias("defects"))
    return agg, explain_to_string(agg)
(PROBLEM_OUTPUTS["3E"], q3e_problem_seconds)=run_timed("3E problem", q3e_problem_action); q3e_problem_df,q3e_problem_plan=PROBLEM_OUTPUTS["3E"]; print(q3e_problem_plan)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3E Investigation cell
INVESTIGATIONS["3E"]={"isStreaming":q3e_problem_df.isStreaming,"problemHasWatermark":"EventTimeWatermark" in q3e_problem_plan,"statefulAggregation":"Aggregate" in q3e_problem_plan,"sourceRowsAvailableForStreaming":q3e_event_count}
record("3E","investigation","complete",INVESTIGATIONS["3E"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3E Fix cell
set_job("3E fix watermark")
def q3e_fix_action():
    s=spark.readStream.table(table_ref("manufacturing_event")).select(F.to_timestamp("manufacturing_event.timestamp").alias("event_ts"),F.col("manufacturing_event.machine_id").alias("machine_id"),F.col("manufacturing_event.defect_detected").cast("int").alias("is_defect"))
    fixed=s.withWatermark("event_ts","2 hours").groupBy(F.window("event_ts","1 hour"),"machine_id").agg(F.count("*").alias("events"),F.sum("is_defect").alias("defects"))
    return fixed, explain_to_string(fixed)
(FIX_OUTPUTS["3E"], q3e_fix_seconds)=run_timed("3E fix", q3e_fix_action); q3e_fix_df,q3e_fix_plan=FIX_OUTPUTS["3E"]; print(q3e_fix_plan)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3E Validation cell
valid=q3e_problem_df.isStreaming and q3e_fix_df.isStreaming and ("EventTimeWatermark" in q3e_fix_plan) and ("EventTimeWatermark" not in q3e_problem_plan)
record("3E","validation","passed" if valid else "failed",{"problemIsStreaming":q3e_problem_df.isStreaming,"fixIsStreaming":q3e_fix_df.isStreaming,"problemHasWatermark":"EventTimeWatermark" in q3e_problem_plan,"fixHasWatermark":"EventTimeWatermark" in q3e_fix_plan,"problemSeconds":q3e_problem_seconds,"fixSeconds":q3e_fix_seconds,**INVESTIGATIONS["3E"]}); assert valid, "3E validation failed"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Final validation summary
# 
# The final cell emits machine-readable markers for CLI validation and Spark Operations correlation.

# CELL ********************

# Final validation summary
set_job("Lab 3 final summary")
validations=[r for r in LAB3_RESULTS if r["phase"]=="validation"]; investigations=[r for r in LAB3_RESULTS if r["phase"]=="investigation"]; failed=[r for r in validations if r["status"]!="passed"]
summary={"sparkApplicationId":spark.sparkContext.applicationId,"validationCount":len(validations),"investigationCount":len(investigations),"failedValidationCount":len(failed),"results":LAB3_RESULTS}
print("LAB3_COMPLETE_FINAL_SUMMARY_START"); print(json.dumps(summary, indent=2, sort_keys=True, default=str)); print("LAB3_COMPLETE_FINAL_SUMMARY_END")
assert len(validations)==5, f"Expected 5 validations, got {len(validations)}"; assert len(investigations)==5, f"Expected 5 investigations, got {len(investigations)}"; assert not failed, f"Failed validations: {failed}"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
