import json, re

with open("04-lab-debugging-capstone.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Known PascalCase column names from LakeGen entities
pascal_cols = [
    "OrderId", "OrderNumber", "OrderDate", "CustomerId", "ShippingCountry",
    "OrderTotal", "LineNumber", "SetNum", "PartNum", "ItemName", "Quantity",
    "UnitPrice", "ExtendedPrice", "Name", "Email", "Country", "LoyaltyTier",
    "MemberSince", "PreferredSource", "OrganizationId", "GeneratedAt",
    "ProductionOrderId", "StartTime", "EndTime", "MachineId", "MoldId",
    "CycleTimeSec", "PartStatus", "DefectType", "CavityCount", "PlasticType",
    "ColorCode", "BatchId", "PartWeight", "PartName", "CategoryId",
    "PartMaterial", "QcInspectionId", "InspectionDate", "InspectorId",
    "InspectionResult", "DefectCount", "SampleSize", "Notes",
    "QcMeasurementId", "MeasurementType", "MeasuredValue", "UpperLimit",
    "LowerLimit", "Unit", "SetInventoryId", "WarehouseId", "QuantityOnHand",
    "ReorderPoint", "LastRestocked", "Status", "Source", "Price", "Currency",
    "EffectiveFrom", "EffectiveTo", "ChangeReason", "ProductionLineId",
    "LineName", "Capacity", "OperatingHours", "ProductionLinePartId",
]

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    found = set()
    for col in pascal_cols:
        # Look for column references (in SQL, Python attribute access, strings)
        if re.search(r'\b' + col + r'\b', src):
            found.add(col)
    if found:
        print(f"Cell [{i}]: {sorted(found)}")
