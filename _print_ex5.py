"""Print Exercise 5 cells for review."""
import json

with open("02-lab-configuration-tuning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(30, 40):
    if i >= len(nb["cells"]):
        break
    cell = nb["cells"][i]
    src = "".join(cell["source"])
    ct = cell["cell_type"]
    print(f"=== Cell {i} ({ct}) ===")
    print(src[:2000])
    if len(src) > 2000:
        print("...[truncated]")
    print()
