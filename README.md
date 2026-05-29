<img src="https://github.com/microsoft/fabric-analytics-roadshow-lab/blob/main/assets/images/spark/analytics.png?raw=true"
     width="80"
     align="left"
     style="margin-right:0px; padding-top:20px;" />

<h1 style="border-bottom: none; padding-bottom: 0; margin-bottom: 0;">
  Fabric Spark Performance Engineering Workshop
</h1>

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
jumpstart._install_with_config(
    logical_id="spark-performance-engineering",
    repo_url="https://github.com/mwc360/fabric-spark-performance-workshop.git",
    repo_ref="main",
    entry_point="00-getting-started.Notebook",
    items_in_scope=["Lakehouse", "Notebook", "Environment", "SparkJobDefinition"],
    workspace_path="workspace/",              # defaults to "{logical_id}/"
    name="Spark Performance Engineering",                         # display name (defaults to logical_id)
    workspace_id="<guid>",                       # target workspace (auto-detected in Fabric)
)
```

> [!NOTE]
>
> It will take approximately 8 minutes to install all lab content into your workspace. You will see a success message once done.