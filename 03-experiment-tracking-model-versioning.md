# 03 · Experiment Tracking & Model Versioning

> **Goal:** make every model reproducible and comparable: *which code, data, parameters and environment produced which metric and artifact?*

**Contents**
1. [What must be tracked](#1-what-must-be-tracked)
2. [MLflow tracking at scale](#2-mlflow-tracking-at-scale)
3. [Weights & Biases](#3-weights--biases)
4. [DVC: data & pipeline versioning](#4-dvc-data--pipeline-versioning)
5. [Reproducibility checklist](#5-reproducibility-checklist)
6. [Edge cases & production failures](#6-edge-cases--production-failures)
7. [Interview Q&A](#7-senior-interview-questions--answers)

---

## 1. What must be tracked

| Dimension | Examples | Where stored |
|---|---|---|
| **Code** | git SHA, dirty flag, diff | MLflow tag `mlflow.source.git.commit` |
| **Data** | dataset hash/DVC rev, row count, time window | DVC `.lock`, MLflow tag/dataset input |
| **Params** | hyperparameters, feature list, seed | MLflow params |
| **Environment** | Docker image digest, `pip freeze`, CUDA/driver | Artifact / tag |
| **Metrics** | train/val/test, per-slice, over steps | MLflow metrics |
| **Artifacts** | model, plots, confusion matrices, SHAP | Artifact store (S3/GCS/MinIO) |
| **Signature** | input/output schema | MLflow model signature |

```
          ┌─────────────────────────────────────────────────────┐
 Client ─▶│ MLflow Tracking Server (REST)                       │
          │   ├─ Backend store  : PostgreSQL (runs, params,     │
          │   │                   metrics, tags, registry)      │
          │   └─ Artifact store : S3 / GCS / MinIO (models,     │
          │                       plots, data samples)          │
          └─────────────────────────────────────────────────────┘
```

## 2. MLflow tracking at scale

### Server for teams (not `mlruns/` on a laptop)

`docker-compose.mlflow.yml`
```yaml
services:
  postgres:
    image: postgres:16
    environment: {POSTGRES_USER: mlflow, POSTGRES_PASSWORD: mlflow, POSTGRES_DB: mlflow}
    volumes: [pgdata:/var/lib/postgresql/data]
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment: {MINIO_ROOT_USER: minio, MINIO_ROOT_PASSWORD: minio12345}
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio:/data]
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    depends_on: [postgres, minio]
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: minio12345
    command: >
      sh -c "pip install -q psycopg2-binary boto3 &&
             mlflow server --host 0.0.0.0 --port 5000
             --backend-store-uri postgresql://mlflow:mlflow@postgres/mlflow
             --artifacts-destination s3://mlflow --serve-artifacts"
    ports: ["5000:5000"]
volumes: {pgdata: {}, minio: {}}
```
(Create the `mlflow` bucket in MinIO once; for production use real secrets, TLS and authentication.)

### A tracking decorator that enforces lineage

```python
import functools, hashlib, os, platform, subprocess, sys
import mlflow

def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"

def tracked(experiment: str, data_path: str | None = None, tags: dict | None = None):
    """Wrap a training function so every run records code/data/env lineage.
    The wrapped function returns (model, metrics: dict, params: dict)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=fn.__name__) as run:
                mlflow.set_tags({
                    "git.sha": _git("rev-parse", "HEAD"),
                    "git.dirty": str(bool(_git("status", "--porcelain"))),
                    "python": platform.python_version(),
                    "image": os.getenv("IMAGE_DIGEST", "local"),
                    **(tags or {}),
                })
                if data_path:
                    h = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
                    mlflow.set_tag("data.sha256", h)
                model, metrics, params = fn(*a, **kw)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_text(subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"], text=True), "env/requirements.txt")
                return run.info.run_id, model
        return wrapper
    return deco
```

Usage with signature + input example (required for safe serving):
```python
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

@tracked("fraud-detection", data_path="data/train.csv", tags={"owner": "risk-ml"})
def train(X_tr, y_tr, X_va, y_va, lr=0.05, n=400):
    m = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=7).fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
    sig = infer_signature(X_va, m.predict_proba(X_va))
    mlflow.sklearn.log_model(m, "model", signature=sig, input_example=X_va.head(3))
    return m, {"val_auc": auc}, {"lr": lr, "n_estimators": n}
```

### Querying and comparing runs
```python
import mlflow
runs = mlflow.search_runs(
    experiment_names=["fraud-detection"],
    filter_string="metrics.val_auc > 0.9 and tags.`git.dirty` = 'False'",
    order_by=["metrics.val_auc DESC"], max_results=10)
best = runs.iloc[0]
print(best.run_id, best["metrics.val_auc"])
```

### Metric logging patterns
- **Step-wise** curves: `mlflow.log_metric("loss", v, step=epoch)`.
- **Slice metrics**: `val_auc.region=eu`, naming convention allows dashboards by slice.
- **Large artifacts**: log once; reference by URI. Do not log 1 GB checkpoints every epoch.
- **System metrics**: `mlflow.enable_system_metrics_logging()` for GPU/CPU/mem.
- **Autolog with care**: `mlflow.autolog()` is convenient but logs many params; pin what you consider contract.

## 3. Weights & Biases

W&B shines for rich visualisation, sweeps and team collaboration; MLflow shines for self-hosting and registry integration.

| Capability | MLflow | W&B |
|---|---|---|
| Hosting | Self-host OSS (or managed) | SaaS / dedicated / self-managed |
| Experiment UI | Good | Excellent (panels, media, comparisons) |
| Hyperparameter sweeps | Via Optuna/Hyperopt integration | Native sweeps (Bayesian, early-stop) |
| Registry | Built-in | W&B Registry/Artifacts |
| Cost | Infra + ops | Seat/usage based |
| Data residency | Full control | Plan dependent |

```python
import wandb

sweep_cfg = {
    "method": "bayes",
    "metric": {"name": "val_auc", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        "max_depth": {"values": [4, 6, 8, 10]},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

def run():
    with wandb.init(project="fraud") as r:
        cfg = r.config
        auc = train_and_eval(lr=cfg.lr, max_depth=cfg.max_depth)   # your function
        wandb.log({"val_auc": auc})
        art = wandb.Artifact("fraud-model", type="model")
        art.add_file("artifacts/model.joblib")
        r.log_artifact(art)

sweep_id = wandb.sweep(sweep_cfg, project="fraud")
wandb.agent(sweep_id, function=run, count=30)
```

## 4. DVC: data & pipeline versioning

Git cannot store large data. DVC stores small **pointer files** (`.dvc`, `dvc.lock`) in Git and the data in remote storage, content-addressed by hash.

```
Git repo                      DVC remote (S3/GCS/Azure/SSH)
├─ data/raw.csv.dvc  ──hash──▶  ab/cdef...  (the actual bytes)
├─ dvc.yaml  (pipeline)
├─ dvc.lock  (hashes of deps/outs per stage)
└─ params.yaml
```

```bash
dvc init
dvc remote add -d storage s3://my-bucket/dvcstore
dvc add data/raw/train.csv          # creates data/raw/train.csv.dvc + .gitignore entry
git add data/raw/train.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "data: add train v1"
dvc push                            # upload bytes

# time travel to reproduce an old model
git checkout <old-sha> && dvc checkout   # restores matching data

# compare experiments
dvc params diff main
dvc metrics diff main
dvc exp run -S train.max_depth=12        # lightweight experiment, no commit needed
dvc exp show
```

Linking DVC to MLflow so a run points to exact data:
```python
import subprocess, mlflow
rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
lock = open("dvc.lock").read()
mlflow.set_tag("dvc.git_rev", rev)
mlflow.log_text(lock, "lineage/dvc.lock")
```

### Choosing a versioning approach

| Need | DVC | lakeFS | Delta/Iceberg time travel |
|---|---|---|---|
| Small–mid files in Git workflow | ✅ | ○ | ○ |
| Petabyte lake, branching | ○ | ✅ | ✅ |
| Table-level snapshot, SQL | ○ | ○ | ✅ |
| Pipeline DAG with caching | ✅ | ✗ | ✗ |

## 5. Reproducibility checklist

1. **Pin dependencies** with a lock file (`uv.lock`, `pip-compile`, conda-lock) and build from a **container image digest** — not `latest`.
2. **Seed everything**: Python `random`, NumPy, framework (`torch.manual_seed`, `torch.use_deterministic_algorithms(True)` when feasible), data loaders' worker seeds.
3. **Version data** (DVC rev / table snapshot id) and record row counts and hashes.
4. **One entrypoint** `python -m project.train --config configs/x.yaml`; no manual notebook steps.
5. **Config as code**: typed, committed; log the *resolved* config.
6. **Record hardware**: GPU model, driver, CUDA, cuDNN; nondeterministic kernels can alter results.
7. **Re-run test**: CI nightly reruns a small config and asserts metric within tolerance.

```python
import os, random, numpy as np, torch

def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
```

## 6. Edge cases & production failures

| Failure | Root cause | Impact | Fix |
|---|---|---|---|
| **"Best" run unreproducible** | Dirty working tree / uncommitted notebook change | Cannot rebuild champion | Block registration when `git.dirty=True`; CI-only promotion |
| **Tracking server DB bloat** | Per-step metrics at high frequency, millions of rows | Slow UI/queries | Log every N steps, aggregate; partition/retention; move to Postgres |
| **Artifact store permission drift** | Training job role lacks `s3:PutObject` after IAM change | Runs "succeed" but model missing | Verify artifact existence at end of run; fail loudly |
| **Metric cherry-picking** | Selecting best test score among 200 runs | Optimistic estimate (leakage through selection) | Select on validation; final test touched once; report multiple-comparison caveat |
| **Pickle incompatibility** | Train sklearn 1.3, serve 1.5 | Load error or silent change | Log env + conda.yaml; serve using MLflow `pyfunc` in same image; use ONNX for portability |
| **Data changed under same path** | Overwritten file in S3 | Old runs referencing path no longer reproducible | Immutable content-addressed storage; version in path |
| **Non-determinism** | Multi-threaded float reductions, GPU atomics | ΔAUC ~0.002 between runs | Track variance across seeds; compare with CIs not point estimates |

## 7. Senior interview questions & answers

**Q1. How would you make sure we can reproduce a model that's been in production for a year?**
- ❌ *Trap:* "We saved the pickle file."
- ✅ *Staff:* The registry version links to a run which records git SHA (clean tree), data version (DVC rev / table snapshot), resolved config, container image digest and dependency lock, plus seeds and hardware. DVC remote and image registry have retention policies at least as long as model retention. I validate this quarterly by *actually* re-running a registered model's pipeline in a sandbox and comparing metrics to tolerance.

**Q2. MLflow vs W&B — choose for a regulated bank.**
- ❌ *Trap:* "W&B has better charts, so W&B."
- ✅ *Staff:* Data residency, audit and air-gapped requirements favour self-hosted MLflow (plus its registry) behind SSO, with Postgres and object storage we control. W&B self-managed is possible, but weigh licence, ops cost and integration. I'd give researchers a good UI via MLflow plus dashboards, and standardise on the registry that the CD system reads.

**Q3. 5,000 hyperparameter trials swamp the tracking server. What do you do?**
- ❌ *Trap:* "Buy a bigger server."
- ✅ *Staff:* Reduce write amplification: log metrics every N steps or batch via `log_batch`; parent/child runs with only summaries on the parent; disable autolog of large artifacts; use early stopping/pruning (Optuna, Hyperband) so fewer trials run to completion; Postgres with indexes and retention policy; consider async logging. Then scale the tracking server horizontally behind a load balancer since it's stateless.

**Q4. What's the difference between experiment tracking and the model registry?**
- ❌ *Trap:* "Same thing."
- ✅ *Staff:* Tracking is an append-only log of *all* attempts for analysis; the registry is a curated, governed list of *candidate-for-deployment* models with versions, aliases, approval state and consumer-facing names. Runs are cheap and numerous; registered versions are few and carry process (reviews, gates, audit).

**Q5. Why DVC rather than committing data to Git LFS?**
- ❌ *Trap:* "DVC is open source."
- ✅ *Staff:* DVC adds pipeline DAG semantics with caching (`dvc repro` runs only affected stages), parameters and metrics diffing, experiments, and works with any object store without LFS server limits. LFS is only file storage. For very large lakes with branching I'd choose lakeFS or table formats with time travel instead.

**Q6. Your validation AUC rose from 0.88 to 0.89 after a code change. Ship it?**
- ❌ *Trap:* "Yes, it improved."
- ✅ *Staff:* Not enough evidence. Quantify seed variance (e.g., 5 seeds, bootstrap CIs); check per-slice metrics and calibration; ensure the validation set wasn't reused for dozens of tuning decisions; verify no leakage introduced by the change; evaluate on a time-forward holdout; consider the cost (latency/size). Ship if gain exceeds noise, no slice regresses beyond tolerance, and the shadow test agrees.

**Q7. How do you handle experiment tracking for notebooks and exploratory work without slowing researchers?**
- ❌ *Trap:* "Ban notebooks."
- ✅ *Staff:* Provide a thin library with autolog defaults and a notebook template that tags notebooks as `stage=exploration`. Exploration runs can't be registered; promotion requires converting to the pipeline entrypoint and passing CI. That preserves speed while keeping production lineage clean.

**Q8. How do you store and compare very large model artifacts (10+ GB)?**
- ❌ *Trap:* "Log them every epoch to MLflow."
- ✅ *Staff:* Keep checkpoints in object storage with lifecycle rules (keep best-N + final), log URIs and checksums, use multipart upload, and store only the final/best model as a registry artifact. For comparisons rely on metrics/eval reports, not downloading weights. Consider deltas/LoRA adapters instead of full-model copies.

---
**Prev:** [← 02](02-data-engineering-feature-stores.md) · **Next:** [04 · Model Registry & Governance →](04-model-registry-governance.md)
