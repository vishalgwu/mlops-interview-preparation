<div align="center">

# 🚀 MLOps Interview Preparation

**A zero-to-hero, production-grade MLOps curriculum — theory, runnable code, failure post-mortems, and staff-level interview Q&A.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-manifests-326CE5?logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%26%20registry-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-945DD6?logo=dvc&logoColor=white)
![Feast](https://img.shields.io/badge/Feast-feature%20store-FF6F00)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD%2FCT-2088FF?logo=githubactions&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Modules](https://img.shields.io/badge/modules-12-blueviolet)

</div>

---

## 🗺️ End-to-End System Map

<p align="center">
  <img src="assets/mlops_end_to_end_architecture.svg" alt="End-to-end MLOps architecture: raw data, validation, feature store, training with DVC, MLflow registry, CI/CD, Kubernetes/Triton deployment, Prometheus/Evidently monitoring, retraining triggers" width="100%"/>
</p>

Every module zooms into one part of this loop. Each includes: **first-principles theory → executable code & manifests → real failure modes → 5–8 scenario questions with a "Trap Answer" and a "Staff-Level Answer" → diagrams.**

## 🌳 How the Modules Connect

<p align="center">
  <img src="assets/tree_00_master_map.svg" alt="Master tree showing how the 12 modules feed into each other" width="100%"/>
</p>

Each module opens with a **decision tree** (which component, and when?) and a **connection map** (what feeds it, what it feeds). The pink arc is the continuous-training loop: Monitoring (08) triggers CI/CD/CT (05).

<details>
<summary><b>All module connections as a table</b></summary>

| From | To | What flows |
|---|---|---|
| [01 Foundations](01-mlops-foundations-system-design.md) | [02 Data & Features](02-data-engineering-feature-stores.md) | data lifecycle |
| [01 Foundations](01-mlops-foundations-system-design.md) | [03 Experiments](03-experiment-tracking-model-versioning.md) | reproducibility |
| [01 Foundations](01-mlops-foundations-system-design.md) | [05 CI/CD/CT](05-cicd-ct-automation-pipelines.md) | maturity levels |
| [01 Foundations](01-mlops-foundations-system-design.md) | [08 Monitoring](08-monitoring-observability-alerting.md) | feedback loops |
| [02 Data & Features](02-data-engineering-feature-stores.md) | [03 Experiments](03-experiment-tracking-model-versioning.md) | versioned data |
| [02 Data & Features](02-data-engineering-feature-stores.md) | [08 Monitoring](08-monitoring-observability-alerting.md) | drift baselines |
| [02 Data & Features](02-data-engineering-feature-stores.md) | [11 Failures](11-production-failures-troubleshooting.md) | skew incidents |
| [03 Experiments](03-experiment-tracking-model-versioning.md) | [04 Registry](04-model-registry-governance.md) | runs become versions |
| [04 Registry](04-model-registry-governance.md) | [05 CI/CD/CT](05-cicd-ct-automation-pipelines.md) | gates + promotion |
| [04 Registry](04-model-registry-governance.md) | [10 Security](10-security-privacy-compliance.md) | RBAC + approvals |
| [05 CI/CD/CT](05-cicd-ct-automation-pipelines.md) | [06 Serving](06-model-serving-architecture.md) | builds images |
| [05 CI/CD/CT](05-cicd-ct-automation-pipelines.md) | [07 Deployment](07-deployment-strategies-traffic-routing.md) | progressive delivery |
| [08 Monitoring](08-monitoring-observability-alerting.md) | [05 CI/CD/CT](05-cicd-ct-automation-pipelines.md) | retrain trigger |
| [06 Serving](06-model-serving-architecture.md) | [07 Deployment](07-deployment-strategies-traffic-routing.md) | versions to route |
| [06 Serving](06-model-serving-architecture.md) | [08 Monitoring](08-monitoring-observability-alerting.md) | metrics + logs |
| [09 Infra & K8s](09-infrastructure-containerization-orchestration.md) | [06 Serving](06-model-serving-architecture.md) | hosts serving |
| [07 Deployment](07-deployment-strategies-traffic-routing.md) | [08 Monitoring](08-monitoring-observability-alerting.md) | canary analysis |
| [07 Deployment](07-deployment-strategies-traffic-routing.md) | [11 Failures](11-production-failures-troubleshooting.md) | rollback |
| [09 Infra & K8s](09-infrastructure-containerization-orchestration.md) | [11 Failures](11-production-failures-troubleshooting.md) | OOM, cold start |
| [10 Security](10-security-privacy-compliance.md) | [06 Serving](06-model-serving-architecture.md) | endpoint hardening |
| [10 Security](10-security-privacy-compliance.md) | [09 Infra & K8s](09-infrastructure-containerization-orchestration.md) | pod + supply-chain security |
| [11 Failures](11-production-failures-troubleshooting.md) | [12 Interview Kit](12-mlops-interview-cheatsheet-system-design.md) | interview stories |

</details>

## 📅 Prefer a Guided Month?

Follow the **[30-Day Course](COURSE-30-DAY.md)**: daily reading, hands-on lab, interview drill and tip, weekly checkpoints, a final project and a self-assessment grid.

<details>
<summary><b>Decision trees for every module</b></summary>

### 01 · [Foundations](01-mlops-foundations-system-design.md)
<img src="assets/tree_01_decisions.svg" alt="Decision tree for Foundations" width="100%"/>

### 02 · [Data & Features](02-data-engineering-feature-stores.md)
<img src="assets/tree_02_decisions.svg" alt="Decision tree for Data & Features" width="100%"/>

### 03 · [Experiments](03-experiment-tracking-model-versioning.md)
<img src="assets/tree_03_decisions.svg" alt="Decision tree for Experiments" width="100%"/>

### 04 · [Registry](04-model-registry-governance.md)
<img src="assets/tree_04_decisions.svg" alt="Decision tree for Registry" width="100%"/>

### 05 · [CI/CD/CT](05-cicd-ct-automation-pipelines.md)
<img src="assets/tree_05_decisions.svg" alt="Decision tree for CI/CD/CT" width="100%"/>

### 06 · [Serving](06-model-serving-architecture.md)
<img src="assets/tree_06_decisions.svg" alt="Decision tree for Serving" width="100%"/>

### 07 · [Deployment](07-deployment-strategies-traffic-routing.md)
<img src="assets/tree_07_decisions.svg" alt="Decision tree for Deployment" width="100%"/>

### 08 · [Monitoring](08-monitoring-observability-alerting.md)
<img src="assets/tree_08_decisions.svg" alt="Decision tree for Monitoring" width="100%"/>

### 09 · [Infra & K8s](09-infrastructure-containerization-orchestration.md)
<img src="assets/tree_09_decisions.svg" alt="Decision tree for Infra & K8s" width="100%"/>

### 10 · [Security](10-security-privacy-compliance.md)
<img src="assets/tree_10_decisions.svg" alt="Decision tree for Security" width="100%"/>

### 11 · [Failures](11-production-failures-troubleshooting.md)
<img src="assets/tree_11_decisions.svg" alt="Decision tree for Failures" width="100%"/>

### 12 · [Interview Kit](12-mlops-interview-cheatsheet-system-design.md)
<img src="assets/tree_12_decisions.svg" alt="Decision tree for Interview Kit" width="100%"/>

</details>

---

## 📚 Curriculum Navigation

| # | Module | Target tools | Key concepts |
|---|---|---|---|
| 01 | [MLOps Foundations & System Design](01-mlops-foundations-system-design.md) | Git, DVC, Python | Code/Data/Model lifecycles, technical debt (CACE, hidden feedback loops), maturity levels 0–2 |
| 02 | [Data Engineering & Feature Stores](02-data-engineering-feature-stores.md) | Feast, Great Expectations, Pydantic, Evidently | Offline vs online store, point-in-time joins, schema/data drift, lineage |
| 03 | [Experiment Tracking & Model Versioning](03-experiment-tracking-model-versioning.md) | MLflow, Weights & Biases, DVC | Reproducibility, metric/artifact logging, data & pipeline versioning |
| 04 | [Model Registry & Governance](04-model-registry-governance.md) | MLflow Registry, GitHub environments | Aliases vs stages, promotion gates, model cards, audit trails, compliance |
| 05 | [CI/CD/CT Automation Pipelines](05-cicd-ct-automation-pipelines.md) | GitHub Actions, Argo Workflows, Kubeflow Pipelines | Testing ML, continuous training, drift-triggered retraining, GitOps |
| 06 | [Model Serving Architecture](06-model-serving-architecture.md) | FastAPI, Triton, TorchServe, Kafka, Ray | Online/batch/stream serving, dynamic batching, gRPC vs REST |
| 07 | [Deployment Strategies & Traffic Routing](07-deployment-strategies-traffic-routing.md) | Istio, Envoy, Argo Rollouts | Blue/green, canary, shadow, A/B testing, rollback |
| 08 | [Monitoring, Observability & Alerting](08-monitoring-observability-alerting.md) | Prometheus, Grafana, Evidently AI, Alertmanager | Data/concept/prior-shift drift, SLOs, delayed-label quality, alert design |
| 09 | [Infrastructure: Containers & Orchestration](09-infrastructure-containerization-orchestration.md) | Docker, Kubernetes, KEDA, Ray, Spark | Multi-stage images, probes, HPA, Ingress, GPU scheduling, distributed compute |
| 10 | [Security, Privacy & Compliance](10-security-privacy-compliance.md) | OIDC/JWT, Istio, Kong, Opacus, Flower | Poisoning, extraction, DP, federated learning, RBAC, gateway authN |
| 11 | [Production Failures & Troubleshooting](11-production-failures-troubleshooting.md) | py-spy, memray, DCGM, circuit breakers | Memory leaks, OOM, cold starts, deadlocks, feature inconsistency, fallbacks |
| 12 | [Interview Cheatsheet & System Design](12-mlops-interview-cheatsheet-system-design.md) | — | Trade-off frameworks, design templates, right-vs-wrong patterns, rapid-fire Q&A |

---

## 🗓️ Study Roadmaps

> Want the full guided version? See the **[30-Day Course](COURSE-30-DAY.md)**.

### ⚡ 3-Day Crash Course (interview in a week)
| Day | Focus | Do |
|---|---|---|
| **1** | Lifecycle & data | Read **01** (all), **02** §2–4, **03** §1–2; draw the system map from memory |
| **2** | Ship & run | Read **05** §1–3, **06** §1–2, **07** §1–3, **08** §1–4; deploy the FastAPI service locally |
| **3** | Fail & design | Read **11** §1–7, **12** (all); rehearse two system designs out loud; answer every ❌/✅ question in 01, 07, 08, 11 |

### 📘 7-Day Deep Dive
| Day | Modules | Hands-on |
|---|---|---|
| 1 | **01 + 02** | Run the DVC pipeline skeleton; build a drift report with `psi()` |
| 2 | **03 + 04** | Start MLflow via Docker Compose; log runs, register a model, move an alias |
| 3 | **05** | Add the CI workflow; write behavioural model tests |
| 4 | **06 + 07** | Run FastAPI + Prometheus metrics; simulate a canary with two Services |
| 5 | **08 + 09** | Grafana dashboard + alert rule; deploy to Minikube with probes and HPA |
| 6 | **10 + 11** | Add JWT auth + rate limit; inject a memory leak and find it with `tracemalloc` |
| 7 | **12** | Mock interviews using the scenarios; review all trap answers |

### 🏆 14-Day Mastery
| Days | Focus | Outcome |
|---|---|---|
| 1–2 | Module **01**; implement the full Level-1 pipeline (validate → train → evaluate → register) | Reproducible pipeline you can explain |
| 3–4 | Module **02**; Feast with Redis, PIT joins, GX validation, skew-detection job | Feature platform with parity tests |
| 5 | Module **03** | W&B sweep + MLflow tracking decorator + DVC remote |
| 6 | Module **04** | Promotion script with gates + protected-environment approval |
| 7 | Module **05**; Argo CronWorkflow on Minikube | Working CT loop |
| 8–9 | Modules **06 + 07**; Triton ONNX model, Istio canary + shadow | Measured latency/throughput curves |
| 10 | Module **08**; Evidently job → Pushgateway → Grafana → Alertmanager → retrain webhook | Closed drift→retrain loop |
| 11 | Module **09**; KEDA, GPU scheduling concepts, Ray Tune | Autoscaling demo |
| 12 | Module **10**; DP-SGD experiment, signed artifacts, RBAC | Security checklist applied |
| 13 | Module **11**; chaos drills: kill feature store, OOM, deadlock | Fallback tiers proven in tests |
| 14 | Module **12**; 3 full mock system designs, time-boxed | Interview-ready |

---

## 🧪 Local Setup & Hands-On Environment

### Prerequisites
| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Code samples |
| Docker Desktop / Engine | 24+ | Containers, Compose |
| kubectl | 1.29+ | Cluster control |
| Minikube (or kind) | 1.33+ | Local Kubernetes |
| Helm | 3.14+ | Install Prometheus/Grafana/Argo |
| Git, DVC | latest | Versioning |

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "mlflow>=2.16" dvc[s3] feast[redis] great-expectations evidently \
            fastapi "uvicorn[standard]" prometheus-client scikit-learn pandas joblib \
            pytest ruff kfp opacus
```

### Option A — Docker Compose (fastest; Modules 02, 03, 06, 08)

Save as `docker-compose.yml` in a working directory:
```yaml
services:
  postgres:
    image: postgres:16
    environment: { POSTGRES_USER: mlflow, POSTGRES_PASSWORD: mlflow, POSTGRES_DB: mlflow }
    volumes: [pgdata:/var/lib/postgresql/data]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment: { MINIO_ROOT_USER: minio, MINIO_ROOT_PASSWORD: minio12345 }
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio:/data]

  minio-init:
    image: minio/mc
    depends_on: [minio]
    entrypoint: >
      sh -c "sleep 5 && mc alias set m http://minio:9000 minio minio12345 &&
             mc mb -p m/mlflow m/dvc || true"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    depends_on: [postgres, minio-init]
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

  redis:
    image: redis:7
    ports: ["6379:6379"]

  prometheus:
    image: prom/prometheus:v2.54.1
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml:ro"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:11.2.0
    environment: { GF_SECURITY_ADMIN_PASSWORD: admin }
    ports: ["3000:3000"]
    depends_on: [prometheus]

volumes: { pgdata: {}, minio: {} }
```
`prometheus.yml`
```yaml
global: { scrape_interval: 15s }
scrape_configs:
  - job_name: fraud-api
    static_configs: [{ targets: ["host.docker.internal:8080"] }]   # FastAPI on the host
```
Run and verify:
```bash
docker compose up -d
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 AWS_ACCESS_KEY_ID=minio AWS_SECRET_ACCESS_KEY=minio12345
python -c "import mlflow; mlflow.set_experiment('smoke'); \
  r=mlflow.start_run(); mlflow.log_metric('ok',1); mlflow.end_run(); print('MLflow OK')"
# start the FastAPI service from Module 06
uvicorn app.main:app --port 8080 &
curl -s localhost:8080/metrics | head
```
UIs: MLflow `:5000` · MinIO `:9001` · Prometheus `:9090` · Grafana `:3000` (admin/admin — local only).

### Option B — Minikube (Modules 05, 07, 09, 11)
```bash
minikube start --cpus=4 --memory=8192 --driver=docker
minikube addons enable ingress
minikube addons enable metrics-server        # required for HPA

# Build the image inside the cluster's Docker daemon
eval $(minikube docker-env)                  # PowerShell: & minikube -p minikube docker-env | Invoke-Expression
docker build -t fraud-api:dev .

kubectl create namespace mlops
kubectl apply -n mlops -f deploy/k8s/        # Deployment, Service, HPA, PDB from Module 09
kubectl -n mlops rollout status deploy/fraud-api

# Observability stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Progressive delivery & CT (optional)
helm repo add argo https://argoproj.github.io/argo-helm
helm install argo-workflows argo/argo-workflows -n argo --create-namespace
helm install argo-rollouts  argo/argo-rollouts  -n argo-rollouts --create-namespace

# Access
kubectl -n mlops port-forward svc/fraud-api 8080:80
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
```
Load-test and watch the HPA:
```bash
pip install locust
locust -f loadtest.py --headless -u 200 -r 20 -t 5m --host http://localhost:8080 &
kubectl -n mlops get hpa -w
```
Clean up: `docker compose down -v` · `minikube delete`.

> **Resource note:** the Compose stack needs ~3 GB RAM; Minikube with monitoring needs ~8 GB. On smaller machines use `kind` and skip Argo/Istio initially.

---

## 🧭 How to Use This Repo

1. **Read** a module's theory and tables, then **run** the code. Understanding comes from breaking things.
2. **Cover the ✅ answers**, answer each interview question aloud, then compare.
3. **Study the ❌ traps:** knowing why a plausible answer fails is what separates senior from junior responses.
4. **Close with [Module 12](12-mlops-interview-cheatsheet-system-design.md)** for frameworks and rapid-fire review.

## 📁 Repository Layout
```
mlops-interview-preparation/
├── README.md
├── COURSE-30-DAY.md
├── assets/
│   ├── mlops_end_to_end_architecture.svg
│   ├── tree_00_master_map.svg
│   └── tree_NN_decisions.svg / tree_NN_connections.svg   (one pair per module)
├── 01-mlops-foundations-system-design.md
├── 02-data-engineering-feature-stores.md
├── 03-experiment-tracking-model-versioning.md
├── 04-model-registry-governance.md
├── 05-cicd-ct-automation-pipelines.md
├── 06-model-serving-architecture.md
├── 07-deployment-strategies-traffic-routing.md
├── 08-monitoring-observability-alerting.md
├── 09-infrastructure-containerization-orchestration.md
├── 10-security-privacy-compliance.md
├── 11-production-failures-troubleshooting.md
└── 12-mlops-interview-cheatsheet-system-design.md
```

## ⚖️ License & Originality
All content is original and written from first principles. Code snippets use open-source tooling; verify versions and adapt security settings (secrets, TLS, credentials shown here are local-only placeholders) before any production use.
