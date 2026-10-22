<div align="center">

<a href="https://mlops-interview-preparation.vercel.app"><img src="docs/screenshots/01-hero.jpg" alt="MLOps Interview Preparation: storefront-style home page" width="100%"></a>

# 🚀 MLOps Interview Preparation

**A zero-to-hero, production-grade MLOps curriculum: 22 modules, 255 concept episodes, runnable code, failure post-mortems, and staff-level interview Q&A.**

[![Live site](https://img.shields.io/badge/Open%20the%20site-live%20site-2874F0?style=for-the-badge&labelColor=172337)](https://mlops-interview-preparation.vercel.app)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
![Modules](https://img.shields.io/badge/modules-22-blueviolet)
![Episodes](https://img.shields.io/badge/concept%20episodes-255-E50914)

</div>

---

## 🛍️ The Storefront-Style Site

The whole course is also a zero-dependency web app (`site/`, plain HTML/CSS/JS): modules are **catalogue items** on shelves, sections are **episodes**, and revision feels like browsing a store (everything is free). The styling takes cues from a familiar blue-and-orange e-commerce layout: a search bar, a category strip, a banner, shelf carousels, a numbered Top 10, and a detail page per module. **[Open the live site](https://mlops-interview-preparation.vercel.app)**.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/02-rows-top10.jpg" alt="Shelves and Top 10 modules"><br><sub><b>Shelves and Top 10</b>: role-based "Top Picks" and a numbered Top 10, with Read now / List buttons on every card.</sub></td>
<td width="50%"><img src="docs/screenshots/05-detail-modal.jpg" alt="Module detail page with episodes"><br><sub><b>Detail page</b>: tools, genres and an episode list for every section.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/03-reader-diagram.jpg" alt="Reader with episode sidebar and decision-tree diagram"><br><sub><b>Reader</b>: light theme, all episodes in a sidebar, progress tracking, decision-tree diagrams inline.</sub></td>
<td width="50%" align="center"><img src="docs/screenshots/04-mobile.jpg" alt="Mobile layout" width="220"><br><sub><b>Mobile</b>: responsive, no horizontal scrolling.</sub></td>
</tr>
</table>

**Features:** interview profiles reorder the recommendations; the reader remembers where you stopped (a "Continue Reading" shelf appears); My List; search over all 255 episode titles; deep links such as `#/read/13/13.2`; and everything stays private in your browser's `localStorage`.

> The visual style is an homage to e-commerce storefronts. It uses no copied logos, ratings, reviews or prices, and this project is not affiliated with or endorsed by Flipkart.

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
| [01 Foundations](chapters/01_mlops_foundations_system_design.md) | [02 Data & Features](chapters/02_data_engineering_feature_stores.md) | data lifecycle |
| [01 Foundations](chapters/01_mlops_foundations_system_design.md) | [03 Experiments](chapters/03_experiment_tracking_model_versioning.md) | reproducibility |
| [01 Foundations](chapters/01_mlops_foundations_system_design.md) | [05 CI/CD/CT](chapters/05_cicd_ct_automation_pipelines.md) | maturity levels |
| [01 Foundations](chapters/01_mlops_foundations_system_design.md) | [08 Monitoring](chapters/08_monitoring_observability_alerting.md) | feedback loops |
| [02 Data & Features](chapters/02_data_engineering_feature_stores.md) | [03 Experiments](chapters/03_experiment_tracking_model_versioning.md) | versioned data |
| [02 Data & Features](chapters/02_data_engineering_feature_stores.md) | [08 Monitoring](chapters/08_monitoring_observability_alerting.md) | drift baselines |
| [02 Data & Features](chapters/02_data_engineering_feature_stores.md) | [11 Failures](chapters/11_production_failures_troubleshooting.md) | skew incidents |
| [03 Experiments](chapters/03_experiment_tracking_model_versioning.md) | [04 Registry](chapters/04_model_registry_governance.md) | runs become versions |
| [04 Registry](chapters/04_model_registry_governance.md) | [05 CI/CD/CT](chapters/05_cicd_ct_automation_pipelines.md) | gates + promotion |
| [04 Registry](chapters/04_model_registry_governance.md) | [10 Security](chapters/10_security_privacy_compliance.md) | RBAC + approvals |
| [05 CI/CD/CT](chapters/05_cicd_ct_automation_pipelines.md) | [06 Serving](chapters/06_model_serving_architecture.md) | builds images |
| [05 CI/CD/CT](chapters/05_cicd_ct_automation_pipelines.md) | [07 Deployment](chapters/07_deployment_strategies_traffic_routing.md) | progressive delivery |
| [08 Monitoring](chapters/08_monitoring_observability_alerting.md) | [05 CI/CD/CT](chapters/05_cicd_ct_automation_pipelines.md) | retrain trigger |
| [06 Serving](chapters/06_model_serving_architecture.md) | [07 Deployment](chapters/07_deployment_strategies_traffic_routing.md) | versions to route |
| [06 Serving](chapters/06_model_serving_architecture.md) | [08 Monitoring](chapters/08_monitoring_observability_alerting.md) | metrics + logs |
| [09 Infra & K8s](chapters/09_infrastructure_containerization_orchestration.md) | [06 Serving](chapters/06_model_serving_architecture.md) | hosts serving |
| [07 Deployment](chapters/07_deployment_strategies_traffic_routing.md) | [08 Monitoring](chapters/08_monitoring_observability_alerting.md) | canary analysis |
| [07 Deployment](chapters/07_deployment_strategies_traffic_routing.md) | [11 Failures](chapters/11_production_failures_troubleshooting.md) | rollback |
| [09 Infra & K8s](chapters/09_infrastructure_containerization_orchestration.md) | [11 Failures](chapters/11_production_failures_troubleshooting.md) | OOM, cold start |
| [10 Security](chapters/10_security_privacy_compliance.md) | [06 Serving](chapters/06_model_serving_architecture.md) | endpoint hardening |
| [10 Security](chapters/10_security_privacy_compliance.md) | [09 Infra & K8s](chapters/09_infrastructure_containerization_orchestration.md) | pod + supply-chain security |
| [11 Failures](chapters/11_production_failures_troubleshooting.md) | [12 Interview Kit](chapters/12_mlops_interview_cheatsheet_system_design.md) | interview stories |

</details>

## 📅 Prefer a Guided Month?

Follow the **[30-Day Course](COURSE-30-DAY.md)**: daily reading, hands-on lab, interview drill and tip, weekly checkpoints, a final project and a self-assessment grid.

<details>
<summary><b>Decision trees for every module</b></summary>

### 01 · [Foundations](chapters/01_mlops_foundations_system_design.md)
<img src="assets/tree_01_decisions.svg" alt="Decision tree for Foundations" width="100%"/>

### 02 · [Data & Features](chapters/02_data_engineering_feature_stores.md)
<img src="assets/tree_02_decisions.svg" alt="Decision tree for Data & Features" width="100%"/>

### 03 · [Experiments](chapters/03_experiment_tracking_model_versioning.md)
<img src="assets/tree_03_decisions.svg" alt="Decision tree for Experiments" width="100%"/>

### 04 · [Registry](chapters/04_model_registry_governance.md)
<img src="assets/tree_04_decisions.svg" alt="Decision tree for Registry" width="100%"/>

### 05 · [CI/CD/CT](chapters/05_cicd_ct_automation_pipelines.md)
<img src="assets/tree_05_decisions.svg" alt="Decision tree for CI/CD/CT" width="100%"/>

### 06 · [Serving](chapters/06_model_serving_architecture.md)
<img src="assets/tree_06_decisions.svg" alt="Decision tree for Serving" width="100%"/>

### 07 · [Deployment](chapters/07_deployment_strategies_traffic_routing.md)
<img src="assets/tree_07_decisions.svg" alt="Decision tree for Deployment" width="100%"/>

### 08 · [Monitoring](chapters/08_monitoring_observability_alerting.md)
<img src="assets/tree_08_decisions.svg" alt="Decision tree for Monitoring" width="100%"/>

### 09 · [Infra & K8s](chapters/09_infrastructure_containerization_orchestration.md)
<img src="assets/tree_09_decisions.svg" alt="Decision tree for Infra & K8s" width="100%"/>

### 10 · [Security](chapters/10_security_privacy_compliance.md)
<img src="assets/tree_10_decisions.svg" alt="Decision tree for Security" width="100%"/>

### 11 · [Failures](chapters/11_production_failures_troubleshooting.md)
<img src="assets/tree_11_decisions.svg" alt="Decision tree for Failures" width="100%"/>

### 12 · [Interview Kit](chapters/12_mlops_interview_cheatsheet_system_design.md)
<img src="assets/tree_12_decisions.svg" alt="Decision tree for Interview Kit" width="100%"/>

</details>

---

## 📚 Curriculum Navigation

| # | Module | Target tools | Key concepts |
|---|---|---|---|
| 01 | [MLOps Foundations & System Design](chapters/01_mlops_foundations_system_design.md) | Git, DVC, Python | Code/Data/Model lifecycles, technical debt (CACE, hidden feedback loops), maturity levels 0–2 |
| 02 | [Data Engineering & Feature Stores](chapters/02_data_engineering_feature_stores.md) | Feast, Great Expectations, Pydantic, Evidently | Offline vs online store, point-in-time joins, schema/data drift, lineage |
| 03 | [Experiment Tracking & Model Versioning](chapters/03_experiment_tracking_model_versioning.md) | MLflow, Weights & Biases, DVC | Reproducibility, metric/artifact logging, data & pipeline versioning |
| 04 | [Model Registry & Governance](chapters/04_model_registry_governance.md) | MLflow Registry, GitHub environments | Aliases vs stages, promotion gates, model cards, audit trails, compliance |
| 05 | [CI/CD/CT Automation Pipelines](chapters/05_cicd_ct_automation_pipelines.md) | GitHub Actions, Argo Workflows, Kubeflow Pipelines | Testing ML, continuous training, drift-triggered retraining, GitOps |
| 06 | [Model Serving Architecture](chapters/06_model_serving_architecture.md) | FastAPI, Triton, TorchServe, Kafka, Ray | Online/batch/stream serving, dynamic batching, gRPC vs REST |
| 07 | [Deployment Strategies & Traffic Routing](chapters/07_deployment_strategies_traffic_routing.md) | Istio, Envoy, Argo Rollouts | Blue/green, canary, shadow, A/B testing, rollback |
| 08 | [Monitoring, Observability & Alerting](chapters/08_monitoring_observability_alerting.md) | Prometheus, Grafana, Evidently AI, Alertmanager | Data/concept/prior-shift drift, SLOs, delayed-label quality, alert design |
| 09 | [Infrastructure: Containers & Orchestration](chapters/09_infrastructure_containerization_orchestration.md) | Docker, Kubernetes, KEDA, Ray, Spark | Multi-stage images, probes, HPA, Ingress, GPU scheduling, distributed compute |
| 10 | [Security, Privacy & Compliance](chapters/10_security_privacy_compliance.md) | OIDC/JWT, Istio, Kong, Opacus, Flower | Poisoning, extraction, DP, federated learning, RBAC, gateway authN |
| 11 | [Production Failures & Troubleshooting](chapters/11_production_failures_troubleshooting.md) | py-spy, memray, DCGM, circuit breakers | Memory leaks, OOM, cold starts, deadlocks, feature inconsistency, fallbacks |
| 12 | [Interview Cheatsheet & System Design](chapters/12_mlops_interview_cheatsheet_system_design.md) | — | Trade-off frameworks, design templates, right-vs-wrong patterns, rapid-fire Q&A |
| 13 | [Testing for ML Systems](chapters/13_testing_for_ml_systems.md) | pytest, invariance and slice tests | Data, model, behavioural and slice tests; golden sets; making silent failures loud before release |
| 14 | [Streaming & Real-Time ML](chapters/14_streaming_realtime_ml.md) | Kafka, Flink | Windows, watermarks, train/serve parity for streaming features, online learning |
| 15 | [Distributed Training, GPU Scheduling & Tuning](chapters/15_distributed_training_gpu_scheduling_tuning.md) | FSDP/ZeRO, Kueue | Parallelism, checkpointing and fault tolerance, queueing and fair share, hyperparameter search, throughput diagnosis |
| 16 | [ML Platform Engineering & FinOps](chapters/16_ml_platform_engineering_finops.md) | Golden paths, policy-as-code | Multi-tenancy, platform as a product, unit economics, cost attribution |
| 17 | [Responsible ML in Practice](chapters/17_responsible_ml_in_practice.md) | SHAP, model cards | Fairness metrics and trade-offs, explainability, calibration, governance in practice |
| 18 | [Domain Playbooks](chapters/18_domain_playbooks.md) | Two-tower, fraud, backtesting | Recommenders, fraud, forecasting, vision/NLP and decision-centric ML in production |
| 19 | [Bridge to LLMOps](chapters/19_bridge_to_llmops.md) | Prompt registry, eval gates | What carries over from MLOps to LLM applications and what changes |
| 20 | [Production Projects & Labs Blueprint](chapters/20_production_projects_labs_blueprint.md) | Pipeline, feature store, CI/CD/CT | Five portfolio-grade builds with acceptance tests and failure drills |
| 21 | [Learning Roadmap & Interview Study Plan](chapters/21_learning_roadmap_interview_study_plan.md) | Rubrics, papers, books | Skill ladder, 72-hour to 8-week plans, curated reading list (complements the 30-day course) |
| 22 | [Mock Interview & System-Design Question Bank](chapters/22_mock_interview_system_design_bank.md) | Fraud, platform and incident designs | Worked designs, incident drills, behavioural frames with scoring |

**Modules 13-22 and the extension sections (numbered `N.15` and up in modules 01-12)** use a five-part *concept* format: **Conceptual Deep Dive**, **Cross-Connection Scenarios**, **Production Anti-Patterns**, **Minimal Working Code** (two runnable blocks) and **Interview Angle** (questions with concise model answers stating a trade-off and a measurable check).

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
4. **Close with [Module 12](chapters/12_mlops_interview_cheatsheet_system_design.md)** for frameworks and rapid-fire review.

## ✅ Quality and Verification

This edition was produced with a **plan, write, audit** loop: a lead agent audited the repo against a total MLOps concept map, writer agents wrote the new material from precise outlines, and a fixer corrected defects in the original modules.

| Check | How | Result |
|---|---|---|
| Structure | `scripts/audit.py`: every `N.M` section has the five parts and a code block; required concepts present | 0 issues |
| Python syntax | every ```python block parsed with `ast` | 0 errors |
| Runtime | stdlib/numpy sections executed in a clean directory (`scripts/check_snippets.py`) | 80 sections ran, 0 failed, 32 skipped (third-party) |
| Cross-references | `scripts/check_xrefs.py` validates section references (sensitivity confirmed with planted errors) | 615 checked, 0 broken |
| PDFs | one per module from the Markdown source | 22 / 22 |

**Honest limits** (details in [`CROSSCHECK.md`](CROSSCHECK.md)):

- Snippets using third-party libraries (MLflow, Feast, Evidently, Kubeflow, Argo, and similar) are syntax-checked but not executed; flags and APIs drift between versions.
- The PDFs do not embed the SVG decision and connection diagrams (the PDF library cannot render SVG); the diagrams appear on GitHub and on the site.
- Manifests in CI/CD examples contain explicit `REPLACE_WITH_...` placeholders where a real checksum or commit SHA cannot be known offline.
- Perishable facts (versions, prices, regulation details, feature lists) are generic or marked "verify against current docs"; fairness and regulatory content is orientation, not legal advice.
- The audit is structural; it cannot prove that an explanation is correct.

## 🤝 Contributing

Corrections and missing edge cases are welcome. Edit the module in `chapters/NN_*.md`, or add a section as `extensions/NNb.md` and run `python scripts/merge_extensions.py`, keep the five-part format for numbered sections, then:

```bash
pip install -r requirements.txt
python scripts/audit.py && python scripts/check_xrefs.py && python scripts/check_snippets.py
python scripts/regen_toc.py && python scripts/build_pdfs.py && python scripts/build_site_data.py
```

## 📁 Repository Layout
```
mlops-interview-preparation/
├── README.md
├── CROSSCHECK.md          # audit report: what changed, what is still missing
├── COURSE-30-DAY.md       # guided 30-day course
├── LICENSE
├── assets/                # SVG architecture, decision and connection diagrams
├── chapters/              # modules 01-22 (Markdown, source of truth)
├── extensions/            # expansion sections (merged into chapters)
├── pdfs/                  # one PDF per module
├── scripts/               # audit, xref check, snippet runner, TOC, PDF + site builders
├── site/                  # the streaming-style web app (deployed on Vercel)
└── docs/screenshots/      # images used in this README
```

## ⚖️ License & Originality
All content is original and written from first principles. Code snippets use open-source tooling; verify versions and adapt security settings (secrets, TLS, credentials shown here are local-only placeholders) before any production use.
