# 📅 30-Day MLOps Course

A one-month, ~2 hours/day plan that turns the 12 modules into a guided course: **learn → build → drill → review**. Each day has a reading target, a concrete lab, an interview drill and a tip.

![Course map](assets/tree_00_master_map.svg)

**How to use it**
- ~2 h per day (45 min read, 60 min lab, 15 min drill). Day 7, 14, 21 are review and capstone days.
- Keep a `notes.md`: one line per day answering *"what would break in production, and how would I notice?"*
- Do the drill **out loud, with a timer**. Cover the ✅ answer, say yours, then compare.
- Start each week by re-reading that week's decision trees (images at the top of each module).

**Prerequisites:** Python 3.11, Docker, Git, basic scikit-learn. Set up the lab from the [README](README.md#-local-setup--hands-on-environment).

---

## Week 1: Foundations, Data and Experiments
*Goal: a reproducible training pipeline with validated data and tracked runs.*

| Day | Read | Lab (build this) | Interview drill | Tip |
|---|---|---|---|---|
| 1 | [01](01-mlops-foundations-system-design.md) §1-2 | Draw the lifecycle (code/data/model) from memory; compare with the [system map](assets/mlops_end_to_end_architecture.svg) | Q1 DevOps vs MLOps | Say "three artifacts, three clocks" |
| 2 | 01 §3-5 | Run the `dvc.yaml` skeleton with `dvc repro`; change one param and watch only one stage rerun | Q3 hidden feedback loop, Q4 notebook-to-prod | Measure lead time before choosing tools |
| 3 | [02](02-data-engineering-feature-stores.md) §1-3 | Pydantic + GX suite on a CSV (bad batch must fail the task); start Redis, `feast apply` + `materialize` | Q2 point-in-time joins | TTL = refresh cadence + buffer |
| 4 | 02 §4-5 | Implement `psi()`, inject drift in one column, emit a lineage event | Q4 drift is not always retrain | PSI + effect size, not p-values |
| 5 | [03](03-experiment-tracking-model-versioning.md) §1-2 | `docker compose up` MLflow + MinIO + Postgres; log a run with the `tracked` decorator | Q1 reproduce a year-old model | Block registration when `git.dirty` |
| 6 | 03 §3-5 | Sweep with early stopping (W&B or Optuna); DVC remote on MinIO; re-run your best run from tags only | Q6 AUC +0.01: ship? | Report intervals across seeds |
| 7 | Review | Capstone W1: validate > train > track > version data, with a stranger-proof README | Rapid-fire table, module 12 §6 | Spaced repetition wins |

**Checkpoint W1:** reproduce any run from its tags and explain point-in-time correctness without notes.

## Week 2: Registry, Automation and Serving
*Goal: gated promotion plus a service that survives real traffic.*

| Day | Read | Lab | Interview drill | Tip |
|---|---|---|---|---|
| 8 | [04](04-model-registry-governance.md) all | Register 3 versions; aliases `challenger`/`champion`; write `promote.py` gates and fail it on purpose | Q1 stages vs aliases, Q2 four-eyes | Serving pins a version, not an alias |
| 9 | [05](05-cicd-ct-automation-pipelines.md) §1-3 | CI workflow: lint, tests, smoke-train, kubeconform; add invariance + directional model tests | Q1 CI for ML | Behaviour tests catch what unit tests can't |
| 10 | 05 §4-5 | Argo CronWorkflow on Minikube (or compile the KFP pipeline); add debounce + cooldown | Q3 CT trigger design | Retrain storms are real |
| 11 | [06](06-model-serving-architecture.md) §1-2 | FastAPI service: warm-up, `/readyz`, `/metrics`, async prediction log | Q7 liveness vs readiness | Never block the event loop |
| 12 | 06 §3-4 | Export sklearn to ONNX; serve in Triton with dynamic batching | Q3 Triton vs FastAPI | Tune queue delay to a latency budget |
| 13 | 06 §5-6 | Batch scoring with atomic publish; Kafka consumer sketch | Q5 exactly-once | Idempotent sinks beat "exactly-once" claims |
| 14 | Review | Capstone W2: PR > CI > gated registry version > image > running service | Mock: "design CI/CD/CT" in 8 min | Show the rollback path |

**Checkpoint W2:** a model change goes from PR to a gated version and a running service through one pipeline.

## Week 3: Delivery, Monitoring and Infrastructure
*Goal: safe rollouts, drift awareness and a cluster that scales correctly.*

| Day | Read | Lab | Interview drill | Tip |
|---|---|---|---|---|
| 15 | [07](07-deployment-strategies-traffic-routing.md) §1-3 | Two Deployments + weighted split (Istio or two Services); flip weights | Q1 canary vs shadow vs A/B | Sticky assignment per user |
| 16 | 07 §4-7 | Shadow compare script; Argo Rollouts AnalysisTemplate; sample-size + SRM helpers | Q4 peeking at A/B | Pre-register metrics |
| 17 | [08](08-monitoring-observability-alerting.md) §1-3 | Evidently report to Pushgateway gauges | Q1 drift types | Two reference windows |
| 18 | 08 §4-6 | Prometheus rules + Grafana dashboard from the panel plan | Q5 alert design | Page on symptoms |
| 19 | [09](09-infrastructure-containerization-orchestration.md) §1-3 | Multi-stage Dockerfile; Deployment with probes; HPA + load test | Q3 CPU limits | Measure RSS before setting limits |
| 20 | 09 §4-5 | NetworkPolicy; Ray Tune HPO run (or KubeRay on Minikube) | Q5 Ray vs Spark | Match tool to workload shape |
| 21 | Review | Capstone W3: canary rollout that auto-aborts on a failing Prometheus query | Mock: "roll out a risky model" | Drill the rollback |

**Checkpoint W3:** a bad model is caught and rolled back automatically, and you can say why each alert exists.

## Week 4: Security, Failures and Interview Readiness
*Goal: harden, break things on purpose, then rehearse.*

| Day | Read | Lab | Interview drill | Tip |
|---|---|---|---|---|
| 22 | [10](10-security-privacy-compliance.md) §1-2 | JWT auth + rate limit + payload cap on the API | Q1 stop model cloning | Return buckets, not probabilities |
| 23 | 10 §3-5 | `cosign` sign/verify a model; DP-SGD run with Opacus; RBAC Role for the trainer | Q3 DP vs FL | Never `pickle` untrusted files |
| 24 | [11](11-production-failures-troubleshooting.md) §1-3 | Inject a leak, find it with `tracemalloc`, add worker recycling; CUDA OOM split-retry wrapper | Q1 100 MB/hour leak | Mitigate first, diagnose second |
| 25 | 11 §4-5 | Reproduce a hang; `py-spy dump`; add timeouts and a bulkhead | Q4 hang with idle CPU | Timeouts everywhere |
| 26 | 11 §6-7 | Tiered fallback + circuit breaker; kill the feature store and watch the tier metric | Q5 fallback strategy | Test fallbacks continuously |
| 27 | [12](12-mlops-interview-cheatsheet-system-design.md) §1-3 | Mock design 1: real-time fraud (45 min, timer on) | S1 accuracy dropped with no deploy | Say trade-offs out loud |
| 28 | 12 §3 | Mock designs 2 and 3: recommender, RAG platform | S3 GPU bill doubled | Do the back-of-envelope math |
| 29 | 12 §4-6 | Rapid-fire table, then the "Right vs Wrong" table | Every trap answer: why does it fail? | Explain the failure, not just the fix |
| 30 | All | Demo the final project; self-assess with the table below | One full mock interview | Close the gaps you found |

**Checkpoint W4:** you can design, operate and defend an end-to-end system, including its failure modes.

---

## Final Project (do alongside Weeks 3–4)

Build `fraud-mlops-demo`:
1. **Data:** synthetic transactions, GX suite, DVC.
2. **Features:** Feast offline + Redis online.
3. **Train:** tracked MLflow run with lineage tags; `promote.py` gates.
4. **Serve:** FastAPI container with metrics, fallback tiers.
5. **Deliver:** GitHub Actions CI; Minikube Deployment + HPA; canary via two Services.
6. **Monitor:** Prometheus + Grafana + Evidently drift gauge; alert → retrain webhook.
7. **Secure:** JWT auth, rate limits, signed model artifact.
8. **Document:** model card, runbook, and a one-page architecture diagram.

**Definition of done:** you can (a) reproduce any prediction's model, (b) roll back in under 5 minutes, (c) show a drift alert causing a retrain run, (d) explain every trade-off in an interview.

## Scoring Yourself

| Skill | Beginner | Solid | Staff-ready |
|---|---|---|---|
| Reproducibility | Saves a pickle | Tags code+data+env | Re-runs registered models quarterly to prove it |
| Serving | Flask wrapper | Warm-up, probes, metrics | Batching, fallbacks, load-tested p99 |
| Delivery | Manual deploy | CI + canary | Auto-analysis + rollback + drills |
| Monitoring | CPU graphs | Drift + latency alerts | Slice + delayed-label + SLO burn |
| Design interviews | Names tools | Gives a pipeline | Frames trade-offs, failure modes, ownership |

## Extra Practice Scenarios

1. A holiday sale changes your traffic mix. Which alerts fire, which should not, and what do you change?
2. Your feature store goes down for 30 minutes at peak. Describe each fallback tier and what you tell downstream consumers.
3. The data team renames a column. How does your system detect it before it reaches training?
4. A regulator requests the exact model and inputs behind a decision from 8 months ago.
5. A new embedding model changes vector space. Plan the index migration.
6. GPU spend doubled. What do you measure first and what do you change?

**Next:** back to the [README](README.md) · jump to [Module 12](12-mlops-interview-cheatsheet-system-design.md)
