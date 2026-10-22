# Cross-Check Report

An account of how this edition was audited and expanded, what changed, and what is still missing. Produced with a plan, write, audit loop: a lead agent audited the repository against a total MLOps concept map; five writer agents wrote the new material from precise outlines; a fixer corrected defects in the original modules; automated checks verified the result.

## What was checked

| Check | Method | Result |
|---|---|---|
| Structure | `scripts/audit.py`: every numbered `N.M` section has Conceptual Deep Dive, Cross-Connection Scenarios, Production Anti-Patterns, Minimal Working Code and Interview Angle, plus a code block; required-concept keywords per module | 0 issues |
| Python syntax | every ```python block parsed with `ast` | 0 errors |
| Runtime | stdlib/numpy-only sections executed in a clean directory | 80 sections ran, 0 failed, 32 skipped (need third-party packages) |
| Cross-references | `scripts/check_xrefs.py` validates "Module NN section X.Y" and bare section numbers against real headings; sensitivity confirmed with planted errors | 615 checked, 0 broken |
| Encoding | scan for mojibake and replacement characters | none |
| PDFs | one per module, Unicode fonts embedded | 22 / 22; SVG diagrams are not embedded (see gaps) |
| Site | static build, SVG diagrams served and rendered in the reader | 22 modules, 255 episodes |

## What changed

| Area | Change |
|---|---|
| Structure | Modules moved to `chapters/` with underscore names; README, course and in-chapter links updated; asset paths rewritten to `../assets/`; the seven unnumbered closing sections of each module were numbered so every section is an episode; contents lists regenerated |
| Original modules 01-12 | 30 new extension sections (numbered from the next free number, 15 to 17) in the five-part concept format |
| New modules 13-22 | Testing for ML systems; streaming and real-time ML; distributed training, GPU scheduling and tuning; ML platform engineering and FinOps; responsible ML in practice; domain playbooks; bridge to LLMOps; production projects and labs; learning roadmap and study plan; mock-interview and system-design bank (49 sections) |
| Tooling | Audit, cross-reference checker, snippet runner, TOC regenerator, PDF and site builders; fence-aware section parsing; merge script inserts new sections before the module's navigation footer |

## Defects found in the original modules (fixer report)

| # | Where | Problem | Status |
|---|---|---|---|
| 1 | Footers and body links, modules 01-12 | Prev/Next and cross-links used hyphenated file names that no longer exist, and README links were wrong relative to `chapters/` | Fixed |
| 2 | Module 02 section 3 | File layout in the prose did not match the code | Fixed |
| 3 | Module 01 section 5 | `dvc.yaml` outs block was malformed | Fixed |
| 4 | Module 08 section 3 | Evidently drift publishing likely raised `KeyError` for the 0.4.x result layout | Fixed, with a version note ("verify against current docs"); not executed |
| 5 | Module 08 section 5 | "LatencySLOBurn" was a plain threshold, not a burn-rate alert; drift alert could fire on stale data | Fixed: renamed, staleness guard added |
| 6 | Module 07 sections 3 and 6 | Canary analysis matched stable and canary pods, so it did not isolate the canary | Fixed (hash-scoped query), hedged on controller and label names |
| 7 | Module 05 section 4 | Argo template referenced an output parameter it never defined | Fixed |
| 8 | Module 05 section 3 | CI/CD workflow gaps and an invalid YAML `concurrency` line | Fixed; kubeconform checksum and trivy-action pin are explicit `REPLACE_WITH_...` placeholders |
| 9 | Module 06 sections 2, 5 | Prometheus metrics broke with multiple workers; "atomic" S3 promote was not atomic | Fixed: multiprocess collector, commit marker |
| 10 | Module 09 section 4 | NetworkPolicy blocked Prometheus scraping and the init-container download | Fixed; model-store CIDR is a placeholder; ingress-nginx retirement note hedged |
| 11 | Module 10 sections 2, 5 | Body-size check ran after parsing; gateway plugin licence wording | Fixed / hedged |
| 12 | Module 04 section 3 | `promote.py` swallowed all exceptions, so missing metrics skipped the gates | Fixed: only "does not exist" is benign |
| 13 | Module 11 section 7 | Circuit breaker lacked a single half-open probe | Fixed |
| 14 | Modules 03, 04, 11 | Snippet import slips and missing imports | Fixed |
| 15 | "More Interview Questions" sections | Possible duplicated questions | Left: needs replacement content, headings must not change |

## Known remaining gaps

- Snippets using third-party libraries (MLflow, Feast, Evidently, Kubeflow, Argo, Kafka/Flink clients, and similar) are syntax-checked only. Flags and APIs drift between releases; check current documentation.
- The PDFs omit the SVG decision and connection diagrams because the PDF library cannot render SVG; they are available in the repository and on the site.
- Placeholders (`REPLACE_WITH_SHA256`, `REPLACE_WITH_COMMIT_SHA`, a model-store CIDR) are deliberate: real values cannot be known offline.
- Perishable facts (versions, prices, feature lists, regulation details) are generic or marked "verify against current docs". Fairness and regulatory content is orientation only, not legal advice.
- The "More Interview Questions" duplication item (above) is unresolved.
- Not covered in depth: bespoke vendor-managed platform configuration, hardware kernel tuning, and jurisdiction-specific legal detail.
- The audit is structural. It cannot prove an explanation is correct; human technical review remains valuable.
