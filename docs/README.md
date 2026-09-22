# Voxora documentation

Voxora is a CPU-only REST API service for speech recognition and synthesis. The documentation is organized in three layers:

## Product documentation

- [API.md](API.md) — REST reference: endpoints, auth, configuration, error contract
- README translations: [简体中文](README.zh-CN.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Italiano](README.it.md) (English [../README.md](../README.md) is authoritative)

## Evaluation report

The measured evidence behind the service's performance claims:

- [EVALUATION.md](EVALUATION.md) — consolidated findings: performance, round-trip accuracy, optimization, stability
- [METHODOLOGY.md](METHODOLOGY.md) — measurement protocol, definitions, threats to validity
- [REPRODUCING.md](REPRODUCING.md) — pinned environments, download channels, known issues
- Raw data: [`../data/`](../data) (results, optimization sweep, stability study, environment fingerprint)

## Survey foundations

Why Voxora ships these engines, on what licensing terms:

- [SURVEY.md](SURVEY.md) — engine landscape, selection criteria, evaluated-but-excluded candidates
- [MODEL_LICENSES.md](MODEL_LICENSES.md) — per-engine and per-checkpoint licensing
