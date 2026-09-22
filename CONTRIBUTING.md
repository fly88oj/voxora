# Contributing

Thanks for considering a contribution! This project is research infrastructure, so **measurement rigor comes first**: a PR that adds a number without its protocol, environment, and raw data cannot be merged.

## Ways to contribute

- **New engines** — see [docs/REPRODUCING.md §6](docs/REPRODUCING.md#6-adding-an-engine) for the checklist (engine class, registry entry, download script, license row, tests).
- **Reproductions on other CPUs** — extremely valuable. Run the suite, open an issue/PR attaching the result JSON (with its embedded environment fingerprint); we will add it to a `data/reproductions/` directory.
- **Protocol improvements** — changes to `voxora/scoring.py` or timing semantics must update `docs/METHODOLOGY.md` and the golden tests in the same PR.
- **Docs and translations** — translations of the README into additional languages are welcome. English is authoritative; translations must carry the header note saying so.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[onnx,dev]"
pytest                 # unit + contract tests, no weights needed
pytest -m integration  # requires VOXORA_INTEGRATION_MODELS
ruff check src tests scripts
```

## Ground rules

1. Every benchmark claim in docs must be traceable to a file under `data/` or to a PR-attached result.
2. Unit tests must pass without any model weights installed (stub engines exist for this reason).
3. Never commit model weights, downloaded corpora, or credentials. The optional API key is read from `VOXORA_API_KEY` only.
4. Match the existing code style (ruff-enforced); keep docstrings factual and specific.
5. By contributing you agree that your contributions are licensed Apache-2.0.
