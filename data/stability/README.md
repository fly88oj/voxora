# Stability data

Replication runs behind [evaluation report §7](../../docs/EVALUATION.md#7-stability-findings); protocol: [METHODOLOGY.md §6](../../docs/METHODOLOGY.md#run-to-run-stability-protocol).

- `asr_<engine>_t<threads>_r5.json` — ONNX ASR engines + Piper, 5 passes per engine
  (`voxora run --repeat 5`, 8-file optimization subset; pass 1 = cold).
- `pinned/asr_*_pinned_r5.json` — same protocol with CPU affinity fixed via
  `taskset`; these files record `environment.runtime.cpu_affinity` and
  `environment.runtime.loadavg_1min`.
- `torch_repeats.json` — PyTorch engines, fresh process per run, warm/cold
  separated for the TTS engines.

The first non-pinned `asr_*_r5.json` batch predates automatic
`environment.runtime` recording; the `pinned/` batch and all future `voxora`
outputs carry affinity + loadavg automatically.
