"""Voxora: a CPU-only REST API service for speech recognition and synthesis.

The package has three layers, in order of importance:

- ``voxora.api``     — the service: FastAPI app exposing every engine behind
  one REST contract (``voxora-api``).
- ``voxora.engines`` — uniform wrappers around the eight speech engines
  (lazy loading, optional heavy dependencies).
- ``voxora.bench``   — the measurement harness that produces the performance
  reference data shipped in ``data/`` (``voxora`` CLI).
"""

__version__ = "0.1"

SCHEMA_VERSION = 1
