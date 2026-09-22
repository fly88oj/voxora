"""Runtime configuration. Every value can come from an environment variable
(prefixed ``VOXORA_``) so that no secret or machine-specific path lives in code.

Credentials policy: the optional API key is read exclusively from the
``VOXORA_API_KEY`` environment variable; it is never logged or echoed back.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOXORA_", env_file=None, frozen=True)

    # Directory holding downloaded model weights (see scripts/download_models.sh).
    models_dir: str = "models"

    # CosyVoice loads from a source checkout; point this at your clone.
    cosyvoice_repo: str = ""

    # API
    host: str = "127.0.0.1"
    port: int = 8300
    api_key: str = ""  # optional; when non-empty, requests must send X-API-Key

    # Engine defaults
    num_threads: int = 8  # measured sweet spot for the ONNX engines on 16C/32T
    default_asr: str = "sensevoice"
    default_tts: str = "piper"


@lru_cache
def get_settings() -> Settings:
    return Settings()
