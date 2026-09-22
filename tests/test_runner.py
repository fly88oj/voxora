"""Bench runner tests with the stub TTS engine (schema + aggregation)."""

import json

import pytest

from voxora.bench import run_tts, write_result


def test_tts_runner_schema(tmp_path, stub_engines):
    result = run_tts("stub-tts", ["alpha", "beta"], models_dir="/nonexistent")
    assert result["schema_version"] == 1
    assert result["kind"] == "tts"
    assert result["engine"] == "stub-tts"
    assert len(result["items"]) == 2
    # aggregate RTF equals total timing over items
    total_audio = sum(i["audio_s"] for i in result["items"])
    assert result["aggregate"]["audio_s"] == pytest.approx(total_audio, abs=0.01)
    assert result["environment"]["python"]
    assert "cpu" in result["environment"]
    # isolation evidence is recorded automatically (METHODOLOGY.md §6)
    assert "cpu_affinity" in result["environment"]["runtime"]
    assert "loadavg_1min" in result["environment"]["runtime"]

    out = tmp_path / "r.json"
    write_result(out, result)
    assert json.loads(out.read_text())["engine"] == "stub-tts"


def test_tts_runner_writes_wavs(tmp_path, stub_engines):
    wav_dir = tmp_path / "wavs"
    run_tts("stub-tts", ["hello"], models_dir="/nonexistent", wav_dir=wav_dir)
    files = list(wav_dir.glob("*.wav"))
    assert len(files) == 1
    assert files[0].read_bytes()[:4] == b"RIFF"


def test_tts_runner_repeat_stats(stub_engines):
    result = run_tts("stub-tts", ["alpha"], models_dir="/nonexistent", repeat=3)
    agg = result["aggregate"]
    assert len(agg["passes"]) == 3
    # deterministic stub: identical (near-zero) timing per pass -> no variance
    assert agg["stats"]["n"] == 3
    assert agg["stats"]["std"] == pytest.approx(0.0, abs=1e-9)
    # cv is None when the mean rounds to zero (documented guard)
    assert agg["stats"]["cv"] is None or agg["stats"]["cv"] == pytest.approx(0.0, abs=1e-9)
    # aggregate.rtf is the across-pass mean
    assert agg["rtf"] == pytest.approx(sum(agg["passes"]) / 3, abs=1e-9)
    assert result["config"]["repeat"] == 3


def test_repeat_below_one_clamped(stub_engines):
    result = run_tts("stub-tts", ["alpha"], models_dir="/nonexistent", repeat=0)
    assert len(result["aggregate"]["passes"]) == 1
