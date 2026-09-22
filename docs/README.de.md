# Voxora (Deutsch)

> Dies ist eine Übersetzung der [englischen README](../README.md). Bei Abweichungen gilt die englische Fassung.

**Voxora ist ein REST-API-Dienst für Spracherkennung (ASR) und Sprachsynthese (TTS), der vollständig auf der CPU läuft.**

- **Eine API, acht Engines** — Transkription und Synthese über einen einzigen OpenAI-kompatiblen Vertrag; Engine pro Anfrage frei wählbar —
  bestehende OpenAI-SDKs funktionieren, wenn `base_url` auf Voxora zeigt
  ([API.md](API.md#openai-sdk-compatibility), Englisch).
- **Nur CPU** — kein GPU/CUDA; die Standardinstallation benötigt nicht einmal PyTorch.
- **Telemetrie pro Anfrage** — jede Antwort enthält Wall-Clock-Zeit und RTF.
- **Gemessen statt beworben** — reproduzierbares Messwerkzeug und Referenzdaten (Rohdaten, Umgebungs-Fingerabdrücke, Methodik) als Grundlage der Kapazitätsplanung; siehe „Gemessene Referenz“ unten.

## Gemessene Referenz

Was liefern diese Engines auf der CPU? Voxoras Antwort: gemessene, versionierte Daten statt Herstellerangaben — vollständige Tabellen, Genauigkeitsergebnisse sowie Optimierungs-/Stabilitätsstudien liegen an einem Ort: der **[Evaluationsbericht](EVALUATION.md)** (Protokoll und Validität: [METHODOLOGY.md](METHODOLOGY.md); Rohdaten unter [`../data/`](../data)).

Drei Erkenntnisse für den Betrieb:

1. **bf16 ist auf AMD Zen 4/5 eine kostenlose Beschleunigung** — 3,5–3,9× für die PyTorch-Engines, Transkripte verifiziert identisch; Threads nie überzeichnen (32 Threads auf 16 Kernen: 5–46× langsamer).
2. **LLM-basierte ASR fährt am besten als int8-ONNX** — 5× schneller als PyTorch fp32 bei 60 % weniger Speicher, nahezu identische Genauigkeit.
3. **Piper ist die einzige Echtzeit-TTS-Ebene** (RTF 0,04–0,06, erste Audio <300 ms); die LLM-TTS-Engines eignen sich auch nach Tuning (RTF 1,4–6,0) nur für asynchrone Synthese.

Vorbehalt: Messungen stammen von einer gemeinsam genutzten, nicht exklusiven Maschine — absolute RTF folgt der Umgebungslast (Last und Affinität werden in jeder Ergebnisdatei aufgezeichnet); Back-to-back gemessene Rankings sind stabil. Details: [Evaluationsbericht §7](EVALUATION.md#7-stability-findings).

## Schnellstart

```bash
pip install -e ".[onnx]"                  # ONNX-Engines, kein PyTorch nötig
scripts/download_models.sh models         # Gewichte (~3 GB)
voxora-api --models-dir models --port 8300
```

```bash
curl -s http://127.0.0.1:8300/v1/audio/transcriptions \
  -F file=@sample.wav -F engine=sensevoice

curl -s http://127.0.0.1:8300/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"text":"Hallo von Voxora.","engine":"piper","language":"de"}' \
  -o out.wav
```

Interaktive OpenAPI-Dokumentation: `http://127.0.0.1:8300/docs`

## Benchmark-Werkzeug

Das Paket bringt außerdem das Messwerkzeug mit, das die Referenzdaten erzeugt hat:

```bash
voxora list                                        # Engine-Katalog
voxora run --engine sensevoice --audio-dir data/fixtures -o r.json
voxora run --engine piper --text "Hallo." --language de -o t.json --wav-dir wavs/
voxora run --engine zipformer --audio-dir data/fixtures --repeat 5 -o zf.json
```

Jede Ergebnisdatei bettet eine Schema-Version, die exakte Engine-Konfiguration und einen Umgebungs-Fingerabdruck ein (CPU, Bibliotheksversionen, **CPU-Affinität und Lastmittel zum Laufzeitpunkt**).

## Dokumentation

- [API.md](API.md) — REST-Referenz (Englisch)
- Vollständige Übersicht: [docs/README.md](README.md) (Englisch)

Evaluationsbericht:

- [EVALUATION.md](EVALUATION.md) — konsolidierter Messbericht (Englisch)
- [METHODOLOGY.md](METHODOLOGY.md) — Messprotokoll und Definitionen (Englisch)
- [REPRODUCING.md](REPRODUCING.md) — gepinnte Umgebungen, Download-Kanäle (Englisch)

Grundlagen der Erhebung:

- [SURVEY.md](SURVEY.md) — Engine-Landschaft und Auswahl (Englisch)
- [MODEL_LICENSES.md](MODEL_LICENSES.md) — Lizenzen pro Checkpoint (Englisch)
- Weitere Übersetzungen: [English](../README.md) · [简体中文](README.zh-CN.md) · [Français](README.fr.md) · [Español](README.es.md) · [Italiano](README.it.md)
## Lizenz

Code: Apache-2.0. Mitgelieferte Testdaten: CC-BY-4.0. Modellgewichte folgen ihren Upstream-Lizenzen; dieses Repository verbreitet keine Gewichte.
