# Voxora (Italiano)

> Traduzione del [README in inglese](../README.md). In caso di divergenza, prevale la versione inglese.

**Voxora è un servizio API REST di riconoscimento vocale (ASR) e sintesi della parola interamente su CPU.**

- **Una API, otto motori** — trascrizione e sintesi tramite un unico contratto compatibile con OpenAI; motore scelto liberamente per ogni richiesta —
  gli SDK OpenAI esistenti funzionano puntando `base_url` a Voxora
  ([API.md](API.md#openai-sdk-compatibility), inglese).
- **Solo CPU** — niente GPU/CUDA; l'installazione predefinita non richiede nemmeno PyTorch.
- **Telemetria per richiesta** — ogni risposta riporta tempi e RTF.
- **Misurato, non pubblicizzato** — strumento di misurazione riproducibile e dati di riferimento (dati grezzi, impronte d'ambiente, metodologia) a fondamento della pianificazione della capacità; si veda «Riferimento misurato» qui sotto.

## Riferimento misurato

Cosa offrono questi motori su CPU? La risposta di Voxora: dati misurati e versionati, non cifre del venditore — tabelle complete, risultati di accuratezza e studi di ottimizzazione/stabilità in un unico posto: il **[rapporto di valutazione](EVALUATION.md)** (protocollo e validità: [METHODOLOGY.md](METHODOLOGY.md); dati grezzi in [`../data/`](../data)).

Tre fatti operativi:

1. **bf16 è velocità gratis su AMD Zen 4/5** — 3,5–3,9× per i motori PyTorch, trascrizioni verificate identiche; mai sobrascrivere i thread (32 thread su 16 core: 5–46× più lento).
2. **L'ASR di tipo LLM si schiera al meglio come ONNX int8** — 5× più veloce di PyTorch fp32 con il 60 % di memoria in meno, accuratezza quasi identica.
3. **Piper è l'unico livello TTS in tempo reale** (RTF 0,03–0,07, primo audio <300 ms); i motori TTS LLM, anche ottimizzati (RTF 1,4–6,0), si adattano alla sintesi asincrona.

Avvertenza: le misure provengono da una macchina condivisa non esclusiva — l'RTF assoluto segue il carico ambientale (carico e affinità registrati in ogni file di risultati); le classifiche misurate back-to-back sono stabili. Dettagli: [rapporto di valutazione §7](EVALUATION.md#7-stability-findings).

## Avvio rapido

```bash
pip install -e ".[onnx]"                  # motori ONNX, senza PyTorch
scripts/download_models.sh models         # pesi (~3 GB)
voxora-api --models-dir models --port 8300
```

```bash
curl -s http://127.0.0.1:8300/v1/audio/transcriptions \
  -F file=@sample.wav -F engine=sensevoice

curl -s http://127.0.0.1:8300/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"text":"Ciao da Voxora.","engine":"piper","language":"en"}' \
  -o out.wav
```

Documentazione OpenAPI interattiva: `http://127.0.0.1:8300/docs`

## Strumento di misurazione integrato

Il pacchetto include anche lo strumento di misurazione che ha prodotto i dati di riferimento:

```bash
voxora list                                        # catalogo dei motori
voxora run --engine sensevoice --audio-dir data/fixtures -o r.json
voxora run --engine piper --text "Ciao." --language it -o t.json --wav-dir wavs/
voxora run --engine zipformer --audio-dir data/fixtures --repeat 5 -o zf.json
```

Ogni file di risultati incorpora una versione di schema, la configurazione esatta del motore e un'impronta d'ambiente (CPU, versioni delle librerie, **affinità CPU e carico medio al momento dell'esecuzione**).

## Documentazione

- [API.md](API.md) — riferimento REST (inglese)
- Indice completo: [docs/README.md](README.md) (inglese)

Rapporto di valutazione:

- [EVALUATION.md](EVALUATION.md) — rapporto di valutazione consolidato (inglese)
- [METHODOLOGY.md](METHODOLOGY.md) — protocollo e definizioni (inglese)
- [REPRODUCING.md](REPRODUCING.md) — ambienti bloccati, canali di download (inglese)

Fondamenti dello studio:

- [SURVEY.md](SURVEY.md) — panorama dei motori e selezione (inglese)
- [MODEL_LICENSES.md](MODEL_LICENSES.md) — licenze per checkpoint (inglese)
- Altre traduzioni: [English](../README.md) · [简体中文](README.zh-CN.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md)
## Licenza

Codice: Apache-2.0. Dati di prova inclusi: CC-BY-4.0. I pesi dei modelli seguono le licenze upstream; questo repository non ne distribuisce alcuno.
