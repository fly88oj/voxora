# Voxora (Español)

> Traducción del [README en inglés](../README.md). En caso de discrepancia, prevalece la versión inglesa.

**Voxora es un servicio API REST de reconocimiento de voz (ASR) y síntesis de habla (TTS) que funciona por completo en CPU.**

- **Una API, ocho motores** — transcripción y síntesis mediante un único contrato compatible con OpenAI; motor elegible por petición —
  los SDK de OpenAI funcionan apuntando `base_url` a Voxora
  ([API.md](API.md#openai-sdk-compatibility), inglés).
- **Solo CPU** — sin GPU/CUDA; la instalación por defecto no requiere ni PyTorch.
- **Telemetría por petición** — cada respuesta incluye tiempos y RTF.
- **Medido, no promocionado** — herramienta de medición reproducible y datos de referencia (datos en bruto, huellas de entorno, metodología) como base del dimensionamiento; véase «Referencia medida» más abajo.

## Referencia medida

¿Qué ofrecen estos motores en CPU? La respuesta de Voxora: datos medidos y versionados, no cifras de fabricante — las tablas completas, los resultados de precisión y los estudios de optimización/estabilidad viven en un solo sitio: el **[informe de evaluación](EVALUATION.md)** (protocolo y validez: [METHODOLOGY.md](METHODOLOGY.md); datos en bruto en [`../data/`](../data)).

Tres hechos operativos:

1. **bf16 es velocidad gratis en AMD Zen 4/5** — 3,5–3,9× para los motores PyTorch, transcripciones verificadas idénticas; nunca sobresuscribir hilos (32 hilos en 16 núcleos: 5–46× más lento).
2. **El ASR tipo LLM se despliega mejor como ONNX int8** — 5× más rápido que PyTorch fp32 con 60 % menos memoria, precisión casi idéntica.
3. **Piper es el único nivel TTS en tiempo real** (RTF 0,03–0,07, primer audio <300 ms); los motores TTS LLM, incluso optimizados (RTF 1,4–6,0), sirven para síntesis asíncrona.

Advertencia: las medidas provienen de una máquina compartida no exclusiva — el RTF absoluto sigue la carga ambiental (carga y afinidad quedan registradas en cada archivo de resultados); las clasificaciones medidas back-to-back son estables. Detalles: [informe de evaluación §7](EVALUATION.md#7-stability-findings).

## Inicio rápido

```bash
pip install -e ".[onnx]"                  # motores ONNX, sin PyTorch
scripts/download_models.sh models         # pesos (~3 GB)
voxora-api --models-dir models --port 8300
```

```bash
curl -s http://127.0.0.1:8300/v1/audio/transcriptions \
  -F file=@sample.wav -F engine=sensevoice

curl -s http://127.0.0.1:8300/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"text":"Hola desde Voxora.","engine":"piper","language":"es"}' \
  -o out.wav
```

Documentación OpenAPI interactiva: `http://127.0.0.1:8300/docs`

## Herramienta de medición integrada

El paquete también incluye la herramienta de medición que produjo los datos de referencia:

```bash
voxora list                                        # catálogo de motores
voxora run --engine sensevoice --audio-dir data/fixtures -o r.json
voxora run --engine piper --text "Hola." --language es -o t.json --wav-dir wavs/
voxora run --engine zipformer --audio-dir data/fixtures --repeat 5 -o zf.json
```

Cada archivo de resultados incrusta una versión de esquema, la configuración exacta del motor y una huella de entorno (CPU, versiones de bibliotecas, **afinidad de CPU y carga media en tiempo de ejecución**).

## Documentación

- [API.md](API.md) — referencia REST (inglés)
- Índice completo: [docs/README.md](README.md) (inglés)

Informe de evaluación:

- [EVALUATION.md](EVALUATION.md) — informe de evaluación consolidado (inglés)
- [METHODOLOGY.md](METHODOLOGY.md) — protocolo y definiciones (inglés)
- [REPRODUCING.md](REPRODUCING.md) — entornos fijados, canales de descarga (inglés)

Fundamentos del estudio:

- [SURVEY.md](SURVEY.md) — panorama de motores y selección (inglés)
- [MODEL_LICENSES.md](MODEL_LICENSES.md) — licencias por punto de control (inglés)
- Otras traducciones: [English](../README.md) · [简体中文](README.zh-CN.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Italiano](README.it.md)
## Licencia

Código: Apache-2.0. Datos de prueba incluidos: CC-BY-4.0. Los pesos de los modelos siguen sus licencias originales; este repositorio no distribuye ninguno.
