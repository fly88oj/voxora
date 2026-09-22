# Voxora (Français)

> Traduction du [README anglais](../README.md). En cas de divergence, la version anglaise fait foi.

**Voxora est un service API REST de reconnaissance vocale (ASR) et de synthèse de la parole (TTS) qui fonctionne entièrement sur CPU.**

- **Une API, huit moteurs** — transcription et synthèse via un contrat unique compatible OpenAI ; moteur choisi librement à chaque requête —
  les SDK OpenAI existants fonctionnent en pointant `base_url` vers Voxora
  ([API.md](API.md#openai-sdk-compatibility), anglais).
- **CPU uniquement** — pas de GPU/CUDA ; l'installation par défaut n'exige même pas PyTorch.
- **Télémétrie par requête** — chaque réponse embarque le temps mesuré et le RTF.
- **Mesuré, pas marketing** — outillage de mesure reproductible et données de référence (données brutes, empreintes d'environnement, méthodologie) pour fonder le dimensionnement ; voir « Référence mesurée » ci-dessous.

## Référence mesurée

Que délivrent ces moteurs sur CPU ? La réponse de Voxora : des données mesurées et versionnées plutôt que des chiffres constructeur — tableaux complets, précision et études d'optimisation/stabilité au même endroit : le **[rapport d'évaluation](EVALUATION.md)** (protocole et validité : [METHODOLOGY.md](METHODOLOGY.md) ; données brutes dans [`../data/`](../data)).

Trois faits opérationnels :

1. **bf16 est un gain gratuit sur AMD Zen 4/5** — 3,5–3,9× pour les moteurs PyTorch, transcriptions vérifiées identiques ; ne jamais sursouscrire les threads (32 threads sur 16 cœurs : 5–46× plus lent).
2. **L'ASR de type LLM se déploie au mieux en ONNX int8** — 5× plus rapide que PyTorch fp32 avec 60 % de mémoire en moins, précision quasi identique.
3. **Piper est la seule couche TTS temps réel** (RTF 0,03–0,07, premier audio <300 ms) ; les moteurs TTS LLM, même optimisés (RTF 1,4–6,0), conviennent à une synthèse asynchrone.

Réserve : mesures issues d'une machine partagée non exclusive — le RTF absolu suit la charge ambiante (charge et affinité enregistrées dans chaque fichier de résultat) ; les classements mesurés back-to-back sont stables. Détails : [rapport d'évaluation §7](EVALUATION.md#7-stability-findings).

## Démarrage rapide

```bash
pip install -e ".[onnx]"                  # moteurs ONNX, sans PyTorch
scripts/download_models.sh models         # poids (~3 Go)
voxora-api --models-dir models --port 8300
```

```bash
curl -s http://127.0.0.1:8300/v1/audio/transcriptions \
  -F file=@sample.wav -F engine=sensevoice

curl -s http://127.0.0.1:8300/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"text":"Bonjour de Voxora.","engine":"piper","language":"fr"}' \
  -o out.wav
```

Documentation OpenAPI interactive : `http://127.0.0.1:8300/docs`

## Outil de mesure intégré

Le paquet embarque aussi l'outillage de mesure qui a produit les données de référence :

```bash
voxora list                                        # catalogue des moteurs
voxora run --engine sensevoice --audio-dir data/fixtures -o r.json
voxora run --engine piper --text "Bonjour." --language fr -o t.json --wav-dir wavs/
voxora run --engine zipformer --audio-dir data/fixtures --repeat 5 -o zf.json
```

Chaque fichier de résultats intègre une version de schéma, la configuration exacte du moteur et une empreinte d'environnement (CPU, versions des bibliothèques, **affinité CPU et charge moyenne au moment de l'exécution**).

## Documentation

- [API.md](API.md) — référence REST (anglais)
- Index complet : [docs/README.md](README.md) (anglais)

Rapport d'évaluation :

- [EVALUATION.md](EVALUATION.md) — rapport d'évaluation consolidé (anglais)
- [METHODOLOGY.md](METHODOLOGY.md) — protocole et définitions (anglais)
- [REPRODUCING.md](REPRODUCING.md) — environnements épinglés, canaux de téléchargement (anglais)

Fondements de l'étude :

- [SURVEY.md](SURVEY.md) — panorama des moteurs et sélection (anglais)
- [MODEL_LICENSES.md](MODEL_LICENSES.md) — licences par point de contrôle (anglais)
- Autres traductions : [English](../README.md) · [简体中文](README.zh-CN.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Italiano](README.it.md)
## Licence

Code : Apache-2.0. Données de test fournies : CC-BY-4.0. Les poids de modèles suivent leurs licences amont ; ce dépôt n'en distribue aucun.
