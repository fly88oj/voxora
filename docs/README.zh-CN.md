# Voxora（简体中文）

> 本文件是 [英文 README](../README.md) 的译文。如有出入，以英文版为准。

**Voxora 是一个只依赖 CPU 的语音识别（ASR）与语音合成（TTS）REST API 服务。**

- **一个 API，八个引擎**：单一 OpenAI 兼容契约完成转写与合成，按请求自由选引擎——现有 OpenAI SDK 指向 `base_url` 即可直接使用（见 [API.md](API.md#openai-sdk-compatibility)，英文）。
- **纯 CPU 设计**：无需 GPU/CUDA，默认安装连 PyTorch 都不需要；任意 x86-64 Linux 机器可部署。
- **逐请求遥测**：每个响应携带墙钟计时与 RTF（实时率）。
- **性能有实测依据**：内置可复现的测量工具与参考数据（原始数据、环境指纹、方法学），容量规划建立在证据之上——见下文“性能实测参考”。

## 性能实测参考

这些引擎在 CPU 上表现如何？Voxora 的回答是可测量、带版本的数据而非厂商口径——完整表格、准确率结果与优化/稳定性研究集中在一处：**[综合测评报告](EVALUATION.md)**（协议与效度见 [METHODOLOGY.md](METHODOLOGY.md)；原始数据在 [`../data/`](../data)）。

三条运营要点：

1. **AMD Zen 4/5 上 bf16 是免费加速**——PyTorch 引擎 3.5–3.9 倍且转写逐字一致；线程切勿超订阅（16 核跑 32 线程劣化 5–46 倍）。
2. **LLM 式 ASR 的最佳部署形态是 int8 ONNX**——比 PyTorch fp32 快 5 倍、省 60% 内存、精度几乎无损。
3. **Piper 是唯一的实时 TTS 档位**（RTF 0.03–0.07，首音频 <300ms）；LLM TTS 引擎即使调优后（RTF 1.4–6.0）也只适合异步合成。

注意：测量来自共享非独占机器——绝对 RTF 随环境负载浮动（每个结果文件都记录了负载与亲和性）；背靠背测量的引擎排序稳定。详见[测评报告 §7](EVALUATION.md#7-stability-findings)。

## 快速开始

```bash
pip install -e ".[onnx]"                  # ONNX 引擎，无需 PyTorch
scripts/download_models.sh models         # 下载权重（约 3 GB）
voxora-api --models-dir models --port 8300
```

```bash
# 识别（任意 libsndfile 格式，服务端自动重采样至 16 kHz 单声道）
curl -s http://127.0.0.1:8300/v1/audio/transcriptions \
  -F file=@sample.wav -F engine=sensevoice

# 合成（返回 audio/wav，附计时响应头）
curl -s http://127.0.0.1:8300/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"text":"你好，来自 Voxora。","engine":"piper","language":"zh"}' \
  -o out.wav -D -
```

交互式 API 文档：`http://127.0.0.1:8300/docs`

## 基准 CLI

```bash
voxora list                                        # 引擎目录
voxora run --engine sensevoice --audio-dir data/fixtures -o r.json
voxora run --engine piper --text "你好。" --language zh -o t.json --wav-dir wavs/
```

每个结果文件内嵌 schema 版本、引擎精确配置与环境指纹（CPU 型号、核心数、库版本）。

## 文档导航

- [API.md](API.md) — REST 接口参考（英文）
- 完整文档索引：[docs/README.md](README.md)

测评报告：

- [EVALUATION.md](EVALUATION.md) — 综合测评报告（英文）
- [METHODOLOGY.md](METHODOLOGY.md) — 测量协议、指标定义、效度威胁（英文）
- [REPRODUCING.md](REPRODUCING.md) — 版本锁定、下载通道、已知问题（英文）

调研基础：

- [SURVEY.md](SURVEY.md) — 引擎调研与选型理由（英文）
- [MODEL_LICENSES.md](MODEL_LICENSES.md) — 各模型许可对照（英文）

其他译本：[English](../README.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Italiano](README.it.md)

## 许可

代码 Apache-2.0；随仓库分发的测试音频与测量数据 CC-BY-4.0；模型权重遵循各自上游许可（本仓库不分发任何权重）。
