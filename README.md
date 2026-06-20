# 🧠 Awesome AI Projects for CPU

![Awesome](https://awesome.re/badge.svg) ![MIT License](https://img.shields.io/badge/license-MIT-brightgreen) ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

> A curated list of **open source AI projects** that run well on **CPU**, no GPU required.  
> Perfect for makers, indie developers, and local AI experiments.

**Last updated:** June 2026 — featuring projects from 2025–2026

-----

## 📑 Table of Contents

- [Language Models](#-language-models-chatbots--text)
- [Inference Engines & Runtimes](#-inference-engines--runtimes)
- [Image Generation and Editing](#️-image-generation-and-editing)
- [Voice and Audio](#-voice-and-audio)
- [Computer Vision](#️-computer-vision)
- [Small Models](#-small-models-perfect-for-cpu)
- [AI Assistants & Agents](#-ai-assistants--agents)
- [Coding Assistants](#-coding-assistants)
- [Document & Knowledge (RAG)](#-document--knowledge-rag)
- [Agentic Workflows & Platforms](#-agentic-workflows--platforms)
- [Embeddings & Vector Databases](#-embeddings--vector-databases)
- [Creative AI](#-creative-ai--miscellaneous)
- [Development Tools](#️-development-tools)
- [Tips for CPU](#️-tips-for-running-on-cpu)

-----

## 🧠 Language Models (Chatbots / Text)

- [**Ollama**](https://github.com/ollama/ollama) — "Docker for local LLMs." Run Llama, Mistral, Gemma, DeepSeek with one command. Excellent CPU support, easy CLI, model library.
- [**GPT4All**](https://github.com/nomic-ai/gpt4all) ⭐77k — Simple interface to run LLMs locally with several CPU-optimized models included.
- [**LM Studio**](https://lmstudio.ai/) — Beautiful GUI app for local LLMs, automatically supports CPU execution.
- [**Text Generation WebUI**](https://github.com/oobabooga/text-generation-webui) — Powerful web interface for LLMs with extensive CPU mode options.
- [**Jan.ai**](https://github.com/janhq/jan) — ChatGPT-like interface that runs 100% offline with a clean, modern UI.
- [**LocalAI**](https://github.com/mudler/LocalAI) ⭐46k — OpenAI-compatible API for running local models. Drop-in replacement that also supports vision, voice, image gen — no GPU required.
- [**Kobold.cpp**](https://github.com/LostRuins/koboldcpp) — Lightweight inference engine for GGUF models with built-in web UI.
- [**Open WebUI**](https://github.com/open-webui/open-webui) — Self-hosted, offline ChatGPT-style interface for Ollama, with RAG, web search, and multi-user support.

-----

## ⚡ Inference Engines & Runtimes

- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) ⭐112k — The gold standard for CPU-optimized LLM inference in C/C++. Powers Ollama, LM Studio, and most local LLM tools.
- [**llamafile**](https://github.com/mozilla-ai/llamafile) ⭐24.9k — Mozilla's single-file LLM executable. Distribute and run LLMs as a standalone binary — no installation, no dependencies, no GPU required. Built on llama.cpp, supports CPU inference out of the box.
- [**BitNet**](https://github.com/microsoft/BitNet) ⭐39k — Microsoft's official inference framework for 1-bit LLMs. Extremely efficient on CPU.
- [**eLLM**](https://github.com/lucienhuangfu/eLLM) ⭐416 — Rust-based inference engine that claims to run LLMs faster on CPU than on GPU through aggressive optimization.
- [**Krasis**](https://github.com/brontoguana/krasis) ⭐455 — Hybrid LLM runtime focusing on efficient execution of larger models on consumer hardware (CPU + limited VRAM).
- [**IPEX-LLM**](https://github.com/intel/ipex-llm) ⭐8.8k — Accelerate local LLM inference on Intel CPUs, iGPUs, and NPUs. Seamless integration with llama.cpp, Ollama, HF Transformers.
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — Cross-platform ML inference acceleration with CPU-optimized execution providers (OpenVINO, XNNPACK, CoreML).
- [**OpenVINO**](https://github.com/openvinotoolkit/openvino) — Intel's optimization toolkit for high-performance CPU inference across vision, language, and audio models.
- [**LLM-D**](https://github.com/llm-d/llm-d) — Achieves state-of-the-art inference performance with innovative architecture design.
- [**CTranslate2**](https://github.com/OpenNMT/CTranslate2) — Fast inference engine for Transformer models. Powers Faster Whisper, optimized for CPU with Intel MKL and ONNX.
- [**Trillim**](https://github.com/Trillim/Trillim) ⭐15 — Local AI stack for CPUs: CLI, Python SDK, and FastAPI server for BitNet and Bonsai (1-bit/ternary) bundles. Includes speech-to-text, text-to-speech, and image generation support.

-----

## 🖼️ Image Generation and Editing

- [**Stable Diffusion (CPU mode)**](https://github.com/CompVis/stable-diffusion) — Image generation model that works on CPU (slower but functional).
- [**Diffusion Bee**](https://diffusionbee.com/) — User-friendly GUI for Stable Diffusion on macOS, fully CPU compatible.
- [**InvokeAI**](https://github.com/invoke-ai/InvokeAI) — Professional Stable Diffusion interface with excellent CPU support.
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) — Node-based UI for image AI pipelines, supports CPU workflow.
- [**Fooocus**](https://github.com/lllyasviel/Fooocus) — Simplified Stable Diffusion, easier to use than ComfyUI.
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) — AI image upscaler, fast on CPU with great results.
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN) — Restores and improves old/blurry faces in photos, runs efficiently on CPU.
- [**Upscayl**](https://github.com/upscayl/upscayl) ⭐40k+ — Cross-platform AI image upscaler with simple GUI. Works great on CPU.

-----

## 🎤 Voice and Audio

- [**Whisper.cpp**](https://github.com/ggerganov/whisper.cpp) ⭐50k — Highly optimized Whisper (OpenAI) for CPU speech recognition. The fastest Whisper implementation for CPU.
- [**Faster Whisper**](https://github.com/SYSTRAN/faster-whisper) — Up to 4x faster than original Whisper using CTranslate2. Excellent CPU performance.
- [**Piper TTS**](https://github.com/rhasspy/piper) ⭐11k — Fast, local text-to-speech with small voice models (5-20MB). _Note: archived but still functional._
- [**Sherpa-ONNX**](https://github.com/k2-fsa/sherpa-onnx) ⭐12.9k — Comprehensive speech processing toolkit powered by ONNX Runtime. Speech-to-text, TTS, speaker diarization, VAD, keyword spotting — all on CPU. Cross-platform (x86, ARM, RISC-V, Android, iOS, Raspberry Pi).
- [**Supertonic**](https://github.com/supertone-inc/supertonic) ⭐9.7k — Lightning-fast, on-device, multilingual TTS running natively via ONNX. Python, JS, Rust, Swift bindings.
- [**MOSS-TTS-Nano**](https://github.com/OpenMOSS/MOSS-TTS-Nano) ⭐3.5k — Ultra-compact (0.1B params) multilingual TTS from OpenMOSS. Runs realtime on a 4-core CPU, supports Chinese + English + more, with ONNX CPU inference and voice cloning. Apache-2.0.
- [**Coqui TTS**](https://github.com/coqui-ai/TTS) ⭐45k — Open source text-to-speech engine with many voices and languages. CPU efficient.
- [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) — Multi-lingual large voice generation model from FunAudioLLM. Supports voice cloning.
- [**Amphion**](https://github.com/open-mmlab/Amphion) ⭐9.8k — Open-MMLab's toolkit for Audio, Music, and Speech Generation. Reproducible research with CPU mode.
- [**Vosk**](https://github.com/alphacep/vosk-api) — Offline speech recognition, very lightweight (50MB models).
- [**Bark (Suno)**](https://github.com/suno-ai/bark) ⭐39k — Realistic voice generation from text with CPU mode available.
- [**Qwen3-TTS**](https://github.com/gabriele-mastrapasqua/qwen3-tts) ⭐51 — Pure C inference engine for Qwen3-TTS. No Python, no PyTorch — just C and BLAS. Supports 0.6B/1.7B models.
- [**RVC (Voice Conversion)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Real-time voice conversion, CPU compatible.
- [**Demucs**](https://github.com/facebookresearch/demucs) — Separate music into vocals/instruments (CPU mode available).
- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Generate music from text descriptions (CPU mode supported).
- [**MusicGPT**](https://github.com/gabotechs/MusicGPT) — Generate music based on natural language prompts. Runs locally on CPU.
- [**acestep.cpp**](https://github.com/ServeurpersoCom/acestep.cpp) ⭐339 — Local AI music generation server with browser UI, powered by GGML. Describe a song + optional lyrics and get stereo 48kHz audio. Runs on CPU via BLAS-accelerated GGML backend with a dedicated CPU build script.
- [**FunMusic**](https://github.com/FunAudioLLM/FunMusic) — Fundamental toolkit for music generation, part of the FunAudioLLM ecosystem.

-----

## 👁️ Computer Vision

- [**OpenCV + DNN**](https://github.com/opencv/opencv) — Industry-standard vision framework with neural networks, fully CPU capable.
- [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics) ⭐57k — YOLOv8, v9, v10+ with `--device cpu`. Real-time object detection on CPU.
- [**MediaPipe**](https://github.com/google/mediapipe) — Google's library for hand, face, pose, and body tracking on CPU.
- [**FaceX**](https://github.com/facex-engine/facex) ⭐195 — Full face stack running entirely in the browser via WebAssembly. Detection, 576-point 3D mesh, recognition, anti-spoof. Zero server needed.
- [**ONNX Models**](https://github.com/onnx/models) — Collection of pre-trained, state-of-the-art ONNX models for vision, text, and audio.

-----

## 🔬 Small Models (Perfect for CPU)

### 💬 Language Models (< 7B parameters)

- [**Phi-3/Phi-4 Mini (3.8B/4B)**](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — Microsoft's ultra-efficient models with excellent quality for their size.
- [**Gemma 2B/3B**](https://huggingface.co/google/gemma-2b-it) — Google's compact models, very fast on CPU.
- [**TinyLlama (1.1B)**](https://github.com/jzhang38/TinyLlama) — Smallest LLaMA-based model, runs on 4GB RAM.
- [**Mistral 7B**](https://huggingface.co/mistralai/Mistral-7B-v0.1) — Best quality/size ratio, quantized versions run smoothly.
- [**LFM (Liquid Foundation Models)**](https://github.com/Liquid4All/cookbook) ⭐2k — Liquid AI's open-weight models with hybrid architecture (convolution + attention). Efficient on CPU, laptops, and edge devices. [Try on HF](https://huggingface.co/LiquidAI).
- [**Qwen 2.5 / 3 (3B/7B)**](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — Alibaba's efficient multilingual models. Qwen3 brings improved reasoning.
- [**DeepSeek 2.5 Lite**](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) — Efficient Mixture-of-Experts model, strong with quantized GGUF.
- [**StableLM 3B**](https://huggingface.co/stabilityai/stablelm-3b-4e1t) — Stability AI's compact yet capable model.
- [**SmolLM2 (135M-1.7B)**](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) — HuggingFace's tiny models for on-device and CPU inference.

### 🖼️ Image Models

- [**Stable Diffusion 1.5**](https://huggingface.co/runwayml/stable-diffusion-v1-5) — Classic version, lighter than v2/XL.
- [**TinySD**](https://huggingface.co/segmind/tiny-sd) — Distilled version, 50% smaller than SD 1.5.
- [**SSD-1B**](https://huggingface.co/segmind/SSD-1B) — 1B parameter SD model, 60% faster than SD 1.5.

### 🎤 Audio Models

- [**Whisper Tiny/Base**](https://huggingface.co/openai/whisper-tiny) — 39M/74M params for speech transcription.
- [**Piper TTS voices**](https://huggingface.co/rhasspy/piper-voices) — Tiny 5-20MB models for fast local TTS.

### 👁️ Vision Models

- [**MobileNetV3**](https://pytorch.org/vision/stable/models/mobilenetv3.html) — 5.4M params, image classification.
- [**YOLOv8n (nano)**](https://github.com/ultralytics/ultralytics) — Smallest YOLO, 3M params for object detection.
- [**EfficientNet-Lite**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) — Lightweight classification models optimized for CPU.

### 🧠 Embedding Models (for RAG/Search)

- [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22M params, fast text embeddings.
- [**BGE-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) — 33M params, excellent for retrieval.
- [**gte-small**](https://huggingface.co/Alibaba-NLP/gte-small) — 33M params, strong multilingual embeddings (Alibaba).

-----

## 🤖 AI Assistants & Agents

- [**Cline**](https://github.com/cline/cline) ⭐62k — Autonomous coding agent as an SDK, IDE extension, or CLI assistant. Works with local LLMs via Ollama/LM Studio.
- [**smolagents**](https://github.com/huggingface/smolagents) ⭐27.9k — HuggingFace's barebones library for agents that think in code. Supports local transformers and Ollama models, runs entirely on CPU.
- [**Open Interpreter**](https://github.com/KillianLucas/open-interpreter) — Code-executing AI assistant (works with local LLMs).
- [**AutoGPT**](https://github.com/Significant-Gravitas/AutoGPT) — Autonomous AI agent (supports local models).
- [**CrewAI**](https://github.com/crewAIInc/crewAI) ⭐52k — Multi-agent orchestration framework. Deploy autonomous agents that collaborate on complex tasks.
- [**LangGraph**](https://github.com/langchain-ai/langgraph) — Stateful, graph-based agent orchestration framework from LangChain.
- [**Dify**](https://github.com/langgenius/dify) ⭐142k — Production-ready platform for agentic workflow development. Visual builder + built-in RAG.
- [**Flowise**](https://github.com/FlowiseAI/Flowise) ⭐53k — Drag-and-drop visual tool to build LLM apps and AI agents. Self-host with Ollama.
- [**RAGFlow**](https://github.com/infiniflow/ragflow) ⭐81k — Leading open-source RAG engine with agent capabilities. Deep document understanding.
- [**AnythingLLM**](https://github.com/Mintplex-Labs/anything-llm) — Chat with your documents (PDFs, text), supports local models.
- [**PrivateGPT**](https://github.com/imartinez/privateGPT) — Ask questions to your documents 100% offline.
- [**CrewAI**](https://github.com/crewAIInc/crewAI) ⭐52k — Multi-agent orchestration for role-playing AI teams.

-----

## 💻 Coding Assistants

- [**Cline**](https://github.com/cline/cline) ⭐62k — Autonomous coding agent. VS Code extension + CLI + SDK. Supports Ollama, LM Studio, and any OpenAI-compatible local backend.
- [**Continue.dev**](https://github.com/continuedev/continue) — Open-source VS Code / JetBrains copilot. Use local models (Qwen 2.5 Coder, DeepSeek Coder) for autocomplete and chat.
- [**Aider**](https://github.com/paul-gauthier/aider) — Terminal-based AI pair programming with git integration. Works with local LLMs.
- [**Tabby**](https://github.com/TabbyML/tabby) — Self-hosted GitHub Copilot alternative. Code completion on CPU with StarCoder models.
- [**OpenCode**](https://github.com/sst/opencode) — The most-starred open-source AI coding agent of 2026. Designed for fast local development workflows.

-----

## 📚 Document & Knowledge (RAG)

- [**RAGFlow**](https://github.com/infiniflow/ragflow) ⭐81k — Deep document understanding RAG engine. PDF, DOCX, Excel — with agentic retrieval.
- [**Dify**](https://github.com/langgenius/dify) ⭐142k — Full-featured LLM app platform with built-in RAG pipeline, knowledge base, and agentic workflow.
- [**AnythingLLM**](https://github.com/Mintplex-Labs/anything-llm) — All-in-one desktop app for document-grounded conversations and private knowledge bases.
- [**PrivateGPT**](https://github.com/imartinez/privateGPT) — Offline Q&A over your documents (PDFs, text, code).
- [**MinerU**](https://github.com/opendatalab/MinerU) — Transforms complex documents (PDF, HTML, scans) into clean Markdown/JSON for RAG pipelines.
- [**Docling**](https://github.com/docling-project/docling) ⭐61.6k — IBM's document understanding library. Parses PDF, DOCX, PPTX, images and more into structured Markdown/JSON with layout preservation. Runs fully on CPU via ONNX Runtime with dedicated CPU-only installation.
- [**VelociRAG**](https://github.com/HaseebKhalid1507/VelociRAG) — Lightning-fast RAG for AI agents. ONNX-powered, 4-layer fusion, MCP server. No PyTorch needed.
- [**RAG-Anything**](https://github.com/hkuds/rag-anything) — All-in-one RAG framework with multiple retrieval strategies.
- [**LightRAG**](https://github.com/HKUDS/LightRAG) ⭐36.6k — Graph-based retrieval-augmented generation system. Indexes text into entity-relation graphs for efficient retrieval. Uses lightweight local embedding models and works with any local LLM backend (Ollama, llama.cpp). [EMNLP 2025].
- [**LlamaIndex**](https://github.com/run-llama/llama_index) — Data framework for connecting LLMs to external data sources (APIs, databases, documents).

-----

## 🔄 Agentic Workflows & Platforms

- [**Dify**](https://github.com/langgenius/dify) ⭐142k — Production-ready platform for building AI agents and workflows. Visual pipeline builder, RAG, MCP support, multi-model.
- [**Flowise**](https://github.com/FlowiseAI/Flowise) ⭐53k — Low-code/no-code platform to build LLM apps, chatbots, and agents visually.
- [**n8n**](https://github.com/n8n-io/n8n) — Advanced workflow automation with native AI capabilities and MCP nodes.
- [**Langflow**](https://github.com/langflow-ai/langflow) — Visual framework for building multi-agent and RAG applications.
- [**Haystack**](https://github.com/deepset-ai/haystack) — End-to-end NLP framework for building search, QA, and RAG pipelines.

-----

## 📊 Embeddings & Vector Databases

- [**Chroma**](https://github.com/chroma-core/chroma) — Lightweight, embedded vector database. Runs entirely on CPU, perfect for local RAG.
- [**Weaviate**](https://github.com/weaviate/weaviate) — Open-source vector search engine with hybrid search (vector + keyword). Runs on CPU.
- [**Qdrant**](https://github.com/qdrant/qdrant) — High-performance vector database with rich filtering. CPU-friendly for moderate scale.
- [**FAISS**](https://github.com/facebookresearch/faiss) — Meta's library for efficient similarity search and dense vector clustering. CPU-optimized.
- [**Voyager**](https://github.com/spotify/voyager) — Spotify's approximate nearest neighbor search library. Lightweight and fast on CPU.

-----

## 💡 Creative AI & Miscellaneous

- [**Amphion**](https://github.com/open-mmlab/Amphion) ⭐9.8k — Audio, music, and speech generation toolkit. TTS, SVC, music gen — all on CPU.
- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Generate music from text descriptions (CPU mode supported).
- [**FunMusic**](https://github.com/FunAudioLLM/FunMusic) — Music generation toolkit from FunAudioLLM.
- [**Diarize**](https://github.com/FoxNoseTech/diarize) ⭐71 — Speaker diarization — "who spoke when?" CPU-only, no API keys, 8x faster than real-time.
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) — CPU-optimized inference for LLaMA and compatible models.
- [**Roop**](https://github.com/s0md3v/roop) — One-click face swap tool (CPU compatible).

-----

## 🛠️ Development Tools

- [**Transformers (Hugging Face)**](https://github.com/huggingface/transformers) — Load and run any model with CPU backend (`device="cpu"`).
- [**Transformers.js**](https://github.com/huggingface/transformers.js) ⭐16.1k — HuggingFace's Transformers for the browser. Run NLP, vision, and audio models directly in JavaScript — no server, no GPU. Powered by ONNX Runtime WebAssembly.
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — Accelerate ML inference on CPU with optimizations (XNNPACK, OpenVINO, CoreML).
- [**OpenVINO**](https://github.com/openvinotoolkit/openvino) — Intel's optimization toolkit for CPU inference across any model.
- [**BitNet**](https://github.com/microsoft/BitNet) ⭐39k — Official framework for 1-bit LLM inference. Revolutionary efficiency.
- [**LMDeploy**](https://github.com/InternLM/lmdeploy) — Model compression and deployment toolkit for efficient CPU serving.
- [**CTranslate2**](https://github.com/OpenNMT/CTranslate2) — Fast transformer inference on CPU. Powers Faster Whisper and many production systems.
- [**MLX**](https://github.com/ml-explore/mlx) — Apple's ML framework optimized for Apple Silicon (M-series CPUs). Excellent for local inference.
- [**Candle**](https://github.com/huggingface/candle) ⭐20k — HuggingFace's minimalist ML framework for Rust with CPU-first design. Run LLMs, vision models, and more locally with zero GPU dependency.
- [**llmfit**](https://github.com/AlexsJones/llmfit) ⭐28.3k — Rust CLI tool that detects your hardware and finds the best LLMs for your RAM, CPU, and GPU. One command to right-size models — scores quality, speed, fit, and context for hundreds of models. Supports Ollama, llama.cpp, MLX, LM Studio backends.

-----

## ⚙️ Tips for Running on CPU

### 🎯 General Optimization

- 🔧 Use **smaller model versions** (`tiny`, `small`, `mini`, `nano`)
- ⚡ Apply **quantization** (Q4, Q5, Q8) to reduce RAM usage by 50-75%
- 🧩 Use optimized runtimes: **GGUF/GGML**, **ONNX Runtime**, or **OpenVINO**
- 🚀 Enable **multi-threading** to utilize all CPU cores
- 📉 Reduce resolution/steps in image generation for faster results
- 🔄 Use batch size 1 for CPU inference (larger batches don't help)

### 📊 Model Size Guide

|Size  |Parameters|RAM Needed|Speed on CPU|Use Case             |
|------|----------|----------|------------|---------------------|
|Tiny  |< 1B      |2-4GB     |⚡⚡⚡⚡        |Testing, edge devices|
|Small |1-3B      |4-8GB     |⚡⚡⚡         |Daily use, chatbots  |
|Medium|3-7B      |8-16GB    |⚡⚡          |Quality balance      |
|Large |7B+       |16GB+     |⚡           |Best quality (slower)|

### 🔄 Quantization Formats

- **GGUF (Q4_K_M)** — Best for llama.cpp/Ollama (4-bit), excellent quality/size ratio
- **GPTQ** — Good compression with decent inference speed
- **AWQ** — Better quality than GPTQ at the same model size
- **ONNX** — Cross-platform optimization, works with many frameworks
- **BitNet (1-bit)** — Next-gen extreme quantization, 90% size reduction

### 💡 2026 Tips & Trends

- **1-bit LLMs are here**: Microsoft's BitNet delivers surprisingly good quality at 1-bit precision. Runs 10x faster on CPU.
- **MoE models save RAM**: Mixture-of-Experts (DeepSeek, Qwen3-MoE) activate only a fraction of parameters per token.
- **Apple Silicon is a CPU powerhouse**: Use MLX or llama.cpp Metal backend on M1/M2/M3/M4 Macs for near-GPU speeds.
- **Hybrid CPU/GPU runtimes**: Tools like Krasis automatically split models across available hardware.
- **WebAssembly AI**: Run models directly in the browser (FaceX, Transformers.js) — zero install.

### 🚀 Quick Start Example

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Run a small model
ollama run phi3:mini

# For transcription with Whisper
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make
./main -m models/ggml-base.bin -f audio.wav

# For local coding assistant (Continue)
# Install the VS Code extension, point it to Ollama/local model
```

-----

## 🤝 Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

- Add new projects that work well on CPU
- Fix broken links or outdated information
- Improve documentation and examples
- Keep star counts and descriptions up to date

-----

## 📜 License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

This list is licensed under CC0 1.0 Universal and follows the [Awesome](https://github.com/sindresorhus/awesome) format.

-----

## ⭐ Star History

If you find this list helpful, please consider giving it a star on GitHub!

-----

**Made with ❤️ by the community** | **Last updated:** June 2026
