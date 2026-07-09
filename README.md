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
- [**GPT4All**](https://github.com/nomic-ai/gpt4all)— Simple interface to run LLMs locally with several CPU-optimized models included.
- [**LM Studio**](https://lmstudio.ai/) — Beautiful GUI app for local LLMs, automatically supports CPU execution.
- [**LMCP**](https://github.com/lmcp-ai/lmcp) — Local Model Control Protocol — run any LLM with a unified API, auto-discovers Ollama, llama.cpp, vLLM backends.
- [**Jan.ai**](https://github.com/janhq/jan) — ChatGPT-like interface that runs 100% offline with a clean, modern UI.
- [**LocalAI**](https://github.com/mudler/LocalAI)— OpenAI-compatible API for running local models. Drop-in replacement that also supports vision, voice, image gen — no GPU required.
- [**Kobold.cpp**](https://github.com/LostRuins/koboldcpp) — Lightweight inference engine for GGUF models with built-in web UI.
- [**Open WebUI**](https://github.com/open-webui/open-webui) — Self-hosted, offline ChatGPT-style interface for Ollama, with RAG, web search, and multi-user support.

-----

## ⚡ Inference Engines & Runtimes

- [**llama.cpp**](https://github.com/ggml-org/llama.cpp)— The gold standard for CPU-optimized LLM inference in C/C++. Powers Ollama, LM Studio, and most local LLM tools.
- [**llamafile**](https://github.com/mozilla-ai/llamafile)— Mozilla's single-file LLM executable. Distribute and run LLMs as a standalone binary — no installation, no dependencies, no GPU required. Built on llama.cpp, supports CPU inference out of the box.
- [**mistral.rs**](https://github.com/EricLBuehler/mistral.rs) — Fast, flexible LLM inference engine in Rust, built on Candle. Run any Hugging Face model or GGUF file with zero config — prebuilt CPU binaries for Linux/Windows and CPU Docker images mean no GPU or CUDA toolkit needed. Smart in-situ quantization (GGUF, GPTQ, AWQ) and hardware-aware tuning optimize for your CPU.
- [**BitNet**](https://github.com/microsoft/BitNet)— Microsoft's official inference framework for 1-bit LLMs. Extremely efficient on CPU.
- [**eLLM**](https://github.com/lucienhuangfu/eLLM)— Rust-based inference engine that claims to run LLMs faster on CPU than on GPU through aggressive optimization.
- [**Krasis**](https://github.com/brontoguana/krasis)— Hybrid LLM runtime focusing on efficient execution of larger models on consumer hardware (CPU + limited VRAM).
- [**IPEX-LLM**](https://github.com/intel/ipex-llm)— Accelerate local LLM inference on Intel CPUs, iGPUs, and NPUs. Seamless integration with llama.cpp, Ollama, HF Transformers.
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — Cross-platform ML inference acceleration with CPU-optimized execution providers (OpenVINO, XNNPACK, CoreML).
- [**OpenVINO**](https://github.com/openvinotoolkit/openvino) — Intel's optimization toolkit for high-performance CPU inference across vision, language, and audio models.
- [**LLM-D**](https://github.com/llm-d/llm-d) — Achieves state-of-the-art inference performance with innovative architecture design.
- [**CTranslate2**](https://github.com/OpenNMT/CTranslate2) — Fast inference engine for Transformer models. Powers Faster Whisper, optimized for CPU with Intel MKL and ONNX.
- [**Trillim**](https://github.com/Trillim/Trillim)— Local AI stack for CPUs: CLI, Python SDK, and FastAPI server for BitNet and Bonsai (1-bit/ternary) bundles. Includes speech-to-text, text-to-speech, and image generation support.
- [**exo**](https://github.com/exo-explore/exo) — Connect multiple devices (Macs, Linux machines, phones) into a single AI cluster. Automatically discovers devices, splits models across them, and runs frontier LLMs on CPU (Linux) or Apple Silicon MLX (macOS). OpenAI-compatible API.

-----

## 🖼️ Image Generation and Editing

- [**FastSD CPU**](https://github.com/rupeshs/fastsdcpu) ⭐2.1k — Fast Stable Diffusion on CPU and AI PC. Supports SDXL, SD 1.5, LCM, and FLUX — optimized for CPU inference.
- [**Fooocus**](https://github.com/lllyasviel/Fooocus) — Simplified Stable Diffusion, easier to use than ComfyUI.
- [**InvokeAI**](https://github.com/invoke-ai/InvokeAI) — Professional Stable Diffusion interface with excellent CPU support.
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) — Node-based UI for image AI pipelines, supports CPU workflow.
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) — AI image upscaler, fast on CPU with great results.
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN) — Restores and improves old/blurry faces in photos, runs efficiently on CPU.
- [**Upscayl**](https://github.com/upscayl/upscayl) — Cross-platform AI image upscaler with simple GUI. Works great on CPU.

-----

## 🎤 Voice and Audio

- [**Whisper.cpp**](https://github.com/ggerganov/whisper.cpp)— Highly optimized Whisper (OpenAI) for CPU speech recognition. The fastest Whisper implementation for CPU.
- [**Faster Whisper**](https://github.com/SYSTRAN/faster-whisper) — Up to 4x faster than original Whisper using CTranslate2. Excellent CPU performance.
- [**FunASR-GGML**](https://github.com/huaxin0/FunASR-GGML) — Pure C++17 speech recognition engine powered by FunASR's SenseVoice architecture. GGML/GGUF format, real-time mic streaming with VAD, single-file models, no Python dependency. Excellent for multilingual and Chinese ASR on CPU.
- [**Piper TTS**](https://github.com/rhasspy/piper)— Fast, local text-to-speech with small voice models (5-20MB). _Note: archived but still functional._
- [**Sherpa-ONNX**](https://github.com/k2-fsa/sherpa-onnx)— Comprehensive speech processing toolkit powered by ONNX Runtime. Speech-to-text, TTS, speaker diarization, VAD, keyword spotting — all on CPU. Cross-platform (x86, ARM, RISC-V, Android, iOS, Raspberry Pi).
- [**Supertonic**](https://github.com/supertone-inc/supertonic)— Lightning-fast, on-device, multilingual TTS running natively via ONNX. Python, JS, Rust, Swift bindings.
- [**MOSS-TTS-Nano**](https://github.com/OpenMOSS/MOSS-TTS-Nano)— Ultra-compact (0.1B params) multilingual TTS from OpenMOSS. Runs realtime on a 4-core CPU, supports Chinese + English + more, with ONNX CPU inference and voice cloning. Apache-2.0.
- [**VoxCPM**](https://github.com/OpenBMB/VoxCPM)— Tokenizer-free TTS from OpenBMB that generates speech via diffusion autoregressive architecture. VoxCPM2 (2B params) supports 30 languages, voice design, controllable voice cloning, and 48kHz audio. Runs on CPU via `--device cpu`, with GGML and ONNX CPU builds available.
- [**Pocket TTS**](https://github.com/kyutai-labs/pocket-tts) — Lightweight, CPU-first TTS from Kyutai Labs. 100M params, runs ~6x faster than real-time on a MacBook Air M4 using only 2 CPU cores. Voice cloning, multi-language (EN/FR/DE/PT/IT/ES), streaming audio with ~200ms latency, and handles infinitely long text. `pip install pocket-tts`.
- [**Kokoro-FastAPI**](https://github.com/remsky/Kokoro-FastAPI) — Dockerized FastAPI wrapper for the Kokoro-82M TTS model with prebuilt CPU images. OpenAI-compatible Speech endpoint, 8 languages, voice mixing, and per-word timestamps. Deploy on any CPU server with a single `docker run`.
- [**Chatterbox TTS Server**](https://github.com/devnen/Chatterbox-TTS-Server) — Self-host Resemble AI's Chatterbox TTS (Original, Multilingual, Turbo) behind an OpenAI-compatible API with Web UI. Voice cloning, audiobook generation, 23 languages. Runs on CPU with automatic GPU fallback.
- [**OmniVoice-Studio**](https://github.com/debpalash/OmniVoice-Studio) — The open-source ElevenLabs alternative: voice cloning, TTS, ASR, dubbing, and dictation in a single desktop app. Supports 646 languages, auto-detects CUDA/MPS/ROCm/CPU and auto-offloads to CPU when no GPU is available. Fully local, no API keys.
- [**omnivoice.cpp**](https://github.com/ServeurpersoCom/omnivoice.cpp) — GGML-powered local TTS with voice cloning and voice design across 646 languages. C++17 port of OmniVoice with Q8_0 quantization, 24 kHz output, and a dedicated CPU build script. Embeddable C API for integration.

- [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) — Multi-lingual large voice generation model from FunAudioLLM. Supports voice cloning.
- [**Amphion**](https://github.com/open-mmlab/Amphion)— Open-MMLab's toolkit for Audio, Music, and Speech Generation. Reproducible research with CPU mode.
- [**Vosk**](https://github.com/alphacep/vosk-api) — Offline speech recognition, very lightweight (50MB models).

- [**Qwen3-TTS**](https://github.com/gabriele-mastrapasqua/qwen3-tts)— Pure C inference engine for Qwen3-TTS. No Python, no PyTorch — just C and BLAS. Supports 0.6B/1.7B models.
- [**RVC (Voice Conversion)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Real-time voice conversion, CPU compatible.
- [**Demucs**](https://github.com/facebookresearch/demucs) — Separate music into vocals/instruments (CPU mode available).
- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Generate music from text descriptions (CPU mode supported).
- [**MusicGPT**](https://github.com/gabotechs/MusicGPT) — Generate music based on natural language prompts. Runs locally on CPU.
- [**acestep.cpp**](https://github.com/ServeurpersoCom/acestep.cpp)— Local AI music generation server with browser UI, powered by GGML. Describe a song + optional lyrics and get stereo 48kHz audio. Runs on CPU via BLAS-accelerated GGML backend with a dedicated CPU build script.
- [**FunMusic**](https://github.com/FunAudioLLM/FunMusic) — Fundamental toolkit for music generation, part of the FunAudioLLM ecosystem.

-----

## 👁️ Computer Vision

- [**OpenCV + DNN**](https://github.com/opencv/opencv) — Industry-standard vision framework with neural networks, fully CPU capable.
- [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics)— YOLOv8, v9, v10+ with `--device cpu`. Real-time object detection on CPU.
- [**MediaPipe**](https://github.com/google/mediapipe) — Google's library for hand, face, pose, and body tracking on CPU.
- [**FaceX**](https://github.com/facex-engine/facex)— Full face stack running entirely in the browser via WebAssembly. Detection, 576-point 3D mesh, recognition, anti-spoof. Zero server needed.
- [**ONNX Models**](https://github.com/onnx/models) — Collection of pre-trained, state-of-the-art ONNX models for vision, text, and audio.

-----

## 🔬 Small Models (Perfect for CPU)

### 💬 Language Models (< 7B parameters)

- [**Phi-4 Mini (3.8B)**](https://huggingface.co/microsoft/Phi-4-mini-instruct) — Microsoft's ultra-efficient model with excellent quality for its size. Strong on math, code, and reasoning.
- [**Gemma 4 (E2B/E4B)**](https://huggingface.co/google/gemma-4-E2B-it) — Google's latest compact models with multimodal support. E2B and E4B are tiny and fast on CPU.
- [**Ministral 3B**](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) — Mistral's small but capable model, excellent for CPU inference.
- [**LFM 2.5 (350M–1.2B)**](https://huggingface.co/LiquidAI/LFM2.5-350M) — Liquid AI's hybrid architecture (conv + attention) models. Vision-capable variants (VL) from 450M to 1.6B. Blazing fast on CPU and edge devices.
- [**Qwen 3.6**](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — Alibaba's latest with 27B dense and 35B-A3B MoE variants. MoE activates only 3B params per token — great speed/quality ratio.
- [**DeepSeek-R1-Distill-Qwen (1.5B–7B)**](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) — Distilled reasoning models from DeepSeek. Tiny 1.5B variant runs great on CPU.
- [**SmolLM3 (3B)**](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) — HuggingFace's latest tiny model, multilingual (8 languages), optimized for on-device and CPU inference.

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
- [**Namo-R1**](https://github.com/lucasjinreal/Namo-R1) — 500M parameter Vision-Language Model trained from scratch, surpassing Moondream2 and SmolVLM. Designed for CPU-first inference with real-time performance on consumer hardware.

### 🧠 Embedding Models (for RAG/Search)

- [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22M params, fast text embeddings.
- [**BGE-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) — 33M params, excellent for retrieval.
- [**gte-small**](https://huggingface.co/Alibaba-NLP/gte-small) — 33M params, strong multilingual embeddings (Alibaba).

-----

## 🤖 AI Assistants & Agents

- [**Cline**](https://github.com/cline/cline)— Autonomous coding agent as an SDK, IDE extension, or CLI assistant. Works with local LLMs via Ollama/LM Studio.
- [**smolagents**](https://github.com/huggingface/smolagents)— HuggingFace's barebones library for agents that think in code. Supports local transformers and Ollama models, runs entirely on CPU.
- [**Open Interpreter**](https://github.com/KillianLucas/open-interpreter) — Code-executing AI assistant (works with local LLMs).
- [**AutoGPT**](https://github.com/Significant-Gravitas/AutoGPT) — Autonomous AI agent (supports local models).
- [**CrewAI**](https://github.com/crewAIInc/crewAI)— Multi-agent orchestration framework. Deploy autonomous agents that collaborate on complex tasks.
- [**LangGraph**](https://github.com/langchain-ai/langgraph) — Stateful, graph-based agent orchestration framework from LangChain.
- [**Dify**](https://github.com/langgenius/dify)— Production-ready platform for agentic workflow development. Visual builder + built-in RAG.
- [**Flowise**](https://github.com/FlowiseAI/Flowise)— Drag-and-drop visual tool to build LLM apps and AI agents. Self-host with Ollama.
- [**RAGFlow**](https://github.com/infiniflow/ragflow)— Leading open-source RAG engine with agent capabilities. Deep document understanding.
- [**AnythingLLM**](https://github.com/Mintplex-Labs/anything-llm) — Chat with your documents (PDFs, text), supports local models.
- [**PrivateGPT**](https://github.com/imartinez/privateGPT) — Ask questions to your documents 100% offline.
- [**Local Deep Research**](https://github.com/LearningCircuit/local-deep-research) — AI-powered research assistant that performs deep, agentic research using local LLMs (Ollama, llama.cpp). Searches across web, academic papers, and your own documents, then synthesizes findings into cited reports. Runs fully on CPU with Docker Compose or pip install.
- [**Page Assist**](https://github.com/n4ze3m/page-assist) — Open-source browser extension that adds a sidebar and web UI for your local AI models. Chat with any webpage, summarize content, and use AI inline while browsing. Works with Ollama, LM Studio, and any OpenAI-compatible endpoint — fully local, no GPU needed.
- [**CrewAI**](https://github.com/crewAIInc/crewAI)— Multi-agent orchestration for role-playing AI teams.

-----

## 💻 Coding Assistants

- [**Cline**](https://github.com/cline/cline)— Autonomous coding agent. VS Code extension + CLI + SDK. Supports Ollama, LM Studio, and any OpenAI-compatible local backend.
- [**Continue.dev**](https://github.com/continuedev/continue) — Open-source VS Code / JetBrains copilot. Use local models (Qwen 2.5 Coder, DeepSeek Coder) for autocomplete and chat.
- [**Aider**](https://github.com/paul-gauthier/aider) — Terminal-based AI pair programming with git integration. Works with local LLMs.
- [**Tabby**](https://github.com/TabbyML/tabby) — Self-hosted GitHub Copilot alternative. Code completion on CPU with StarCoder models.
- [**OpenCode**](https://github.com/sst/opencode) — The most-starred open-source AI coding agent of 2026. Designed for fast local development workflows.
- [**Crush**](https://github.com/charmbracelet/crush) — Terminal-based agentic coding assistant from Charm. Auto-discovers local models from Ollama, LM Studio, litellm, and any OpenAI-compatible backend — run it fully offline on CPU. LSP-enhanced, MCP-extensible, cross-platform (macOS, Linux, Windows, BSD).

-----

## 📚 Document & Knowledge (RAG)

- [**RAGFlow**](https://github.com/infiniflow/ragflow)— Deep document understanding RAG engine. PDF, DOCX, Excel — with agentic retrieval.
- [**Dify**](https://github.com/langgenius/dify)— Full-featured LLM app platform with built-in RAG pipeline, knowledge base, and agentic workflow.
- [**AnythingLLM**](https://github.com/Mintplex-Labs/anything-llm) — All-in-one desktop app for document-grounded conversations and private knowledge bases.
- [**PrivateGPT**](https://github.com/imartinez/privateGPT) — Offline Q&A over your documents (PDFs, text, code).
- [**MinerU**](https://github.com/opendatalab/MinerU) — Transforms complex documents (PDF, HTML, scans) into clean Markdown/JSON for RAG pipelines.
- [**Marker**](https://github.com/datalab-to/marker) — Converts PDF, image, PPTX, DOCX, XLSX, HTML, and EPUB to Markdown and JSON quickly and accurately. Works on GPU, CPU, or MPS. Formats tables, equations, code blocks, and extracts images. Optionally boosts accuracy with LLMs (Ollama/Gemini).
- [**Docling**](https://github.com/docling-project/docling)— IBM's document understanding library. Parses PDF, DOCX, PPTX, images and more into structured Markdown/JSON with layout preservation. Runs fully on CPU via ONNX Runtime with dedicated CPU-only installation.
- [**VelociRAG**](https://github.com/HaseebKhalid1507/VelociRAG) — Lightning-fast RAG for AI agents. ONNX-powered, 4-layer fusion, MCP server. No PyTorch needed.
- [**RAG-Anything**](https://github.com/hkuds/rag-anything) — All-in-one RAG framework with multiple retrieval strategies.
- [**LightRAG**](https://github.com/HKUDS/LightRAG)— Graph-based retrieval-augmented generation system. Indexes text into entity-relation graphs for efficient retrieval. Uses lightweight local embedding models and works with any local LLM backend (Ollama, llama.cpp). [EMNLP 2025].
- [**LlamaIndex**](https://github.com/run-llama/llama_index) — Data framework for connecting LLMs to external data sources (APIs, databases, documents).

-----

## 🔄 Agentic Workflows & Platforms

- [**Dify**](https://github.com/langgenius/dify)— Production-ready platform for building AI agents and workflows. Visual pipeline builder, RAG, MCP support, multi-model.
- [**Flowise**](https://github.com/FlowiseAI/Flowise)— Low-code/no-code platform to build LLM apps, chatbots, and agents visually.
- [**n8n**](https://github.com/n8n-io/n8n) — Advanced workflow automation with native AI capabilities and MCP nodes.
- [**Langflow**](https://github.com/langflow-ai/langflow) — Visual framework for building multi-agent and RAG applications.
- [**Haystack**](https://github.com/deepset-ai/haystack) — End-to-end NLP framework for building search, QA, and RAG pipelines.
- [**Open Multi-Agent**](https://github.com/open-multi-agent/open-multi-agent) — TypeScript-native multi-agent orchestration framework. Describe a goal and a coordinator automatically decomposes it into a task DAG that runs on any LLM, including local models via Ollama — no GPU required.

-----

## 📊 Embeddings & Vector Databases

- [**Chroma**](https://github.com/chroma-core/chroma) — Lightweight, embedded vector database. Runs entirely on CPU, perfect for local RAG.
- [**Weaviate**](https://github.com/weaviate/weaviate) — Open-source vector search engine with hybrid search (vector + keyword). Runs on CPU.
- [**Qdrant**](https://github.com/qdrant/qdrant) — High-performance vector database with rich filtering. CPU-friendly for moderate scale.
- [**FAISS**](https://github.com/facebookresearch/faiss) — Meta's library for efficient similarity search and dense vector clustering. CPU-optimized.
- [**Voyager**](https://github.com/spotify/voyager) — Spotify's approximate nearest neighbor search library. Lightweight and fast on CPU.
- [**zvec**](https://github.com/alibaba/zvec) — Alibaba's lightweight, in-process vector database. Blazing-fast similarity search with dense + sparse vectors, full-text search, and hybrid retrieval. Embedded library — no servers, no config, runs on CPU anywhere your code runs. Python, Node.js, Go, Rust SDKs.
- [**LEANN**](https://github.com/StarTrail-org/LEANN) — Innovative vector database that uses graph-based selective recomputation to cut storage by 97%. Index millions of documents and run RAG entirely on your laptop with a dedicated `leann[cpu]` install. MCP-native, published at MLSys 2026.

-----

## 💡 Creative AI & Miscellaneous

- [**Amphion**](https://github.com/open-mmlab/Amphion)— Audio, music, and speech generation toolkit. TTS, SVC, music gen — all on CPU.
- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Generate music from text descriptions (CPU mode supported).
- [**FunMusic**](https://github.com/FunAudioLLM/FunMusic) — Music generation toolkit from FunAudioLLM.
- [**Diarize**](https://github.com/FoxNoseTech/diarize)— Speaker diarization — "who spoke when?" CPU-only, no API keys, 8x faster than real-time.
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) — CPU-optimized inference for LLaMA and compatible models.
- [**FaceFusion**](https://github.com/facefusion/facefusion) ⭐29k — Next-gen face swap, lip-sync, and enhancement. CPU and GPU, 30+ languages, actively maintained.

-----

## 🛠️ Development Tools

- [**Transformers (Hugging Face)**](https://github.com/huggingface/transformers) — Load and run any model with CPU backend (`device="cpu"`).
- [**Transformers.js**](https://github.com/huggingface/transformers.js)— HuggingFace's Transformers for the browser. Run NLP, vision, and audio models directly in JavaScript — no server, no GPU. Powered by ONNX Runtime WebAssembly.
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — Accelerate ML inference on CPU with optimizations (XNNPACK, OpenVINO, CoreML).
- [**OpenVINO**](https://github.com/openvinotoolkit/openvino) — Intel's optimization toolkit for CPU inference across any model.
- [**BitNet**](https://github.com/microsoft/BitNet)— Official framework for 1-bit LLM inference. Revolutionary efficiency.
- [**LMDeploy**](https://github.com/InternLM/lmdeploy) — Model compression and deployment toolkit for efficient CPU serving.
- [**CTranslate2**](https://github.com/OpenNMT/CTranslate2) — Fast transformer inference on CPU. Powers Faster Whisper and many production systems.
- [**MLX**](https://github.com/ml-explore/mlx) — Apple's ML framework optimized for Apple Silicon (M-series CPUs). Excellent for local inference.
- [**Candle**](https://github.com/huggingface/candle)— HuggingFace's minimalist ML framework for Rust with CPU-first design. Run LLMs, vision models, and more locally with zero GPU dependency.
- [**llmfit**](https://github.com/AlexsJones/llmfit)— Rust CLI tool that detects your hardware and finds the best LLMs for your RAM, CPU, and GPU. One command to right-size models — scores quality, speed, fit, and context for hundreds of models. Supports Ollama, llama.cpp, MLX, LM Studio backends.
- [**whichllm**](https://github.com/Andyyyy64/whichllm) — Find the best local LLM that actually runs on your hardware. Auto-detects GPU, CPU, and RAM, then ranks models from HuggingFace by real-time benchmarks and quality scores. CPU-only mode supported — `pip install whichllm` and you're ready.

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
