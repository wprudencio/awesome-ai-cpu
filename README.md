# 🧠 Awesome AI Projects for CPU

![Awesome](https://awesome.re/badge.svg) ![MIT License](https://img.shields.io/badge/license-MIT-brightgreen) ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

> A curated list of **open source AI projects** that run well on **CPU**, no GPU required.  
> Perfect for makers, indie developers, and local AI experiments.

-----

## 📑 Table of Contents

- [Language Models](#-language-models-chatbots--text)
- [Image Generation and Editing](#️-image-generation-and-editing)
- [Voice and Audio](#-voice-and-audio)
- [Computer Vision](#️-computer-vision)
- [Small Models](#-small-models-perfect-for-cpu)
- [AI Assistants & Agents](#-ai-assistants--agents)
- [Document & Knowledge](#-document--knowledge)
- [Creative AI](#-creative-ai--miscellaneous)
- [Development Tools](#️-development-tools)
- [Tips for CPU](#️-tips-for-running-on-cpu)

-----

## 🧠 Language Models (Chatbots / Text)

- [**GPT4All**](https://github.com/nomic-ai/gpt4all) — Simple interface to run LLMs locally with several CPU-optimized models included.
- [**Ollama**](https://github.com/ollama/ollama) — Run LLaMA, Mistral, Gemma models with excellent CPU support and easy CLI.
- [**LM Studio**](https://lmstudio.ai/) — Beautiful GUI app for local LLMs, automatically supports CPU execution.
- [**Text Generation WebUI**](https://github.com/oobabooga/text-generation-webui) — Powerful web interface for LLMs with extensive CPU mode options.
- [**Jan.ai**](https://github.com/janhq/jan) — ChatGPT-like interface that runs 100% offline with a clean, modern UI.
- [**LocalAI**](https://github.com/mudler/LocalAI) — OpenAI-compatible API for running local models, drop-in replacement.
- [**Kobold.cpp**](https://github.com/LostRuins/koboldcpp) — Lightweight inference engine for GGUF models with built-in web UI.

-----

## 🖼️ Image Generation and Editing

- [**Stable Diffusion (CPU mode)**](https://github.com/CompVis/stable-diffusion) — Image generation model that works on CPU (slower but functional).
- [**Diffusion Bee**](https://diffusionbee.com/) — User-friendly GUI for Stable Diffusion on macOS, fully CPU compatible.
- [**InvokeAI**](https://github.com/invoke-ai/InvokeAI) — Professional Stable Diffusion interface with excellent CPU support.
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) — Node-based UI for image AI pipelines, supports CPU workflow.
- [**Fooocus**](https://github.com/lllyasviel/Fooocus) — Simplified Stable Diffusion, easier to use than ComfyUI.
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) — AI image upscaler, fast on CPU with great results.
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN) — Restores and improves old/blurry faces in photos, runs efficiently on CPU.
- [**Upscayl**](https://github.com/upscayl/upscayl) — Cross-platform AI image upscaler with simple GUI.

-----

## 🎤 Voice and Audio

- [**Whisper.cpp**](https://github.com/ggerganov/whisper.cpp) — Highly optimized Whisper (OpenAI) for CPU speech recognition.
- [**Piper TTS**](https://github.com/rhasspy/piper) — Fast, local text-to-speech with small voice models (5-20MB).
- [**Coqui TTS**](https://github.com/coqui-ai/TTS) — Open source text-to-speech engine, efficient on CPU with many voices.
- [**Vosk**](https://github.com/alphacep/vosk-api) — Offline speech recognition, very lightweight (50MB models).
- [**Bark (Suno)**](https://github.com/suno-ai/bark) — Realistic voice generation from text with CPU mode available.
- [**RVC (Voice Conversion)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Real-time voice conversion, CPU compatible.
- [**Demucs**](https://github.com/facebookresearch/demucs) — Separate music into vocals/instruments (CPU mode available).

-----

## 👁️ Computer Vision

- [**OpenCV + DNN**](https://github.com/opencv/opencv) — Industry-standard vision framework with neural networks, fully CPU capable.
- [**YOLOv5 (CPU mode)**](https://github.com/ultralytics/yolov5) — Real-time object detection with CPU option `--device cpu`.
- [**YOLOv8**](https://github.com/ultralytics/ultralytics) — Latest YOLO version with improved CPU performance.
- [**MediaPipe**](https://github.com/google/mediapipe) — Google’s library for hand, face, pose, and body tracking on CPU.

-----

## 🔬 Small Models (Perfect for CPU)

### 💬 Language Models (< 7B parameters)

- [**Phi-3 Mini (3.8B)**](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — Microsoft’s ultra-efficient model with excellent quality.
- [**Gemma 2B**](https://huggingface.co/google/gemma-2b-it) — Google’s compact model, very fast on CPU.
- [**TinyLlama (1.1B)**](https://github.com/jzhang38/TinyLlama) — Smallest LLaMA-based model, runs on 4GB RAM.
- [**Mistral 7B**](https://huggingface.co/mistralai/Mistral-7B-v0.1) — Best quality/size ratio, quantized versions run smoothly.
- [**Qwen 2.5 (3B/7B)**](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — Alibaba’s efficient multilingual model.
- [**StableLM 3B**](https://huggingface.co/stabilityai/stablelm-3b-4e1t) — Stability AI’s compact yet capable model.

### 🖼️ Image Models

- [**Stable Diffusion 1.5**](https://huggingface.co/runwayml/stable-diffusion-v1-5) — Classic version, lighter than v2/XL.
- [**TinySD**](https://huggingface.co/segmind/tiny-sd) — Distilled version, 50% smaller than SD 1.5.

### 🎤 Audio Models

- [**Whisper Tiny/Base**](https://huggingface.co/openai/whisper-tiny) — 39M/74M params for speech transcription.

### 👁️ Vision Models

- [**MobileNetV3**](https://pytorch.org/vision/stable/models/mobilenetv3.html) — 5.4M params, image classification.
- [**YOLOv8n (nano)**](https://github.com/ultralytics/ultralytics) — Smallest YOLO, 3M params for object detection.

### 🧠 Embedding Models (for RAG/Search)

- [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22M params, fast text embeddings.
- [**BGE-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) — 33M params, excellent for retrieval.

-----

## 🤖 AI Assistants & Agents

- [**Open Interpreter**](https://github.com/KillianLucas/open-interpreter) — Code-executing AI assistant (works with local LLMs).
- [**AutoGPT**](https://github.com/Significant-Gravitas/AutoGPT) — Autonomous AI agent (supports local models).

-----

## 📚 Document & Knowledge

- [**AnythingLLM**](https://github.com/Mintplex-Labs/anything-llm) — Chat with your documents (PDFs, text), supports local models.
- [**PrivateGPT**](https://github.com/imartinez/privateGPT) — Ask questions to your documents 100% offline.

-----

## 💡 Creative AI & Miscellaneous

- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Generate music from text descriptions (CPU mode supported).
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) — CPU-optimized inference for LLaMA and compatible models.
- [**Roop**](https://github.com/s0md3v/roop) — One-click face swap tool (CPU compatible).

-----

## 🛠️ Development Tools

- [**Transformers (Hugging Face)**](https://github.com/huggingface/transformers) — Load and run any model with CPU backend.
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — Accelerate ML inference on CPU with optimizations.
- [**OpenVINO**](https://github.com/openvinotoolkit/openvino) — Intel’s optimization toolkit for CPU inference.

-----

## ⚙️ Tips for Running on CPU

### 🎯 General Optimization

- 🔧 Use **smaller model versions** (`tiny`, `small`, `mini`, `nano`)
- ⚡ Apply **quantization** (Q4, Q5, Q8) to reduce RAM usage by 50-75%
- 🧩 Use optimized runtimes: **GGUF/GGML**, **ONNX Runtime**, or **OpenVINO**
- 🚀 Enable **multi-threading** to utilize all CPU cores
- 📉 Reduce resolution/steps in image generation for faster results

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
```

-----

## 🤝 Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

- Add new projects that work well on CPU
- Fix broken links or outdated information
- Improve documentation and examples

-----

## 📜 License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

This list is licensed under CC0 1.0 Universal and follows the [Awesome](https://github.com/sindresorhus/awesome) format.

-----

## ⭐ Star History

If you find this list helpful, please consider giving it a star on GitHub!

-----

**Made with ❤️ by the community** | **Last updated:** 2025