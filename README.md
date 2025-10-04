# 🧠 Awesome AI Projects for CPU
![Awesome](https://awesome.re/badge.svg) ![MIT License](https://img.shields.io/badge/license-MIT-brightgreen)
> A curated list of **open source AI projects** that run well on **CPU**, no GPU required.  
> Perfect for makers, indie developers, and local AI experiments.

---

## 🧠 Language Models (Chatbots / Text)

- [**GPT4All**](https://github.com/nomic-ai/gpt4all) — Simple interface to run LLMs locally (several CPU-optimized models included).  
- [**Ollama**](https://github.com/ollama/ollama) — Run LLaMA, Mistral, Gemma models with excellent CPU support.  
- [**LM Studio**](https://lmstudio.ai/) — GUI app for local LLMs, automatically supports CPU execution.  
- [**Text Generation WebUI**](https://github.com/oobabooga/text-generation-webui) — Powerful web interface for LLMs, CPU mode available.

--

### 🎤 Speech & Audio (Compact)
- **[Whisper Tiny/Base](https://huggingface.co/openai/whisper-tiny)** — 39M/74M params, transcription
- **[Piper TTS (small voices)](https://github.com/rhasspy/piper)** — 5-20MB voice models
- **[Vosk Small Models](https://alphacephei.com/vosk/models)** — 50MB speech recognition

---

## 🖼️ Image Generation and Editing

- [**Stable Diffusion (CPU mode)**](https://github.com/CompVis/stable-diffusion) — Image generation model; works on CPU (slower).  
- [**Diffusion Bee**](https://diffusionbee.com/) — User-friendly GUI for Stable Diffusion on macOS, CPU compatible.  
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) — Image upscaler, fast on CPU.  
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN) — Restores and improves old faces/photos, runs on CPU.  

---

## 🎤 Voice and Audio

- [**Whisper.cpp**](https://github.com/ggerganov/whisper.cpp) — Optimized Whisper (OpenAI) for CPU speech recognition.  
- [**Bark (Suno)**](https://github.com/suno-ai/bark) — Realistic voice generation from text (CPU mode available).  
- [**RVC (Retrieval-based Voice Conversion)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Real-time voice conversion, CPU compatible.  
- [**Coqui TTS**](https://github.com/coqui-ai/TTS) — Open source text-to-speech engine, efficient on CPU.  

---

## 👁️ Computer Vision

- [**OpenCV + DNN**](https://github.com/opencv/opencv) — Vision framework with neural networks, fully CPU capable.  
- [**YOLOv5 (CPU mode)**](https://github.com/ultralytics/yolov5) — Real-time object detection, CPU option `--device cpu`.  
- [**MediaPipe**](https://github.com/google/mediapipe) — Google library for hand, face, and body tracking, works well on CPU.  

---

## 💡 Creative AI & Miscellaneous

- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Generate music from text (CPU mode supported).  
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) — CPU-optimized LLaMA model execution.  
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) — Modular UI for image AI pipelines, supports CPU.  

---

## ⚙️ Tips for Running on CPU

- 🔧 Use smaller model versions (`small`, `tiny`, `mini`).  
- ⚡ Apply **quantization** (INT8/INT4) to reduce RAM usage.  
- 🧩 Use optimized runtimes: **ONNX Runtime**, **GGUF**, **GGML**, or **OpenVINO**.  
- 🚀 Utilize multiple CPU cores and reduce resolution/steps in image generation.  

---

### 📜 License
This list follows the [Awesome](https://github.com/sindresorhus/awesome) format.  
Feel free to clone, contribute, and adapt!