# 🦙 haven-llama

**haven-llama** is a native C++ zero-drift LLM engine extension built on [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for high-performance AI companion serving.

It introduces **native C++ logit masking**, **sub-millisecond streaming**, **1-tap GGUF model hot-swapping**, and **zero-drift pronoun enforcement** at the tensor sampling layer.

---

## ✨ Features

- **🛡️ Native C++ Zero-Drift Sampler (`haven_sampler`)**: Implements `llama_sampler_i` to dynamically mask logit probabilities for pronouns (`he/him/his` or `she/her/hers`) at the tensor level. The model *mathematically cannot* generate hallucinated pronoun drift.
- **⚡ High-Performance SSE HTTP Server**: Built-in multi-threaded C++ web server powered by `cpp-httplib` and `nlohmann/json`.
- **🔄 1-Tap Model Hot-Swapping**: Load and switch GGUF models on demand via `/v1/haven/model/load` without restarting the process.
- **📦 Update-Proof Plugin Architecture**: Designed to live cleanly in `examples/haven-server/` inside `llama.cpp`. Upstream CUDA/Vulkan performance updates integrate seamlessly with a simple `git pull`.

---

## 🚀 Quick Start

### 1. Clone inside `llama.cpp`

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp/examples
git clone https://github.com/Haven-AI-Companion/haven-llama.cmd haven-server
```

Add `add_subdirectory(haven-server)` to `examples/CMakeLists.txt`.

### 2. Build with CMake (CUDA / Vulkan / CPU)

```bash
cd ..
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
cmake --build build --config Release --target haven-server
```

### 3. Launch Engine Server

```bash
./build/bin/Release/haven-server.exe --model "path/to/model.gguf" --port 8088
```

---

## 📡 API Endpoints

### `GET /health`
Returns server telemetry and active GGUF model state.

### `POST /v1/haven/chat/completions`
Streams response tokens via Server-Sent Events (`text/event-stream`).

```json
{
  "prompt": "<|im_start|>system\nYou are a companion...<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
  "user_gender": "female",
  "user_name": "Daniel",
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### `POST /v1/haven/model/load`
Hot-swaps the active GGUF model in memory.

---

## 📄 License

MIT License. Built with ❤️ for the [Haven AI Companion](https://github.com/Haven-AI-Companion) Ecosystem.
