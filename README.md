# Cathey — Offline Smart Home Assistant

EECS 6908 Final Project · Columbia University

Cathey is a fully offline voice assistant deployable on a **Raspberry Pi 5**.  
It listens for the wake word "Cathey", classifies user intent into 4 categories, executes smart home commands via GPIO, and learns user preferences over time using a four-layer memory system.

---

## Project Structure

```
Embedded-AI-Final-Project/
├── cathey.py               # Entry point — run the full assistant
├── config.py               # All constants (model paths, audio params, memory thresholds)
├── agent.py                # CatheyAgent — intent routing + dialogue state machine
├── llm_parser.py           # LLM loading + 3 inference methods (dual backend)
├── rule_based.py           # Rule-based fast path for unambiguous commands
├── schema.py               # Device schema, command validation
├── gpio_executor.py        # GPIO hardware execution (lights, curtain, AC, window)
├── memory.py               # Four-layer memory system (working/episodic/semantic/procedural)
├── audio.py                # STT (Whisper) + TTS (Piper) + VAD audio listener
├── bt_diag.py              # Bluetooth speaker diagnostics
├── bt-speaker.service      # systemd unit for Bluetooth speaker auto-connect
├── deploy.sh               # One-command deploy to Raspberry Pi over SSH
├── requirements_pi.txt     # Pi-specific Python dependencies
├── lora_training.ipynb     # LoRA fine-tuning notebook
├── voices/                 # Piper TTS voice model files
├── cathey_memory/          # Persistent memory storage (ChromaDB + JSON)
└── tests/
    ├── test_rule_based.py
    ├── test_gpio_executor.py
    ├── test_agent_memory.py
    ├── test_benchmark.py
    ├── test_bt_diag.py
    ├── test_hardware_extension.py
    ├── text_test.ipynb
    └── audio_test.ipynb
```

---

## Intent Categories

| Category | Trigger | Example | 
|---|---|---|
| `direct_command` | Device + explicit action | "Cathey, turn on the light." |
| `needs_clarification` | Vague feeling / preference | "Cathey, it's a bit dark." |
| `general_qa` | Non-device question | "Cathey, how do I eat an apple?" |
| `invalid` | No wake word / unrecognised | "Turn on the light." |

---

## Four-Layer Memory

| Layer | Storage | Lifetime | Purpose |
|---|---|---|---|
| Working | RAM `deque` | Current session | Conversation window context |
| Episodic | ChromaDB (local) | Persistent | RAG retrieval of similar past interactions |
| Semantic | JSON file | Persistent | User preferences (e.g. preferred AC temp) |
| Procedural | JSON file | Persistent | Learned trigger → action patterns (skip re-asking) |

---

## LLM Backend

The LLM backend is selected automatically in `config.py`:

| Hardware | Backend | Model |
|---|---|---|
| Raspberry Pi 5 (CPU) | `llama-cpp-python` | `qwen2.5-3b-instruct-q3_k_m.gguf` |
| Mac / GPU (MPS / CUDA) | HuggingFace transformers | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

---

## Quick Start (Development — Mac/Linux)

### 1. Clone the repo

```bash
git clone https://github.com/Ezreal-Mi/Embedded-AI-Final-Project.git
cd Embedded-AI-Final-Project
```

### 2. Create a Python environment

```bash
python -m venv cathey_env
source cathey_env/bin/activate
```

### 3. Install dependencies

```bash
pip install faster-whisper transformers accelerate sentencepiece torch \
            sounddevice soundfile piper-tts numpy \
            sentence-transformers llama-cpp-python
```

> **Pi only:** `lgpio` and `rpi-lgpio` are also required (see `requirements_pi.txt`).

### 4. Run the assistant

```bash
python cathey.py
```

Cathey will warm up the embedding model, then start listening continuously via VAD. Press `Ctrl+C` to stop.

---

## Deployment on Raspberry Pi 5

`deploy.sh` handles everything over SSH in one command:

```bash
./deploy.sh pi@<PI_IP> <BLUETOOTH_MAC>
```

What it does:
1. Rsyncs all source files to `~/nova/` on the Pi
2. Installs system packages (`libportaudio2`, `bluetooth`, `bluez`, `libopenblas-dev`)
3. Installs Python dependencies from `requirements_pi.txt`
4. Builds `llama-cpp-python` with OpenBLAS acceleration
5. Downloads the GGUF model (`qwen2.5-3b-instruct-q4_k_m.gguf`, ~2 GB) if not present
6. Downloads the Piper voice model (`en_US-lessac-medium.onnx`) if not present
7. Sets the USB microphone as the default PipeWire input
8. Installs and enables `bt-speaker.service` and `cathey.service` as systemd units

**Bluetooth pairing (first deploy only):**

```bash
bluetoothctl
  power on
  agent on
  scan on          # wait for speaker MAC to appear
  pair   <BT_MAC>
  trust  <BT_MAC>
  connect <BT_MAC>
  exit
```

---

## Running Tests

### Unit tests (no hardware needed)

```bash
pytest tests/
```

Test files:
- `test_rule_based.py` — rule-based fast path coverage
- `test_gpio_executor.py` — GPIO executor (mocked hardware)
- `test_agent_memory.py` — agent + memory integration
- `test_benchmark.py` — benchmark helper functions
- `test_bt_diag.py` — Bluetooth diagnostics
- `test_hardware_extension.py` — hardware extension scenarios

### Notebook tests

```bash
jupyter notebook tests/text_test.ipynb   # text-level intent tests, no microphone
jupyter notebook tests/audio_test.ipynb  # microphone required
```

---

## LoRA Fine-tuning

Fine-tuning teaches the model to reliably output valid JSON and correctly handle hard cases.

```bash
jupyter notebook lora_training.ipynb
```

LoRA config (set in `config.py`):

| Parameter | Default |
|---|---|
| `LORA_R` | 8 |
| `LORA_ALPHA` | 16 |
| `LORA_ADAPTER_DIR` | `cathey_lora_adapter` |
| `LORA_MERGED_DIR` | `cathey_lora_merged` |

---

## Configuration Reference

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_GGUF_PATH` | `models/qwen2.5-3b-instruct-q3_k_m.gguf` | GGUF model path (Pi) |
| `LLM_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HF model name (Mac/GPU) |
| `WHISPER_MODEL_SIZE` | `tiny.en` | STT model size |
| `PIPER_MODEL_PATH` | `voices/en_US-lessac-medium.onnx` | TTS voice model |
| `ENERGY_THRESHOLD` | `0.05` | VAD silence cutoff |
| `SKILL_SIM_THRESHOLD` | `0.92` | Procedural memory cosine similarity cutoff |
| `EPISODE_DIST_CUTOFF` | `0.6` | Episodic RAG relevance cutoff |
| `WORKING_MAXLEN` | `8` | Max turns kept in working memory |

---

## How Each Module Connects

```
cathey.py
    │
    ├── LLMParser (llm_parser.py)
    │       ├── parse_unified()       ← classify intent → JSON
    │       ├── resolve_followup()    ← resolve clarification reply → JSON
    │       └── answer_qa()          ← RAG-augmented plain-text answer
    │
    ├── RuleBased (rule_based.py)
    │       └── try_rule_based()     ← fast regex path, skips LLM entirely
    │
    ├── MemoryManager (memory.py)
    │       ├── push_working()       ← append to session RAM
    │       ├── save_episode()       ← write to ChromaDB
    │       ├── update_pref()        ← write to user_prefs.json
    │       ├── record_skill()       ← write to skills.json
    │       ├── lookup_skill()       ← cosine search in procedural memory
    │       └── build_context()      ← aggregate all layers → RAG prompt
    │
    ├── CatheyAgent (agent.py)
    │       └── handle(text)         ← routes text through full pipeline
    │
    ├── GPIOExecutor (gpio_executor.py)
    │       └── execute(command)     ← drives GPIO pins for hardware actions
    │
    └── AudioListener (audio.py)
            ├── run_one_round()      ← fixed-duration recording
            └── continuous_loop()   ← VAD-gated streaming loop
```

---

## Troubleshooting

**`No module named 'sounddevice'`**
→ On Linux/Pi: `sudo apt install portaudio19-dev` then `pip install sounddevice`.

**TTS has no audio output on Pi**
→ Check Bluetooth speaker is connected: `sudo systemctl status bt-speaker`. Run `bt_diag.py` for diagnostics.

**VAD loop captures too much background noise**
→ Increase `ENERGY_THRESHOLD` in `config.py` (try `0.07`–`0.10`).

**LLM latency is too high on CPU**
→ Confirm `LLM_BACKEND = "llama_cpp"` in `config.py` and that the GGUF model is downloaded to `models/`.

**ChromaDB error on first run**
→ The `cathey_memory/` directory is created automatically. If corrupted, delete it: `rm -rf cathey_memory/`

**USB microphone not detected after deploy**
→ Re-run `deploy.sh` with the mic plugged in, or manually: `pactl set-default-source <usb-source-name>`.
