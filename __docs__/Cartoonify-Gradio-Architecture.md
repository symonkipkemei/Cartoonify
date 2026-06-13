# Cartoonify — System Architecture

> Satirical image-to-image transformation using FLUX.1-dev + ControlNet Depth + custom LoRA.
> Runs on Google Colab Pro (A100 GPU). Single-user Gradio interface.

---

## System Overview

```
Google Colab Pro (A100 GPU)
│
├── 07_Cartoonify_Gradio.ipynb
│     │
│     ├── Depth estimator     Depth-Anything-V2-Small (CPU)
│     ├── ControlNet          Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0 (bfloat16, GPU)
│     ├── FLUX pipeline       black-forest-labs/FLUX.1-dev (bfloat16, GPU)
│     └── LoRA weights        .safetensors loaded from Google Drive
│
└── Google Drive
      ├── <lora>.safetensors   active LoRA weights
      └── outputs/             auto-saved generation results
```

Models are loaded once per session and held in GPU VRAM (~38 GB total). The Gradio interface runs in the same process and communicates directly with the loaded pipeline — no server hops, no API calls.

---

## Notebook Structure

The deliverable is a single notebook: `model/notebooks/interface/07_Cartoonify_Gradio.ipynb`

| Cell | Purpose |
|---|---|
| `cell-gpu` | Confirms A100 allocation via `nvidia-smi` |
| `cell-install` | Installs `diffusers`, `transformers`, `accelerate`, `gradio`, `huggingface_hub` |
| `cell-restart` | Kills kernel so fresh package versions are picked up |
| `cell-imports` | Imports torch, gradio, diffusers, PIL, numpy |
| `cell-drive` | Mounts Google Drive; clears stale mountpoint if present |
| `cell-token` | Reads `HF_TOKEN` from Colab Secrets and logs in to Hugging Face |
| `cell-config` | All mutable configuration — LoRA source, paths, trigger word, defaults |
| `cell-models` | Loads depth estimator, ControlNet, FLUX pipeline, and LoRA weights |
| `cell-inference` | `cartoonify()` function — depth map generation + FLUX inference + Drive save |
| `cell-ui` | Gradio Blocks UI — layout, CSS, component wiring, `demo.launch()` |

---

## Configuration

All runtime variables live in `cell-config`. Nothing is hardcoded in the inference function or UI.

```python
# ── LoRA source ────────────────────────────────────────────────────
LORA_SOURCE      = 'drive'          # 'huggingface' | 'drive'
LORA_HF_REPO     = 'strangerzonehf/Ghibli-Flux-Cartoon-LoRA'
LORA_HF_WEIGHT   = 'Ghibili-Cartoon-Art.safetensors'
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/02_FLUX.1/weights_FLUX.1/gdo_cartoon/gdo_cartoon.safetensors'

# ── Trigger word ───────────────────────────────────────────────────
DEFAULT_TRIGGER  = 'gdo_cartoon'

# ── Generation defaults ────────────────────────────────────────────
DEFAULT_PROMPT   = 'satirical cartoon illustration, bold outlines, vivid flat colours, expressive exaggerated features'
DEFAULT_GUIDANCE = 3.5
DEFAULT_STEPS    = 28
DEFAULT_CN_SCALE = 0.8
DEFAULT_SEED     = 42

# ── Base models ────────────────────────────────────────────────────
BASE_MODEL       = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_MODEL = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'
DEPTH_MODEL      = 'depth-anything/Depth-Anything-V2-Small-hf'

# ── Output ─────────────────────────────────────────────────────────
OUTPUT_DIR = '/content/drive/MyDrive/cartoonify/outputs'
```

All UI sliders and text fields are initialised from these variables. Changing a value in config and re-running the UI cell updates the interface defaults without reloading models.

---

## Gradio Interface

The UI is built with `gr.Blocks` and launched with `share=True` to produce a public ngrok URL from Colab.

**Left column — inputs:**
- Image upload (drag-and-drop or clipboard)
- Style prompt textarea (pre-filled from `DEFAULT_PROMPT`)
- LoRA trigger word textbox (editable live — no restart required)
- Collapsible accordion: guidance scale, inference steps, ControlNet scale, seed

**Right column — outputs:**
- Cartoonified result image (downloadable)
- Collapsible depth map preview

Button click calls `cartoonify()` and writes both outputs simultaneously.

---

## LoRA Management

The active LoRA is set in `cell-config` via `LORA_SOURCE` and either `LORA_DRIVE_PATH` or `LORA_HF_REPO` + `LORA_HF_WEIGHT`.

To swap the LoRA without restarting the runtime:

1. Upload the new `.safetensors` to Google Drive
2. Update `LORA_DRIVE_PATH` and `DEFAULT_TRIGGER` in `cell-config`
3. Re-run `cell-config` and `cell-models`

The Gradio interface picks up the new trigger word default immediately. No other cells need to be re-run.

---

## Constraints

| Constraint | Decision |
|---|---|
| No CUDA locally | Runs exclusively on Colab Pro A100 |
| Single user | No concurrency handling |
| Fixed output size | Always 1024×1024 |
| Stateless sessions | No generation history between Gradio sessions |
| Trigger word mutable | Editable in UI and config — never hardcoded |

---

## Runtime Requirements

| Resource | Requirement |
|---|---|
| GPU | NVIDIA A100 40 GB (Colab Pro) |
| VRAM at idle | ~38 GB after all models loaded |
| VRAM per generation | +1–2 GB peak during inference |
| First-run download | ~24 GB (FLUX.1-dev + ControlNet + depth model) |
| Warm cache load time | ~1 minute |
| Per-generation time | ~30–45 seconds at 28 steps |
