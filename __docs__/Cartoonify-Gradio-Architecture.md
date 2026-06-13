# Cartoonify — Story Interface Architecture

> `08_Cartoonify_Story_Gradio.ipynb`
> Extends the base Cartoonify pipeline with a Gemini-powered story-to-prompt layer.

---

## System Overview

```
Google Colab Pro (A100 GPU)
│
├── 08_Cartoonify_Story_Gradio.ipynb
│     │
│     ├── Gemini 2.5 Flash Lite   story → structured prompt  (remote API, no GPU)
│     ├── Depth estimator          Depth-Anything-V2-Small   (CPU)
│     ├── ControlNet               Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0  (bfloat16, GPU)
│     ├── FLUX pipeline            black-forest-labs/FLUX.1-dev  (bfloat16, GPU)
│     └── LoRA weights             .safetensors loaded from Google Drive
│
└── Google Drive
      ├── <lora>.safetensors   active LoRA weights
      └── outputs/             auto-saved generation results
```

Gemini is a remote API call — it adds no GPU memory and runs in under 2 seconds. All other components are identical to `07_Cartoonify_Gradio.ipynb`.

---

## Notebook Structure

| Cell | Purpose |
|---|---|
| `cell-gpu` | Confirms A100 allocation via `nvidia-smi` |
| `cell-install` | Installs `diffusers`, `transformers`, `accelerate`, `gradio`, `huggingface_hub`, `google-genai` |
| `cell-restart` | Kills kernel so fresh package versions are picked up |
| `cell-imports` | Imports torch, gradio, `google.genai`, diffusers, PIL, numpy |
| `cell-drive` | Mounts Google Drive; clears stale mountpoint if present |
| `cell-token` | Reads `HF_TOKEN` and `GOOGLE_API_KEY` from Colab Secrets |
| `cell-config` | All mutable configuration — LoRA, trigger word, generation defaults, `GOOGLE_MODEL` |
| `cell-models` | Loads depth estimator, ControlNet, FLUX pipeline, and LoRA weights |
| `cell-gemini` | `GEMINI_SYSTEM_PROMPT` constant + `build_prompt_from_story()` function |
| `cell-inference` | `cartoonify()` — depth map + FLUX inference + Drive auto-save |
| `cell-ui` | Gradio Blocks — story accordion, prompt box, controls, output panels |

---

## What Changed vs `07_Cartoonify_Gradio.ipynb`

| Area | 07 | 08 |
|---|---|---|
| Install | diffusers, gradio, huggingface_hub | + `google-genai` |
| Secrets | `HF_TOKEN` | + `GOOGLE_API_KEY` |
| Config | LoRA + generation defaults | + `GOOGLE_MODEL = 'gemini-2.5-flash-lite'` |
| New cell | — | `cell-gemini` — system prompt + `build_prompt_from_story()` |
| UI — story | Not present | Story accordion above prompt box |
| UI — buttons | Generate only | **Build Prompt** (Gemini) + **Cartoonify** (FLUX) |

The `cartoonify()` inference function and all model loading cells are unchanged.

---

## Configuration

All runtime variables live in `cell-config`. Nothing is hardcoded in the inference or UI cells.

```python
# ── LoRA ──────────────────────────────────────────────────────────
LORA_SOURCE      = 'drive'          # 'huggingface' | 'drive'
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/02_FLUX.1/weights_FLUX.1/gdo_cartoon/gdo_cartoon.safetensors'
DEFAULT_TRIGGER  = 'gdo_cartoon'

# ── Prompt default ─────────────────────────────────────────────────
DEFAULT_PROMPT   = 'satirical cartoon illustration, bold outlines, vivid flat colours, expressive exaggerated features'

# ── Generation defaults ────────────────────────────────────────────
DEFAULT_GUIDANCE = 3.5
DEFAULT_STEPS    = 28
DEFAULT_CN_SCALE = 0.8
DEFAULT_SEED     = 42

# ── Base models ────────────────────────────────────────────────────
BASE_MODEL       = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_MODEL = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'
DEPTH_MODEL      = 'depth-anything/Depth-Anything-V2-Small-hf'

# ── Gemini ─────────────────────────────────────────────────────────
GOOGLE_MODEL     = 'gemini-2.5-flash-lite'
```

Changing `GOOGLE_MODEL` swaps the Gemini model. The system prompt lives in `cell-gemini` and can be edited without touching `cell-config`.

---

## Gemini Layer

`build_prompt_from_story()` in `cell-gemini`:

```
User story (free text)
        │
        ▼
  GEMINI_SYSTEM_PROMPT
  + three few-shot examples
  + user story as the message
        │
        ▼
  Gemini 2.5 Flash Lite
  (temperature=0.7, max_output_tokens=300)
        │
        ▼
  Structured prompt string
  (seven-layer, pipe-separated)
        │
        ▼
  Trigger word substitution
  (replaces gdo_cartoon if user changed trigger)
        │
        ▼
  prompt_input textbox updated
```

The system prompt is a constant in `cell-gemini`. It contains:
- The seven-layer output format specification
- Hard rules (always include fixed medium/technique layers)
- Three few-shot examples drawn from the training caption vocabulary

---

## Gradio Interface

**Left column — inputs (top to bottom):**

1. **Source Image** — drag-and-drop or clipboard upload
2. **What's the Story?** accordion *(collapsed by default)*
   - Story textarea — free-text description
   - **Build Prompt** button — calls Gemini, writes result into the prompt box below
3. `or write a prompt directly` divider
4. **Style Prompt** — pre-filled by Gemini or written manually; always editable
5. **LoRA Trigger Word** — editable live; Gemini substitutes this automatically
6. **Advanced Controls** accordion — guidance scale, steps, ControlNet scale, seed
7. **Cartoonify** button — runs the full FLUX pipeline

**Right column — outputs:**
- Cartoonified result (downloadable)
- Depth map preview (collapsed accordion)

Users who skip the story accordion and type a prompt directly have an identical experience to `07_Cartoonify_Gradio.ipynb`.

---

## API Keys Required

| Secret name | Where to get it | Used for |
|---|---|---|
| `HF_TOKEN` | `huggingface.co/settings/tokens` | Downloading gated FLUX.1-dev weights |
| `GOOGLE_API_KEY` | `aistudio.google.com/apikey` | Calling Gemini story-to-prompt |

Both are read via `google.colab.userdata.get()` — never hardcoded.

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
| Gemini call latency | < 2 seconds (remote API, no GPU cost) |
