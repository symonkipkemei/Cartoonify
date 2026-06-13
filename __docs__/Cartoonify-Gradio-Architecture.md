# Cartoonify — System Architecture

> Unified reference for all four Cartoonify notebooks.
> One runtime, one LoRA, one Gradio pattern — three image-conditioning modes.

---

## System Overview

```
Google Colab Pro  (A100 GPU, 40 GB VRAM)
│
├── Notebook (one of four)
│     │
│     ├── Gemini 2.5 Flash Lite ──── story → structured prompt   (remote API, no GPU)
│     │
│     ├── Image conditioning  ─────── mode-specific (see table below)
│     │
│     ├── FLUX diffusion pipeline ─── bfloat16, GPU
│     │     ├── Base model
│     │     └── LoRA weights  ─────── .safetensors from Google Drive
│     │
│     └── Gradio Blocks UI ──────────  share=True → public ngrok URL
│
└── Google Drive
      ├── <lora>.safetensors    active LoRA weights
      └── outputs/              auto-saved generation results
```

---

## Notebooks at a Glance

| Notebook | Mode | Preprocessing | Pipeline | Prompt-driven |
|---|---|---|---|---|
| `07_Cartoonify_Gradio.ipynb` | Depth | Depth-Anything-V2 | `FluxControlNetPipeline` | Manual prompt only |
| `08_Cartoonify_Story_Gradio.ipynb` | Depth | Depth-Anything-V2 | `FluxControlNetPipeline` | + Gemini story layer |
| `09_Cartoonify_Kontext_Gradio.ipynb` | Kontext | None | `FluxKontextPipeline` | + Gemini story layer |
| `10_Cartoonify_Canny_Gradio.ipynb` | Canny | OpenCV Canny edges | `FluxControlNetPipeline` | + Gemini story layer |

All notebooks share the same LoRA, the same Gemini system prompt (08 onward), and the same Gradio UI layout. They differ only in what happens between image upload and the FLUX pipeline call.

---

## Notebook Cell Structure (Common Pattern)

Every notebook follows the same cell sequence. Only the content of highlighted cells differs per mode.

| Cell ID | Purpose | Varies per mode? |
|---|---|---|
| `cell-gpu` | `nvidia-smi` — confirm A100 | No |
| `cell-install` | pip dependencies | Only in `10` (adds `opencv-python-headless`) |
| `cell-restart` | `os.kill(os.getpid(), 9)` — kernel restart for clean package state | No |
| `cell-imports` | torch, PIL, diffusers, gradio, google.genai | Pipeline class import differs |
| `cell-drive` | Mount Google Drive; clear stale mountpoint | No |
| `cell-token` | `HF_TOKEN` + `GOOGLE_API_KEY` from Colab Secrets | No |
| `cell-config` | All mutable variables — LoRA, trigger word, defaults | Mode-specific parameters |
| `cell-models` | Load pipeline + LoRA; apply PEFT torchao patch | Pipeline class differs |
| `cell-gemini` | `GEMINI_SYSTEM_PROMPT` + `build_prompt_from_story()` | Absent in `07` |
| `cell-inference` | `cartoonify()` — preprocessing + FLUX call + Drive save | Preprocessing differs |
| `cell-ui` | Gradio Blocks UI + `demo.launch(share=True)` | Mode-specific controls |

---

## Models Used Per Notebook

| Model | `07` | `08` | `09` | `10` |
|---|---|---|---|---|
| `black-forest-labs/FLUX.1-dev` (~24 GB) | ✓ | ✓ | — | ✓ |
| `black-forest-labs/FLUX.1-Kontext-dev` (~24 GB) | — | — | ✓ | — |
| `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` (~4 GB) | ✓ | ✓ | — | ✓ |
| `depth-anything/Depth-Anything-V2-Small-hf` (~300 MB) | ✓ | ✓ | — | — |
| `gdo_cartoon.safetensors` (~600 MB, from Drive) | ✓ | ✓ | ✓ | ✓ |
| `gemini-2.5-flash-lite` (remote API) | — | ✓ | ✓ | ✓ |

**FLUX.1-dev vs FLUX.1-Kontext-dev:** Kontext is a fine-tune of the FLUX.1-dev transformer. LoRA weights trained on FLUX.1-dev apply to the same layer types and load onto Kontext cleanly — no retraining required.

**ControlNet mode codes** (ControlNet-Union-Pro-2.0):
- `control_mode=2` — Depth (used in 07, 08)
- `control_mode=0` — Canny (used in 10)

---

## Shared: LoRA Loading

All notebooks load the LoRA using an identical pattern after pipeline creation:

```python
import peft.import_utils as _peft_utils
import peft.tuners.lora.torchao as _peft_torchao
_orig = _peft_utils.is_torchao_available
def _safe():
    try: return _orig()
    except ImportError: return False
_peft_utils.is_torchao_available = _safe
_peft_torchao.is_torchao_available = _safe

pipe.load_lora_weights(LORA_DRIVE_PATH)
```

The PEFT torchao patch is required because `peft` attempts to call `torchao` at import time and raises `ImportError` in the Colab environment. The patch intercepts the call safely without disabling anything else.

---

## Shared: Gemini Story-to-Prompt Layer

Notebooks 08, 09, and 10 all use the same `build_prompt_from_story()` function and the same `GEMINI_SYSTEM_PROMPT` constant (defined in `cell-gemini`). Nothing changes between versions.

The function takes a plain-text story and returns a structured seven-layer prompt:

```
gdo_cartoon <medium>, <technique>, <color>, <mood>, <commentary>, <composition>
```

| Layer | Fixed or derived | Value |
|---|---|---|
| Trigger | Fixed | `gdo_cartoon` (or user-overridden trigger) |
| Medium | Fixed | `editorial cartoon \| political cartoon \| caricature \| newspaper illustration` |
| Technique | Fixed | `pen and ink \| cross-hatching \| bold outlines \| hand-drawn linework \| varied line weight` |
| Color | Default | `black and white \| monochrome \| print cartoon \| white background` |
| Mood | Derived from story | e.g. `satirical \| ironic \| dark humour \| critical \| bleak` |
| Commentary | Derived from story | e.g. `political commentary \| scathing \| power critique \| deadpan` |
| Composition | Derived from story | e.g. `two-group layout \| standing figure left \| queue right \| speech bubble top` |

Gemini call parameters: `temperature=0.7`, `max_output_tokens=300`. The call takes under 2 seconds and uses no GPU memory.

---

## Shared: Configuration Pattern

All notebooks keep every mutable value in `cell-config`. Nothing is hardcoded in inference or UI cells.

**Common across all notebooks:**

```python
LORA_SOURCE      = 'drive'           # 'drive' | 'huggingface'
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/…/gdo_cartoon.safetensors'
DEFAULT_TRIGGER  = 'gdo_cartoon'
DEFAULT_PROMPT   = 'satirical cartoon illustration, bold outlines, vivid flat colours, …'
DEFAULT_STEPS    = 28
DEFAULT_SEED     = 42
GOOGLE_MODEL     = 'gemini-2.5-flash-lite'
OUTPUT_DIR       = '/content/drive/MyDrive/cartoonify/outputs'
```

**Mode-specific additions:**

| Variable | `07` / `08` (Depth) | `09` (Kontext) | `10` (Canny) |
|---|---|---|---|
| `DEFAULT_GUIDANCE` | `3.5` | `2.5` | `3.5` |
| `DEFAULT_CN_SCALE` | `0.8` | — | `0.7` |
| `DEFAULT_CN_END` | — | — | `0.8` |
| `CONTROLNET_MODEL` | Shakker-Labs/Union-Pro-2.0 | — | Shakker-Labs/Union-Pro-2.0 |
| `DEPTH_MODEL` | Depth-Anything-V2-Small | — | — |
| `BASE_MODEL` | FLUX.1-dev | FLUX.1-Kontext-dev | FLUX.1-dev |
| `DEFAULT_CANNY_LOW` | — | — | `50` |
| `DEFAULT_CANNY_HIGH` | — | — | `200` |

---

## Shared: Gradio UI Layout

Every notebook uses the same two-column Gradio Blocks layout.

**Left column (inputs, top to bottom):**
1. Source image upload (drag-and-drop or clipboard)
2. **What's the Story?** accordion — story textarea + **Build Prompt** button (08/09/10 only)
3. `or write a prompt directly` divider (08/09/10 only)
4. Style Prompt textbox
5. LoRA Trigger Word textbox
6. Advanced Controls accordion — guidance scale, steps, seed, plus mode-specific sliders
7. **Cartoonify** button

**Right column (outputs):**
- Cartoonified result (download button enabled)
- Preprocessing preview in a collapsible accordion (depth map in 07/08; Canny edge map in 10; absent in 09)

**Mode differences in the UI:**

| Element | `07` | `08` | `09` | `10` |
|---|---|---|---|---|
| Story accordion | — | ✓ | ✓ | ✓ |
| Build Prompt button | — | ✓ | ✓ | ✓ |
| ControlNet scale slider | ✓ | ✓ | — | ✓ |
| Canny threshold sliders | — | — | — | ✓ (×2) |
| ControlNet guidance end slider | — | — | — | ✓ |
| Preprocessing preview panel | Depth map | Depth map | — | Canny edge map |

---

## API Keys Required

| Secret name | Where to get it | Used by |
|---|---|---|
| `HF_TOKEN` | huggingface.co/settings/tokens (Read scope) | All notebooks — downloading gated FLUX weights |
| `GOOGLE_API_KEY` | aistudio.google.com/apikey | 08, 09, 10 — Gemini story-to-prompt |

**FLUX.1-Kontext-dev licence:** The Kontext model (`09`) is gated separately from FLUX.1-dev. Accept the licence at `huggingface.co/black-forest-labs/FLUX.1-Kontext-dev` before running notebook 09.

Both secrets are read via `google.colab.userdata.get()` — never hardcoded in the notebook.

---

## LoRA Swap Procedure

Changing the LoRA requires updating two variables in `cell-config` only. The pipeline does not need to be reloaded — re-run `cell-models` to swap weights:

```python
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/…/new_lora.safetensors'
DEFAULT_TRIGGER  = 'new_trigger_word'
```

The trigger word is also picked up by the Gemini system prompt automatically through the `trigger_word` parameter passed to `build_prompt_from_story()`.

---

## Runtime Requirements

| Resource | Requirement |
|---|---|
| GPU | NVIDIA A100 40 GB (Google Colab Pro) |
| VRAM — Depth mode (07/08) | ~38 GB after loading FLUX + ControlNet + Depth model + LoRA |
| VRAM — Kontext mode (09) | ~30 GB after loading Kontext + LoRA |
| VRAM — Canny mode (10) | ~38 GB after loading FLUX + ControlNet + LoRA |
| First-run download — Depth/Canny | ~29 GB (FLUX.1-dev + ControlNet + Depth-Anything-V2) |
| First-run download — Kontext | ~25 GB (FLUX.1-Kontext-dev + LoRA) |
| Warm-cache load time | ~1 minute |
| Per-generation time | ~30–45 seconds at 28 steps |
| Gemini latency | < 2 seconds (remote API, no GPU cost) |

**Warm-cache note:** FLUX.1-dev + ControlNet-Union-Pro-2.0 weights are shared between notebooks 07, 08, and 10. If any of those notebooks have already run in the same Colab session, the weights load from cache instantly.
