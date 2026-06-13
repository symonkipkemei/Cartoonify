# Cartoonify — System Architecture

> Unified reference for all five Cartoonify notebooks.
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
| `01_Cartoonify_Gradio_Depth.ipynb` | Depth | Depth-Anything-V2 | `FluxControlNetPipeline` | Manual prompt only |
| `02_Cartoonify_Gradio_Depth_Story.ipynb` | Depth | Depth-Anything-V2 | `FluxControlNetPipeline` | + Gemini story layer |
| `03_Cartoonify_Gradio_Kontext.ipynb` | Kontext | None | `FluxKontextPipeline` | + Gemini story layer |
| `04_Cartoonify_Gradio_Canny.ipynb` | Canny | OpenCV Canny edges | `FluxControlNetPipeline` | + Gemini story layer |
| `05_Cartoonify_Gradio_Unified.ipynb` | All three | Mode-dependent | Mode-dependent | + Gemini (silent on generate) |

All notebooks share the same LoRA, the same Gemini system prompt (02 onward), and the same Gradio UI layout. They differ in what happens between image upload and the FLUX pipeline call. Notebook `05` combines all three modes in one interface with dynamic pipeline loading.

---

## Notebook Cell Structure (Common Pattern)

Every notebook follows the same cell sequence. Only the content of highlighted cells differs per mode.

| Cell ID | Purpose | Varies per mode? |
|---|---|---|
| `cell-gpu` | `nvidia-smi` — confirm A100 | No |
| `cell-install` | pip dependencies | Only in `04` (adds `opencv-python-headless`) |
| `cell-restart` | `os.kill(os.getpid(), 9)` — kernel restart for clean package state | No |
| `cell-imports` | torch, PIL, diffusers, gradio, google.genai | Pipeline class import differs |
| `cell-drive` | Mount Google Drive; clear stale mountpoint | No |
| `cell-token` | `HF_TOKEN` + `GOOGLE_API_KEY` from Colab Secrets | No |
| `cell-config` | All mutable variables — LoRA, trigger word, defaults | Mode-specific parameters |
| `cell-models` | Load pipeline + LoRA; apply PEFT torchao patch | Pipeline class differs |
| `cell-gemini` | `GEMINI_SYSTEM_PROMPT` + `build_prompt_from_story()` | Absent in `01` |
| `cell-inference` | `cartoonify()` — preprocessing + FLUX call + Drive save | Preprocessing differs |
| `cell-ui` | Gradio Blocks UI + `demo.launch(share=True)` | Mode-specific controls |

---

## Models Used Per Notebook

| Model | `01` | `02` | `03` | `04` | `05` |
|---|---|---|---|---|---|
| `black-forest-labs/FLUX.1-dev` (~24 GB) | ✓ | ✓ | — | ✓ | ✓ (Scene/Portrait) |
| `black-forest-labs/FLUX.1-Kontext-dev` (~24 GB) | — | — | ✓ | — | ✓ (Reimagine) |
| `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` (~4 GB) | ✓ | ✓ | — | ✓ | ✓ (Scene/Portrait) |
| `depth-anything/Depth-Anything-V2-Small-hf` (~300 MB) | ✓ | ✓ | — | — | ✓ (Scene only) |
| `gdo_cartoon.safetensors` (~600 MB, from Drive) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gemini-2.5-flash-lite` (remote API) | — | ✓ | ✓ | ✓ | ✓ |

**Notebook 05 note:** Only one pipeline loads at a time. `load_pipeline(mode)` unloads the current pipeline before loading the new one — the models listed for `05` are never in VRAM simultaneously.

**FLUX.1-dev vs FLUX.1-Kontext-dev:** Kontext is a fine-tune of the FLUX.1-dev transformer. LoRA weights trained on FLUX.1-dev apply to the same layer types in the Kontext transformer and load cleanly — no retraining required.

**ControlNet mode codes** (ControlNet-Union-Pro-2.0):
- `control_mode=2` — Depth (used in 01, 02)
- `control_mode=0` — Canny (used in 04)

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

Notebooks 02, 03, and 04 all use the same `build_prompt_from_story()` function and the same `GEMINI_SYSTEM_PROMPT` constant (defined in `cell-gemini`). Nothing changes between versions.

Notebook 05 adds a second Gemini function, `build_wild_settings()`, for Wild settings mode.

### `build_prompt_from_story()` — Default mode

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

Call parameters: `temperature=0.7`, `max_output_tokens=300`. Under 2 seconds, no GPU memory.

### `build_wild_settings()` — Wild mode (notebook 05 only)

Returns `(prompt_str, settings_dict)`. The settings dict contains generation parameters Gemini suggests for maximum satirical impact based on the story content.

Output format — two lines from Gemini:
```
Line 1: [standard seven-layer prompt]
Line 2: {"mode": "...", "guidance": ..., "steps": ..., "cn_scale": ..., "cn_end": ..., "canny_low": ..., "canny_high": ..., "rationale": "..."}
```

Call parameters: `temperature=0.7`, `max_output_tokens=500`. The richer system prompt includes worked examples and parameter tuning rules keyed on story signals (confrontational tone → high guidance; face identity → tight Canny; crowd scene → Scene mode with high CN scale).

The settings dict drives slider updates in the Gradio UI — sliders change value visually during generation to show what's being used. The rationale string appears in the processing log.

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

| Variable | `01` / `02` (Depth) | `03` (Kontext) | `04` (Canny) | `05` (Unified) |
|---|---|---|---|---|
| `DEFAULT_GUIDANCE` | `3.5` | `2.5` | `3.5` | Per mode via `DEFAULTS` dict |
| `DEFAULT_CN_SCALE` | `0.8` | — | `0.7` | Per mode via `DEFAULTS` dict |
| `DEFAULT_CN_END` | — | — | `0.8` | Per mode via `DEFAULTS` dict |
| `CONTROLNET_MODEL` | Shakker-Labs/Union-Pro-2.0 | — | Shakker-Labs/Union-Pro-2.0 | Shakker-Labs/Union-Pro-2.0 |
| `DEPTH_MODEL` | Depth-Anything-V2-Small | — | — | Depth-Anything-V2-Small |
| `BASE_MODEL` | FLUX.1-dev | FLUX.1-Kontext-dev | FLUX.1-dev | Both (mode-dependent) |
| `DEFAULT_CANNY_LOW` | — | — | `50` | Per mode via `DEFAULTS` dict |
| `DEFAULT_CANNY_HIGH` | — | — | `200` | Per mode via `DEFAULTS` dict |
| `DEFAULT_MODE` | — | — | — | `'Reimagine'` |
| `DEFAULT_SETTINGS_MODE` | — | — | — | `'Default'` |

---

## Shared: Gradio UI Layout

Every notebook uses the same two-column Gradio Blocks layout.

**Left column (inputs, top to bottom):**
1. Source image upload (drag-and-drop or clipboard)
2. **What's the Story?** accordion — story textarea + **Build Prompt** button (02/03/04 only)
3. `or write a prompt directly` divider (02/03/04 only)
4. Style Prompt textbox
5. LoRA Trigger Word textbox
6. Advanced Controls accordion — guidance scale, steps, seed, plus mode-specific sliders
7. **Cartoonify** button

**Right column (outputs):**
- Cartoonified result (download button enabled)
- Preprocessing preview in a collapsible accordion (depth map in 01/02; Canny edge map in 04; absent in 03)

**Mode differences in the UI:**

| Element | `01` | `02` | `03` | `04` | `05` |
|---|---|---|---|---|---|
| Story accordion | — | ✓ | ✓ | ✓ | Always open (no accordion) |
| Build Prompt button | — | ✓ | ✓ | ✓ | — (Gemini runs on generate) |
| Mode selector | — | — | — | — | Three-pill radio (Reimagine / Scene / Portrait) |
| Settings mode toggle | — | — | — | — | Two-option radio (Default / Wild) |
| ControlNet scale slider | ✓ | ✓ | — | ✓ | ✓ (hidden for Reimagine) |
| Canny threshold sliders | — | — | — | ✓ (×2) | ✓ (Portrait only, ×2) |
| ControlNet guidance end slider | — | — | — | ✓ | ✓ (Portrait only) |
| Preprocessing preview panel | Depth map | Depth map | — | Canny edge map | — |
| Processing log | — | — | — | — | ✓ (streaming narrative) |

---

## API Keys Required

| Secret name | Where to get it | Used by |
|---|---|---|
| `HF_TOKEN` | huggingface.co/settings/tokens (Read scope) | All notebooks — downloading gated FLUX weights |
| `GOOGLE_API_KEY` | aistudio.google.com/apikey | 02, 03, 04 — Gemini story-to-prompt |

**FLUX.1-Kontext-dev licence:** The Kontext model (`03`) is gated separately from FLUX.1-dev. Accept the licence at `huggingface.co/black-forest-labs/FLUX.1-Kontext-dev` before running notebook 03.

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
| VRAM — Depth mode (01/02) | ~38 GB after loading FLUX + ControlNet + Depth model + LoRA |
| VRAM — Kontext mode (03) | ~30 GB after loading Kontext + LoRA |
| VRAM — Canny mode (04) | ~38 GB after loading FLUX + ControlNet + LoRA |
| VRAM — Unified (05) | One mode loaded at a time — same profiles as above |
| First-run download — Depth/Canny | ~29 GB (FLUX.1-dev + ControlNet + Depth-Anything-V2) |
| First-run download — Kontext | ~25 GB (FLUX.1-Kontext-dev + LoRA) |
| Warm-cache load time | ~1 minute |
| Mode switch time (05 only) | ~60–90 seconds |
| Per-generation time | ~30–45 seconds at 28 steps |
| Gemini latency | < 2 seconds (remote API, no GPU cost) |

**Warm-cache note:** FLUX.1-dev + ControlNet-Union-Pro-2.0 weights are shared between notebooks 01, 02, 04, and notebook 05 when running Scene or Portrait mode. If any of those notebooks have already run in the same Colab session, the weights load from cache instantly.
