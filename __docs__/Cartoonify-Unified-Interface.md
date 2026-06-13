# Cartoonify — Unified Interface

> Notebook `05_Cartoonify_Gradio_Unified.ipynb`
> One story. One button. Three rendering styles — all in a single interface.

---

## What This Notebook Is

Notebooks 02–04 each expose a single pipeline behind a dedicated Gradio UI. Choosing between them requires understanding the technical difference between ControlNet depth conditioning, Canny edge conditioning, and native image-to-image recomposition before you've seen a single result.

The unified notebook flips that — the user writes a story, picks a rendering style in plain language, and clicks one button. Gemini builds the structured prompt silently on generate. Pipeline loading and VRAM management are handled automatically.

---

## Flow

```
① What's the story?  (textarea — always visible, always first)
        │
        ▼ on generate
Gemini 2.5 Flash Lite
        │  structured seven-layer prompt
        ▼
② Upload a photo  →  Resize to 1024 × 1024
        │
        ③ Rendering style
        │
        ├── Reimagine ──→ FluxKontextPipeline (no preprocessing)
        │
        ├── Scene ──────→ Depth-Anything-V2 (CPU) → FluxControlNetPipeline (mode=2)
        │
        └── Portrait ───→ cv2.Canny → FluxControlNetPipeline (mode=0)
                │
                ▼
        LoRA  +  Structured prompt
                │
                ▼
        1024 × 1024 cartoon PNG  →  Gradio display  +  Drive auto-save
```

The processing log on the right narrates each step as it runs.

---

## Rendering Styles — Plain Language Mapping

| UI label | Technical mode | Pipeline | Control signal | Use when |
|---|---|---|---|---|
| **Reimagine** | Kontext | `FluxKontextPipeline` | Full image content (semantic) | Scene needs to change — new staging, exaggerated proportions, full recomposition |
| **Scene** | Depth | `FluxControlNetPipeline` (mode=2) | Near/far depth map | Crowds, architecture, landscapes — spatial layout must be preserved |
| **Portrait** | Canny | `FluxControlNetPipeline` (mode=0) | Hard geometric edge lines | Specific person must be immediately recognisable — follows face outlines |

---

## What Changed vs Individual Notebooks

| Behaviour | Notebooks 02–04 | Notebook 05 |
|---|---|---|
| Story input | Collapsed accordion | Always open, first element |
| Prompt building | Separate **Build Prompt** button | Gemini runs silently on **Cartoonify** |
| Mode selection | Separate notebooks | Three-pill radio selector in one UI |
| Pipeline loading | Fixed at notebook start | `load_pipeline(mode)` — loads on first use, switches on mode change |
| Processing feedback | Silent spinner | Live narrative log (streaming `yield`) |
| Prompt access | Always visible textbox | Buried in `Advanced Settings → Edit prompt directly` |
| Trigger word | Visible in main UI | Moved to Advanced Settings |

---

## VRAM Switching — `load_pipeline()`

The A100 (40 GB) cannot hold all three pipelines simultaneously. `load_pipeline(mode)` manages this:

```python
def load_pipeline(mode: str) -> None:
    global pipe, controlnet, depth_estimator, active_mode

    if mode == active_mode:
        return  # already loaded — no-op

    # Unload current
    del pipe, controlnet, depth_estimator   # whichever are not None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Load new
    if mode == 'reimagine':
        pipe = FluxKontextPipeline.from_pretrained(...)
    elif mode in ('scene', 'portrait'):
        controlnet = FluxControlNetModel.from_pretrained(...)
        pipe = FluxControlNetPipeline.from_pretrained(...)
        if mode == 'scene':
            depth_estimator = hf_pipeline('depth-estimation', ...)

    _peft_patch()
    pipe.load_lora_weights(LORA_DRIVE_PATH)
    active_mode = mode
```

**Switching cost:** ~60–90 seconds (one-time per switch per session). Within the same mode, each generation is ~30–45 s.

**LoRA reloads on every switch** — the LoRA weights live in the pipeline; when the pipeline is replaced the LoRA must be reloaded onto the new one.

**VRAM profiles:**

| Mode | What loads | Approx. VRAM |
|---|---|---|
| Reimagine | FLUX.1-Kontext-dev + LoRA | ~30 GB |
| Scene | FLUX.1-dev + ControlNet + Depth-Anything-V2 + LoRA | ~38 GB |
| Portrait | FLUX.1-dev + ControlNet + LoRA | ~38 GB |

---

## Mode Selector Auto-Adjustments

When the user switches rendering style, `mode_selector.change` fires `update_mode()`, which:

| Setting | What happens |
|---|---|
| Guidance Scale | Resets to `2.5` (Reimagine) or `3.5` (Scene / Portrait) |
| ControlNet Scale | Resets to mode default; hidden for Reimagine |
| ControlNet Guidance End | Resets to `0.8`; visible only for Portrait |
| Canny Low / High | Reset to `50` / `200`; visible only for Portrait |
| Inference Steps | Unchanged |
| Seed | Unchanged |

These are hardcoded defaults from the `DEFAULTS` dict in `cell-config`. This is not LLM-suggested settings — parameter recommendations based on story content are a separate, later feature.

---

## Processing Log — Narrative by Mode

The `cartoonify()` function is a Python generator. Each `yield (image_or_None, log_text)` call updates both the result panel and the log textbox in real time.

**Reimagine:**
```
⏳ Gemini building your prompt...
✓ Prompt ready
⏳ FLUX Kontext rendering...
✓ Done — saved to Drive
```

**Scene:**
```
⏳ Gemini building your prompt...
✓ Prompt ready
⏳ Reading scene depth...
✓ Depth map ready
⏳ FLUX rendering...
✓ Done — saved to Drive
```

**Portrait:**
```
⏳ Gemini building your prompt...
✓ Prompt ready
⏳ Extracting portrait outlines...
✓ Outlines extracted
⏳ FLUX rendering...
✓ Done — saved to Drive
```

If the user has typed in `Edit prompt directly`, the Gemini call is skipped and the log reads `✓ Using manual prompt override`.

---

## Prompt Priority Order

```
1. prompt_override (Edit prompt directly)  → skips Gemini entirely
2. story (non-empty)                       → Gemini builds structured prompt
3. No story, no override                   → DEFAULT_PROMPT from cell-config
```

Trigger word deduplication: `cartoonify()` checks `prompt.startswith(trigger)` before prepending, preventing the double-trigger bug that was present in notebooks 02–04.

---

## Configuration

All mutable values live in `cell-config`. The key addition vs individual notebooks is the `DEFAULTS` dict and `DEFAULT_MODE`:

```python
DEFAULT_MODE  = 'Reimagine'  # starting mode — controls initial pipeline load

DEFAULTS = {
    'reimagine': {'guidance': 2.5, 'cn_scale': 0.7, 'cn_end': 0.8, 'canny_low': 50, 'canny_high': 200},
    'scene':     {'guidance': 3.5, 'cn_scale': 0.8, 'cn_end': 0.8, 'canny_low': 50, 'canny_high': 200},
    'portrait':  {'guidance': 3.5, 'cn_scale': 0.7, 'cn_end': 0.8, 'canny_low': 50, 'canny_high': 200},
}
```

Both base model variables are present (Kontext and FLUX.1-dev) since `load_pipeline()` may load either depending on the selected mode.

---

## Cell Structure

| Cell | Purpose | Notes |
|---|---|---|
| `cell-gpu` | `!nvidia-smi` | Unchanged |
| `cell-install` | pip dependencies | Includes `opencv-python-headless` (required for Portrait) |
| `cell-restart` | Kernel restart | Unchanged |
| `cell-imports` | All imports for all three modes | Adds `FluxKontextPipeline`, `FluxControlNetPipeline`, `FluxControlNetModel`, `cv2` |
| `cell-drive` | Mount Google Drive | Unchanged |
| `cell-token` | `HF_TOKEN` + `GOOGLE_API_KEY` | Unchanged |
| `cell-config` | All mutable variables | Adds `DEFAULTS`, `DEFAULT_MODE`, both base model IDs |
| `cell-models` | `_peft_patch()` + `load_pipeline()` + initial load | New: `load_pipeline()` manages all three modes |
| `cell-gemini` | `GEMINI_SYSTEM_PROMPT` + `build_prompt_from_story()` | Unchanged from 02–04 |
| `cell-inference` | `extract_canny()` + `cartoonify()` generator | New: generator pattern; mode dispatch; trigger dedup |
| `cell-ui` | Gradio Blocks UI + `demo.launch()` | New: story-first layout; three-pill mode selector; live log |

---

## Runtime

| Resource | Value |
|---|---|
| GPU | NVIDIA A100 40 GB (Google Colab Pro) |
| First-run download (Reimagine) | ~25 GB |
| First-run download (Scene / Portrait) | ~29 GB |
| Warm-cache load | ~1 minute |
| Mode switch | ~60–90 seconds |
| Per-generation | ~30–45 seconds at 28 steps |
| Gemini | < 2 seconds (remote API, no GPU cost) |

**Note:** Switching between Scene and Portrait does not require reloading FLUX.1-dev or ControlNet — both modes share those weights. Only the depth estimator is added/removed. In practice, the Scene → Portrait switch reloads the pipeline anyway (because `active_mode` changes) unless a future optimisation detects that only the depth estimator needs to be dropped.
