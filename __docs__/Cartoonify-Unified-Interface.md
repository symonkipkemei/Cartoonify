# Cartoonify — Unified Interface

> Notebook `05_Cartoonify_Gradio_Unified.ipynb`
> One story. One button. Three rendering styles — all in a single interface.
> Two settings modes: Default (hardcoded) and Wild (Gemini-suggested for maximum satirical impact).

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

## Settings Modes — Default vs Wild

The Settings section (between the rendering style selector and Advanced Settings) has a two-option toggle:

| Mode | What happens |
|---|---|
| **Default** | Hardcoded parameters from `DEFAULTS` dict in `cell-config`; mode switch resets sliders to mode defaults |
| **Wild** | Gemini reads the story and returns both the structured prompt AND parameter suggestions optimised for maximum satirical impact; sliders update visually to show what's being used |

### Default mode — auto-adjustments on style switch

When the user switches rendering style, `mode_selector.change` fires `update_mode()`, which resets:

| Setting | Reimagine | Scene | Portrait |
|---|---|---|---|
| Guidance Scale | 2.5 | 3.5 | 3.5 |
| ControlNet Scale | hidden | 0.8 | 0.7 |
| ControlNet Guidance End | hidden | hidden | 0.8 |
| Canny Low / High | hidden | hidden | 50 / 200 |
| Inference Steps | unchanged | unchanged | unchanged |
| Seed | unchanged | unchanged | unchanged |

### Wild mode — Gemini parameter suggestions

When the user clicks Cartoonify with Wild selected and a story entered, `build_wild_settings()` is called. It returns `(prompt, settings_dict)` where `settings_dict` contains:

| Key | Range | What it controls |
|---|---|---|
| `mode` | reimagine / scene / portrait | Suggested rendering style (advisory — shown in log; does not auto-switch) |
| `guidance` | 2.0–6.0 | Prompt dominance; higher = scathing tone holds harder |
| `steps` | 28–40 | More = more detail; Wild pushes 35–40 for complex/confrontational content |
| `cn_scale` | 0.3–1.2 | ControlNet conditioning strength |
| `cn_end` | 0.50–1.0 | Step fraction ControlNet stays active; lower gives FLUX more finishing freedom |
| `canny_low` | 10–100 | Canny lower threshold; lower = more edges (tighter face adherence) |
| `canny_high` | 80–400 | Canny upper threshold |
| `rationale` | string | One sentence explaining the parameter choices — displayed in the processing log |

**Slider updates:** When Wild applies settings, the sliders in Advanced Settings update to show the values being used. The user can open Advanced Settings to see them.

**Mode suggestion:** If Gemini's recommended rendering style differs from the one selected, the log shows an advisory note (e.g. `⚡ Wild suggests Portrait for this story (currently Reimagine)`). The rendering style is never switched automatically.

**Fallback:** If Gemini fails or the settings JSON cannot be parsed, Wild falls back to a standard Gemini prompt call with current slider values.

### Gemini call comparison

| Aspect | Default mode | Wild mode |
|---|---|---|
| Function | `build_prompt_from_story()` | `build_wild_settings()` |
| System prompt | `GEMINI_SYSTEM_PROMPT` | `WILD_SYSTEM_PROMPT` |
| Output lines | 1 (prompt only) | 2 (prompt + settings JSON) |
| `max_output_tokens` | 300 | 500 |
| Returns | `str` | `(str, dict)` |

Wild mode uses a richer system prompt with parameter tuning rules and three worked examples. The rules bias Gemini toward more aggressive parameter choices — higher guidance for confrontational stories, tighter Canny for face-critical caricature, more steps for complex compositions.

**Parameter tuning rules in Wild mode (summary):**

| Story signal | Wild choices |
|---|---|
| Confrontational / scathing / dark | guidance 4.5–5.5, steps 35–40 |
| Whimsical / absurdist / ironic | guidance 2.5–3.5, steps 28–32 |
| Face identity critical | Portrait, canny_low 20–40, canny_high 100–160, cn_scale 0.85–1.0 |
| Crowd / hierarchy scene | Scene, cn_scale 0.85–0.95, steps 34–38 |
| Total recomposition / allegory | Reimagine, guidance 2.5–3.0 |
| More artistic hand-drawn feel | cn_end 0.70 |

---

## Processing Log — Narrative by Mode

The `cartoonify()` function is a Python generator. Each `yield` updates the result panel, the log textbox, and (in Wild mode) the slider values simultaneously.

**Default mode — Reimagine:**
```
⧗ Gemini building your prompt...
✓ Prompt ready
⧗ FLUX Kontext rendering...
✓ Done — saved to Drive
```

**Default mode — Scene:**
```
⧗ Gemini building your prompt...
✓ Prompt ready
⧗ Reading scene depth...
✓ Depth map ready
⧗ FLUX rendering...
✓ Done — saved to Drive
```

**Default mode — Portrait:**
```
⧗ Gemini building your prompt...
✓ Prompt ready
⧗ Extracting portrait outlines...
✓ Outlines extracted
⧗ FLUX rendering...
✓ Done — saved to Drive
```

**Wild mode (any style):**
```
⧗ Wild — Gemini building prompt and tuning for satire...
✓ Wild settings applied
   <rationale sentence from Gemini>
   ⚡ Wild suggests Portrait for this story (currently Reimagine)   ← only if mode differs
⧗ FLUX rendering...
✓ Done — saved to Drive
```

If the user has typed in `Edit prompt directly`, the Gemini call is skipped entirely and the log reads `✓ Using manual prompt override`.

---

## Prompt Priority Order

```
1. prompt_override (Edit prompt directly)     → skips Gemini entirely; Default slider values used
2. Wild mode + story (non-empty)              → build_wild_settings() → prompt + slider updates
3. Default mode + story (non-empty)           → build_prompt_from_story() → prompt only
4. No story, no override (either mode)        → DEFAULT_PROMPT from cell-config
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
| `cell-imports` | All imports for all three modes | `FluxKontextPipeline`, `FluxControlNetPipeline`, `FluxControlNetModel`, `cv2`, `json` |
| `cell-drive` | Mount Google Drive | Unchanged |
| `cell-token` | `HF_TOKEN` + `GOOGLE_API_KEY` | Unchanged |
| `cell-config` | All mutable variables | Adds `DEFAULTS`, `DEFAULT_MODE`, `DEFAULT_SETTINGS_MODE`, both base model IDs |
| `cell-models` | `_peft_patch()` + `load_pipeline()` + initial load | `load_pipeline()` manages all three modes; VRAM unload on switch |
| `cell-gemini` | Both Gemini functions | `GEMINI_SYSTEM_PROMPT` + `build_prompt_from_story()` (Default) + `WILD_SYSTEM_PROMPT` + `build_wild_settings()` (Wild) |
| `cell-inference` | `extract_canny()` + `cartoonify()` generator | Generator yields 9-tuple: `(image, log, spinner_html, guidance, steps, cn_scale, cn_end, canny_low, canny_high)` |
| `cell-ui` | Gradio Blocks UI + `demo.launch()` | Story-first layout; mode selector; settings mode toggle; sliders in generate outputs |

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
