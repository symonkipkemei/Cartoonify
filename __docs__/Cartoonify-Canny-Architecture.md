# Cartoonify — Canny Architecture

> Reference documentation for `10_Cartoonify_Canny_Gradio.ipynb`.
> Secondary mode: FLUX.1-dev + Canny ControlNet + LoRA + Gemini story-to-prompt.

---

## What This Is

Notebook `10` is the secondary image-to-image mode for Cartoonify. It uses Canny edge detection as the ControlNet conditioning signal: OpenCV extracts the hard geometric outlines of the source photo, and FLUX regenerates the image constrained to those edge lines.

**Use this mode when:**
- The subject must remain visually recognisable as a specific person (tight face outlines)
- Portrait caricature matters more than compositional freedom
- You want the cartoon to stay close to the original framing

**Use Kontext (`09`) instead when:**
- Full scene recomposition is needed
- Exaggeration of body language, scale, and staging is more important than face precision

---

## System Architecture

```
User story (plain text)
    │
    ▼
Gemini 2.5 Flash Lite
    │  seven-layer structured prompt
    ▼
User reviews / edits prompt
    │
    ▼
Photo upload  →  resize to 1024 × 1024
    │
    ▼
OpenCV cv2.Canny(image, low_threshold, high_threshold)
    │  hard geometric edge map (white lines on black)
    ▼
FluxControlNetPipeline
    ├── control_image   = canny edge map
    ├── control_mode    = 0  (Canny in Union Pro 2.0)
    ├── LoRA weights    (gdo_cartoon cartoon style)
    └── structured prompt (mood, composition, commentary)
    ▼
1024 × 1024 cartoon PNG  →  saved to Drive  →  shown in Gradio
```

---

## Components

| Component | Model / Library | Purpose |
|---|---|---|
| Story-to-prompt | `gemini-2.5-flash-lite` via `google-genai` | Converts plain text to seven-layer structured prompt |
| Edge extraction | `cv2.Canny()` from `opencv-python-headless` | Extracts hard geometric outlines from source photo |
| Base diffusion | `black-forest-labs/FLUX.1-dev` (~24 GB) | Flow-matching text-to-image backbone |
| ControlNet | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` (~4 GB) | Constrains generation to Canny edge structure |
| LoRA | `gdo_cartoon.safetensors` (~600 MB, from Drive) | Editorial cartoon style |
| UI | Gradio Blocks | Browser interface + public share link |

**Shared weights with `07` / `08`:** The base FLUX.1-dev weights and the ControlNet-Union-Pro-2.0 weights are identical to those used in the depth-based notebooks — they load from the Colab cache on warm runs.

---

## Notebook Cell Structure

| Cell ID | Purpose |
|---|---|
| `cell-gpu` | `nvidia-smi` — confirm A100 |
| `cell-install` | pip: diffusers, transformers, accelerate, gradio, google-genai, **opencv-python-headless** |
| `cell-restart` | `os.kill(os.getpid(), 9)` — force kernel restart for clean state |
| `cell-imports` | torch, cv2, PIL, diffusers, google.genai, gradio |
| `cell-drive` | Mount Google Drive |
| `cell-token` | HF_TOKEN + GOOGLE_API_KEY from Colab Secrets |
| `cell-config` | All user-editable variables |
| `cell-models` | Load FluxControlNetModel + FluxControlNetPipeline + LoRA |
| `cell-gemini` | `GEMINI_SYSTEM_PROMPT` + `build_prompt_from_story()` |
| `cell-inference` | `extract_canny()` + `cartoonify()` — returns `(canny_image, result)` |
| `cell-ui` | Gradio Blocks UI + `demo.launch(share=True)` |

---

## Canny Edge Extraction

```python
def extract_canny(pil_image: Image.Image, low: int, high: int) -> Image.Image:
    img_np = np.array(pil_image)
    edges  = cv2.Canny(img_np, low, high)
    edges  = edges[:, :, None]
    edges  = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)
```

The output is a three-channel RGB image with white lines on a black background — the format expected by ControlNet.

**Threshold behaviour:**

| Threshold setting | Edge density | Effect on output |
|---|---|---|
| Low=20, High=100 | Dense — many fine edges | Output stays very close to source geometry |
| Low=50, High=200 (default) | Moderate | Balanced structural constraint |
| Low=100, High=350 | Sparse — only dominant edges | More stylistic freedom, less geometric fidelity |

Edges between the two thresholds are kept only if connected to a strong edge (hysteresis). This suppresses noise while preserving fine lines that are part of continuous contours.

---

## ControlNet Call

```python
result = pipe(
    prompt=full_prompt,
    control_image=canny_image,
    control_mode=0,                              # 0 = Canny in Union Pro 2.0
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.8,                    # ControlNet active for first 80%
    generator=torch.Generator('cuda').manual_seed(seed),
    joint_attention_kwargs={'scale': 1.0},       # LoRA active
    max_sequence_length=512,
).images[0]
```

**`control_guidance_end=0.8`:** The ControlNet conditions only the first 80% of diffusion steps. FLUX finishes the last 20% freely — this softens hard aliasing at edge boundaries while preserving the overall structural layout.

---

## Configuration Reference

| Variable | Default | Notes |
|---|---|---|
| `LORA_SOURCE` | `'drive'` | `'drive'` or `'huggingface'` |
| `LORA_DRIVE_PATH` | `/content/drive/MyDrive/cartoonify/…/gdo_cartoon.safetensors` | Update to your Drive path |
| `DEFAULT_TRIGGER` | `'gdo_cartoon'` | Must match the LoRA training trigger word |
| `DEFAULT_GUIDANCE` | `3.5` | Standard FLUX range (3–5) |
| `DEFAULT_STEPS` | `28` | 20–35 is a good range |
| `DEFAULT_CN_SCALE` | `0.7` | How strongly edges constrain the output |
| `DEFAULT_CN_END` | `0.8` | Fraction of steps where ControlNet is active |
| `DEFAULT_CANNY_LOW` | `50` | Lower Canny threshold |
| `DEFAULT_CANNY_HIGH` | `200` | Upper Canny threshold |
| `DEFAULT_SEED` | `42` | Fixed seed for reproducibility |
| `CONTROLNET_MODEL` | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | Shared with notebooks 07 and 08 |

---

## Gradio Interface

The UI is identical to notebook `08` with these differences:

| Element | Notebook 08 (depth) | Notebook 10 (Canny) |
|---|---|---|
| Output column | Result + depth map preview | Result + Canny edge map preview |
| Advanced controls | Guidance, steps, CN scale, seed | + Canny Low threshold, + Canny High threshold, + CN Guidance End |
| Output preview label | "Depth Map" | "Canny Edge Map" |
| `cartoonify()` returns | `(depth_rgb, result)` | `(canny_image, result)` |
| Save filename suffix | `_cartoonify.png` | `_cartoonify_canny.png` |
| Header badge | `Depth ControlNet` | `Secondary Mode · Canny` |

---

## Gemini Story-to-Prompt Layer

The `cell-gemini` cell and `build_prompt_from_story()` function are carried forward from notebooks `08` and `09` without changes. The same `GEMINI_SYSTEM_PROMPT`, same model (`gemini-2.5-flash-lite`), same seven-layer output format, same `temperature=0.7`, same `max_output_tokens=300`.

---

## What Differs from Notebook 09 (Kontext)

| Area | `09` Kontext | `10` Canny |
|---|---|---|
| Pipeline | `FluxKontextPipeline` | `FluxControlNetPipeline` |
| Base model | `FLUX.1-Kontext-dev` | `FLUX.1-dev` |
| Preprocessing | None | `cv2.Canny()` edge map |
| Image to FLUX | `image=src` (full content) | `control_image=canny_image` (edges only) |
| ControlNet | Not used | ControlNet-Union-Pro-2.0, `control_mode=0` |
| Structural constraint | Semantic (FLUX reads the scene) | Geometric (hard edges constrain generation) |
| Compositional freedom | High | Low — output follows source outlines |
| Default guidance scale | `2.5` | `3.5` |
| Output panels | Result only | Result + Canny edge map preview |

---

## Runtime Requirements

- **GPU:** A100 (40 GB VRAM) — Google Colab Pro required
- **Secrets:** `HF_TOKEN` (Hugging Face read token) · `GOOGLE_API_KEY` (Google AI Studio)
- **First run downloads:** ~28 GB total (FLUX.1-dev ~24 GB + ControlNet ~4 GB + LoRA ~600 MB)
- **Warm run:** model weights cached in Colab — loads in ~1 minute

The FLUX.1-dev + ControlNet weights are shared with notebooks `07` and `08`. If those notebooks have run previously in the same Colab session, the cache is warm.
