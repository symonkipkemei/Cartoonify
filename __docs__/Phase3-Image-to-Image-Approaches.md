# Phase 3 — Image-to-Image Approaches

> Why the depth pipeline was replaced, what was evaluated, and what was chosen.

---

## Why Depth ControlNet Is Limiting for Satire

Phases 1 and 2 use a depth map extracted from the source photo as the ControlNet conditioning signal. FLUX generates a new image constrained to that spatial layout.

The depth map encodes one thing only: how far each pixel is from the camera. It discards everything else — faces, expressions, clothing, gesture, body language. These are precisely the elements that make satirical cartoons land. The result is a cartoon that is recognisable through silhouette and spatial arrangement, but no deeper.

Phase 3 moves from geometric constraint to **semantic understanding**. Instead of extracting a proxy signal from the image, the pipeline passes the image directly to a model that reads its full content and reinterprets it through the prompt.

---

## Approaches Evaluated

Five image-to-image modes were evaluated against the reference notebooks in `__sample__/google_colab/02_IMAGE/`.

---

### FLUX Kontext — `FluxKontextPipeline`
**Reference notebook:** `08_FLUX_Kontext.ipynb`

```
Source Photo
    │
    ▼
FluxKontextPipeline
    ├── reads full image content (subjects, scene, context)
    ├── text prompt steers the reinterpretation
    └── generates a transformed version of the same scene
    ▼
Cartoon output
```

Kontext is FLUX's native image-to-image model. It takes the source image directly — no depth map, no edge extraction — and reads the entire image as context. The model understands who is in the image, what they are doing, and what the scene means, then recomposes it according to the text prompt.

**Characteristics for satirical use:**
- Characters remain recognisable through content understanding, not silhouette alone
- Composition can shift — figures can be repositioned, scale can be exaggerated
- Gemini-generated structured prompts plug in directly
- No preprocessing step; the image goes straight into the pipeline
- `guidance_scale = 2.5` — lower than standard FLUX, giving the model creative latitude to reinterpret rather than copy

**LoRA compatibility:** Kontext is a fine-tune of the FLUX.1-dev transformer. LoRA weights trained on FLUX.1-dev apply to the same layer types and load cleanly.

**Pipeline call:**
```python
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
result = pipe(
    image=src,
    prompt=full_prompt,
    guidance_scale=2.5,
    num_inference_steps=28,
    joint_attention_kwargs={'scale': 1.0},
).images[0]
```

**Status: selected as the Phase 3 primary pipeline — implemented in `09_Cartoonify_Kontext_Gradio.ipynb`**

---

### ControlNet Canny — `FluxControlNetPipeline` with edge map
**Reference notebook:** `06_FLUX_Image-to-Image_1_Canny.ipynb`

```
Source Photo
    │
    ▼
cv2.Canny(image, low_threshold, high_threshold)
    │ hard geometric edges
    ▼
FluxControlNetPipeline (control_mode=0)
    ▼
Cartoon output
```

Canny extracts hard geometric outlines — silhouettes, face contours, clothing folds, structural lines. FLUX generates a new image constrained to stay within those edges.

**Characteristics for satirical use:**
- Facial outlines are preserved — the subject stays recognisable as a specific person
- Edge density is tunable via thresholds — lower values produce tighter constraint, higher values are looser
- The bold-outline quality produced maps naturally to the `bold outlines | hand-drawn linework` vocabulary in the LoRA prompts
- More rigid than Kontext — compositional exaggeration is limited by the edge geometry

**Best fit:** Portrait-focused caricature where tight face recognition matters more than compositional freedom.

**Status: secondary mode — implemented in `10_Cartoonify_Canny_Gradio.ipynb`**

---

### FLUX Redux — `FluxPriorReduxPipeline` + ControlNet
**Reference notebook:** `07_FLUX_Redux.ipynb`

```
Reference cartoon image         Source photo
        │                            │
        ▼                            ▼
FluxPriorReduxPipeline          cv2.Canny()
(visual embeddings)             (structural edges)
        │                            │
        └──────────────┬─────────────┘
                       ▼
         FluxControlNetPipeline
         (text encoder disabled)
                       ▼
               Cartoon output
```

Redux encodes a reference image as visual embeddings that replace the text prompt entirely. The aesthetic of the reference — colours, textures, line quality — is transferred to the structural layout of the source photo. Text prompts are disabled; generation is steered by the visual embeddings alone.

**Characteristics:**
- Strong aesthetic transfer from a reference cartoon image
- Text prompts cannot be used — the Gemini story-to-prompt layer is incompatible with this pipeline
- Requires a high-quality reference cartoon to be effective; results depend entirely on what is provided

**Fit for Cartoonify:** Limited. The story-driven prompt pipeline (Phase 2) is the core of what Cartoonify does. Redux disables that capability entirely.

**Status: not selected — incompatible with the structured prompt approach**

---

### ControlNet Depth — Phase 1 & 2 approach
**Reference notebook:** `06_FLUX_Image-to-Image_3_Depth.ipynb`

The existing pipeline used in `07_Cartoonify_Gradio.ipynb` and `08_Cartoonify_Story_Gradio.ipynb`. Retained as a reference and as a fallback for scene-heavy images (landscapes, crowd scenes, architectural context) where spatial layout matters more than individual character recognition.

**Status: implemented in phases 1 & 2, available as a fallback**

---

### FLUX Inpainting — `FluxControlInpaintPipeline`
**Reference notebook:** `12_Inpainting_2_FLUX.ipynb`

Takes an image, a drawn mask, and a depth map. Regenerates only the masked region while leaving the rest of the image untouched. The mask is drawn interactively in a Gradio sketch widget.

**Fit for Cartoonify:** Not a primary mode. Designed for targeted regional editing, not full-image transformation. Relevant as a post-processing step — e.g., revising a specific element in an already-cartoonified output.

**Status: not selected for primary use**

---

## Decision Summary

| Approach | Pipeline | Preprocessing | Prompt-driven | Compositional freedom | Status |
|---|---|---|---|---|---|
| **Kontext** | `FluxKontextPipeline` | None | Yes | High | **Phase 3 primary** |
| Canny ControlNet | `FluxControlNetPipeline` | Edge map | Yes | Low | **Secondary — `10_Cartoonify_Canny_Gradio.ipynb`** |
| Redux | `FluxPriorReduxPipeline` | Edge map | No — disabled | Medium | Not selected |
| Depth ControlNet | `FluxControlNetPipeline` | Depth map | Yes | Low | Phases 1 & 2 |
| Inpainting | `FluxControlInpaintPipeline` | Depth map + mask | Yes | Regional only | Post-processing |

---

## What Changes in Phase 3

| Area | Phase 2 (`08`) | Phase 3 (`09`) |
|---|---|---|
| Pipeline | `FluxControlNetPipeline` + ControlNet | `FluxKontextPipeline` |
| Preprocessing | Depth-Anything-V2 → depth map | None |
| Image constraint | Spatial depth layout | Full semantic content |
| Compositional freedom | Low | High |
| Text prompt | Gemini structured prompt | Gemini structured prompt (unchanged) |
| LoRA | `gdo_cartoon` | `gdo_cartoon` (unchanged) |
| Default guidance scale | 3.5 | 2.5 |
| Model download | Shakker ControlNet (~4 GB, cached) | FLUX.1-Kontext-dev (~24 GB) |
| Depth preview panel | Shown in UI | Removed |
| ControlNet scale slider | Present | Removed |

The Gemini story-to-prompt layer is carried forward from Phase 2 without changes.
