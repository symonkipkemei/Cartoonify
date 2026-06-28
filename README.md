# Cartoonify

Transforms any photo into a satirical editorial cartoon using fine-tuned diffusion pipelines.

**Stack:** FLUX.1-dev · ControlNet · FLUX.1-Kontext-dev · custom LoRA · Gemini 2.5 Flash Lite · Gradio · Google Colab Pro (A100)

---

## How it works

Three conditioning modes — pick based on what you need:

```
Photo upload  →  Resize to 1024 × 1024
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    Depth map        (direct)       Canny edges
  (Depth-Anything)                 (cv2.Canny)
          │               │               │
  FluxControlNet   FluxKontext    FluxControlNet
   (mode=2)       Pipeline        (mode=0)
          │               │               │
          └───────────────┴───────────────┘
                          │
              LoRA  +  Structured Prompt
                          │
                  1024×1024 cartoon PNG
```

| Mode | Notebook | Use when |
|---|---|---|
| **Depth** | `01` / `02` | Scene-heavy photos, landscapes, crowds |
| **Kontext** | `03` | Portrait satire — subject must stay recognisable |
| **Canny** | `04` | Tight caricature — follow source outlines closely |

All modes use the same Gemini story-to-prompt layer (notebooks `02`–`05`).
Notebook `05` combines all three modes in a single interface — recommended starting point.

---

## Project layout

```
Cartoonify/
│
├── notebooks/
│   ├── cartoonify/
│   │   ├── 05_Cartoonify_Gradio_Unified.ipynb      # ← start here — all three modes
│   │   └── archive/                                 # 01–04 kept for reference
│   │
│   └── lora-training/
│       ├── _template/                               # copy to add a new cartoonist LoRA
│       └── gado-cartoon/                            # first cartoonist instance
│           ├── 01_FLUX_LoRA_Preparation.ipynb
│           ├── 02_FLUX_LoRA_Train.ipynb
│           ├── Cartoonify_FLUX_Captions.xlsx
│           └── data/
│               ├── images/   # CTN001–CTN0087 source cartoons
│               └── captions/ # structured caption metadata
│
├── projects/                                        # content runs
│   └── africa-iconic-buildings/                     # 10 African buildings demo set
│
├── __docs__/                # Pipeline architecture and workflow docs
│
└── __sample__/              # Reference material (gitignored)
    └── google_colab/        # Upstream reference notebooks
```

---

## Quickstart

**Requires:** Google Colab Pro → Runtime → Change runtime type → **A100 GPU**

1. Open [notebooks/cartoonify/05_Cartoonify_Gradio_Unified.ipynb](notebooks/cartoonify/05_Cartoonify_Gradio_Unified.ipynb) in Colab
2. Add `HF_TOKEN` and `GOOGLE_API_KEY` to Colab Secrets (left sidebar → key icon)
3. Run all cells in order — first run downloads ~24 GB of model weights (~4 min)
4. Open the public Gradio URL printed at the end of the last cell
5. Write your story → upload a photo → pick a rendering style → click **Cartoonify**

Subsequent runs use the Colab model cache and load in ~1 minute.

---

## The LoRA

The pipeline uses a LoRA fine-tuned on editorial and political cartoons. The trigger word activates the style — without it the output defaults to generic FLUX output.

**Current LoRA:** `gdo_cartoon` (loaded from Google Drive)

**Training data format** — each image paired with a seven-layer structured caption:

```
gdo_cartoon editorial cartoon | political cartoon | caricature | newspaper illustration,
pen and ink | cross-hatching | bold outlines | hand-drawn linework | varied line weight,
black and white | monochrome | print cartoon | white background,
satirical | confrontational | exaggerated | humorous tension | absurdist,
dry wit | political commentary | ironic | sharp | deadpan,
three figure layout | horizontal spread | frontal view | eye level | speech bubble top left
```

Swapping the LoRA requires updating two variables in `cell-config` only:

```python
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/...'
DEFAULT_TRIGGER  = 'your_trigger_word'
```

---

## Key parameters

| Parameter | Depth (01/02) | Kontext (03) | Canny (04) | Unified 05 — Reimagine | Unified 05 — Scene | Unified 05 — Portrait |
|---|---|---|---|---|---|---|
| Guidance Scale | 3.5 | 2.5 | 3.5 | 2.5 | 3.5 | 3.5 |
| Inference Steps | 28 | 28 | 28 | 28 | 28 | 28 |
| ControlNet Scale | 0.8 | — | 0.7 | — | 0.8 | 0.7 |
| ControlNet End | — | — | 0.8 | — | — | 0.8 |
| Canny Low / High | — | — | 50 / 200 | — | — | 50 / 200 |
| Seed | 42 | 42 | 42 | 42 | 42 | 42 |

Notebook `05` auto-adjusts these defaults when the rendering style is changed in the UI.

---

## Notebooks reference

| Notebook | Mode | Description |
|---|---|---|
| [05_Cartoonify_Gradio_Unified.ipynb](notebooks/cartoonify/05_Cartoonify_Gradio_Unified.ipynb) | **All three** | **Recommended** — story-first UI, three rendering styles, dynamic pipeline loading |
| [archive/01_Cartoonify_Gradio_Depth.ipynb](notebooks/cartoonify/archive/01_Cartoonify_Gradio_Depth.ipynb) | Depth | Baseline — manual prompt, no Gemini *(archived)* |
| [archive/02_Cartoonify_Gradio_Depth_Story.ipynb](notebooks/cartoonify/archive/02_Cartoonify_Gradio_Depth_Story.ipynb) | Depth + Gemini | Story → Gemini → structured prompt → depth ControlNet *(archived)* |
| [archive/03_Cartoonify_Gradio_Kontext.ipynb](notebooks/cartoonify/archive/03_Cartoonify_Gradio_Kontext.ipynb) | Kontext | Full image content → FluxKontextPipeline *(archived)* |
| [archive/04_Cartoonify_Gradio_Canny.ipynb](notebooks/cartoonify/archive/04_Cartoonify_Gradio_Canny.ipynb) | Canny | cv2.Canny edges → FluxControlNetPipeline *(archived)* |
| [01_FLUX_LoRA_Preparation.ipynb](notebooks/lora-training/gado-cartoon/01_FLUX_LoRA_Preparation.ipynb) | Training | Dataset preparation and caption structuring |
| [02_FLUX_LoRA_Train.ipynb](notebooks/lora-training/gado-cartoon/02_FLUX_LoRA_Train.ipynb) | Training | LoRA fine-tuning |

---

## Documentation

**Architecture (shared across all notebooks):**
- [System Architecture](__docs__/Cartoonify-Gradio-Architecture.md) — component comparison, Gemini layer, LoRA loading, Gradio layout, runtime requirements

**Unified interface:**
- [Unified Interface](__docs__/Cartoonify-Unified-Interface.md) — notebook 05 — flow, VRAM switching, mode selector, processing log, configuration

**Image workflows (one per conditioning mode):**
- [Depth ControlNet Workflow](__docs__/Cartoonify-Depth%20ControlNet-Workflow.md) — notebooks 01/02 — Depth-Anything-V2 → FluxControlNetPipeline
- [Kontext Workflow](__docs__/Cartoonify-Kontext-Image-Workflow.md) — notebook 03 — full image content → FluxKontextPipeline
- [Canny ControlNet Workflow](__docs__/Cartoonify-Canny%20ControlNet%20Workflow.md) — notebook 04 — cv2.Canny edges → FluxControlNetPipeline

---

## Published work

- [Cartoonify: Buildings as Political Objects](https://blog.iaac.net/cartoonify-buildings-as-political-objects/) — 10 iconic African buildings cartoonified using the pipeline, IAAC Blog, 2025
