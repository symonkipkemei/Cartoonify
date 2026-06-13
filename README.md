# Cartoonify

Transforms any photo into a satirical editorial cartoon using a fine-tuned diffusion pipeline.

**Stack:** FLUX.1-dev · ControlNet Depth · custom LoRA · Gradio · Google Colab Pro (A100)

---

## How it works

```
Photo upload
    │
    ▼
Depth-Anything-V2  →  depth map (scene structure)
    │
    ▼
FLUX.1-dev + ControlNet + LoRA  →  cartoon image
    │
    ▼
1024×1024 PNG  →  Gradio UI + auto-save to Drive
```

The depth map locks the spatial composition; the LoRA applies the cartoon style; the prompt steers the mood and detail. None of these copy pixels from the source — FLUX generates from scratch, constrained by structure.

Full walkthrough: [\_\_docs\_\_/IMAGE_WORKFLOW.md](__docs__/IMAGE_WORKFLOW.md)  
System architecture: [\_\_docs\_\_/ARCHITECTURE.md](__docs__/ARCHITECTURE.md)

---

## Project layout

```
Cartoonify/
│
├── data/
│   ├── images/              # 17 editorial cartoons used as training reference (CTN001–CTN0017)
│   └── captions/            # Cartoonify_FLUX_Captions.xlsx — structured prompt metadata
│
├── model/
│   └── notebooks/
│       ├── lora-training/   # LoRA preparation, training, and evaluation notebooks
│       │   └── dataset_FLUX.1/   # 30 image+caption pairs (bof_aar_lacha trigger)
│       ├── image-image/     # FLUX image-to-image experiments (Canny, Depth)
│       ├── text-image/      # FLUX text-to-image experiments
│       └── interface/       # Gradio UI notebooks
│           └── 07_Cartoonify_Gradio.ipynb   ← main deliverable (working)
│
├── __docs__/                # Project documentation
│   ├── ARCHITECTURE.md          # System components, notebook structure, runtime requirements
│   ├── IMAGE_WORKFLOW.md        # Step-by-step data flow through the pipeline
│   └── PROPOSAL.md              # "What's the Story?" — next phase proposal
│
└── __sample__/              # Reference scaffolds
    ├── EXECUTION_PROPOSAL.md    # Original proposal (source)
    ├── IMAGE_WORKFLOW.md        # Original workflow doc (source)
    └── frontend/            # React/TypeScript UI scaffold (future use)
```

---

## Quickstart

**Requires:** Google Colab Pro → Runtime → A100 GPU

1. Open [model/notebooks/interface/07_Cartoonify_Gradio.ipynb](model/notebooks/interface/07_Cartoonify_Gradio.ipynb) in Colab
2. Add `HF_TOKEN` and `GOOGLE_API_KEY` to Colab Secrets (left sidebar → key icon)
3. Run all cells in order — first run downloads ~24 GB of model weights (~4 min)
4. Open the public Gradio URL printed at the end of the last cell
5. Drop a photo, hit **Cartoonify**

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

Swapping the LoRA requires updating two variables in the config cell:

```python
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/...'
DEFAULT_TRIGGER  = 'your_trigger_word'
```

---

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| Guidance Scale | 3.5 | FLUX sweet spot is 3–5 |
| Inference Steps | 28 | 20–35 is a good range |
| ControlNet Scale | 0.8 | Shakker-Labs recommended value |
| Seed | 42 | Fix for reproducibility; change to explore |

---

## Notebooks reference

| Notebook | Purpose |
|---|---|
| [07_Cartoonify_Gradio.ipynb](model/notebooks/interface/07_Cartoonify_Gradio.ipynb) | Main app — working prototype |
| [08_Cartoonify_Story_Gradio.ipynb](model/notebooks/interface/08_Cartoonify_Story_Gradio.ipynb) | Adds Gemini story-to-prompt layer |
| [09_Cartoonify_Kontext_Gradio.ipynb](model/notebooks/interface/09_Cartoonify_Kontext_Gradio.ipynb) | Phase 3 primary — FLUX Kontext native image-to-image |
| [10_Cartoonify_Canny_Gradio.ipynb](model/notebooks/interface/10_Cartoonify_Canny_Gradio.ipynb) | Phase 3 secondary — Canny edge map + ControlNet |
| [06_Gradio_FLUX_Depth_LoRA.ipynb](model/notebooks/interface/06_Gradio_FLUX_Depth_LoRA.ipynb) | Predecessor: FLUX + depth + LoRA (no UI polish) |
| [02_FLUX_LoRA_Train.ipynb](model/notebooks/lora-training/02_FLUX_LoRA_Train.ipynb) | LoRA fine-tuning |
| [01_FLUX_LoRA_Preparation.ipynb](model/notebooks/lora-training/01_FLUX_LoRA_Preparation.ipynb) | Dataset preparation and caption structuring |
| [06_FLUX_Image-to-Image_3_Depth.ipynb](model/notebooks/image-image/06_FLUX_Image-to-Image_3_Depth.ipynb) | Depth ControlNet experiments |

---

## What's next

**Phase 4 — "What's the Story?"**
A Gemini 2.5 Flash Lite layer that converts a plain-language story into a structured LoRA-aligned prompt. Users describe what they want in natural language; Gemini fills the seven caption layers automatically; the result pre-fills the prompt box for review before generation.

See [\_\_docs\_\_/PROPOSAL.md](__docs__/PROPOSAL.md) for the full design.

---

## Documentation

- [Architecture](__docs__/ARCHITECTURE.md) — system components, notebook structure, runtime requirements
- [Image Workflow](__docs__/IMAGE_WORKFLOW.md) — step-by-step pipeline walkthrough
- [What's the Story? Proposal](__docs__/PROPOSAL.md) — Gemini prompt-builder design
- [Phase 3 — Image-to-Image Approaches](__docs__/Phase3-Image-to-Image-Approaches.md) — approaches evaluated, decision rationale, what changed
- [Canny Architecture](__docs__/Cartoonify-Canny-Architecture.md) — Canny edge map pipeline, thresholds, ControlNet call reference
