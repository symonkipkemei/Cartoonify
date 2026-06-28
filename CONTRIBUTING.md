# Contributing to Cartoonify

## Adding a new cartoonist LoRA

Each cartoonist lives at `notebooks/lora-training/<cartoonist-slug>/`. The slug becomes the folder name, the LoRA filename on Google Drive, and the basis for the trigger word.

### 1. Set up the dataset folder

```
notebooks/lora-training/your-cartoonist-slug/
├── data/
│   ├── images/        ← source cartoons (JPG, minimum 50 recommended)
│   └── captions/      ← caption metadata (CSV or XLSX)
```

Use `notebooks/lora-training/_template/` as your starting point — it has the preparation and training notebooks pre-configured for this layout.

### 2. Write structured captions

Each image needs a 6-layer pipe-separated caption:

```
trigger_word editorial cartoon | technique | color | mood | commentary | composition
```

The trigger word must be unique per cartoonist (e.g. `gdo_cartoon` for Gado). It activates the style at inference time — without it the output defaults to generic FLUX output.

See `notebooks/lora-training/gado-cartoon/Cartoonify_FLUX_Captions.xlsx` for 87 worked examples.

### 3. Run the training notebooks

On Google Colab Pro (A100 GPU):

1. `01_FLUX_LoRA_Preparation.ipynb` — resizes images to 1024×1024, pairs with captions, writes to `dataset_FLUX.1/`
2. `02_FLUX_LoRA_Train.ipynb` — fine-tunes FLUX.1-dev, saves LoRA weights to Google Drive

Training 87 images for 2000 steps takes ~45 minutes on A100.

### 4. Wire into the pipeline

Open `notebooks/cartoonify/05_Cartoonify_Gradio_Unified.ipynb` and update two variables in `cell-config`:

```python
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/your-cartoonist-slug.safetensors'
DEFAULT_TRIGGER  = 'your_trigger_word'
```

That's the only change needed — the rest of the pipeline is cartoonist-agnostic.

---

## Adding a new content project

Content runs (a set of images cartoonified for a specific purpose) go in `projects/<project-slug>/`. See `projects/africa-iconic-buildings/` for the file naming convention:

```
projects/your-project/
├── README.md              ← what the project is, credits, blog link if published
├── originals/             ← source photographs
├── *_1024.png             ← resized pipeline inputs
├── *_cartoon.png          ← generated outputs
└── *_story.txt            ← satirical captions (Gemini input)
```

---

## Reporting issues

Open an issue on GitHub. Include: notebook version, GPU type, error cell output, and the story/prompt that triggered the problem.
