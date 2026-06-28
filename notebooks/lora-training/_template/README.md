# LoRA Training Template

Copy this folder to `../your-cartoonist-slug/` to train a new cartoonist LoRA.

## Steps

1. **Copy** this folder: `cp -r _template your-cartoonist-slug`
2. **Prepare dataset** — put source images in `data/images/` (JPG, at least 50 recommended)
3. **Write captions** — follow the 6-layer pipe-separated format (see below)
4. **Run notebooks in order:**
   - `01_Preparation.ipynb` — resizes images, pairs each with its caption, writes to `dataset_FLUX.1/`
   - `02_Train.ipynb` — fine-tunes FLUX.1-dev with your dataset, saves LoRA to Google Drive
5. **Wire into the pipeline** — update two variables in `05_Cartoonify_Gradio_Unified.ipynb` cell-config:

```python
LORA_DRIVE_PATH  = '/content/drive/MyDrive/cartoonify/your-cartoonist-slug.safetensors'
DEFAULT_TRIGGER  = 'your_trigger_word'
```

## Caption format (6 layers, pipe-separated)

```
trigger_word editorial cartoon | style descriptor | color palette | mood | commentary style | composition
```

Example from `gado-cartoon`:
```
gdo_cartoon editorial cartoon | pen and ink, cross-hatching, bold outlines | black and white, monochrome | satirical, exaggerated | dry wit, political commentary | three figure layout, eye level
```

## Reference implementation

`../gado-cartoon/` is the first completed instance of this pattern — 87 editorial cartoons, FLUX.1-dev, trained on Google Colab Pro A100.
