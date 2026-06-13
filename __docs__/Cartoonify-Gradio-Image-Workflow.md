# Image Workflow — How Cartoonify Transforms a Photo

> What happens between "upload image" and "download cartoon"

---

## Pipeline Overview

```
Source Photo
    │
    ▼
① Resize → 1024 × 1024
    │
    ▼
② Depth Estimation  (Depth-Anything-V2-Small)
    │ greyscale depth map
    ▼
③ FLUX.1-dev Inference
    ├── ControlNet    depth map → spatial structure
    ├── LoRA          style bias → cartoon aesthetic
    └── Text prompt   mood, detail, composition
    ▼
④ 1024 × 1024 PNG  →  Gradio display + Drive save
```

---

## Step 1 — Resize

The source image is resized to **1024 × 1024 pixels** using `Image.LANCZOS` before any processing begins.

- No cropping, no padding — the full image is squeezed into a square
- FLUX.1-dev was trained at 1024 px; deviating reduces output quality
- Resizing happens on CPU before GPU work begins

---

## Step 2 — Depth Estimation

**Depth-Anything-V2-Small** runs on CPU and produces a greyscale depth map from the resized source image.

```
Source Photo            Depth Map
┌────────────┐          ┌────────────┐
│  subject   │          │ ░░░░░░░░░░ │  ← near (light)
│  in front  │   →→→   │ ▒▒▒▒▒▒▒▒▒▒ │
│  of wall   │          │ ████████████│  ← far (dark)
└────────────┘          └────────────┘
```

The raw depth array is normalised to 0–255 and converted to RGB (the format ControlNet expects). The resulting image is shown in the depth preview panel in the UI.

**Why depth instead of Canny edges:**
Canny captures geometric outlines — good for architecture, poor for people and complex scenes. Depth captures the *spatial story* — what is in front of what — which survives radical style changes. The cartoon output maintains the same compositional weight as the original photo because the depth structure is preserved throughout diffusion.

---

## Step 3 — FLUX Inference

The core transformation. Three components act simultaneously during the 28-step diffusion process.

### ControlNet — spatial structure

The depth map is fed into **ControlNet-Union-Pro-2.0** at every diffusion step.

```
Depth Map ──→ ControlNet ──→ spatial signal injected
                               at each of 28 steps
                                      │
                                      ▼
                            FLUX output matches
                            the depth layout
```

- `control_mode = 2` — instructs Union Pro to interpret the input as a depth map (not edges, pose, etc.)
- `controlnet_conditioning_scale = 0.8` — the Shakker-Labs recommended value; balances structure fidelity against creative freedom

The depth signal is a *loose constraint*, not a pixel-level lock. Foreground subjects remain in front, background stays behind, major shapes hold their position — but FLUX retains significant creative latitude within those bounds.

### LoRA — cartoon style

The LoRA is a ~600 MB set of weight adjustments trained on top of FLUX.1-dev that biases generation toward the cartoon/satirical aesthetic.

```
FLUX.1-dev base weights   (24 GB, frozen)
         +
LoRA delta weights        (~600 MB)
         =
FLUX biased toward editorial cartoon style
```

- The **trigger word** (e.g. `gdo_cartoon`) is required to activate the style — it was baked into every training caption and the model associates it with the visual vocabulary
- `joint_attention_kwargs = {"scale": 1.0}` — LoRA applied at full strength

### Text prompt — mood and detail

The prompt is processed by FLUX's T5 text encoder and steers generation alongside the depth and LoRA.

Prompts follow the seven-layer vocabulary the LoRA was trained on:

```
gdo_cartoon <medium>, <technique>, <color>, <mood>, <commentary>, <composition>
```

| Layer | Example |
|---|---|
| Trigger | `gdo_cartoon` |
| Medium | `editorial cartoon \| political cartoon \| caricature \| newspaper illustration` |
| Technique | `pen and ink \| cross-hatching \| bold outlines \| hand-drawn linework \| varied line weight` |
| Color | `black and white \| monochrome \| print cartoon \| white background` |
| Mood | `satirical \| confrontational \| absurdist \| dark humour \| grotesque` |
| Commentary | `political commentary \| ironic \| deadpan \| scathing \| sharp` |
| Composition | `three figure layout \| frontal view \| eye level \| speech bubble top left \| wide shot` |

`guidance_scale = 3.5` controls how literally the prompt is followed. FLUX performs best between 3 and 5 — higher values produce more literal, less creative output.

---

## Step 4 — Output

The pipeline returns a **1024 × 1024 PIL Image**.

- Displayed immediately in the Gradio result panel
- Auto-saved to Google Drive as `outputs/<YYYYMMDD_HHMMSS>_cartoonify.png`
- Downloadable via the Gradio download button

After each generation, `gc.collect()` and `torch.cuda.empty_cache()` flush GPU memory to prevent fragmentation across multiple runs in the same session.

---

## What the Depth Map Controls

| Controls | Does not control |
|---|---|
| Foreground / background separation | Pixel colours |
| Relative subject positions | Textures |
| Major shape proportions | Lighting direction |
| Near / far spatial relationships | Artistic style |
| Overall compositional weight | Fine linework detail |

---

## Parameter Reference

| Parameter | Default | What increasing it does |
|---|---|---|
| Guidance Scale | 3.5 | More literal prompt adherence, less creative variation |
| Inference Steps | 28 | Higher quality, slower generation |
| ControlNet Scale | 0.8 | Stricter depth structure, less style freedom |
| Seed | 42 | Different value → different stylistic variation on same inputs |
| Trigger Word | `gdo_cartoon` | Must match LoRA training word — activates the style |
