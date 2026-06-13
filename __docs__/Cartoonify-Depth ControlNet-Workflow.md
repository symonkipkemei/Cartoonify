# Cartoonify — Depth ControlNet Workflow

> Notebooks `01_Cartoonify_Gradio_Depth.ipynb` and `02_Cartoonify_Gradio_Depth_Story.ipynb`
> What happens between "upload a photo" and "download a cartoon" using Depth ControlNet.

---

## Pipeline Overview

```
User story (plain text)              [02 only — skip to ③ in 01]
    │
① Gemini 2.5 Flash Lite
    │  structured seven-layer prompt
    ▼
② User reviews / edits prompt
    │
    ▼
③ Photo upload  →  Resize to 1024 × 1024
    │
    ▼
④ Depth Estimation  (Depth-Anything-V2-Small, CPU)
    │  greyscale depth map → converted to RGB
    ▼
⑤ FLUX.1-dev Inference
    ├── ControlNet (control_mode=2)  depth map → spatial structure
    ├── LoRA                         style bias → cartoon aesthetic
    └── Text prompt                  mood, detail, composition
    ▼
⑥ 1024 × 1024 PNG  →  Gradio display + Drive auto-save
```

Steps ①–② are present only in `02`. Step ③ onward is identical in both notebooks.

---

## Step 1 — Story to Prompt (Gemini) `02 only`

The user writes a plain-language description of what they want to illustrate. Gemini converts it into a structured prompt aligned with the LoRA training vocabulary.

```
GEMINI_SYSTEM_PROMPT
  (output format spec + hard rules + three few-shot examples)
        +
User message:  "Story: <user text>"
        │
        ▼
Gemini 2.5 Flash Lite
temperature = 0.7  ·  max_output_tokens = 300
        │
        ▼
Structured prompt — one line, seven layers
        │
        ▼
Trigger word substitution if user changed the trigger field
        │
        ▼
prompt_input textbox updated in Gradio
```

Temperature `0.7` gives creative variation in mood and composition while keeping the vocabulary grounded. The Gemini call takes under 2 seconds and uses no GPU memory.

**Example:**

Story input:
> A politician hands out empty promises while people queue for food.

Gemini output:
```
gdo_cartoon editorial cartoon | political cartoon | caricature | newspaper illustration,
pen and ink | cross-hatching | bold outlines | hand-drawn linework | varied line weight,
black and white | monochrome | print cartoon | white background,
satirical | ironic | dark humour | critical | bleak,
political commentary | scathing | power critique | deadpan | accusatory,
two-group layout | standing figure left | queue of figures right | speech bubble top | eye level | wide shot
```

---

## Step 2 — User Reviews the Prompt `02 only`

The structured prompt populates the **Style Prompt** textbox. The user can accept it, edit specific keywords, or discard it and write a prompt manually. The box is always editable — Gemini output is a starting point, not a locked value.

---

## Step 3 — Resize

```python
src = Image.fromarray(image).convert('RGB')
src = src.resize((1024, 1024), Image.LANCZOS)
```

The source photo is resized to **1024 × 1024 pixels** using LANCZOS resampling. No cropping or padding — the full image is scaled into a square. FLUX.1-dev was trained at this resolution; deviating from it reduces output quality. This step runs on CPU before any GPU work begins.

---

## Step 4 — Depth Estimation

**Depth-Anything-V2-Small** runs on CPU and produces a greyscale depth map from the source photo.

```
Source Photo            Depth Map (greyscale)
┌────────────┐          ┌────────────┐
│  subject   │          │ ░░░░░░░░░░ │  ← near (light / white)
│  in front  │  →→→→   │ ▒▒▒▒▒▒▒▒▒▒ │
│  of wall   │          │ ████████████│  ← far (dark / black)
└────────────┘          └────────────┘
```

The raw output is normalised to 0–255 and converted to RGB (three identical channels) for ControlNet. This converted version is shown in the depth preview panel in the Gradio UI.

**What the depth map encodes:**

| Encodes | Does not encode |
|---|---|
| Foreground / background separation | Pixel colours |
| Relative subject positions | Textures |
| Major shape proportions | Lighting direction |
| Near / far spatial relationships | Artistic style |
| Overall compositional weight | Faces, expressions, body language |

This is the key limitation of depth-based conditioning: expressions, gestures, and identity details — the elements that make political satire land — are stripped out. The cartoon output is recognisable through silhouette alone, not through content understanding. See Kontext mode (`03`) for the alternative.

---

## Step 5 — FLUX Inference

Three components act simultaneously across `num_inference_steps` diffusion steps.

### ControlNet — spatial structure

```python
result = pipe(
    prompt=full_prompt,
    control_image=depth_image,
    control_mode=2,                          # 2 = Depth in Union Pro 2.0
    controlnet_conditioning_scale=0.8,
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator('cuda').manual_seed(seed),
    joint_attention_kwargs={'scale': 1.0},
    max_sequence_length=512,
).images[0]
```

- `control_mode=2` — tells Union Pro to interpret the control image as a depth map (not edges or pose)
- `controlnet_conditioning_scale=0.8` — how strongly the depth signal constrains each diffusion step
- The depth signal is a loose constraint, not a pixel lock — FLUX retains significant creative latitude within the major shapes

### LoRA — cartoon style

```
FLUX.1-dev base weights   (~24 GB, bfloat16, frozen)
         +
LoRA delta weights        (~600 MB)
         =
FLUX biased toward editorial cartoon style
```

- The trigger word (e.g. `gdo_cartoon`) activates the LoRA — it was baked into every training caption
- `joint_attention_kwargs={'scale': 1.0}` — LoRA applied at full strength

### Text prompt — mood and composition

The structured prompt is encoded by FLUX's T5 text encoder and steers generation at every step alongside the depth conditioning signal. `guidance_scale=3.5` — FLUX performs best between 3 and 5.

---

## Step 6 — Output

The pipeline returns a **1024 × 1024 PIL Image**.

- Displayed in the Gradio result panel (downloadable)
- Auto-saved to Google Drive as `outputs/<YYYYMMDD_HHMMSS>_cartoonify.png`
- Depth map shown in a collapsible preview accordion in the right output column
- GPU memory flushed after each generation (`gc.collect()` + `torch.cuda.empty_cache()`)

---

## Parameter Reference

| Parameter | Default | What changing it does |
|---|---|---|
| Guidance Scale | 3.5 | Higher → more literal prompt adherence, less creative variation |
| Inference Steps | 28 | Higher → better quality, slower generation |
| ControlNet Scale | 0.8 | Higher → stricter depth structure, less stylistic freedom |
| Seed | 42 | Different value → different stylistic variation on same inputs |
| Trigger Word | `gdo_cartoon` | Must match the LoRA training trigger — activates the cartoon style |
| Gemini temperature | 0.7 (fixed) | Not exposed in the UI — edit `cell-gemini` to change |
