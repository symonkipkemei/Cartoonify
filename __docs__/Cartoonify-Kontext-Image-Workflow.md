# Cartoonify — Kontext Workflow

> Notebook `09_Cartoonify_Kontext_Gradio.ipynb`
> What happens between "upload a photo" and "download a cartoon" using FLUX Kontext.

---

## Pipeline Overview

```
User story (plain text)
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
④ FLUX.1-Kontext-dev Inference
    ├── image=src                full image content read directly
    ├── LoRA weights             cartoon style applied to the transformer
    └── Text prompt              mood, exaggeration, composition
    ▼
⑤ 1024 × 1024 PNG  →  Gradio display + Drive auto-save
```

No preprocessing step. The resized photo goes directly into the pipeline — no depth map, no edge extraction. FLUX Kontext reads the full image.

---

## Step 1 — Story to Prompt (Gemini)

Identical to notebook 08. The user writes a plain-language description; Gemini converts it into the seven-layer structured prompt the LoRA was trained on.

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

**Example:**

Story input:
> Two leaders shake hands in front of cameras while their shadows fight each other behind them.

Gemini output:
```
gdo_cartoon editorial cartoon | political cartoon | caricature | newspaper illustration,
pen and ink | cross-hatching | bold outlines | hand-drawn linework | varied line weight,
black and white | monochrome | print cartoon | white background,
satirical | ironic | duplicitous | tense | darkly comic,
political commentary | diplomatic hypocrisy | sharp | deadpan | allegorical,
two figure layout | shadow duality | frontal handshake foreground | fighting silhouettes background | centred composition | eye level
```

---

## Step 2 — User Reviews the Prompt

The structured prompt populates the **Style Prompt** textbox. The user can accept it, edit specific keywords, or discard it entirely and write a prompt manually. The box is always editable.

---

## Step 3 — Resize

```python
src = Image.fromarray(image).convert('RGB')
src = src.resize((1024, 1024), Image.LANCZOS)
```

The source photo is resized to **1024 × 1024 pixels** using LANCZOS resampling. No cropping or padding — the full image is scaled into a square. This step runs on CPU before GPU work begins.

---

## Step 4 — FLUX Kontext Inference

Kontext takes the source image directly as the `image=` parameter. No proxy signal (depth map, edge map) is extracted — the model reads the full pixel content of the photo.

```python
result = pipe(
    image=src,
    prompt=full_prompt,
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=2.5,
    generator=torch.Generator('cuda').manual_seed(seed),
    joint_attention_kwargs={'scale': 1.0},
    max_sequence_length=512,
).images[0]
```

### What Kontext understands from the image

Kontext is FLUX's native image-to-image model — a fine-tune of the FLUX.1-dev transformer trained to read an image as context and recompose it according to a text prompt. Unlike ControlNet-based modes, which extract a proxy signal (depth or edges) and use that as structure, Kontext reads:

- Who is in the image (faces, identities, relative prominence)
- What they are doing (posture, gesture, action)
- What the scene means (spatial relationships, foreground/background, props)
- Compositional weight (what is large and central vs small and peripheral)

The text prompt then steers how FLUX reinterprets that content — exaggerating features, shifting staging, applying cartoon style through the LoRA.

### guidance_scale = 2.5

Kontext operates at a lower guidance scale than standard FLUX (default 3.5). At 2.5, the model balances between the image content it read and the text prompt — giving it creative latitude to reinterpret rather than copy. Raising guidance pushes the output closer to the literal prompt and further from the source image.

### LoRA — cartoon style

```
FLUX.1-Kontext-dev base weights  (~24 GB, bfloat16, frozen)
         +
LoRA delta weights               (~600 MB)
         =
Kontext biased toward editorial cartoon style
```

Kontext is a fine-tune of the FLUX.1-dev transformer. LoRA weights trained on FLUX.1-dev apply to the same layer types in the Kontext transformer and load cleanly without retraining. The trigger word (`gdo_cartoon`) activates the cartoon style exactly as in the depth-based notebooks.

### What Kontext can and cannot do

| Can do | Cannot do |
|---|---|
| Recognise and preserve specific faces | Guarantee exact pixel-level pose matching |
| Exaggerate scale, expression, gesture via prompt | Override strong compositional geometry in the source |
| Reposition figures within the scene | Fully invent elements absent from the source |
| Change lighting, colour, atmosphere | Reproduce every fine garment detail |

Compositional freedom is high — the cartoon output can look quite different from the source photo while still clearly depicting the same subject in a recognisable situation.

---

## Step 5 — Output

The pipeline returns a **1024 × 1024 PIL Image**.

- Displayed in the Gradio result panel (downloadable, height 640 px)
- Auto-saved to Google Drive as `outputs/<YYYYMMDD_HHMMSS>_cartoonify_kontext.png`
- No preprocessing preview panel in the UI (nothing was extracted from the image)
- GPU memory flushed after each generation (`gc.collect()` + `torch.cuda.empty_cache()`)

---

## Parameter Reference

| Parameter | Default | What changing it does |
|---|---|---|
| Guidance Scale | 2.5 | Higher → closer to prompt, less image-inspired recomposition |
| Inference Steps | 28 | Higher → better quality, slower generation |
| Seed | 42 | Different value → different stylistic variation on same inputs |
| Trigger Word | `gdo_cartoon` | Must match the LoRA training trigger — activates the cartoon style |
| Gemini temperature | 0.7 (fixed) | Not exposed in the UI — edit `cell-gemini` to change |

**No ControlNet scale or Canny threshold sliders** — there is no conditioning signal to tune. The only constraint on the output is the image content plus the text prompt.

---

## Kontext vs Depth ControlNet

| Question | Depth mode (07/08) | Kontext mode (09) |
|---|---|---|
| How is the image used? | Depth proxy extracted; pixels discarded | Full image content read directly |
| Are faces preserved? | Silhouette only | Yes — faces and identities understood |
| Compositional freedom | Low — tied to depth layout | High — FLUX reinterprets the scene |
| Preprocessing step | Depth-Anything-V2 on CPU | None |
| Default guidance scale | 3.5 | 2.5 |
| Extra model to load | Depth estimator + ControlNet | None beyond base FLUX |
| Best for | Scene-heavy images, crowds, architecture | Portrait satire, recognisable subjects |
