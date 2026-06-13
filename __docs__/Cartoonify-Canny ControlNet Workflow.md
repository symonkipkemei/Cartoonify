# Cartoonify — Canny ControlNet Workflow

> Notebook `04_Cartoonify_Gradio_Canny.ipynb`
> What happens between "upload a photo" and "download a cartoon" using Canny edge detection.

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
④ Canny Edge Extraction  (cv2.Canny, CPU)
    │  white geometric lines on black background
    ▼
⑤ FLUX.1-dev Inference
    ├── ControlNet (control_mode=0)  edge map → structural constraint
    ├── LoRA                         style bias → cartoon aesthetic
    └── Text prompt                  mood, detail, composition
    ▼
⑥ 1024 × 1024 PNG  →  Gradio display + Drive auto-save
        +
    Canny edge map shown in preview accordion
```

---

## Step 1 — Story to Prompt (Gemini)

Identical to notebooks 02 and 03. The user writes a plain-language description; Gemini converts it into the seven-layer structured prompt the LoRA was trained on.

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
> A general sits behind a massive desk covered in medals while tiny soldiers march across a map below him.

Gemini output:
```
gdo_cartoon editorial cartoon | political cartoon | caricature | newspaper illustration,
pen and ink | cross-hatching | bold outlines | hand-drawn linework | varied line weight,
black and white | monochrome | print cartoon | white background,
absurdist | dark | pompous | grotesque | confrontational,
political condemnation | war metaphor | accusatory | ironic | scathing,
large figure dominates upper frame | tiny figures lower third | desk as stage | top-heavy composition | eye level
```

---

## Step 2 — User Reviews the Prompt

The structured prompt populates the **Style Prompt** textbox. The user can accept it, edit specific keywords, or discard it and write a prompt manually. The box is always editable.

---

## Step 3 — Resize

```python
src = Image.fromarray(image).convert('RGB')
src = src.resize((1024, 1024), Image.LANCZOS)
```

The source photo is resized to **1024 × 1024 pixels** using LANCZOS resampling. No cropping or padding — the full image is scaled into a square. This step runs on CPU before GPU work begins.

---

## Step 4 — Canny Edge Extraction

OpenCV's Canny detector finds the hard geometric boundaries in the resized photo.

```python
def extract_canny(pil_image: Image.Image, low: int, high: int) -> Image.Image:
    img_np = np.array(pil_image)
    edges  = cv2.Canny(img_np, low, high)
    edges  = edges[:, :, None]
    edges  = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)
```

The output is a three-channel RGB image with **white lines on a black background** — the format expected by ControlNet.

### What the edge map captures

```
Source Photo               Canny Edge Map
┌────────────┐             ┌────────────┐
│  face      │             │   ╭──╮     │  ← face outline
│  in suit   │  →→→→→→   │  ╱    ╲    │
│  at desk   │             │  ╰──╯     │  ← suit / collar
└────────────┘             │  ─────── │  ← desk edge
                           └────────────┘
```

Canny encodes the hard geometric contours — silhouettes, face outlines, clothing folds, hair lines, desk edges — while ignoring flat textures and gradual tone changes. FLUX regenerates the image constrained to stay within these lines.

### Threshold behaviour

The Canny algorithm uses two thresholds with hysteresis: edges stronger than `high_threshold` are always kept; edges weaker than `low_threshold` are always discarded; edges between the two are kept only if connected to a strong edge.

| Low threshold | High threshold | Edge density | Effect on output |
|---|---|---|---|
| 20 | 80 | Very dense — fine details captured | Output very close to source geometry |
| 50 (default) | 200 (default) | Moderate — dominant contours | Balanced structural constraint |
| 100 | 350 | Sparse — only strong boundaries | More stylistic freedom within structure |

---

## Step 5 — FLUX Inference

Three components act simultaneously across `num_inference_steps` diffusion steps.

### ControlNet — structural constraint

```python
result = pipe(
    prompt=full_prompt,
    control_image=canny_image,
    control_mode=0,                          # 0 = Canny in Union Pro 2.0
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.8,               # ControlNet active for first 80% of steps
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator('cuda').manual_seed(seed),
    joint_attention_kwargs={'scale': 1.0},
    max_sequence_length=512,
).images[0]
```

- `control_mode=0` — tells Union Pro to interpret the control image as a Canny edge map
- `controlnet_conditioning_scale=0.7` — how strongly edges constrain each diffusion step; lower values (0.3–0.5) give more stylistic freedom
- `control_guidance_end=0.8` — ControlNet conditions only the first 80% of steps; FLUX finishes the last 20% freely. This softens hard aliasing at edge boundaries while preserving the overall structural layout

### LoRA — cartoon style

```
FLUX.1-dev base weights   (~24 GB, bfloat16, frozen)
         +
LoRA delta weights        (~600 MB)
         =
FLUX biased toward editorial cartoon style
```

The trigger word (`gdo_cartoon`) activates the LoRA. `joint_attention_kwargs={'scale': 1.0}` applies the LoRA at full strength. The LoRA shifts the visual style toward editorial cartoon linework without overriding the edge structure.

### Text prompt — mood and composition

The structured prompt is encoded by FLUX's T5 text encoder and steers generation alongside the Canny conditioning signal. `guidance_scale=3.5` — the same value used in depth mode; higher than Kontext because ControlNet modes benefit from stronger prompt steering.

### Why `control_guidance_end=0.8` matters

Without this parameter, the ControlNet forces every one of the 28 diffusion steps to follow the edge lines exactly. At full strength across all steps, FLUX produces cartoon lines that look mechanical — clean and precise but lacking the hand-drawn quality. Setting `end=0.8` lets FLUX work freely in the final 20% (steps 23–28), where it adds organic variation at edge boundaries and softens the most rigid-looking artefacts.

---

## Step 6 — Output

The pipeline returns a **1024 × 1024 PIL Image**.

- Displayed in the Gradio result panel (downloadable, height 480 px)
- Auto-saved to Google Drive as `outputs/<YYYYMMDD_HHMMSS>_cartoonify_canny.png`
- Canny edge map shown in a collapsible preview accordion in the right output column
- GPU memory flushed after each generation (`gc.collect()` + `torch.cuda.empty_cache()`)

---

## Parameter Reference

| Parameter | Default | What changing it does |
|---|---|---|
| Canny Low Threshold | 50 | Lower → more edges extracted → tighter geometric constraint |
| Canny High Threshold | 200 | Higher → only strong edges kept → looser, more expressive output |
| Guidance Scale | 3.5 | Higher → more literal prompt adherence |
| Inference Steps | 28 | Higher → better quality, slower generation |
| ControlNet Scale | 0.7 | Higher → edges constrain output more rigidly |
| ControlNet Guidance End | 0.8 | Lower → FLUX gets more steps to finish freely |
| Seed | 42 | Different value → different stylistic variation on same inputs |
| Trigger Word | `gdo_cartoon` | Must match LoRA training trigger — activates cartoon style |

---

## Canny vs Depth ControlNet

| Question | Depth mode (01/02) | Canny mode (04) |
|---|---|---|
| What is extracted? | Near/far distance per pixel | Hard geometric edge lines |
| What is preserved? | Spatial layout and proportions | Silhouettes, face contours, clothing folds |
| Face identity preserved? | Silhouette only | Outlines yes — more recognisable |
| Default CN scale | 0.8 | 0.7 |
| Extra parameter | — | Canny low/high thresholds |
| Preview panel | Greyscale depth map | White-on-black edge map |
| Best for | Scene-heavy compositions | Portrait caricature, tight subject recognition |

## Canny vs Kontext

| Question | Kontext mode (03) | Canny mode (04) |
|---|---|---|
| Preprocessing | None | cv2.Canny edge extraction |
| Image constraint | Semantic (content understood) | Geometric (edge lines) |
| Compositional freedom | High — FLUX reinterprets the scene | Low — output follows source outlines |
| Face identity | Read semantically | Preserved through geometric contours |
| ControlNet | Not used | ControlNet-Union-Pro-2.0 |
| Default guidance scale | 2.5 | 3.5 |
| Best for | Scene recomposition, satirical staging | Portrait caricature, close-to-source rendering |
