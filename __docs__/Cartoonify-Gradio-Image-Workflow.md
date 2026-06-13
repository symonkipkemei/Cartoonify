# Cartoonify — Story Interface Workflow

> `08_Cartoonify_Story_Gradio.ipynb`
> What happens between "write a story" and "download a cartoon"

---

## Full Pipeline Overview

```
User story (plain text)
    │
    ▼
① Gemini 2.5 Flash Lite
    │ structured seven-layer prompt
    ▼
② User reviews / edits prompt
    │
    ▼
③ Photo upload  →  Resize to 1024 × 1024
    │
    ▼
④ Depth Estimation  (Depth-Anything-V2-Small, CPU)
    │ greyscale depth map
    ▼
⑤ FLUX.1-dev Inference
    ├── ControlNet    depth map → spatial structure
    ├── LoRA          style bias → cartoon aesthetic
    └── Text prompt   mood, detail, composition
    ▼
⑥ 1024 × 1024 PNG  →  Gradio display + Drive save
```

Steps ③–⑥ are identical to `07_Cartoonify_Gradio.ipynb`. Steps ① and ② are new.

---

## Step 1 — Story to Prompt (Gemini)

The user writes a plain-language description of what they want to illustrate. Gemini converts it into a structured prompt that matches the vocabulary the LoRA was trained on.

### Why a structured prompt matters

The LoRA was trained on captions with a fixed seven-layer structure. Prompts that use the same vocabulary and layer order activate the LoRA more reliably — the model has stronger associations between those exact keywords and the visual style.

### Prompt structure

```
<trigger>, <medium>, <technique>, <color>, <mood>, <commentary>, <composition>
```

| Layer | Fixed or derived | Example |
|---|---|---|
| Trigger | Fixed | `gdo_cartoon` |
| Medium | Fixed | `editorial cartoon \| political cartoon \| caricature \| newspaper illustration` |
| Technique | Fixed | `pen and ink \| cross-hatching \| bold outlines \| hand-drawn linework \| varied line weight` |
| Color | Default or derived | `black and white \| monochrome \| print cartoon \| white background` |
| Mood | Derived from story | `satirical \| ironic \| dark humour \| critical \| bleak` |
| Commentary | Derived from story | `political commentary \| scathing \| power critique \| deadpan \| accusatory` |
| Composition | Derived from story | `two-group layout \| standing figure left \| queue right \| speech bubble top \| eye level` |

The first three layers are always the same — Gemini fills the last four from the user's story.

### How the Gemini call works

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
One-line structured prompt string
        │
        ▼
Trigger word substitution
(replaces gdo_cartoon if user changed the trigger field)
        │
        ▼
prompt_input textbox updated in Gradio
```

- Temperature `0.7` gives creative variation in mood and composition while keeping the vocabulary grounded
- `max_output_tokens=300` fits a full seven-layer prompt (~150 tokens typical)
- The call takes under 2 seconds and uses no GPU memory

### Example

**Story input:**
> A politician hands out empty promises while people queue for food.

**Gemini output:**
```
gdo_cartoon editorial cartoon | political cartoon | caricature | newspaper illustration,
pen and ink | cross-hatching | bold outlines | hand-drawn linework | varied line weight,
black and white | monochrome | print cartoon | white background,
satirical | ironic | dark humour | critical | bleak,
political commentary | scathing | power critique | deadpan | accusatory,
two-group layout | standing figure left | queue of figures right | speech bubble top | eye level | wide shot
```

---

## Step 2 — User Reviews the Prompt

The structured prompt populates the **Style Prompt** textbox. The user can:

- Accept it and click **Cartoonify** immediately
- Edit specific keywords before generating
- Discard it and write a prompt manually

The prompt box is always editable — Gemini output is a starting point, not a locked value.

---

## Step 3 — Resize

The source photo is resized to **1024 × 1024 pixels** using `Image.LANCZOS`.

- No cropping, no padding — the full image is squeezed into a square
- FLUX.1-dev was trained at 1024 px; deviating reduces output quality
- Happens on CPU before GPU work begins

---

## Step 4 — Depth Estimation

**Depth-Anything-V2-Small** runs on CPU and produces a greyscale depth map.

```
Source Photo            Depth Map
┌────────────┐          ┌────────────┐
│  subject   │          │ ░░░░░░░░░░ │  ← near (light)
│  in front  │   →→→   │ ▒▒▒▒▒▒▒▒▒▒ │
│  of wall   │          │ ████████████│  ← far (dark)
└────────────┘          └────────────┘
```

The raw array is normalised to 0–255 and converted to RGB for ControlNet. The result is shown in the depth preview panel.

Depth is used instead of Canny edges because it preserves the *spatial story* — what is in front of what — which survives radical style changes. The cartoon output maintains the same compositional weight as the source photo.

---

## Step 5 — FLUX Inference

Three components act simultaneously across 28 diffusion steps.

### ControlNet — spatial structure

The depth map is fed into **ControlNet-Union-Pro-2.0** at every step.

```
Depth Map ──→ ControlNet ──→ spatial signal injected
                               at each of 28 steps
                                      │
                                      ▼
                            FLUX output matches
                            the depth layout
```

- `control_mode = 2` — interprets the input as depth (not edges or pose)
- `controlnet_conditioning_scale = 0.8` — balances structure fidelity against creative freedom

The depth signal is a loose constraint, not a pixel lock. Major shapes and foreground/background relationships are preserved; FLUX retains significant creative latitude within those bounds.

### LoRA — cartoon style

```
FLUX.1-dev base weights   (24 GB, frozen)
         +
LoRA delta weights        (~600 MB)
         =
FLUX biased toward editorial cartoon style
```

- The trigger word (e.g. `gdo_cartoon`) activates the LoRA — it was baked into every training caption
- `joint_attention_kwargs = {"scale": 1.0}` — LoRA applied at full strength
- The LoRA shifts visual style without overriding the depth structure

### Text prompt — mood and composition

The structured prompt (from Gemini or written manually) is encoded by FLUX's T5 text encoder and steers generation at every step.

`guidance_scale = 3.5` — FLUX performs best between 3 and 5. Higher values produce more literal, less creative output.

---

## Step 6 — Output

The pipeline returns a **1024 × 1024 PIL Image**.

- Displayed in the Gradio result panel
- Auto-saved to Google Drive as `outputs/<YYYYMMDD_HHMMSS>_cartoonify.png`
- Downloadable via the Gradio download button

GPU memory is flushed after each generation (`gc.collect()` + `torch.cuda.empty_cache()`).

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
| Gemini temperature | 0.7 | Built into the function — not exposed in the UI |
