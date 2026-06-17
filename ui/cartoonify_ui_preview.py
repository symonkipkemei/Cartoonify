"""
Cartoonify — UI preview (no GPU, no FLUX, no Gemini)
Run:  python cartoonify_ui_preview.py

Only requires: gradio, Pillow
"""

import time
import random
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_TRIGGER = 'gdo_cartoon'
DEFAULT_PROMPT  = 'satirical cartoon illustration, bold outlines, vivid flat colours'
DEFAULT_MODE    = 'Reimagine'
DEFAULT_STEPS   = 28
DEFAULT_SEED    = 42

DEFAULTS = {
    'reimagine': {'guidance': 2.5, 'cn_scale': 0.7, 'cn_end': 0.8, 'canny_low': 50,  'canny_high': 200},
    'scene':     {'guidance': 3.5, 'cn_scale': 0.8, 'cn_end': 0.8, 'canny_low': 50,  'canny_high': 200},
    'portrait':  {'guidance': 3.5, 'cn_scale': 0.7, 'cn_end': 0.8, 'canny_low': 50,  'canny_high': 200},
}

MODE_DESCRIPTIONS = {
    'Reimagine': (
        'FLUX recomposes the full image freely. '
        'The scene, staging, background and framing can all shift. '
        'Best for bold creative reinterpretation where you want maximum satirical transformation.'
    ),
    'Scene': (
        'A depth map locks the spatial layout and figure positions. '
        'Foreground and background relationships stay intact. '
        'Best for crowd scenes, protests, or architecture where structure must be preserved.'
    ),
    'Portrait': (
        'Canny edge detection traces face and body outlines before rendering. '
        'The person stays immediately recognisable in the cartoon. '
        'Best when satirising a specific individual — politician, CEO, public figure.'
    ),
}

# ── Placeholder image (replaces real FLUX output) ─────────────────────────────
_PALETTES = {
    'reimagine': [(15, 10, 45),  (80, 40, 160),  (160, 100, 240)],
    'scene':     [(10, 30, 20),  (40, 120, 80),  (120, 200, 140)],
    'portrait':  [(40, 15, 15),  (160, 60, 40),  (240, 160, 100)],
}

def _make_placeholder(mode: str, story: str) -> Image.Image:
    bg, mid, hi = _PALETTES.get(mode, _PALETTES['reimagine'])
    img  = Image.new('RGB', (1024, 1024), bg)
    draw = ImageDraw.Draw(img)
    for i in range(7):
        r = 50 + i * 80
        draw.ellipse([512 - r, 512 - r, 512 + r, 512 + r],
                     outline=mid if i % 2 == 0 else hi, width=4 - (i % 2))
    draw.text((512, 480), f'[ {mode.upper()} ]', fill=hi, anchor='mm')
    snippet = (story[:55] + '…') if len(story) > 55 else story
    if snippet:
        draw.text((512, 530), snippet, fill=mid, anchor='mm')
    draw.text((512, 960), '— mock output · no GPU —', fill=mid, anchor='mm')
    return img


# ── Status HTML helpers ───────────────────────────────────────────────────────
def _status(msg: str, state: str = 'processing') -> str:
    colors = {'idle': '#3f3f46', 'processing': '#a78bfa', 'done': '#34d399', 'warn': '#fbbf24'}
    c    = colors.get(state, colors['processing'])
    anim = ' animation:dot-pulse 1.2s ease-in-out infinite;' if state == 'processing' else ''
    return (
        f'<div class="cfy-status">'
        f'<div class="cfy-dot" style="background:{c};{anim}"></div>'
        f'<span>{msg}</span>'
        f'</div>'
    )

def _progress(step: int, total: int) -> str:
    pct = int(100 * step / max(total, 1))
    return (
        f'<div class="cfy-status">'
        f'<div class="cfy-dot" style="background:#a78bfa;animation:dot-pulse 1.2s ease-in-out infinite;"></div>'
        f'<div class="cfy-progress-wrap">'
        f'<div class="cfy-progress-label"><span>FLUX rendering</span>'
        f'<span style="color:#a78bfa;font-weight:700">{step} / {total}</span></div>'
        f'<div class="cfy-progress-track">'
        f'<div class="cfy-progress-fill" style="width:{pct}%"></div>'
        f'</div></div></div>'
    )

_STATUS_IDLE = _status('Upload a photo, describe your story, then hit Cartoonify.', 'idle')


# ── Image upload → show on canvas, reset result ───────────────────────────────
def on_image_upload(image):
    if image is None:
        return gr.update(value=None), None, gr.update(visible=False), None
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    return gr.update(value=pil), pil, gr.update(visible=False), None


# ── Toggle between original and cartoon ───────────────────────────────────────
def toggle_view(choice, original, result):
    img = original if choice == 'Original' else result
    return gr.update(value=img)


# ── Mock cartoonify generator ─────────────────────────────────────────────────
def cartoonify(
    story, image, mode_label, wild_mode,
    trigger_word, guidance_scale, num_steps,
    cn_scale, cn_end, canny_low, canny_high,
    seed, prompt_override,
):
    """Generator — yields (canvas, status, result_state, view_toggle, g, s, cn_s, cn_e, clow, chigh)."""
    if image is None:
        raise gr.Error('Upload a photo first.')

    mode = mode_label.lower()

    def emit(canvas=None, status=None, result=None, show_toggle=None,
             g=None, s=None, cn_s=None, cn_e=None, clow=None, chigh=None):
        if show_toggle is None:
            toggle_upd = gr.update()
        elif show_toggle:
            toggle_upd = gr.update(visible=True, value='Cartoon')
        else:
            toggle_upd = gr.update(visible=False)
        return (
            gr.update() if canvas is None else gr.update(value=canvas),
            gr.update() if status is None else gr.update(value=status),
            gr.update() if result is None else result,
            toggle_upd,
            gr.update() if g     is None else gr.update(value=g),
            gr.update() if s     is None else gr.update(value=s),
            gr.update() if cn_s  is None else gr.update(value=cn_s),
            gr.update() if cn_e  is None else gr.update(value=cn_e),
            gr.update() if clow  is None else gr.update(value=clow),
            gr.update() if chigh is None else gr.update(value=chigh),
        )

    g_val     = guidance_scale
    s_val     = int(num_steps)
    cn_s_val  = cn_scale
    cn_e_val  = cn_end
    clow_val  = int(canny_low)
    chigh_val = int(canny_high)

    # ── Step 1: Prompt ────────────────────────────────────────────────────────
    if prompt_override.strip():
        yield emit(status=_status('✓ Using manual prompt override', 'done'))
        time.sleep(0.3)

    elif wild_mode and story.strip():
        yield emit(status=_status('⚡ Wild — Gemini building prompt and tuning for satire…'))
        time.sleep(1.2)
        g_val     = round(random.uniform(3.5, 5.5), 1)
        s_val     = random.choice([32, 35, 36, 38])
        cn_s_val  = round(random.uniform(0.75, 0.95), 2)
        cn_e_val  = round(random.uniform(0.70, 0.85), 2)
        clow_val  = random.choice([20, 28, 35, 40])
        chigh_val = random.choice([120, 140, 160, 180])
        rationale = 'Mock: confrontational tone → high guidance, tight Canny for face detail'
        yield emit(
            status=_status(f'✓ Wild applied — {rationale} · trigger: {trigger_word}', 'done'),
            g=g_val, s=s_val, cn_s=cn_s_val,
            cn_e=cn_e_val, clow=clow_val, chigh=chigh_val,
        )
        time.sleep(0.4)

    elif story.strip():
        yield emit(status=_status('Gemini building your prompt…'))
        time.sleep(0.9)
        yield emit(status=_status(f'✓ Prompt ready · trigger: {trigger_word}', 'done'))
        time.sleep(0.3)

    else:
        yield emit(status=_status('No story — using default prompt', 'idle'))
        time.sleep(0.3)

    # ── Step 2: Pipeline switch (mock) ────────────────────────────────────────
    yield emit(status=_status(f'Loading {mode_label} pipeline…'))
    time.sleep(0.5)
    yield emit(status=_status(f'✓ {mode_label} pipeline ready', 'done'))
    time.sleep(0.2)

    # ── Step 3: Fake inference with CSS progress bar ──────────────────────────
    for step in range(0, s_val + 1):
        time.sleep(0.06)
        yield emit(status=_progress(step, s_val))

    # ── Step 4: Result ────────────────────────────────────────────────────────
    result_img = _make_placeholder(mode, story)
    yield emit(
        canvas=result_img,
        status=_status('✓ Done — toggle to compare with your original', 'done'),
        result=result_img,
        show_toggle=True,
    )


# ── Mode update (slider visibility + description) ─────────────────────────────
def update_mode(mode):
    d           = DEFAULTS[mode.lower()]
    is_kontext  = (mode == 'Reimagine')
    is_portrait = (mode == 'Portrait')
    return (
        MODE_DESCRIPTIONS[mode],
        gr.update(value=d['guidance']),
        gr.update(visible=not is_kontext, value=d['cn_scale']),
        gr.update(visible=is_portrait,    value=d['cn_end']),
        gr.update(visible=is_portrait,    value=d['canny_low']),
        gr.update(visible=is_portrait,    value=d['canny_high']),
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = '''
/* ── Tokens ──────────────────────────────────────────────── */
:root {
    --bg:         #0d0d0f;
    --surface:    #17171a;
    --surface-2:  #1e1e22;
    --border:     #2a2a30;
    --accent:     #a78bfa;
    --accent-dim: rgba(167,139,250,0.12);
    --text:       #f4f4f5;
    --muted:      #71717a;
    --amber:      #fbbf24;
    --amber-dim:  rgba(251,191,36,0.10);
    --radius:     12px;
}

/* ── Page reset ──────────────────────────────────────────── */
.gradio-container {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    color: var(--text) !important;
}
.gradio-container * { box-sizing: border-box; }
.dark, body, .main, .wrap, .contain, .block { background: transparent !important; }

/* ── Outer row (sidebar | canvas) ───────────────────────── */
#cfy-app { gap: 0 !important; }
#cfy-app > .wrap { gap: 0 !important; }

/* ── SIDEBAR ─────────────────────────────────────────────── */
#cfy-sidebar {
    min-width: 268px !important;
    max-width: 268px !important;
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding: 1.25rem 1rem !important;
    overflow-y: auto !important;
    align-self: stretch !important;
}

.cfy-logo {
    display: block;
    font-size: 1.25rem; font-weight: 900; letter-spacing: -0.8px;
    color: var(--text); padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--border); margin-bottom: 0.75rem;
}
.cfy-badge {
    display: inline-block; margin-left: 0.4rem;
    background: var(--amber-dim); color: var(--amber);
    border: 1px solid rgba(251,191,36,0.25);
    font-size: 0.58rem; font-weight: 800; padding: 0.1rem 0.4rem;
    border-radius: 20px; letter-spacing: 0.08em; text-transform: uppercase;
    vertical-align: middle;
}

.cfy-section-label {
    display: block;
    font-size: 0.62rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted);
    margin: 1rem 0 0.3rem;
}

/* Mode radio — vertical list */
#cfy-mode-nav { background: transparent !important; border: none !important; padding: 0 !important; }
#cfy-mode-nav .wrap { flex-direction: column !important; gap: 1px !important; }
#cfy-mode-nav label {
    padding: 0.45rem 0.6rem !important;
    border-radius: 7px !important; border: none !important;
    background: transparent !important;
    color: var(--muted) !important; font-weight: 500 !important;
    font-size: 0.85rem !important; cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
}
#cfy-mode-nav label:has(input:checked) {
    background: var(--accent-dim) !important; color: var(--accent) !important;
}
#cfy-mode-nav label:hover:not(:has(input:checked)) {
    background: var(--surface-2) !important; color: var(--text) !important;
}
#cfy-mode-nav input[type="radio"] { display: none !important; }

/* Mode description */
#cfy-mode-desc { margin-top: 0.2rem !important; margin-bottom: 0.5rem !important; }
#cfy-mode-desc p {
    color: var(--muted) !important; font-size: 0.76rem !important;
    line-height: 1.55 !important; margin: 0 !important;
    padding: 0.25rem 0.65rem !important;
}

/* Wild toggle */
#cfy-wild {
    padding: 0.5rem 0.6rem !important; border-radius: 8px !important;
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    transition: background 0.2s, border-color 0.2s !important;
}
#cfy-wild:has(input:checked) { background: var(--amber-dim) !important; border-color: rgba(251,191,36,0.3) !important; }
#cfy-wild label { color: var(--text) !important; font-weight: 600 !important; font-size: 0.79rem !important; cursor: pointer !important; gap: 0.5rem !important; }
#cfy-wild input[type="checkbox"] {
    appearance: none !important; -webkit-appearance: none !important;
    width: 2rem !important; height: 1rem !important;
    background: var(--border) !important; border-radius: 999px !important;
    position: relative !important; cursor: pointer !important; flex-shrink: 0 !important;
    transition: background 0.2s !important;
}
#cfy-wild input[type="checkbox"]:checked { background: var(--amber) !important; }
#cfy-wild input[type="checkbox"]::after {
    content: '' !important; position: absolute !important;
    top: 2px !important; left: 2px !important;
    width: 12px !important; height: 12px !important;
    background: white !important; border-radius: 50% !important;
    transition: transform 0.2s !important;
}
#cfy-wild input[type="checkbox"]:checked::after { transform: translateX(16px) !important; }

/* Accordion */
.gradio-container details {
    background: transparent !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; margin-top: 0.65rem !important;
}
.gradio-container details summary {
    color: var(--muted) !important; font-size: 0.78rem !important;
    font-weight: 600 !important; padding: 0.45rem 0.6rem !important; cursor: pointer !important;
}
.gradio-container details summary:hover { color: var(--text) !important; }
.gradio-container details[open] summary { color: var(--text) !important; border-bottom: 1px solid var(--border) !important; }
.gradio-container details .block { padding: 0.6rem !important; }

/* Sliders */
.gradio-container input[type="range"] { accent-color: var(--accent) !important; }
.gradio-container label span { color: var(--muted) !important; font-size: 0.76rem !important; }

/* ── CANVAS AREA ─────────────────────────────────────────── */
#cfy-canvas-area {
    padding: 1.25rem !important; min-width: 0 !important;
}
#cfy-canvas-area > .wrap { display: flex; flex-direction: column; gap: 0.65rem; }

/* Canvas */
#cfy-canvas {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
#cfy-canvas img { border-radius: 10px !important; object-fit: contain !important; }

/* Original / Cartoon toggle — full-width segmented bar */
#cfy-toggle {
    width: 100% !important;
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: 4px !important;
}
#cfy-toggle .wrap { display: flex !important; gap: 4px !important; width: 100% !important; }
#cfy-toggle label {
    flex: 1 !important; text-align: center !important;
    padding: 0.48rem 0 !important; border-radius: 8px !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    color: var(--muted) !important; cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    border: none !important; background: transparent !important;
}
#cfy-toggle label:has(input:checked) { background: var(--accent) !important; color: white !important; }
#cfy-toggle input[type="radio"] { display: none !important; }

/* Activity / status box */
#cfy-status-box {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    min-height: 2.4rem !important;
}
.cfy-status {
    display: flex; align-items: center; gap: 0.55rem;
    min-height: 1.4rem; padding: 0;
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
    font-size: 0.73rem; color: var(--muted);
}
.cfy-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.cfy-progress-wrap { flex: 1; display: flex; flex-direction: column; gap: 3px; }
.cfy-progress-label { display: flex; justify-content: space-between; font-size: 0.73rem; color: var(--muted); }
.cfy-progress-track { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.cfy-progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.08s linear; }
@keyframes dot-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.3; transform: scale(0.7); }
}

/* Bottom input bar */
#cfy-input-row {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem !important; gap: 0.75rem !important;
}
#cfy-input-row > .wrap { align-items: stretch !important; }

/* Text inputs */
.gradio-container textarea, .gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important; font-size: 0.875rem !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: var(--muted) !important; }
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: var(--accent) !important; outline: none !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
}

/* Photo upload thumb — compact, no bleed */
.cfy-upload {
    border: 2px dashed var(--border) !important; border-radius: 8px !important;
    background: var(--surface-2) !important; transition: border-color 0.2s !important;
    overflow: hidden !important;
}
.cfy-upload:hover { border-color: var(--accent) !important; }
/* Hide the verbose upload instructions; keep only the small icon */
.cfy-upload .wrap { overflow: hidden !important; height: 100% !important; }
.cfy-upload p,
.cfy-upload .upload-text,
.cfy-upload span.or { display: none !important; }
.cfy-upload button.upload { display: none !important; }
.cfy-upload svg { width: 20px !important; height: 20px !important; opacity: 0.4 !important; }

/* Generate button */
#cfy-btn button {
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    height: 46px !important; width: 100% !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3) !important;
    transition: opacity 0.15s, transform 0.12s !important;
}
#cfy-btn button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#cfy-btn button:active { transform: translateY(0) !important; }
#cfy-btn button:disabled { opacity: 0.45 !important; cursor: wait !important; }

footer { display: none !important; }
'''


# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title='Cartoonify') as demo:

    original_state = gr.State(None)
    result_state   = gr.State(None)

    # ── Sidebar | Canvas ─────────────────────────────────────────────────────
    with gr.Row(elem_id='cfy-app', equal_height=False):

        # ── LEFT SIDEBAR ─────────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=268, elem_id='cfy-sidebar'):

            gr.HTML('<span class="cfy-logo">&#127912; Cartoonify'
                    '<span class="cfy-badge">UI Preview</span></span>')

            gr.HTML('<span class="cfy-section-label">Mode</span>')
            mode_selector = gr.Radio(
                choices=['Reimagine', 'Scene', 'Portrait'],
                value=DEFAULT_MODE, label='', show_label=False,
                elem_id='cfy-mode-nav',
            )
            mode_desc = gr.Markdown(
                value=MODE_DESCRIPTIONS[DEFAULT_MODE],
                elem_id='cfy-mode-desc',
            )

            gr.HTML('<span class="cfy-section-label">Options</span>')
            wild_toggle = gr.Checkbox(
                value=False,
                label='⚡ Wild — Gemini tunes for max satire',
                elem_id='cfy-wild',
            )

            with gr.Accordion('Fine-tune', open=False):
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=10.0,
                    value=DEFAULTS[DEFAULT_MODE.lower()]['guidance'],
                    step=0.5, label='Guidance Scale',
                )
                steps_slider = gr.Slider(
                    minimum=10, maximum=50, value=DEFAULT_STEPS,
                    step=1, label='Inference Steps',
                )
                cn_scale_slider = gr.Slider(
                    minimum=0.1, maximum=1.5,
                    value=DEFAULTS[DEFAULT_MODE.lower()]['cn_scale'],
                    step=0.05, label='ControlNet Scale',
                    visible=DEFAULT_MODE != 'Reimagine',
                )
                cn_end_slider = gr.Slider(
                    minimum=0.3, maximum=1.0,
                    value=DEFAULTS[DEFAULT_MODE.lower()]['cn_end'],
                    step=0.05, label='ControlNet Guidance End',
                    visible=DEFAULT_MODE == 'Portrait',
                )
                canny_low_slider = gr.Slider(
                    minimum=0, maximum=200,
                    value=DEFAULTS[DEFAULT_MODE.lower()]['canny_low'],
                    step=10, label='Canny Low Threshold',
                    visible=DEFAULT_MODE == 'Portrait',
                )
                canny_high_slider = gr.Slider(
                    minimum=50, maximum=500,
                    value=DEFAULTS[DEFAULT_MODE.lower()]['canny_high'],
                    step=10, label='Canny High Threshold',
                    visible=DEFAULT_MODE == 'Portrait',
                )
                seed_input    = gr.Number(value=DEFAULT_SEED, label='Seed', precision=0)
                trigger_input = gr.Textbox(value=DEFAULT_TRIGGER, label='LoRA Trigger Word', lines=1)
                with gr.Accordion('Edit prompt directly', open=False):
                    prompt_input = gr.Textbox(
                        label='Prompt override',
                        placeholder='Leave empty — Gemini builds the prompt from your story.',
                        lines=3, max_lines=6, value='',
                    )

        # ── CANVAS + INPUT ────────────────────────────────────────────────────
        with gr.Column(scale=7, elem_id='cfy-canvas-area'):

            # Main canvas — shows uploaded photo, then cartoon result
            canvas_display = gr.Image(
                label='', type='pil', interactive=False,
                elem_id='cfy-canvas', height=490,
            )

            # Original / Cartoon toggle (hidden until generation completes)
            view_toggle = gr.Radio(
                choices=['Original', 'Cartoon'],
                value='Cartoon', label='', show_label=False,
                visible=False, elem_id='cfy-toggle',
            )

            gr.HTML('<span class="cfy-section-label" style="margin:0 0 0.2rem">Activity</span>')
            status_output = gr.HTML(value=_STATUS_IDLE, elem_id='cfy-status-box')

            # Bottom bar: story text (left) + photo upload thumb (right)
            with gr.Row(elem_id='cfy-input-row', equal_height=True):
                with gr.Column(scale=5):
                    story_input = gr.Textbox(
                        label='', lines=3, max_lines=6,
                        placeholder=(
                            'Describe your story or what to change…\n'
                            'e.g. A politician handing out empty promises while people queue for food'
                        ),
                    )
                with gr.Column(scale=2, min_width=130):
                    img_input = gr.Image(
                        label='Photo', type='numpy', height=104,
                        elem_classes=['cfy-upload'],
                        sources=['upload', 'clipboard'],
                    )

            generate_btn = gr.Button(
                'Cartoonify',
                variant='primary', elem_id='cfy-btn',
            )

    # ── Event wiring ──────────────────────────────────────────────────────────

    mode_selector.change(
        fn=update_mode, inputs=[mode_selector],
        outputs=[mode_desc, guidance_slider, cn_scale_slider,
                 cn_end_slider, canny_low_slider, canny_high_slider],
    )

    # Upload → show on canvas, store as original, clear prior result + toggle
    img_input.change(
        fn=on_image_upload, inputs=[img_input],
        outputs=[canvas_display, original_state, view_toggle, result_state],
    )

    # Generate → stream updates to canvas/status/sliders; reveal toggle at end
    generate_btn.click(
        fn=cartoonify,
        inputs=[
            story_input, img_input, mode_selector, wild_toggle,
            trigger_input, guidance_slider, steps_slider,
            cn_scale_slider, cn_end_slider, canny_low_slider, canny_high_slider,
            seed_input, prompt_input,
        ],
        outputs=[
            canvas_display, status_output, result_state,
            view_toggle,
            guidance_slider, steps_slider,
            cn_scale_slider, cn_end_slider,
            canny_low_slider, canny_high_slider,
        ],
        api_name='cartoonify',
    )

    # Toggle → swap canvas between stored original and result
    view_toggle.change(
        fn=toggle_view,
        inputs=[view_toggle, original_state, result_state],
        outputs=[canvas_display],
    )


if __name__ == '__main__':
    demo.launch(
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.zinc,
            font=gr.themes.GoogleFont('Inter'),
        ),
        show_error=True,
    )
