"""
Cartoonify — UI preview (no GPU, no FLUX, no Gemini)
Run:  python cartoonify_ui_preview.py

Only requires: gradio, Pillow
"""

import time
import random
import gradio as gr
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_TRIGGER       = 'gdo_cartoon'
DEFAULT_PROMPT        = 'satirical cartoon illustration, bold outlines, vivid flat colours'
DEFAULT_MODE          = 'Reimagine'
DEFAULT_STEPS         = 28
DEFAULT_SEED          = 42

DEFAULTS = {
    'reimagine': {'guidance': 2.5, 'cn_scale': 0.7, 'cn_end': 0.8, 'canny_low': 50,  'canny_high': 200},
    'scene':     {'guidance': 3.5, 'cn_scale': 0.8, 'cn_end': 0.8, 'canny_low': 50,  'canny_high': 200},
    'portrait':  {'guidance': 3.5, 'cn_scale': 0.7, 'cn_end': 0.8, 'canny_low': 50,  'canny_high': 200},
}

MODE_DESCRIPTIONS = {
    'Reimagine': 'FLUX recomposes the full image freely — best for creative reinterpretation where scene and staging can change.',
    'Scene':     'Depth map locks spatial layout and figure positions — best for crowds and architecture where structure matters.',
    'Portrait':  'Canny edges preserve face outlines — best when a specific person must be immediately recognisable.',
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
    """Return an HTML status line. state: idle | processing | done | warn"""
    colors = {
        'idle':       '#3f3f46',
        'processing': '#a78bfa',
        'done':       '#34d399',
        'warn':       '#fbbf24',
    }
    dot_color = colors.get(state, colors['processing'])
    anim = ' animation:dot-pulse 1.2s ease-in-out infinite;' if state == 'processing' else ''
    return (
        f'<div class="cfy-status">'
        f'<div class="cfy-dot" style="background:{dot_color};{anim}"></div>'
        f'<span>{msg}</span>'
        f'</div>'
    )

def _progress(step: int, total: int) -> str:
    pct  = int(100 * step / max(total, 1))
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

_STATUS_IDLE = _status('Click Cartoonify to begin.', 'idle')


# ── Mock cartoonify generator ─────────────────────────────────────────────────
def cartoonify(
    story, image, mode_label, wild_mode,
    trigger_word, guidance_scale, num_steps,
    cn_scale, cn_end, canny_low, canny_high,
    seed, prompt_override,
):
    """Generator — yields (result_image, status_html, g, s, cn_s, cn_e, clow, chigh)."""
    if image is None:
        raise gr.Error('Upload a photo first.')

    mode = mode_label.lower()

    def emit(image_val=None, status=None, show_result=None,
             g=None, s=None, cn_s=None, cn_e=None, clow=None, chigh=None):
        if show_result is False:
            res = gr.update(visible=False)
        elif image_val is not None:
            res = gr.update(value=image_val, visible=True)
        else:
            res = gr.update()
        return (
            res,
            gr.update() if status is None else gr.update(value=status),
            gr.update() if g    is None else gr.update(value=g),
            gr.update() if s    is None else gr.update(value=s),
            gr.update() if cn_s is None else gr.update(value=cn_s),
            gr.update() if cn_e is None else gr.update(value=cn_e),
            gr.update() if clow is None else gr.update(value=clow),
            gr.update() if chigh is None else gr.update(value=chigh),
        )

    # Working param values (may be overridden by Wild)
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
    yield emit(status=_progress(0, s_val), show_result=False)
    for step in range(1, s_val + 1):
        time.sleep(0.06)
        yield emit(status=_progress(step, s_val))

    # ── Step 4: Result ────────────────────────────────────────────────────────
    result = _make_placeholder(mode, story)
    yield emit(image_val=result, status=_status('✓ Done — mock output (no GPU)', 'done'))


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
/* ── Reset & tokens ─────────────────────────────────────── */
:root {
    --bg:           #0d0d0f;
    --surface:      #17171a;
    --surface-2:    #1e1e22;
    --border:       #2a2a30;
    --accent:       #a78bfa;
    --accent-dim:   rgba(167,139,250,0.12);
    --text:         #f4f4f5;
    --text-muted:   #71717a;
    --success:      #34d399;
    --amber:        #fbbf24;
    --amber-dim:    rgba(251,191,36,0.10);
    --radius:       14px;
}

/* ── Page ────────────────────────────────────────────────── */
.gradio-container {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 1.25rem 1.5rem 2rem !important;
    color: var(--text) !important;
}
.gradio-container * { box-sizing: border-box; }

/* override Gradio Soft/Base surface colours */
.dark, body, .main, .wrap, .contain, .block {
    background: transparent !important;
}

/* ── Header ──────────────────────────────────────────────── */
#cfy-header {
    padding: 2.25rem 2rem 1.75rem;
    text-align: center;
    margin-bottom: 0;
}
.cfy-title {
    display: block;
    font-size: 2.75rem; font-weight: 900; letter-spacing: -2.5px;
    color: var(--text); line-height: 1;
    margin-bottom: 0.5rem;
}
.cfy-subtitle {
    display: block;
    font-size: 0.92rem; color: var(--text-muted);
    font-weight: 400; line-height: 1.6;
}
.cfy-badge {
    display: inline-block;
    background: var(--amber-dim); color: var(--amber);
    border: 1px solid rgba(251,191,36,0.25);
    font-size: 0.65rem; font-weight: 800; padding: 0.18rem 0.55rem;
    border-radius: 20px; letter-spacing: 0.1em;
    text-transform: uppercase; margin-top: 0.75rem;
}

/* ── Mode nav tab row ─────────────────────────────────────── */
#cfy-mode-nav-row {
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
    padding: 0 0.5rem;
}
#cfy-mode-nav { background: transparent !important; border: none !important; padding: 0 !important; }
#cfy-mode-nav .wrap { display: flex !important; gap: 0 !important; }
#cfy-mode-nav label {
    flex: 0 1 auto !important;
    padding: 0.65rem 1.5rem !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    cursor: pointer !important;
    transition: color 0.15s, border-color 0.15s !important;
    margin-bottom: -1px !important;
    letter-spacing: 0.01em !important;
}
#cfy-mode-nav label:has(input:checked) {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
#cfy-mode-nav label:hover:not(:has(input:checked)) {
    color: var(--text) !important;
}
#cfy-mode-nav input[type="radio"] { display: none !important; }

/* ── Cards ───────────────────────────────────────────────── */
#cfy-left, #cfy-right {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
}

/* ── Section labels ──────────────────────────────────────── */
.cfy-label {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.68rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.4rem; margin-top: 1rem;
}
.cfy-label:first-child { margin-top: 0; }
.cfy-step {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent-dim); color: var(--accent);
    font-size: 0.6rem; font-weight: 900; flex-shrink: 0;
}

/* ── Inputs ──────────────────────────────────────────────── */
.gradio-container textarea, .gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 0.875rem !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: var(--text-muted) !important; }
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ── Upload zone ─────────────────────────────────────────── */
.cfy-upload { border: 2px dashed var(--border) !important; border-radius: 12px !important; background: var(--surface-2) !important; transition: border-color 0.2s !important; }
.cfy-upload:hover { border-color: var(--accent) !important; }

/* ── Mode description ────────────────────────────────────── */
#cfy-mode-desc p {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    line-height: 1.55 !important;
    margin: 0 !important;
    font-style: italic !important;
}

/* ── Wild toggle (checkbox styled as pill) ───────────────── */
#cfy-wild { padding: 0.6rem 0.75rem !important; border-radius: 10px !important; background: var(--surface-2) !important; border: 1px solid var(--border) !important; transition: background 0.2s, border-color 0.2s !important; }
#cfy-wild:has(input:checked) { background: var(--amber-dim) !important; border-color: rgba(251,191,36,0.3) !important; }
#cfy-wild label { color: var(--text) !important; font-weight: 700 !important; font-size: 0.82rem !important; cursor: pointer !important; gap: 0.6rem !important; }
#cfy-wild input[type="checkbox"] {
    appearance: none !important; -webkit-appearance: none !important;
    width: 2.25rem !important; height: 1.125rem !important;
    background: var(--border) !important; border-radius: 999px !important;
    position: relative !important; cursor: pointer !important;
    transition: background 0.2s !important; flex-shrink: 0 !important;
}
#cfy-wild input[type="checkbox"]:checked { background: var(--amber) !important; }
#cfy-wild input[type="checkbox"]::after {
    content: '' !important; position: absolute !important;
    top: 2px !important; left: 2px !important;
    width: 14px !important; height: 14px !important;
    background: white !important; border-radius: 50% !important;
    transition: transform 0.2s !important;
}
#cfy-wild input[type="checkbox"]:checked::after { transform: translateX(18px) !important; }

/* ── Fine-tune accordion ─────────────────────────────────── */
.gradio-container details { background: transparent !important; border: 1px solid var(--border) !important; border-radius: 10px !important; margin-top: 0.75rem !important; }
.gradio-container details summary { color: var(--text-muted) !important; font-size: 0.8rem !important; font-weight: 600 !important; padding: 0.6rem 0.75rem !important; }
.gradio-container details summary:hover { color: var(--text) !important; }
.gradio-container details[open] summary { color: var(--text) !important; border-bottom: 1px solid var(--border) !important; }
.gradio-container details .block { padding: 0.75rem !important; }

/* Sliders */
.gradio-container input[type="range"] { accent-color: var(--accent) !important; }
.gradio-container label span { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* ── Status line ─────────────────────────────────────────── */
.cfy-status {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.4rem 0; min-height: 2rem;
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
    font-size: 0.78rem; color: var(--text-muted);
}
.cfy-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.cfy-progress-wrap { flex: 1; display: flex; flex-direction: column; gap: 4px; }
.cfy-progress-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-muted); }
.cfy-progress-track { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.cfy-progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.08s linear; }
@keyframes dot-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.7); }
}

/* ── Result image ────────────────────────────────────────── */
#cfy-result { border-radius: 12px !important; overflow: hidden !important; }
#cfy-result img { border-radius: 10px !important; }

/* ── Generate button ─────────────────────────────────────── */
#cfy-btn button {
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important; font-weight: 700 !important;
    height: 52px !important; width: 100% !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
    transition: opacity 0.15s, transform 0.12s !important;
}
#cfy-btn button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#cfy-btn button:active { transform: translateY(0) !important; }
#cfy-btn button:disabled { opacity: 0.45 !important; cursor: wait !important; }

footer { display: none !important; }
'''


# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title='Cartoonify') as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML('''
        <div id="cfy-header">
            <span class="cfy-title">&#127912; Cartoonify</span>
            <span class="cfy-subtitle">Voice up. Turn your story into a satirical illustration.</span>
            <div><span class="cfy-badge">UI Preview — no GPU</span></div>
        </div>
    ''')

    # ── Mode nav (tab row above both columns) ────────────────────────────────
    with gr.Row(elem_id='cfy-mode-nav-row'):
        mode_selector = gr.Radio(
            choices=['Reimagine', 'Scene', 'Portrait'],
            value=DEFAULT_MODE,
            label='',
            elem_id='cfy-mode-nav',
        )

    # ── Main two-column layout ───────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left column — Inputs ─────────────────────────────────────────────
        with gr.Column(scale=5, min_width=340, elem_id='cfy-left'):

            gr.HTML('<div class="cfy-label"><span class="cfy-step">1</span>What\'s the story?</div>')
            story_input = gr.Textbox(
                label='', lines=5, max_lines=10,
                placeholder=(
                    'A politician hands out empty promises while people queue for food…\n'
                    'A general behind a desk of medals while tiny soldiers march across a map below…'
                ),
            )

            gr.HTML('<div class="cfy-label"><span class="cfy-step">2</span>Upload a photo</div>')
            img_input = gr.Image(
                label='', type='numpy', height=210,
                elem_classes=['cfy-upload'],
                sources=['upload', 'clipboard'],
            )

            # Mode description (updates on tab switch)
            mode_desc = gr.Markdown(
                value=MODE_DESCRIPTIONS[DEFAULT_MODE],
                elem_id='cfy-mode-desc',
            )

            gr.HTML('<div class="cfy-label" style="margin-top:1.25rem">Settings</div>')

            # Wild mode toggle
            wild_toggle = gr.Checkbox(
                value=False,
                label='⚡ Wild — Gemini tunes parameters for maximum satirical impact',
                elem_id='cfy-wild',
            )

            # Fine-tune accordion (conditional sliders per mode)
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

        # ── Right column — Output ─────────────────────────────────────────────
        with gr.Column(scale=6, min_width=480, elem_id='cfy-right'):

            result_output = gr.Image(
                label='', type='pil', height=520,
                interactive=False, elem_id='cfy-result',
            )

            status_output = gr.HTML(value=_STATUS_IDLE)

            generate_btn = gr.Button(
                '\U0001f3a8  Cartoonify  →',
                variant='primary',
                elem_id='cfy-btn',
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    mode_selector.change(
        fn=update_mode,
        inputs=[mode_selector],
        outputs=[mode_desc, guidance_slider, cn_scale_slider,
                 cn_end_slider, canny_low_slider, canny_high_slider],
    )

    generate_btn.click(
        fn=cartoonify,
        inputs=[
            story_input, img_input, mode_selector, wild_toggle,
            trigger_input, guidance_slider, steps_slider,
            cn_scale_slider, cn_end_slider, canny_low_slider, canny_high_slider,
            seed_input, prompt_input,
        ],
        outputs=[
            result_output, status_output,
            guidance_slider, steps_slider,
            cn_scale_slider, cn_end_slider,
            canny_low_slider, canny_high_slider,
        ],
        api_name='cartoonify',
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
