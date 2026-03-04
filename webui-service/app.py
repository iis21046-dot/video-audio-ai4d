import os
import requests
import tempfile
import gradio as gr

TTS_API   = os.getenv("TTS_API",   "http://tts:5002")
VIDEO_API = os.getenv("VIDEO_API", "http://video:8000")

LANGUAGES = ["el", "en", "fr", "de", "es", "it", "pt", "pl", "tr", "ru", "zh-cn", "ja"]

# ── TTS ───────────────────────────────────────────────────
def tts_generate(text: str, language: str):
    if not text.strip():
        return None, "⚠️ Δεν έδωσες κείμενο."
    try:
        resp = requests.get(
            f"{TTS_API}/api/tts",
            params={"text": text, "language_id": language},
            timeout=60,
        )
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(resp.content)
        tmp.flush()
        return tmp.name, "✅ Ολοκληρώθηκε!"
    except Exception as e:
        return None, f"❌ Σφάλμα TTS: {e}"

# ── Video ─────────────────────────────────────────────────
def video_generate(prompt: str, num_frames: int, steps: int, seed: int):
    if not prompt.strip():
        return None, "⚠️ Δεν έδωσες prompt."
    try:
        payload = {
            "prompt": prompt,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "seed": seed if seed > 0 else None,
        }
        resp = requests.post(f"{VIDEO_API}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        video_resp = requests.get(f"{VIDEO_API}{data['download_url']}", timeout=60)
        video_resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_resp.content)
        tmp.flush()
        return tmp.name, f"✅ Έτοιμο! Job ID: {data['job_id']}"
    except Exception as e:
        return None, f"❌ Σφάλμα Video: {e}"

# ── UI ────────────────────────────────────────────────────
with gr.Blocks(title="AI Media Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️🎬 AI Media Studio\nText-to-Speech & Text-to-Video powered by local models")

    with gr.Tab("🎤 Text-to-Speech"):
        with gr.Row():
            with gr.Column():
                tts_text  = gr.Textbox(label="Κείμενο", lines=4,
                                       placeholder="Γράψε κείμενο εδώ…")
                tts_lang  = gr.Dropdown(LANGUAGES, value="el", label="Γλώσσα")
                tts_btn   = gr.Button("🔊 Δημιούργησε ήχο", variant="primary")
            with gr.Column():
                tts_audio  = gr.Audio(label="Αποτέλεσμα", type="filepath")
                tts_status = gr.Textbox(label="Status", interactive=False)

        tts_btn.click(tts_generate, [tts_text, tts_lang], [tts_audio, tts_status])

    with gr.Tab("🎬 Text-to-Video"):
        with gr.Row():
            with gr.Column():
                vid_prompt = gr.Textbox(label="Prompt", lines=3,
                                        placeholder="A astronaut riding a horse on Mars…")
                with gr.Row():
                    vid_frames = gr.Slider(16, 97, value=49, step=8, label="Frames")
                    vid_steps  = gr.Slider(10, 100, value=50, step=5,  label="Steps")
                    vid_seed   = gr.Number(value=0, label="Seed (0 = τυχαίο)")
                vid_btn = gr.Button("🎬 Δημιούργησε βίντεο", variant="primary")
            with gr.Column():
                vid_video  = gr.Video(label="Αποτέλεσμα")
                vid_status = gr.Textbox(label="Status", interactive=False)

        vid_btn.click(video_generate,
                      [vid_prompt, vid_frames, vid_steps, vid_seed],
                      [vid_video, vid_status])

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)