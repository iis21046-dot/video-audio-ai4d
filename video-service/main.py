import os
import uuid
import torch
import imageio
import numpy as np
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import CogVideoXPipeline

# ── Config ────────────────────────────────────────────────
MODEL_ID      = os.getenv("MODEL_ID",       "THUDM/CogVideoX-5b")
NUM_FRAMES    = int(os.getenv("NUM_FRAMES",  "49"))
GUIDANCE      = float(os.getenv("GUIDANCE_SCALE", "6"))
OUTPUT_DIR    = Path("/tmp/video_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model once at startup ────────────────────────────
print(f"[video-service] Loading {MODEL_ID} …")
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()   # saves VRAM when idle
print("[video-service] Model ready ✓")

# ── FastAPI ───────────────────────────────────────────────
app = FastAPI(title="Text-to-Video API", version="1.0")

class VideoRequest(BaseModel):
    prompt: str
    num_frames: int   = NUM_FRAMES
    guidance_scale: float = GUIDANCE
    num_inference_steps: int = 50
    seed: int | None  = None

class VideoResponse(BaseModel):
    job_id: str
    download_url: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/generate", response_model=VideoResponse)
def generate_video(req: VideoRequest):
    job_id = str(uuid.uuid4())
    out_path = OUTPUT_DIR / f"{job_id}.mp4"

    generator = None
    if req.seed is not None:
        generator = torch.Generator("cuda").manual_seed(req.seed)

    result = pipe(
        prompt=req.prompt,
        num_frames=req.num_frames,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        generator=generator,
    )

    frames = result.frames[0]  # list of PIL images
    frames_np = [np.array(f) for f in frames]
    imageio.mimwrite(str(out_path), frames_np, fps=8, quality=8)

    return VideoResponse(job_id=job_id, download_url=f"/video/{job_id}")

@app.get("/video/{job_id}")
def download_video(job_id: str):
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"{job_id}.mp4")