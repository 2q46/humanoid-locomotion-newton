"""Record isolated Newton evaluation rollouts without touching training state."""
import json
import os
from pathlib import Path
import subprocess
import sys


def record_video(checkpoint, motions, output, iteration, *, device="cuda:0", steps=300,
                 fps=25, velocity=0.7, seed=123, timeout=180):
    directory = Path(output) / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"iteration-{iteration:08d}"
    video, metrics, log = stem.with_suffix(".mp4"), stem.with_suffix(".json"), stem.with_suffix(".log")
    command = [sys.executable, "-m", "g1_amp.play", "--checkpoint", str(Path(checkpoint).resolve()),
               "--motions", str(Path(motions).resolve()), "--device", device,
               "--steps", str(steps), "--video-fps", str(fps), "--velocity", str(velocity),
               "--seed", str(seed), "--video", str(video.resolve()), "--metrics-json", str(metrics.resolve())]
    # A fresh process isolates GL/EGL and evaluation physics from the learner state.
    environment = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1"}
    with log.open("w") as stream:
        try:
            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Evaluation video failed; see {log}") from error
    if not video.exists() or video.stat().st_size == 0 or not metrics.exists():
        raise RuntimeError(f"Evaluation did not produce video/metrics; see {log}")
    return video, json.loads(metrics.read_text())
