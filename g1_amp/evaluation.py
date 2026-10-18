"""Record isolated batched Newton rollouts without touching learner state."""
import json
import os
from pathlib import Path
import subprocess
import sys


def record_videos(checkpoint, motions, output, iteration, *, device="cuda:0", steps=300,
                  fps=25, velocity=0.7, seed=123, num_envs=32, timeout=600):
    directory = Path(output) / "videos" / f"iteration-{iteration:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    metrics, log = directory / "metrics.json", directory / "evaluation.log"
    command = [sys.executable, "-m", "g1_amp.video_rollouts", "--checkpoint", str(Path(checkpoint).resolve()),
               "--motions", str(Path(motions).resolve()), "--device", device,
               "--steps", str(steps), "--fps", str(fps), "--velocity", str(velocity),
               "--seed", str(seed), "--num-envs", str(num_envs), "--output-dir", str(directory.resolve())]
    environment = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1"}
    with log.open("w") as stream:
        try:
            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Evaluation videos failed; see {log}") from error
    report = json.loads(metrics.read_text())
    if report["video_count"] != num_envs or len(report["rollouts"]) != num_envs:
        raise RuntimeError(f"Wrong number of evaluation videos; see {log}")
    for row in report["rollouts"]:
        video = directory / row["video"]
        if not video.exists() or video.stat().st_size == 0:
            raise RuntimeError(f"Missing evaluation video {video}; see {log}")
    return directory, report
