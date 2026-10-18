"""Compare steady end-to-end PPO/AMP throughput across environment counts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.warmup < args.iterations:
        parser.error("warmup must be positive and smaller than iterations")
    job = os.environ.get("SLURM_JOB_ID", "local")
    base = Path("runs") / f"scaling_{job}"
    base.mkdir(parents=True, exist_ok=True)
    results = []
    for count in args.envs:
        output = base / f"envs_{count}"
        gpu_log = base / f"gpu_{count}.csv"
        command = [sys.executable, "-m", "g1_amp.train", "--num-envs", str(count),
                   "--iterations", str(args.iterations), "--output", str(output),
                   "--video-every", "0", "--log-every", "1", "--checkpoint-every", str(args.iterations),
                   "--wandb-name", f"scaling-{job}-{count}-envs", "--wandb-group", f"scaling-{job}"]
        print(f"Benchmarking {count} environments", flush=True)
        with gpu_log.open("w") as gpu_stream, (base / f"train_{count}.log").open("w") as log:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", "--loop=1"], stdout=gpu_stream, stderr=subprocess.STDOUT)
            try:
                returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired:
                returncode = 124
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        if returncode:
            result = {"num_envs": count, "failed": True, "log": str(base / f"train_{count}.log")}
        else:
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            first, last = rows[args.warmup - 1], rows[-1]
            first_elapsed = first["env_steps"] / first["steps_per_second"]
            last_elapsed = last["env_steps"] / last["steps_per_second"]
            result = {"num_envs": count, "steady_steps_per_second":
                (last["env_steps"] - first["env_steps"]) / (last_elapsed - first_elapsed),
                "timed_iterations": len(rows) - args.warmup}
            samples = []
            for line in gpu_log.read_text().splitlines():
                try:
                    utilization, memory = map(float, line.split(","))
                    samples.append((utilization, memory))
                except ValueError:
                    continue
            if samples:
                result["peak_gpu_memory_mib"] = max(x[1] for x in samples)
                result["mean_gpu_utilization_including_startup"] = sum(x[0] for x in samples) / len(samples)
        results.append(result)
        (base / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    valid = [r for r in results if not r.get("failed")]
    if not valid:
        raise RuntimeError(f"All environment counts failed; see {base}")
    best = max(valid, key=lambda r: r["steady_steps_per_second"])
    (base / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(f"Best: {best}", flush=True)


if __name__ == "__main__":
    main()
