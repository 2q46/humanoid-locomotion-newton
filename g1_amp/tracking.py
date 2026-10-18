"""W&B tracking with persisted run identity and versioned learner checkpoints."""
import json
from pathlib import Path
import wandb


class RunTracker:
    def __init__(self, args, output, config):
        self.output = Path(output)
        identity_path = self.output / "wandb_run.json"
        previous_path = args.resume.parent / "wandb_run.json" if args.resume else identity_path
        previous = json.loads(previous_path.read_text()) if args.resume and previous_path.exists() else {}
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity or previous.get("entity"),
            name=args.wandb_name or previous.get("name") or self.output.name,
            group=args.wandb_group, job_type="train", mode=args.wandb_mode,
            id=previous.get("id"),
            resume="allow" if previous and args.wandb_mode == "online" else None,
            dir=str(self.output), config=config,
            settings=wandb.Settings(init_timeout=120),
        )
        self.run.define_metric("env_steps")
        self.run.define_metric("*", step_metric="env_steps")
        self.run.config.update(config, allow_val_change=True)
        identity_path.write_text(json.dumps({"id": self.run.id, "entity": self.run.entity,
            "project": self.run.project, "name": self.run.name, "url": self.run.url,
            "mode": args.wandb_mode}, indent=2) + "\n")
        print(f"W&B {args.wandb_mode}: {self.run.url or self.run.id}", flush=True)

    def log(self, report):
        # Use environment transitions as the chart axis while W&B owns its append-only step.
        self.run.log({key: value for key, value in report.items() if value is not None})

    def checkpoint(self, path, iteration, env_steps):
        artifact = wandb.Artifact(f"g1-amp-{self.run.id}", type="model",
                                  metadata={"iteration": iteration, "env_steps": env_steps})
        artifact.add_file(str(path), name="checkpoint.msgpack")
        artifact.add_file(str(self.output / "config.json"), name="config.json")
        artifact.add_file(str(self.output / "checkpoint.json"), name="checkpoint.json")
        self.run.log_artifact(artifact, aliases=["latest", f"iteration-{iteration}"])

    def video(self, path, metrics, iteration, env_steps, fps):
        values = {"env_steps": env_steps, "iteration": iteration,
                  "eval/video": wandb.Video(str(path), format="mp4",
                       caption=f"Update {iteration}; standing start; command {metrics['command_velocity']} m/s; seed {metrics['seed']}; {fps} fps; resets after falls"),
                  "eval/task_reward": metrics["mean_task_reward"], "eval/falls": metrics["falls"],
                  "eval/steps": metrics["steps"]}
        lengths = metrics["completed_episode_lengths"]
        if lengths:
            values["eval/episode_length"] = sum(lengths) / len(lengths)
        self.run.log(values)

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)
