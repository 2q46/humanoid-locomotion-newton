"""Download pinned G1 assets and a small, real walking reference set."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

MENAGERIE_REV = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
MOTION_REV = "ce1572906efe6157840e8474d5a0d7aa87481e74"
ROOT = Path(os.environ.get("G1_AMP_WORKDIR", Path.cwd())).expanduser().resolve()
MODEL_DIR = ROOT / "assets/unitree_g1"
MOTION_DIR = ROOT / "data/lafan_g1"
CLIPS = ("walk1_subject1", "walk1_subject2", "walk2_subject1")


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "newton-g1-amp-example"})
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = response.read()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)
    print(f"Downloaded {path.relative_to(ROOT)}", flush=True)


def prepare_assets():
    base = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE_REV}/unitree_g1"
    for name in ("g1_mjx.xml", "LICENSE", "README.md"):
        download(f"{base}/{name}", MODEL_DIR / name)
    xml = ET.parse(MODEL_DIR / "g1_mjx.xml")
    files = sorted({m.attrib["file"] for m in xml.findall(".//mesh") if "file" in m.attrib})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda name: download(f"{base}/assets/{name}", MODEL_DIR / "assets" / name), files))
    prepare_newton_xml()


def prepare_newton_xml():
    # The upstream MJX model disables broadphase contacts in favor of scene-level
    # explicit pairs. Newton adds its own ground, so enable physical colliders here.
    xml = ET.parse(MODEL_DIR / "g1_mjx.xml")
    defaults = xml.getroot().find("default/default[@class='g1']")
    collision = defaults.find("default[@class='collision']/geom")
    collision.set("contype", "1")
    collision.set("conaffinity", "1")
    collision.set("condim", "3")
    collision.set("friction", "0.8 0.005 0.0001")
    visual = defaults.find("default[@class='visual']/geom")
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    xml.write(MODEL_DIR / "g1_newton.xml", encoding="unicode")


def prepare_motions(clips=CLIPS):
    from .motion import convert_csv
    base = f"https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/{MOTION_REV}"
    download(f"{base}/README.md", MOTION_DIR / "DATASET_CARD.md")
    manifest = {"dataset": "lvhaidong/LAFAN1_Retargeting_Dataset", "revision": MOTION_REV,
                "source_fps": 30, "target_fps": 50,
                "license": "CC-BY-NC-ND-4.0 (LAFAN1 motion content)", "clips": []}
    for clip in clips:
        src = MOTION_DIR / "raw" / f"{clip}.csv"
        url = f"{base}/g1/{clip}.csv"
        download(url, src)
        dst = MOTION_DIR / f"{clip}.npz"
        report = convert_csv(src, dst)
        manifest["clips"].append({"name": clip, "url": url, "sha256": hashlib.sha256(src.read_bytes()).hexdigest(), **report})
    (MOTION_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets-only", action="store_true")
    args = p.parse_args()
    prepare_assets()
    if not args.assets_only:
        prepare_motions()


if __name__ == "__main__":
    main()
