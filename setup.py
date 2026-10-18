"""Installable Newton G1 locomotion example with Brax PPO and AMP."""
from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
requirements = [line.strip() for line in (HERE / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]

setup(
    name="newton-g1-amp",
    version="0.1.0",
    description="Unitree G1 locomotion with Newton, Brax PPO, and adversarial motion priors",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(include=["g1_amp", "g1_amp.*"]),
    install_requires=requirements,
    extras_require={"viewer": ["newton[examples]==1.7.0"]},
    entry_points={"console_scripts": [
        "g1-amp-prepare=g1_amp.prepare:main",
        "g1-amp-train=g1_amp.train:main",
        "g1-amp-play=g1_amp.play:main",
        "g1-amp-validate=g1_amp.validate:main",
        "g1-amp-diagnose=g1_amp.diagnose:main",
    ]},
)
