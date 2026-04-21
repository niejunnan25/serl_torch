from __future__ import annotations

from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent

WORKSPACE_PACKAGES = [
    "serl_torch",
    "serl_torch.examples",
    "serl_torch.examples.libero",
    "serl_torch.examples.libero.env",
    "serl_torch.examples.libero.runtime",
    "serl_torch.examples.libero.scripts",
    "serl_torch.examples.agibot_real",
    "serl_torch.examples.agibot_real.env",
    "serl_torch.examples.agibot_real.robot",
]


setup(
    name="serl_torch",
    version="0.0.0",
    description="Editable workspace package for serl_torch examples.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=WORKSPACE_PACKAGES,
    package_dir={"serl_torch": "."},
    include_package_data=True,
    zip_safe=False,
)
