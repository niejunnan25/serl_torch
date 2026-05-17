from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.3",
    description="library for rl experiments",
    url="https://github.com/rail-berkeley/serl",
    author="auth",
    license="MIT",
    install_requires=[
        "typing",
        "typing_extensions",
        "hydra-core>=1.3.2",
        "numpy>=1.24.3,<1.27",
        "torch>=2.0",
        "torchvision>=0.15",
        "agentlace @ git+https://github.com/niejunnan25/agentlace.git@885f5fc",
    ],
    packages=find_packages(),
    zip_safe=False,
)
