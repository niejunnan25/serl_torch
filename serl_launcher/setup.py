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
        "torch>=2.0",
        "torchvision>=0.15",
        "agentlace @ git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",
    ],
    packages=find_packages(),
    zip_safe=False,
)
