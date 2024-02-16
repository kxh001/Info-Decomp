from setuptools import setup

setup(
    name="infodp",
    author="Xianghao Kong",
    url="https://github.com/kxh001/Info-Decomp",
    author_email="xkong016@ucr.edu",
    description="Interpretable Diffusion via Information Decomposition",
    py_modules=["utils"],
    install_requires=[
        "tqdm>=4.66.1",
        "pandas>=2.0.0",
        "scipy>=1.10.1",
        "numpy>=1.25.2",
        "Pillow>=10.0.0",
        "seaborn>=0.12.0",
        "diffusers==0.20.0",
        "matplotlib==3.7.2",
        "pycocotools>=2.0.0",
        "setuptools>=65.5.0",
        "accelerate>=0.20.3",
        "torchvision>=0.14.0",
        "transformers>=4.33.1",
        "torchmetrics >= 1.1.0",
        "pytorch-lightning>=2.0.0",
        "spacy>=3.1.1",
        "nltk>=3.7",
        "easydict",
        "json",
        "random",
        "subprocess"
    ],
)
