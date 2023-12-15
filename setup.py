from setuptools import setup, find_packages

setup(
    name='medAI', 
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy", 
        "simple_parsing", 
        "pandas", 
        "scikit-learn",
        "scikit-image",
        "submitit", 
        "matplotlib",
        "einops", 
        "tqdm", 
        "coolname", 
        "wandb",
        "mat73"
    ], 
)