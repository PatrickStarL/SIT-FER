from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sit-fer",
    version="0.1.0",
    author="PatrickStarL",
    description="Integration of Semantic-, Instance-, Text-level Information for Semi-supervised Facial Expression Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PatrickStarL/SIT-FER",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.21.5",
        "opencv-python>=4.6.0",
        "scikit-image>=0.19.3",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "tensorboard>=2.11.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
    ],
)
