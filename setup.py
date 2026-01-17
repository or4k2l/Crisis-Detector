"""Setup script for Crisis Detector package."""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crisis-detector",
    version="1.0.0",
    author="or4k2l",
    description="A unified framework for detecting anomalies and crises across multiple domains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/or4k2l/Crisis-Detector",
    py_modules=["crisis_detector"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)
