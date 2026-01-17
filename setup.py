"""Setup script for Crisis Detector package."""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements (always installed)
core_requirements = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.4.0",
]

# Optional requirements for different domains
extras_require = {
    "finance": ["yfinance>=0.1.70", "statsmodels>=0.13.0"],
    "seismic": ["obspy>=1.3.0"],
    "gravitational": ["gwpy>=3.0.0"],
    "neuro": ["mne>=1.0.0"],
    "dev": [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "pytest-cov>=3.0.0",
    ],
}

# Add 'all' option to install everything
extras_require["all"] = [req for reqs in extras_require.values() for req in reqs]

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
    install_requires=core_requirements,
    extras_require=extras_require,
)
