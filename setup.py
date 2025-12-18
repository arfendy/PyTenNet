"""Setup script for PyTenNet."""
from setuptools import setup, find_packages

setup(
    name="pytennet",
    version="0.2.0",
    packages=find_packages(include=["tensornet", "tensornet.*"]),
    install_requires=["torch>=2.0.0"],
    python_requires=">=3.9",
)
