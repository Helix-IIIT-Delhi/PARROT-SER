from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="parrot-ser",
    version="1.0.0",
    author="Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Jaya Sai Kiran Patibandla, Arun Balaji Buduru, Rajesh Sharma",
    description="PARROT: Heterogeneous PTM Fusion via Optimal Transport for Speech Emotion Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helix-IIIT-Delhi/PARROT-SER",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
