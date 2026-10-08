from setuptools import setup, find_packages

setup(
    name="tokencraft",
    version="1.0.0",
    description="Build, compare, and visualize tokenizers (BPE, WordPiece, Character) from scratch",
    author="Sujan Venkat",
    author_email="vsdharan@gmail.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "tokencraft=tokencraft.cli:main",
        ],
    },
    python_requires=">=3.8",
)
