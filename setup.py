from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="MEA_Absorption_Column",  # Replace with your project's name
    version="0.5.0",
    author="Tanner Polley",
    author_email="tannerwpolley@gmail.com",
    description="A Python based MEA Absorption Column model using a sequential shooting method",
    long_description=long_description,
    url="https://github.com/tannerpolley/Absorption_Column.git",  # Add project URL if any
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Engineers",
        "Development Status :: 3 - Beta",
    ],
    python_requires='>=3.9',
)