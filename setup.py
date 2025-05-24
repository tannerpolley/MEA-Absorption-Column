from setuptools import setup, find_packages

setup(
    name="MEA-Absorption-Column",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        # …any other runtime deps…
    ],
    include_package_data=True,
    package_data={
        # include all CSVs in your data/ folder
        "mea_absorption_column": ["data/*.csv"],
    },
)
