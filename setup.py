from setuptools import setup, find_packages

setup(
    name="MEA-Absorption-Column",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "scipy>=1.11",
        "matplotlib>=3.8",
        "openpyxl>=3.1",
    ],
    include_package_data=True,
    package_data={
        # include all CSVs in your data/ folder
        "mea_absorption_column": [
            "data/*.csv",
            "data/epcsaft_neutral/*.json",
            "data/epcsaft_datasets/*/*.json",
            "data/epcsaft_datasets/*/pure/*.csv",
            "data/epcsaft_datasets/*/mixed/binary_interaction/*.csv",
            "data/epcsaft_datasets/*/mixed/rel_perm/*.csv",
        ],
    },
)
