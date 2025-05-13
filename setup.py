from setuptools import setup, find_packages

setup(
    name="PAGEpy",
    version="1.0.4",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<1.26",
        "pandas>=1.3,<2.1",
        "scikit-learn>=1.0,<1.5",
        "tensorflow>=2.11,<2.16",
        "scanpy>=1.9.1,<1.10",
        "anndata>=0.8.0,<0.11",
        "matplotlib>=3.4,<3.8",
        "scipy>=1.7.0,<1.12",
        "seaborn>=0.11.0,<0.13"
    ],
)