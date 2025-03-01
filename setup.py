from setuptools import setup, find_packages

setup(
    name="quietalpha",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tensorflow",
        "keras",
        "xgboost",
        "ib_insync",
        "ta",
        "joblib",
    ],
    author="QuietAlpha Team",
    author_email="your.email@example.com",
    description="A low-risk trading bot using AI for portfolio management and trading decisions",
)