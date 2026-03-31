from setuptools import setup, find_packages

setup(
    name="oulad-dropout-prediction",
    version="1.0.0",
    description=(
        "Early dropout detection in online higher education using "
        "hybrid tabular ML and NLP on the Open University Learning "
        "Analytics Dataset (OULAD)."
    ),
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/oulad-dropout-prediction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "shap>=0.42.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
