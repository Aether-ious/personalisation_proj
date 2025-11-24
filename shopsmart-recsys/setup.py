from setuptools import setup, find_packages

setup(
    name="shopsmart_recsys",
    version="0.1.0",
    description="A Two-Tower Recommendation Engine with FastAPI serving",
    author="Aakash",
    packages=find_packages(),
    install_requires=[
        "torch",
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "faiss-cpu",
        "pyyaml"
    ],
    python_requires=">=3.9",
)