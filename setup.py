from setuptools import setup, find_packages

setup(
    name="py2tensor",
    version="2.0.0",
    description="Convert any Python function to GPU. No training, exact results, 3959x speedup.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tehlikeli107",
    url="https://github.com/Tehlikeli107/py2tensor",
    packages=find_packages(),
    install_requires=["torch>=2.0"],
    extras_require={
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "triton": ["triton>=3.0"],
    },
    python_requires=">=3.10",
    keywords="gpu cuda tensor pytorch accelerate parallel compiler",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
