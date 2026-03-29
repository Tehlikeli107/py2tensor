from setuptools import setup

setup(
    name="py2tensor",
    version="0.14.0",
    description="Convert any Python function to GPU tensor operation. No training, exact results, 8000x speedup.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tehlikeli107",
    url="https://github.com/Tehlikeli107/py2tensor",
    py_modules=["py2tensor"],
    install_requires=["torch>=2.0"],
    extras_require={
        "numpy": ["numpy"],
        "pandas": ["pandas"],
    },
    python_requires=">=3.10",
    keywords="gpu cuda tensor pytorch accelerate jit compile parallel",
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
