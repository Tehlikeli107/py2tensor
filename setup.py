from setuptools import setup

setup(
    name="py2tensor",
    version="0.7.0",
    description="Convert any Python function to GPU tensor operation. No training, exact results.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tehlikeli107",
    url="https://github.com/Tehlikeli107/py2tensor",
    py_modules=["py2tensor"],
    install_requires=["torch>=2.0"],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
    ],
)
