#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

#with open("README.rst") as readme_file:
#    readme = readme_file.read()

#with open("HISTORY.rst") as history_file:
#    history = history_file.read()

# WESTPA-2.0 with westpa.analysis is not yet in Conda
#requirements = ["westpa>=v2.0b5"]
requirements = [
    "scikit-learn>=0.24,<1.1",
    "scipy>=1.5",
    "numpy>=1.16.5",
    "mdtraj>=1.9",
    "ray>=1.0",
    "h5py>=3.1",
    "tqdm",
    "rich",
    "toml"
]

setup_requirements = [
    "pytest-runner",
]

# TODO: Can maybe remove this?
test_requirements = ["pytest>=3"]

EXTRAS_REQUIRE = {
    "tests": ["pytest>=3", "pytest-timeout", "mdtraj"],
}

setup(
    author="John Russo",
    author_email="russojd@ohsu.edu",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Code for Markov state modeling of weighted ensemble trajectories.",
    entry_points={"console_scripts": ["msm_we=msm_we.cli:main"]},
    install_requires=requirements,
    license="MIT license",
#    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="msm_we",
    name="msm_we",
    packages=find_packages(include=["msm_we", "msm_we.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=EXTRAS_REQUIRE,
    url="https://github.com/jdrusso/msm_we",
    version="0.1.25",
    zip_safe=False,
)
