"""Setup script for DJI Tello Collision Avoidance with PyDNet."""

import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, ".github", "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dji-tello-collision-avoidance",
    version="2.0.0",
    author="dronefreak",
    author_email="",
    description="Real-time monocular depth estimation for DJI Tello drone collision avoidance using PyDNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "tello": ["djitellopy>=2.4.0"],
        "gpu": ["tensorflow[and-cuda]>=2.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "ml": ["keras>=3.0.0"],  # Alternative ML backend
    },
    entry_points={
        "console_scripts": [
            "tello-depth-demo=tello_demo:main",
            "webcam-depth-demo=webcam_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="drone tello depth-estimation computer-vision pytorch collision-avoidance autonomous-navigation",
    project_urls={
        "Bug Reports": "https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet/issues",
        "Source": "https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet",
    },
)
