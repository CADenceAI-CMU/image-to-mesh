from setuptools import setup, find_packages
import os

# Core dependencies - pymeshlab removed (not on PyPI)
REQUIRED = [
    'tqdm>=4.66.3',
    'numpy',
    'ninja',
    'diffusers',
    'pybind11',
    'opencv-python',
    'einops',
    'transformers>=4.48.0',
    'omegaconf',
    'trimesh',
    'gradio',
    'torch>=2.0.0',
    'torchvision',
    'Pillow',
    'huggingface-hub',
]

setup(
    name="hy3dgen",
    version="2.0.2",
    author="Kanupriya Anand",
    author_email="kanand@andrew.cmu.edu",
    description="Hunyuan3D-2: Multiview to 3D generation",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/kanupriyaanand/multiview-3d-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    install_requires=REQUIRED,
    entry_points={
        'console_scripts': [
            'multiview-3d-gen=hy3dgen.cli:main',
        ],
    },
)