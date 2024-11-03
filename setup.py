from setuptools import setup, find_packages

setup(
    name="image_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.3',
        'tensorflow>=2.8.0',
        'matplotlib>=3.4.3',
        'scikit-image>=0.18.3',
        'pytest>=6.2.5',
        'jupyter>=1.0.0',
        'tqdm>=4.4.0',
        'pillow>=8.3.2',
    ],
    author="Dhanraj Rateria",
    author_email="dhanrajrateria@gmail.com",
    description="Image processing project for denoising and sharpening techniques",
    keywords="image processing, denoising, sharpening, computer vision",
    python_requires=">=3.8",
)