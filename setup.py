import setuptools

setuptools.setup(
    name="dipr",
    version="0.31",
    author='Arcturus Industries',
    description="Deep Inertial Prediction Challenge Toolkit",
    long_description='',
    packages=setuptools.find_namespace_packages("dipr.*"),
    python_requires=">=3.7",
    ext_modules=[],
    install_requires=[
        'numpy >=1.16.4, <1.22',
        'tqdm',
        'h5py',
        'matplotlib',
        'opencv-python',
        'numpy-quaternion',
        'scipy',
        'termcolor',
        'numba',
        'seaborn'
    ],
    scripts=[],
)
