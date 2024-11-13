from setuptools import setup, find_packages

setup(
    name='LoRE-SD',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'dwi2decomposition=dwi2decomposition:main',
            'angular_correlation=scripts.angular_correlation:main',
            'decomposition2contrast=scripts.decomposition2contrast:main',
        ],
    },
    author='Siebe Leysen',
    author_email='siebeleysen@hotmail.com',
    description='A package for DWI decomposition: LoRE-SD decomposes the dMRI data into voxel-level ODFs and response functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SiebeLeysen/LoRE_SD',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)