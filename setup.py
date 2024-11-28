from setuptools import setup, find_packages

setup(
    name='lore_sd',
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
            'dwi2decomposition=lore_sd.dwi2decomposition:main',
            'angular_correlation=lore_sd.angular_correlation:main',
            'decomposition2contrast=lore_sd.decomposition2contrast:main',
        ],
    },
    author='Siebe Leysen',
    author_email='siebe.leysen@kuleuven.be',
    description='A package for DWI decomposition: LoRE-SD decomposes the dMRI data into voxel-level ODFs and response functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SiebeLeysen/LoRE_SD',
    python_requires='>=3.6',
)