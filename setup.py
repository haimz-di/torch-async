from setuptools import setup, find_packages

with open('requirements.txt') as r:
    requirements = r.read().splitlines()

setup(
    name='torch-async',
    version='0.1.alpha',
    description='Torch Async is an asynchronous data loader and model trainer for PyTorch.',
    author='Guillaume Sicard et al.',
    author_email='guitch21@gmail.com',
    url='https://github.com/gsicard/torch-async',
    license='Apache-2.0',
    packages=find_packages(exclude=['tests*', 'examples*']),
    long_description='''
    Torch Async performs asynchronous data loading and model training for PyTorch.
    It was design to overcome the limitations of the sequential nature of PyTorch standard training loop by removing locks in the data loading and model training process.
    There are two classes to be subclassed:
    - the `ChunkDataloader` class: loads the data by chunks to be processed by the model's logic
    - the `Model` class: derived from PyTorch's `Module` class and provides a `fit` method
    ''',
    long_description_content_type='text/markdown',
    keywords=['data science', 'AI', 'machine learning', 'deep learning', 'pytorch'],
    python_requires='>=3.7',
    install_requires=requirements,
    project_urls={
        'Source': 'https://github.com/gsicard/torch-async',
    },
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
