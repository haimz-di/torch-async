from setuptools import setup, find_packages

with open('requirements.txt') as r:
    requirements = r.read().splitlines()

setup(
    name='torch-async',
    version='0.1.dev',
    description='Torch Async is an asynchronous data loader and model trainer for PyTorch.',
    author='Guillaume Sicard et al.',
    author_email='guitch21@gmail.com',
    url='https://github.com/gsicard/torch-async',
    license='Apache-2.0',
    packages=find_packages(exclude=['tests*', 'examples*']),
    # TODO: write long description
    long_description='''
    ''',
    long_description_content_type='text/markdown',
    keywords=['deep learning', 'pytorch', 'AI'],
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
