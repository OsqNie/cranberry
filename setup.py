
from setuptools import setup, find_packages

setup(
    name='cranberry',
    packages=find_packages(),
    version='0.1.0',
    description='The Cranberry Deep Learning library',
    author='Oskar Niemenoja',
    license='MIT',
    install_requires=['numpy', 'matplotlib']
)