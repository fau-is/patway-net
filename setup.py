
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xprem',
    version='1.0.0',
    description='Explainable Predictions for Medical Pathways',
    long_description=readme,
    author='xx',
    author_email='xx',
    url='https://github.com/fau-is/explainable-process-predictions-healthcare',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

