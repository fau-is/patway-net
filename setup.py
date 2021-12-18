
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='patway-net',
    version='1.0.0',
    description='Explainable Predictions for Medical Pathways',
    long_description=readme,
    author=['Sandra Zilker'],
    author_email=['sandra.zilker@fau.de'],
    url='https://github.com/fau-is/xprem',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

