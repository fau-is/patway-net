
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
    author=['Mathias Kraus', 'Sven Weinzierl'],
    author_email=['mathias.kraus@fau.de', 'sven.weinzierl@fau.de'],
    url='https://github.com/fau-is/explainable-process-predictions-healthcare',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

