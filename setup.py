
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='patway-net',
    version='1.0.0',
    description='Interpretable Predictions for Patient Pathways',
    long_description=readme,
    author="Sandra Zilker, Sven Weinzierl, Mathias Kraus, Patrick Zschech, Martin Matzner",
    author_email="sandra.zilker@fau.de, sven.weinzierl@fau.de, mathias.kraus@fau.de, patrick.zschech@fau.de, martin.matzner@fau.de",
    url='https://github.com/fau-is/patway-net',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

