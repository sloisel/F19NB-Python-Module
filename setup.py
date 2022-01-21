import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "F19NB",
    version = "1.0.0",
    author = "Sébastien Loisel",
    description = ("Tools for the F19NB class at Heriot-Watt University"),
    license = "BSD",
    url = "http://example.com/helloworld",
    packages=['F19NB'],
    long_description=read('README.md'),
    install_requires=[
        'matplotlib','numpy','scipy'
    ]
)