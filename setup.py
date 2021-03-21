from setuptools import setup
from os import path

VERSION = "0.9.1"

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="dl4dp",
    packages=["dl4dp"],
    version=VERSION,
    license="MIT",
    description="A Python library for morphological tagging, lemmatization and dependency parsing",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=u"Peter Bednár",
    author_email="peter.bednar@tuke.sk",
    url="https://github.com/peterbednar/dl4dp",
    install_requires=["numpy", "torch", "pyyaml", "requests", "conllutils"],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Text Processing :: Linguistic',
    ]
)
