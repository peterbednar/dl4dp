# dl4dp

**dl4dp** is a Python NLP library, which provides tools for morphological tagging, lemmatization and dependency parsing.
The main motivation for this library is to provide state-of-the-art tools for Slovak language, but the models can be build
for any language with training data in [Universal Dependencies](https://universaldependencies.org).

### Installation

The library supports Python 3.6 and later.

#### pip

The dl4dp is available on [PyPi](https://pypi.python.org/pypi) and can be installed via `pip`. To install simply
run:
```
pip install dl4dp
```

To upgrade the previous installation to the newest release, use:
```
pip install dl4dp -U
```

#### From source

Alternatively, you can also install library from this git repository, which will give you more flexibility and allows
you to start contributing to the dl4dp code. For this option, run:
```
git clone https://github.com/peterbednar/dl4dp.git
cd dl4dp
pip install -e .
```

### Getting started with dl4dp

The library provides a command-line interface which allows you to train own model, create pipeline package for production deployment, or parse data. The following command will download [Universal Dependencies](https://universaldependencies.org)  archive and train morphological tagger model on English EWT treebank:

```
python -m dl4dp train tagger -t en_ewt
```

Similarly, the following command will train a model for dependency parser:

```
python -m dl4dp train parser -t en_ewt
```

Subsequently, you can create and install a pipeline package:

```
python -m dl4dp package install -t en_ewt
```

After the installation of the pipeline, it can be used for the parsing of input data in CoNLL-U format:

```
python -m dl4dp parse ~/.dl4dp/treebanks/en_ewt/en_ewt-ud-test.conllu output.conllu -m en_ewt-0.1.0
```

### LICENSE

dl4dp is released under the MIT License. See the [LICENSE](https://github.com/peterbednar/dl4dp/blob/master/LICENSE)
file for more details.
