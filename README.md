# Hidden Markov Model-based POS-tagger.

# Setup.

1. Install Python 3.6 or later. If you are not sure about the version you have installed, type `python -V` or `python3 -V` in your terminal.
2. 

# Running the included examples.
There is a 

# Quick start example.

First of all you need to import the model wrapper that includes everything you need for training and decoding.

```python
from HMM.Markovify import *
```

For this guide we will use a corpora from the [NLTK library](https://github.com/nltk/nltk). If you want to use another corpora make sure it is in the correct format. The standard format for a text is having every word followed by '/TAG', for example:

```I/pronoun am/verb a/preposition correctly/adjective tagged/verb text/noun ./.```

The model is tag-agnostic, so you can use any tagset. If you are not sure about what tagset to use, it is recommended to use the universal tagset.

```python
from nltk.corpus import brown

corpus = brown.tagged_sents(categories='news', tagset='universal')
```

The NLTK library provides the corpora as a list of sentences, each sentence represented by a list of words. Each word is a tuple (word, tag). For example:

```python
[[("Sentence", "NOUN"), ("1", "Number")],
 [("Sentence", "NOUN"), ("2", "Number")],
 ...
]
```

To create the model, first we call the HMM constructor with the tunning parameters:

```python
model = Markovify(smoothing='laplace', alpha=1)
```

For this example we will use a default Hidden Markov Model with Laplace (or add-one) smoothing.

To train the model you just have to call the function `fit` with the corpus. There are multiple ways of specifying the training corpus, check the documentation to see what fits your needs better.

```python
model = model.fit(sentences=corpus)
```

Once the model is trained, we create a decoder:

```python
decoder = model.predict(["I", "went", "to", "school", "yesterday", "."])
```

Do you want more examples? Please check the included Jupyter notebook!
