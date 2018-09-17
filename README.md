# Hidden Markov Model-based POS-tagger.

# Setup.

1. Install Python 3.6 or later as described in the [official guide](https://docs.python.org/3.6/using/index.html). You can check whether you have the correct version by typing `python3 -V` in your terminal.
2. Install [pip](https://pip.pypa.io/en/stable/installing/) for Python3. In ubuntu:
sudo aptitude install python3-pip
or
sudo apt-get install python3-pip

3. Install the following Python libraries using pip3:
    - pandas
    - nltk
    - numpy
    - scipy
    - sklearn

In Ubuntu:

```sh
pip3 install pandas
pip3 install nltk
pip3 install numpy
pip3 install scipy
pip3 install sklearn
```

Download the NLTK resources using python's console: (run python3 in a ubuntu terminal)

```python
import nltk #Fetch the base library

nltk.download('brown') # Download and install the assets
nltk.download('universal_tagset')

quit() # Exit console
```

To test if you have done everything properly, run the `HelloWorld.py` script:

```
python3 HelloWorld.py
```

# Running the included examples.
There is a Jupyter notebook with some examples and experiments named `Examples.py`. To run it, you need to have Jupyter notebook installed (check the guide at [Jupyter's documentation](https://jupyter.readthedocs.io/en/latest/install.html)).

If you are on Ubuntu, you can directly type `sudo apt-get install jupyter-notebook` (or `sudo aptitude install jupyter-notebook`) to download and set up Jupyter Notebook.

And the following Python libraries:

    - Matplotlib (`pip3 install matplotlib`)
    - Seaborn (`pip3 install seaborn`)

To run the notebook, navigate to the project's folder (`cd path/to/folder`) and use the command `jupyter notebook .` (if it doesn't work, try `jupyter-notebook .`). In a few seconds a browser window will open and you will be able to read and run the file `Examples.ipynb`.

# Quick start example.

First of all you need to import the model wrapper.

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

The NLTK library provides the corpora as a list of sentences, each represented as a list of words. Each word is a tuple (word, tag). For example:

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

Once the model is trained, you can start decoding text:

```python
decoder = model.predict(["I", "went", "to", "school", "yesterday", "."])
```

For more examples, please check the example Jupyter notebook!
