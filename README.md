# summarizer

## Setup

### Clone this repository
```
$ git clone https://github.com/nisargjhaveri/summarizer
```

### Python dependencies
- numpy
- scipy
- sklearn
- nltk
- requests
- polyglot
- regex

Use your preferred method. You may use `pip` as followed.
```
$ pip install nltk numpy scipy sklearn requests regex
```

Also install nltk packages called `stopwords` and `punkt`.

### Setup CLUTO
http://glaros.dtc.umn.edu/gkhome/cluto/cluto/download

Set an environment variable `CLUTO_BIN_PATH` with the path of directory containing `vcluster` binary file.

### Setup ROUGE 1.5.5
This is required only if you plan to evaluate the summaries using ROUGE score. You may skip this.

Obtain and setup ROUGE 1.5.5 according to the instructions there.

Set an environment variable `ROUGE_HOME` with the path to ROUGE root directory, the one containing `ROUGE-1.5.5.pl` file.

### Setup Stanford CoreNLP
https://stanfordnlp.github.io/CoreNLP/index.html#download

Get and setup Stanford CoreNLP and set an environment variable `CORENLP_JAR` with the path to `stanford-corenlp-*.jar` file

## Use
Once the setup is done, you should be able to use it.
