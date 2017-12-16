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

Use your preferred method. You may use `pip` as followed.
```
$ pip install nltk numpy scipy sklearn requests
```

Also install nltk packages called `stopwords` and `punkt`.

### Setup CLUTO
http://glaros.dtc.umn.edu/gkhome/cluto/cluto/download

Set an environment variable `CLUTO_BIN_PATH` with the path of directory containing `vcluster` binary file.

### Setup ROUGE 1.5.5
This is required only if you plan to evaluate the summaries using ROUGE score. You may skip this.

Obtain and setup ROUGE 1.5.5 according to the instructions there.

Set an environment variable `ROUGE_HOME` with the path to ROUGE root directory, the one containing `ROUGE-1.5.5.pl` file.

## Use
Once the setup is done, you should be able to use it.
