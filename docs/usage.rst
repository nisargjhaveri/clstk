Usage
=====

Summarize
---------
``sum.py`` is used to summarize a document set. It allows selecting specific CLS method and parameters for that particular method.

.. code-block:: console

  $ python sum.py --help
  usage: sum.py [-h] {linBilmes,coRank,simFusion} ...

  Automatically summarize a set of documents

  optional arguments:
    -h, --help            show this help message and exit

  methods:
    Summarization method

    {linBilmes,coRank,simFusion}

The following command shows help for selected method.

.. code-block:: console

  $ python sum.py {method} --help

Following is the common pattern to run a CLS method on one document set.

.. code-block:: console

  $ python sum.py {method} [options] {source_directory}

All files stored in the directory ``source_directory`` are read and treated as a part of document set to summarize.
The files are expected to be plain text files.

Required arguments
^^^^^^^^^^^^^^^^^^^
  :source_directory:      Directory containing a set of files to be summarized.

Common options
^^^^^^^^^^^^^^
Here is a list of common optional arguments across all CLS methods.

  -h, --help            show this help message and exit
  -v, --verbose         Show verbose information messages
  --no-colors           Don't show colors in verbose log
  -s N, --size N        Maximum size of the summary
  -w, --words           Caluated size as number of words instead of characters
  --source-lang lang    Two-letter language code of the source documents
                        language. Defaults to `en`
  -l lang, --target-lang lang
                        Two-letter language code to generate cross-lingual
                        summary. Defaults to source language.

Evaluate
--------
Another script called ``evaluate.py`` is used to run and evaluate CLS methods over a CLS evaluation dataset.

Similar to ``sum.py``, this script also needs the CLS method as first argument and other argument follows depending on the selected method.

.. code-block:: console

  $ python evaluate.py {method} [options] {source_path} {models_path} {summaries_path}


Required arguments
^^^^^^^^^^^^^^^^^^
  :source_path:           Directory containing all the source files to be summarized. Each set of documents are expected to be in different directories inside this path.
  :models_path:           Directory containing all the model summaries. Each set of summaires are expected to be in different directory inside this path, having the same name as the corresponding directory in the source directory.
  :summaries_path:        Directory to store the generated summaries. The directory will be created if not already exists.

Common options
^^^^^^^^^^^^^^
  -h, --help            show this help message and exit
  --only-rouge          Do not run summarizer. Only compule ROUGE score for
                        existing summaries in summaries_path
  -s N, --size N        Maximum size of the summary
  -w, --words           Caluated size as number of words instead of characters
  --source-lang lang    Two-letter language code of the source documents
                        language. Defaults to `en`
  -l lang, --target-lang lang
                        Two-letter language code to generate cross-lingual
                        summary. Defaults to source language.
