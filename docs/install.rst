Installation
============

Clone repository
----------------
.. code-block:: bash

 $ git clone https://github.com/nisargjhaveri/summarizer


Python dependencies
-------------------
The dependencies are listed in ``requirements.txt``.

To install all the dependencies, run ``pip`` as followed.

.. code-block:: bash

 $ pip install --upgrade -r requirements.txt


Also install nltk packages called ``stopwords`` and ``punkt``.

.. code-block:: bash

 $ python -m nltk.downloader stopwords punkt -d $NLTK_DATA


Setup CLUTO (optional, required for linBilmes summarizer)
---------------------------------------------------------
http://glaros.dtc.umn.edu/gkhome/cluto/cluto/download

Set an environment variable ``CLUTO_BIN_PATH`` with the path of directory containing ``vcluster`` binary file.


Setup ROUGE 1.5.5 (optional, required for evaluating summaries)
---------------------------------------------------------------
https://github.com/nisargjhaveri/ROUGE-1.5.5-unicode

This is required only if you plan to evaluate the summaries using ROUGE score. You may skip this.

Obtain and setup ROUGE 1.5.5 according to the instructions there.

Set an environment variable ``ROUGE_HOME`` with the path to ROUGE root directory, the one containing ``ROUGE-1.5.5.pl`` file.


Setup dependencies for TQE (optional)
-------------------------------------
https://github.com/nisargjhaveri/tqe
