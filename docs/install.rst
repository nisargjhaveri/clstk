Installation
============

Clone repository
----------------
.. code-block:: console

 $ git clone https://github.com/nisargjhaveri/clstk


Python dependencies
-------------------
The dependencies are listed in ``requirements.txt``.

To install all the dependencies, run ``pip`` as followed.

.. code-block:: console

 $ pip install --upgrade -r requirements.txt


Also install nltk packages called ``stopwords`` and ``punkt``.

.. code-block:: console

 $ python -m nltk.downloader stopwords punkt -d $NLTK_DATA


Setup CLUTO (optional)
---------------------------------------------------------
http://glaros.dtc.umn.edu/gkhome/cluto/cluto/download

This is required if you want to use "linBilmes" summarizer.

Set an environment variable ``CLUTO_BIN_PATH`` with the path of directory containing ``vcluster`` binary file.


Setup ROUGE 1.5.5 (optional)
---------------------------------------------------------------
https://github.com/nisargjhaveri/ROUGE-1.5.5-unicode

This is required only if you plan to evaluate the summaries using ROUGE score.

Obtain and setup ROUGE 1.5.5 according to the instructions there.

Set an environment variable ``ROUGE_HOME`` with the path to ROUGE root directory, the one containing ``ROUGE-1.5.5.pl`` file.


Setup dependencies for TQE (optional)
-------------------------------------
https://github.com/nisargjhaveri/tqe

Install dependencies for ``tqe`` module according to the details provided in the link above.


Setup NeuralTextSimplification (optional)
-----------------------------------------
https://github.com/senisioi/NeuralTextSimplification

Setup system from above URL and set ``NTS_OPENNMT_PATH``, ``NTS_MODEL_PATH`` and ``NTS_GPUS`` variables accordingly.
