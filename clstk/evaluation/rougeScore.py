"""
Python implementation of ROUGE score.

Taken and adopted from:

- https://github.com/miso-belica/sumy/blob/master/sumy/evaluation/rouge.py
- https://github.com/google/seq2seq/blob/master/seq2seq/metrics/rouge.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import six

import numpy as np
import regex


class RougeScore(object):
    """
    Implementation of ROUGE score.
    """

    def __init__(self, tokenizer=None, stemmer=None):
        self._tokenize = tokenizer if tokenizer else self.dummy_tokenizer
        self._stemmer = stemmer if stemmer else self.dummy_stemmer

    def dummy_tokenizer(self, sentence):
        sentence = regex.sub(r'-', ' - ', sentence)
        sentence = regex.sub(r'[^\w]', ' ', sentence)
        sentence = regex.sub(r'^\s+', '', sentence)
        sentence = regex.sub(r'\s+$', '', sentence)
        sentence = regex.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip().lower()

        r = regex.compile(r'^\w')

        return filter(r.match, regex.split(r'\s+', sentence))
        # return sentence.split()

    def dummy_stemmer(self, token):
        return token

    def _get_ngrams(self, n, text):
        """Calcualtes n-grams.
        Args:
            n: which n-grams to calculate
            text: An array of tokens
        Returns:
            A set of n-grams
        """
        ngram_list = list()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_list.append(tuple(text[i:i + n]))
        return ngram_list

    def _split_into_words(self, sentences):
        """Splits multiple sentences into words and flattens the result"""
        return map(self._stemmer, sum(map(self._tokenize, sentences), []))

    def _get_word_ngrams(self, n, sentences):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0

        words = self._split_into_words(sentences)
        return self._get_ngrams(n, words)

    def _count_overlap(self, ngrams1, ngrams2):
        counter1 = collections.Counter(ngrams1)
        counter2 = collections.Counter(ngrams2)

        result = 0
        for k, v in six.iteritems(counter1):
            result += min(v, counter2[k])
        return result

    def rouge_n(self, summary, model_summaries, n=2):
        """
        Computes ROUGE-N of two text collections of sentences.
        """
        """
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf
        Args:
            summary: The sentences that have been picked by the
                     summarizer
            model_summaries: List of reference summaries, each containing
                             list of sentences
            n: Size of ngram.    Defaults to 2.
        Returns:
            A tuple (f1, precision, recall) for ROUGE-N
        Raises:
            ValueError: raises exception if a param has len <= 0
        """
        if len(summary) <= 0 or len(model_summaries) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        summary_ngrams = self._get_word_ngrams(n, summary)

        summary_count = 0
        model_count = 0
        overlap_count = 0

        for model in model_summaries:
            model_ngrams = self._get_word_ngrams(n, model)
            model_count += len(model_ngrams)
            summary_count += len(summary_ngrams)

            # Gets the overlapping ngrams between evaluated and reference
            overlap_count += self._count_overlap(summary_ngrams, model_ngrams)

        # Handle edge case.
        # This isn't mathematically correct, but it's good enough
        precision = 0.0 if summary_count == 0 \
            else overlap_count / summary_count

        recall = 0.0 if model_count == 0 else overlap_count / model_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

        return f1_score, precision, recall

    def _len_lcs(self, x, y):
        """
        Returns the length of the Longest Common Subsequence between sequences
        x and y.
        Source:
            http://www.algorithmist.com/index.php/Longest_Common_Subsequence
        Args:
            x: sequence of words
            y: sequence of words
        Returns
            integer: Length of LCS between x and y
        """
        table = self._lcs(x, y)
        n, m = len(x), len(y)
        return table[n, m]

    def _lcs(self, x, y):
        """
        Computes the length of the longest common subsequence (lcs) between two
        strings. The implementation below uses a DP programming algorithm and
        runs in O(nm) time where n = len(x) and m = len(y).
        Source:
            http://www.algorithmist.com/index.php/Longest_Common_Subsequence
        Args:
            x: collection of words
            y: collection of words
        Returns:
            Table of dictionary of coord and len lcs
        """
        n, m = len(x), len(y)
        table = dict()
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 or j == 0:
                    table[i, j] = 0
                elif x[i - 1] == y[j - 1]:
                    table[i, j] = table[i - 1, j - 1] + 1
                else:
                    table[i, j] = max(table[i - 1, j], table[i, j - 1])
        return table

    def rouge_l_sentence_level(self, evaluated_sentences, reference_sentences):
        """
        Computes ROUGE-L (sentence level) of two text collections of sentences.
        """
        """
        http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Calculated according to:
        R_lcs = LCS(X,Y)/m
        P_lcs = LCS(X,Y)/n
        F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
        where:
        X = reference summary
        Y = Candidate summary
        m = length of reference summary
        n = length of candidate summary
        Args:
            evaluated_sentences: The sentences that have been picked by the
                                 summarizer
            reference_sentences: The sentences from the referene set
        Returns:
            A float: F_lcs
        Raises:
            ValueError: raises exception if a param has len <= 0
        """
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")
        reference_words = self._split_into_words(reference_sentences)
        evaluated_words = self._split_into_words(evaluated_sentences)
        m = len(reference_words)
        n = len(evaluated_words)
        lcs = self._len_lcs(evaluated_words, reference_words)
        return self._f_p_r_lcs(lcs, m, n)

    def _f_p_r_lcs(self, llcs, m, n):
        """
        Computes the LCS-based F-measure score
        Source:
            http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Args:
            llcs: Length of LCS
            m: number of words in reference summary
            n: number of words in candidate summary
        Returns:
            Float. LCS-based F-measure score
        """
        r_lcs = llcs / m
        p_lcs = llcs / n
        beta = p_lcs / (r_lcs + 1e-12)
        num = (1 + (beta**2)) * r_lcs * p_lcs
        denom = r_lcs + ((beta**2) * p_lcs)
        f_lcs = num / (denom + 1e-12)
        return f_lcs, p_lcs, r_lcs

    def _recon_lcs(self, x, y):
        """
        Returns the Longest Subsequence between x and y.
        Source:
            http://www.algorithmist.com/index.php/Longest_Common_Subsequence
        Args:
            x: sequence of words
            y: sequence of words
        Returns:
            sequence: LCS of x and y
        """
        i, j = len(x), len(y)
        table = self._lcs(x, y)

        def _recon(i, j):
            """private recon calculation"""
            if i == 0 or j == 0:
                return []
            elif x[i - 1] == y[j - 1]:
                return _recon(i - 1, j - 1) + [(x[i - 1], i)]
            elif table[i - 1, j] > table[i, j - 1]:
                return _recon(i - 1, j)
            else:
                return _recon(i, j - 1)

        recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
        return recon_tuple

    def _union_lcs(self, evaluated_sentences, reference_sentence):
        """
        Returns LCS_u(r_i, C) which is the LCS score of the union longest
        common subsequence between reference sentence ri and candidate summary
        C. For example if r_i= w1 w2 w3 w4 w5, and C contains two sentences:
        c1 = w1 w2 w6 w7 w8 and c2 = w1 w3 w8 w9 w5, then the longest common
        subsequence of r_i and c1 is "w1 w2" and the longest common subsequence
        of r_i and c2 is "w1 w3 w5". The union longest common subsequence of
        r_i, c1, and c2 is "w1 w2 w3 w5" and LCS_u(r_i, C) = 4/5.
        Args:
            evaluated_sentences: The sentences that have been picked by the
                                 summarizer
            reference_sentence: One of the sentences in the reference summaries
        Returns:
            float: LCS_u(r_i, C)
        ValueError:
            Raises exception if a param has len <= 0
        """
        if len(evaluated_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        lcs_union = set()
        reference_words = self._split_into_words([reference_sentence])
        combined_lcs_length = 0
        for eval_s in evaluated_sentences:
            evaluated_words = self._split_into_words([eval_s])
            lcs = set(self._recon_lcs(reference_words, evaluated_words))
            combined_lcs_length += len(lcs)
            lcs_union = lcs_union.union(lcs)

        union_lcs_count = len(lcs_union)
        union_lcs_value = union_lcs_count / combined_lcs_length
        return union_lcs_value

    def rouge_l_summary_level(self, evaluated_sentences, reference_sentences):
        """
        Computes ROUGE-L (summary level) of two text collections of sentences.
        """
        """
        http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Calculated according to:
        R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
        P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
        F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
        where:
        SUM(i,u) = SUM from i through u
        u = number of sentences in reference summary
        C = Candidate summary made up of v sentences
        m = number of words in reference summary
        n = number of words in candidate summary
        Args:
            evaluated_sentences: The sentences that have been picked by the
                                 summarizer
            reference_sentence: One of the sentences in the reference summaries
        Returns:
            A float: F_lcs
        Raises:
            ValueError: raises exception if a param has len <= 0
        """
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        # total number of words in reference sentences
        m = len(self._split_into_words(reference_sentences))

        # total number of words in evaluated sentences
        n = len(self._split_into_words(evaluated_sentences))

        union_lcs_sum_across_all_references = 0
        for ref_s in reference_sentences:
            union_lcs_sum_across_all_references += self._union_lcs(
                evaluated_sentences, ref_s)
        return self._f_p_r_lcs(union_lcs_sum_across_all_references, m, n)

    def _print_result(self, rouge_type, rouge_all, print_all=False):
        rouge_f, rouge_p, rouge_r = map(np.mean, zip(*rouge_all))

        print("ROUGE-%s Average  R:%0.5f  P:%0.5f  F:%0.5f"
              % (rouge_type, rouge_r, rouge_p, rouge_f))

        if print_all:
            print("---------------------------------")
            for i, rouge in enumerate(rouge_all):
                rouge_f, rouge_p, rouge_r = rouge
                print("ROUGE-%s Eval %d  R:%0.5f  P:%0.5f  F:%0.5f"
                      % (rouge_type, i, rouge_r, rouge_p, rouge_f))
            print("---------------------------------")

    def rouge(self, hyp_refs_pairs, print_all=False):
        """
        Calculates and prints average rouge scores for a list of hypotheses
        and references

        :param hyp_refs_pairs: List containing pairs of path to summary and
                               list of paths to reference summaries
        :param print_all: Print every evaluation along with averages
        """

        rouge_1_all = []
        rouge_2_all = []

        for hyp_refs_pair in hyp_refs_pairs:
            hyp_path, ref_paths = hyp_refs_pair

            with open(hyp_path) as hyp_file:
                hyp = map(lambda x: x.decode('utf-8'), list(hyp_file))

            refs = []
            for ref_path in ref_paths:
                with open(ref_path) as ref_file:
                    refs.append(map(lambda x: x.decode('utf-8'),
                                    list(ref_file)))

            rouge_1_all.append(self.rouge_n(hyp, refs, 1))

            rouge_2_all.append(self.rouge_n(hyp, refs, 2))

            # rouge_l = [
            #     self.rouge_l_sentence_level(hyp, ref) for ref in refs
            # ]
            # rouge_l_all.append(map(np.mean, zip(*rouge_l)))

        self._print_result("1", rouge_1_all, print_all)
        self._print_result("2", rouge_2_all, print_all)
