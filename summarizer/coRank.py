from corpus import Corpus
from summary import Summary

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

import logging
logger = logging.getLogger("coRank.py")


def _row_normalize(M):
    return preprocessing.normalize(M, axis=1, norm='l1')


def normalize(M):
    return M / np.sum(M)


def summarize(inDir, params):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir).load(params, translate=True)

    logger.info("Setting up summarizer")
    M_en = cosine_similarity(
        c.getSentenceVectors()
    )
    np.fill_diagonal(M_en, 0)

    M_cn = cosine_similarity(
        c.getTranslationSetenceVectors()
    )
    np.fill_diagonal(M_cn, 0)

    M_encn = np.sqrt(M_en * M_cn)

    M_en = _row_normalize(M_en)
    M_cn = _row_normalize(M_cn)
    M_encn = _row_normalize(M_encn)

    alpha = params['alpha']

    logger.info("Iteratively computing sentence saliency")
    u = normalize(np.random.random((M_cn.shape[0],)))
    v = normalize(np.random.random((M_en.shape[0],)))
    for i in xrange(params['max_iter']):
        u_prev = u
        v_prev = v

        u = alpha * np.dot(M_cn.T, u) + (1 - alpha) * np.dot(M_encn.T, v)
        v = alpha * np.dot(M_en.T, v) + (1 - alpha) * np.dot(M_encn.T, u)

        u = normalize(u)
        v = normalize(v)

        if np.all(np.isclose(u_prev, u)) and np.all(np.isclose(v_prev, v)):
            # Converged
            break

    logger.info("Optimization completed in %d iterations" % (i + 1))

    # summary = optimizer.greedy(params["size"], objective, c)
    logger.info("Computing final sentence scores including redundancy penalty")
    sentence_scores = u.copy()
    sentence_order = []
    for i in xrange(len(sentence_scores)):
        best_sentence = np.argmax(sentence_scores)
        sentence_order.append(best_sentence)

        sentence_scores -= M_cn[:, best_sentence] * u[best_sentence]
        sentence_scores[best_sentence] = float('-inf')

    logger.info("Generating final summary")

    sizeBudget, countTokens = params['size']
    sizeName = "tokens" if countTokens else "chars"

    def sentenceSize(sent):
        return sent.tokenCount() if countTokens else sent.charCount()

    def summarySize(summary):
        return summary.tokenCount() if countTokens else summary.charCount()

    logger.info("Summary budget: %d %s", sizeBudget, sizeName)

    summary = Summary()
    sentences = c.getSentences()

    for sentence_id in sentence_order:
        if summarySize(summary) >= sizeBudget:
            break

        sentence_size = sentenceSize(sentences[sentence_id])

        if summarySize(summary) + sentence_size <= sizeBudget:
            logger.info("Sentence added with size: %d, ", sentence_size)
            summary.addSentence(sentences[sentence_id])

    logger.info("Optimization done, summary size: %d chars, %d tokens",
                summary.charCount(), summary.tokenCount())

    return summary


def setupArgparse(parser):
    def run(args, silent=False):
        params = {
            'size': (args.size, args.words),
            'sourceLang': args.source_lang,
            'targetLang': args.target_lang or args.source_lang,
            'alpha': args.alpha,
            'max_iter': args.max_iter
        }

        summary = summarize(args.source_directory, params)

        if not silent:
            logging.info("Printing source summary\n" +
                         summary.getSummary().encode('utf-8'))

            logging.info("Printing target summary")
            print summary.getTargetSummary().encode('utf-8')

        return summary

    parser.add_argument('-s', '--size', type=int, default=665, metavar="N",
                        help='Maximum size of the summary')
    parser.add_argument('-w', '--words', action="store_true",
                        help='Caluated size as number of words instead of '
                        'characters')
    parser.add_argument('--source-lang', type=str, default='en',
                        metavar="lang", help='Two-letter language code of '
                        'the source documents language. Defaults to `en`')
    parser.add_argument('-l', '--target-lang', type=str, default=None,
                        metavar="lang", help='Two-letter language code to '
                        'generate cross-lingual summary. '
                        'Defaults to source language.')

    parser.add_argument('--alpha', type=float, default=.5,
                        help='Relative contributions to the final saliency '
                        'scores from the information in the same language and '
                        'the information in the other language')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum iterations for the iterative algorithm')

    parser.set_defaults(func=run)
