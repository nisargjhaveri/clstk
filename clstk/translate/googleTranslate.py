"""
Translate using Google Translate.

To use this, environmental variable ``GOOGLE_APPLICATION_CREDENTIALS`` needs
to be set with file continaining your key for Google Cloud account.

See https://cloud.google.com/translate/docs/reference/libraries
"""
import shelve

from google.cloud import translate as googleTranslate

translate_client = None


def _translateText(text, source, target):
    global translate_client

    if not translate_client:
        translate_client = googleTranslate.Client()

    if len(text) >= 4500:
        lines = text.split("\n")

        if len(lines) <= 1:
            raise RuntimeError("SENTENCE_TOO_LARGE")

        half = len(lines) / 2

        translation1 = _translateText("\n".join(lines[:half]),
                                      source, target)
        translation2 = _translateText("\n".join(lines[half:]),
                                      source, target)

        return (
            "\n".join([translation1, translation2])
        )

    translation = translate_client.translate(text,
                                             format_='text',
                                             source_language=source,
                                             target_language=target)

    return translation['translatedText']


def translate(text, sourceLang, targetLang):
    """
    Translate text

    :param text: Text, each line contains one sentence
    :param sourceLang: Two-letter code for source language
    :param targetLang: Two-letter code for target language

    :returns: translated text and list of translated sentences
    :rtype: (translation, sentences)
    """
    def cacheKey(text):
        return "_".join([
            text.strip(), sourceLang, targetLang
        ]).encode('utf-8')

    cache = shelve.open('.translation-cache.google')

    sentencesToTranslate = []
    sourceSentences = text.split("\n")

    for sentence in sourceSentences:
        if cacheKey(sentence) not in cache:
            sentencesToTranslate.append(sentence)

    if len(sentencesToTranslate):
        textToTranslate = "\n".join(sentencesToTranslate)
        translation = _translateText(textToTranslate, sourceLang, targetLang)

        translatedSentences = translation.split("\n")

        if (len(sentencesToTranslate) != len(translatedSentences)):
            raise RuntimeError("GOOGLE_TRANSLATION_ERROR")
        else:
            for source, target in zip(sentencesToTranslate,
                                      translatedSentences):
                cache[cacheKey(source)] = target

    sentences = []
    for sentence in sourceSentences:
        sentences.append({
            "source": sentence.strip(),
            "target": cache[cacheKey(sentence)]
        })

    translation = "\n".join(map(lambda s: s['target'], sentences))

    cache.close()

    return translation, sentences
