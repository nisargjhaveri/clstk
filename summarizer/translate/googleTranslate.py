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


def translate(text, source, target):
    def cacheKey(text):
        return "_".join([
            text, source, target
        ]).encode('utf-8')

    cache = shelve.open('.translation-cache')

    if cacheKey(text) in cache:
        return cache[cacheKey(text)]

    translation = _translateText(text, source, target)

    sentences = []
    sourceSentences = text.split("\n")
    targetSentences = translation.split("\n")

    if (len(sourceSentences) != len(targetSentences)):
        raise RuntimeError("GOOGLE_TRANSLATION_ERROR")
    else:
        for i in xrange(len(sourceSentences)):
            sentences.append({
                "source": sourceSentences[i].strip(),
                "target": targetSentences[i].strip()
            })

    cache[cacheKey(text)] = (translation, sentences)
    cache.close()

    return translation, sentences
