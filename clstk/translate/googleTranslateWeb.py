# Adopted to Python from:
# 1. https://github.com/matheuss/google-translate-token and
# 2. https://github.com/nisargjhaveri/news-access/tree/master/translate

"""
**DO NOT** use this for commercial purpuses
"""

import time
import requests
import re
import json
import shelve

window = {
    # 'TKK': config.get('TKK') or '0' TODO
    'TKK': '0'
}


# // BEGIN
def _xr(a, b):
    c = 0
    while c < len(b) - 2:
        d = b[c + 2]
        d = ord(d[0]) - 87 if "a" <= d else int(d)
        d = (a % 0x100000000) >> d if "+" == b[c + 1] else a << d
        a = a + d & 4294967295 if "+" == b[c] else a ^ d
        c += 3
    return a


def _sM(a):
    b = window["TKK"] if "TKK" in window else ""

    d = b.split(".")
    b = int(d[0]) or 0
    e = []
    f = 0
    g = 0
    while g < len(a):
        l = ord(a[g])
        if 128 > l:
            e.append(l)
        else:
            if 2048 > l:
                e.append(l >> 6 | 192)
            else:
                if 55296 == (l & 64512) \
                   and g + 1 < len(a) \
                   and 56320 == (ord(a[g + 1]) & 64512):
                    g += 1
                    l = 65536 + ((l & 1023) << 10) + (ord(a[g]) & 1023)
                    e.append(l >> 18 | 240)
                    e.append(l >> 12 & 63 | 128)
                else:
                    e.append(l >> 12 | 224)
                    e.append(l >> 6 & 63 | 128)
            e.append(l & 63 | 128)
        g += 1
    a = b
    for f in xrange(len(e)):
        a += e[f]
        a = _xr(a, "+-a^+6")
    a = _xr(a, "+-3^+b+-f")
    a ^= int(d[1]) or 0
    if 0 > a:
        a = (a & 2147483647) + 2147483648
    a %= 1E6
    a = int(a)
    return str(a) + "." + str(a ^ b)
# // END


def _evalTKK(TKK):
    TKK = TKK.decode('string_escape')
    evalStatments = map(str.strip,
                        TKK[len("eval('((function(){"):-len("})")].split(";"))

    for statement in evalStatments:
        if statement.startswith("var"):
            var, val = statement[len("var "):].split("=", 1)
            if var == 'a':
                a = eval(val)
            elif var == 'b':
                b = eval(val)
        elif statement.startswith("return"):
            statement = statement[len("return "):]
            TKK = str(statement.split("+")[0]) + '.' + str(a + b)

    return TKK


def _getToken(text):
    # Update token if needed
    now = int(time.time() / 3600000)
    if int(window['TKK'].split('.')[0]) != now:
        r = requests.get('https://translate.google.com')

        TKK = _evalTKK(re.findall(r"TKK=(.*?)\(\)\)'\);", r.text)[0])
        window['TKK'] = TKK

    # Generate token for text
    tk = _sM(text)

    return tk


def _translateText(text, source, target):
    if len(text) >= 4500:
        lines = text.split("\n")

        if len(lines) <= 1:
            raise RuntimeError("SENTENCE_TOO_LARGE")

        half = len(lines) / 2

        translation1, sentences1 = _translateText("\n".join(lines[:half]),
                                                  source, target)
        translation2, sentences2 = _translateText("\n".join(lines[half:]),
                                                  source, target)

        return (
            "\n".join([translation1, translation2]), sentences1 + sentences2
        )

    url = 'https://translate.google.com/translate_a/single'
    data = {
        'client': 't',
        'sl': source,
        'tl': target,
        'hl': target,
        'dt': ['at', 'bd', 'ex', 'ld', 'md', 'qca', 'rw', 'rm', 'ss', 't'],
        'ie': 'UTF-8',
        'oe': 'UTF-8',
        'otf': 1,
        'ssel': 0,
        'tsel': 0,
        'kc': 7,
        'q': text,
        'tk': _getToken(text)
    }

    req = requests.post(url, data)
    res = json.loads(req.text)

    translation = ""
    sentences = []

    for sentence in res[0]:
        if sentence[0]:
            translation += sentence[0]
            sentences.append({
                "source": sentence[1].strip(),
                "target": sentence[0].strip()
            })

    return translation, sentences


def translate(text, sourceLang, targetLang, sentencePerLine=True):
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
        translation, sentences = _translateText(textToTranslate,
                                                sourceLang, targetLang)

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
