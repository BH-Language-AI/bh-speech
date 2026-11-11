"""WER/CER sa bosanskom normalizacijom. Bez vanjskih zavisnosti."""
import re
import unicodedata


def normalize(text):
    """Mala slova, bez interpunkcije, uredeni razmaci. Dokumentuj svaku izmjenu pravila."""
    text = unicodedata.normalize('NFC', text.lower())
    text = re.sub(r'[^\w\sčćdžđšž]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(hyp, ref, norm=True):
    """Word Error Rate. Vraca None za prazan ref (izbjegni dijeljenje nulom)."""
    if norm:
        hyp, ref = normalize(hyp), normalize(ref)
    h, r = hyp.split(), ref.split()
    if not r:
        return None
    return _lev(h, r) / len(r)


def cer(hyp, ref, norm=True):
    """Character Error Rate."""
    if norm:
        hyp, ref = normalize(hyp), normalize(ref)
    if not ref:
        return None
    return _lev(hyp, ref) / len(ref)
