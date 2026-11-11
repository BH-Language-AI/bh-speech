from bh_speech.evaluation import cer, normalize, wer


def test_normalize_dijakritici():
    assert normalize("ČĆŽŠĐ") == "čćžšđ"
    assert normalize("  Zdravo,  svijete! ") == "zdravo svijete"


def test_wer_osnovno():
    assert wer("zdravo svijete", "zdravo svijete") == 0.0
    assert wer("zdravo", "zdravo svijete") == 0.5
    assert wer("x", "") is None


def test_cer_osnovno():
    assert cer("abc", "abc") == 0.0
    assert cer("", "abc") == 1.0
