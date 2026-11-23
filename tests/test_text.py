from stc.text import clean, tokenize


def test_clean_removes_urls_handles_and_entities():
    assert clean("@united Great &amp; fast http://t.co/abc") == "great & fast"


def test_negation_is_preserved():
    assert tokenize(clean("I can't believe it, they didn't call")) == [
        "i", "can", "not", "believe", "it", "they", "did", "not", "call"]


def test_punctuation_and_emoji_tokens():
    assert tokenize(clean("Delayed again!! 😡")) == ["delayed", "again", "!", "!", "😡"]
