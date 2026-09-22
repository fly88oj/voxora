"""Golden-case tests for the scoring/normalization layer (methodology-critical)."""

import pytest

from voxora.scoring import cer, normalize_cer, normalize_words, primary_error, strip_tags, wer


class TestStripTags:
    def test_sensevoice_tags_removed(self):
        assert strip_tags("<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好") == "你好"

    def test_plain_text_untouched(self):
        assert strip_tags("Hello, world!") == "Hello, world!"


class TestCER:
    def test_identical_zh(self):
        assert cer("今天天气真不错，我们一起去公园散步吧。", "今天天气真不错，我们一起去公园散步吧。") == 0.0

    def test_case_and_punctuation_insensitive(self):
        assert cer("Hello, World!", "hello world") == 0.0

    def test_one_substitution_of_ten_chars(self):
        assert cer("一二三四五六七八九十", "一二三四五六十百九十") == pytest.approx(2 / 10)

    def test_empty_reference_is_full_error(self):
        assert cer("", "abc") == 1.0

    def test_whitespace_ignored(self):
        assert cer("张 三", "张三") == 0.0


class TestWER:
    def test_identical_en(self):
        assert wer("the quick brown fox", "the quick brown fox") == 0.0

    def test_one_word_substitution(self):
        # 4 reference words, 1 substitution -> 0.25 (whole-word granularity)
        assert wer("the quick brown fox", "the quick red fox") == pytest.approx(0.25)

    def test_single_char_typo_is_full_word_error(self):
        # This granularity is intentional and documented (METHODOLOGY.md).
        assert wer("day after day", "days after day") == pytest.approx(1 / 3)

    def test_russian_yo_kept_distinct(self):
        # Documented artifact: no unicode folding, ё != е
        assert wer("с каждым днём", "с каждым днем") == pytest.approx(1 / 3)

    def test_punctuation_becomes_boundary(self):
        assert wer("nice day, isn't it", "nice day isnt it") == 0.0


class TestPrimaryError:
    @pytest.mark.parametrize("lang", ["zh", "yue", "ja", "ko"])
    def test_cjk_uses_cer(self, lang):
        assert primary_error("你好世界", "你好时间", lang) == cer("你好世界", "你好时间")

    @pytest.mark.parametrize("lang", ["en", "de", "fr", "ru", "es"])
    def test_european_uses_wer(self, lang):
        assert primary_error("a b c", "a b d", lang) == wer("a b c", "a b d")

    def test_case_insensitive_lang_code(self):
        assert primary_error("你好", "你号", "ZH") == 0.5


class TestNormalizationInternals:
    def test_normalize_cer_drops_all_space(self):
        assert normalize_cer("a b c") == "abc"

    def test_normalize_words_keeps_boundaries(self):
        assert normalize_words("a, b! c") == ["a", "b", "c"]
