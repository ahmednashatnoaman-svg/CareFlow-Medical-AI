"""Egyptian Arabic text normalization and spoken answer-option matching.

Supports the voice interview flow: a patient may answer a multiple-choice question
either by saying an ordinal ("الاختيار التالت") or by speaking the option's words.
"""

import difflib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Harakat/tanween (U+064B-U+0652), superscript alef (U+0670), tatweel (U+0640)
_DIACRITICS = re.compile(r"[ً-ْٰـ]")

_LETTER_FOLDING = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ة": "ه",
        "ى": "ي", "ئ": "ي",
        "ؤ": "و",
        "ﻻ": "لا",
    }
)

# Arabic-Indic (U+0660-) and Extended Arabic-Indic (U+06F0-) digits
_DIGIT_FOLDING = str.maketrans(
    {chr(0x0660 + d): str(d) for d in range(10)} | {chr(0x06F0 + d): str(d) for d in range(10)}
)

# 1-based option position -> spoken forms (MSA + Egyptian colloquial + cardinals)
_ORDINAL_FORMS: Dict[int, Tuple[str, ...]] = {
    1: ("الاول", "اول", "الاولاني", "اولاني", "الاولى", "واحد"),
    2: ("الثاني", "ثاني", "التاني", "تاني", "الثانيه", "اتنين", "اثنين"),
    3: ("الثالث", "ثالث", "التالت", "تالت", "الثالثه", "تلاته", "ثلاثه"),
    4: ("الرابع", "رابع", "الرابعه", "اربعه"),
    5: ("الخامس", "خامس", "الخامسه", "خمسه"),
}

# Words that merely carry the ordinal ("option", "answer", "number") and add no meaning.
_CARRIER_WORDS = frozenset(
    {"الاختيار", "اختيار", "الخيار", "خيار", "الاجابه", "اجابه", "رقم", "نمره", "the", "option", "answer", "number"}
)

# Negation markers (Egyptian colloquial + MSA + English). A transcript containing one of
# these next to an option's own words denies that option rather than affirming it --
# e.g. "مفيش سخونية" (no fever) must NOT match the "Fever" option.
_NEGATION_WORDS = frozenset(
    {
        "مفيش", "مافيش", "لا", "لأ", "مش", "مو", "بدون", "غير", "خالص", "لسه", "لسة",
        "no", "not", "never", "none", "nothing", "without", "isn't", "wasn't", "don't", "doesn't", "didn't",
    }
)


def _has_negation(normalized_text: str) -> bool:
    """Detects a negation marker among the (already-normalized) transcript's tokens."""
    tokens = set(normalized_text.split())
    return bool(tokens & _NEGATION_WORDS)


def strip_diacritics(text: str) -> str:
    """Removes harakat and tatweel while preserving letter forms, spelling, and word order.

    Display-safe: the result stays readable Arabic, unlike `normalize_arabic`.
    """
    if not text:
        return ""
    return " ".join(_DIACRITICS.sub("", unicodedata.normalize("NFKC", text)).split())


def normalize_arabic(text: str) -> str:
    """Folds Arabic orthographic variation so equivalent spellings compare equal.

    Strips diacritics and tatweel, unifies alef/yaa/taa-marbuta forms, converts
    Arabic-Indic digits to ASCII, and collapses whitespace.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _DIACRITICS.sub("", normalized)
    normalized = normalized.translate(_LETTER_FOLDING).translate(_DIGIT_FOLDING)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split()).casefold()


def parse_ordinal(text: str, option_count: Optional[int] = None) -> Optional[int]:
    """Extracts a spoken 1-based option position, or None if the text names no position.

    Recognizes "الاختيار الثالث", "التالت", "رقم ٣", and a bare "3". Requires the
    utterance to be only a position reference, so a sentence that merely contains a
    number ("بقالي تلات ساعات") is left for fuzzy matching or free text.
    """
    normalized = normalize_arabic(text)
    if not normalized:
        return None

    tokens = [t for t in normalized.split() if t not in _CARRIER_WORDS]
    if not tokens:
        return None

    matched: Optional[int] = None
    for token in tokens:
        candidate: Optional[int] = None
        if token.isdigit():
            candidate = int(token)
        else:
            for position, forms in _ORDINAL_FORMS.items():
                if token in (normalize_arabic(f) for f in forms):
                    candidate = position
                    break

        # A non-positional word means this is a sentence, not a bare selection.
        if candidate is None:
            return None
        if matched is not None and matched != candidate:
            return None
        matched = candidate

    if matched is None or matched < 1:
        return None
    if option_count is not None and matched > option_count:
        return None
    return matched


def _similarity(transcript: str, option_text: str) -> float:
    """Blends sequence ratio with token overlap so word-order differences still score."""
    if not transcript or not option_text:
        return 0.0

    ratio = difflib.SequenceMatcher(None, transcript, option_text).ratio()

    option_tokens = set(option_text.split())
    transcript_tokens = set(transcript.split())
    if not option_tokens:
        return ratio

    overlap = len(option_tokens & transcript_tokens) / len(option_tokens)
    return max(ratio, round(ratio * 0.4 + overlap * 0.6, 4))


def match_option(
    transcript: str,
    options: Sequence[Dict[str, Any]],
    threshold: float = 0.80,
) -> Tuple[Optional[int], float]:
    """Resolves a spoken answer to a 1-based option index.

    Tries an explicit ordinal first, then fuzzy-matches the normalized transcript
    against each option's Arabic and English text.

    Returns:
        Tuple[Optional[int], float]: (1-based index or None, confidence 0.0-1.0).
            None means the utterance should be treated as a free-text answer.
    """
    selectable = [o for o in options if not o.get("is_free_text")]
    if not transcript or not selectable:
        return None, 0.0

    # Bound against the full visible list (selectable + free-text slot) since that's what
    # the patient was shown, but an ordinal landing on the free-text slot is not itself an
    # answer -- fall through so the raw utterance is treated as free text.
    ordinal = parse_ordinal(transcript, option_count=len(options))
    if ordinal is not None:
        if any(o.get("index") == ordinal and o.get("is_free_text") for o in options):
            return None, 0.0
        if ordinal in {o.get("index") for o in selectable}:
            return ordinal, 1.0
        return None, 0.0

    normalized_transcript = normalize_arabic(transcript)
    transcript_negated = _has_negation(normalized_transcript)
    best_index: Optional[int] = None
    best_score = 0.0

    for option in selectable:
        index = option.get("index")
        if not isinstance(index, int):
            continue
        for key in ("text_arabic", "text_english"):
            option_normalized = normalize_arabic(str(option.get(key, "")))
            # A negated transcript can only match an option that is itself phrased as a
            # negative/denial (e.g. "No other symptoms", "Nothing helps") -- otherwise
            # word/phrase overlap with an affirmative option (e.g. "Fever") would record
            # the opposite of what the patient said.
            if transcript_negated and not _has_negation(option_normalized):
                continue
            score = _similarity(normalized_transcript, option_normalized)
            if score > best_score:
                best_score, best_index = score, index

    if best_index is None or best_score < threshold:
        return None, best_score
    return best_index, best_score
