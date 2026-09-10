"""Filter Interactions to English ones.

The English filter is the stage after interaction stitching (ADR-0003): it
reads the normalized Interactions JSONL and keeps the Interactions whose
opening Customer Message is English, since the intent classifier that follows
is English-only and would otherwise misclassify non-English messages quietly
(decision 11 in ``docs/decisions.md``).

Detection uses Lingua (ADR-0004) with a conservative rule:

* handles and links are stripped first — they carry no language signal;
* the detector is restricted to English plus the languages actually present in
  the dataset and the major world languages, which sharpens decisions on short
  messages;
* a minimum relative distance between the two most likely languages makes the
  detector answer "no confident signal" instead of guessing. Those
  Interactions are *kept*: the filter only drops messages with positive
  non-English evidence, never ambiguous ones.

The report counts retained English, retained-but-ambiguous (no signal), and
filtered volume by detected language, so the report can state honestly how
much data the stage removed.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from query.interactions import Interaction

# English plus the languages observed in the SpotifyCares interactions
# (Indonesian, Tagalog, Dutch, French, Swedish, Spanish, Turkish, ...) and the
# major world languages, so confident detections name the right language
# instead of a Latin-script neighbour.
LANGUAGES = (
    Language.ENGLISH,
    Language.SPANISH,
    Language.PORTUGUESE,
    Language.FRENCH,
    Language.GERMAN,
    Language.DUTCH,
    Language.ITALIAN,
    Language.SWEDISH,
    Language.BOKMAL,
    Language.DANISH,
    Language.FINNISH,
    Language.POLISH,
    Language.CZECH,
    Language.SLOVAK,
    Language.TURKISH,
    Language.INDONESIAN,
    Language.TAGALOG,
    Language.VIETNAMESE,
    Language.ROMANIAN,
    Language.CATALAN,
    Language.AFRIKAANS,
    Language.SLOVENE,
    Language.CROATIAN,
    Language.ESTONIAN,
    Language.HUNGARIAN,
    Language.LITHUANIAN,
    Language.LATVIAN,
    Language.RUSSIAN,
    Language.UKRAINIAN,
    Language.BULGARIAN,
    Language.ARABIC,
    Language.PERSIAN,
    Language.HEBREW,
    Language.GREEK,
    Language.HINDI,
    Language.THAI,
    Language.CHINESE,
    Language.JAPANESE,
    Language.KOREAN,
)

URL_PATTERN = re.compile(r"https?://\S+|t\.co/\S+")
MENTION_PATTERN = re.compile(r"@\w+")

# Relative distance between the two most likely languages below which the
# detection counts as ambiguous. Chosen on the full dataset: tighter values
# drop genuine English short messages, looser ones keep obvious non-English.
MINIMUM_RELATIVE_DISTANCE = 0.25


@dataclass(frozen=True)
class EnglishFilterReport:
    """Counts of the English filter stage, for honest reporting."""

    total: int
    retained: int
    filtered: int
    no_signal: int
    filtered_by_language: Mapping[str, int]

    @property
    def filter_rate(self) -> float:
        return self.filtered / self.total if self.total else 0.0


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    return (
        LanguageDetectorBuilder.from_languages(*LANGUAGES)
        .with_minimum_relative_distance(MINIMUM_RELATIVE_DISTANCE)
        .build()
    )


def detect_language(text: str) -> str | None:
    """Name the confident language of a text, or None when there is no signal.

    Handles and links are stripped before detection. None means the detector
    could not confidently separate the top languages (a link- or emoji-only
    message, a very short one); it is the keep signal, not an error.
    """
    stripped = MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", text))
    language = _detector().detect_language_of(stripped)
    return language.name if language is not None else None


def filter_english(
    interactions: Sequence[Interaction],
) -> tuple[tuple[Interaction, ...], EnglishFilterReport]:
    """Keep Interactions whose opening Customer Message is English.

    Returns the retained Interactions in input order and a report with
    retained, no-signal, and filtered volume by detected language.
    """
    retained: list[Interaction] = []
    filtered_by_language: dict[str, int] = {}
    no_signal = 0
    for interaction in interactions:
        language = detect_language(interaction.opening_message.text)
        if language is None:
            no_signal += 1
            retained.append(interaction)
        elif language == Language.ENGLISH.name:
            retained.append(interaction)
        else:
            filtered_by_language[language] = filtered_by_language.get(language, 0) + 1
    report = EnglishFilterReport(
        total=len(interactions),
        retained=len(retained),
        filtered=len(interactions) - len(retained),
        no_signal=no_signal,
        filtered_by_language=dict(
            sorted(filtered_by_language.items(), key=lambda item: (-item[1], item[0]))
        ),
    )
    return tuple(retained), report
