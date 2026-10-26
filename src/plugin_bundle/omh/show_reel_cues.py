"""Motion-production admission shared by scoring and standalone awareness.

A format noun is not a request to author that format. `show-reel` is admitted
only by a request that opens on producing one: the first sentence, after the
courtesy openers (`request_lead.COURTESY_OPENERS`), is a verb from a closed
list, at most one indefinite determiner, at most two modifiers from a closed
list, one of the skill's own noun phrases as whole words, and then nothing, a
comma or colon, or a word from `_FOLLOWERS`. A film named as the object of
another request ("make sure the launch film is uploaded", "write a tweet
announcing the launch film"), a causative ("make the launch film shorter",
"make our showreel public") or a bare noun ("showreel") is not one. Explicit
invocation of the skill by name is the router's, not this reader's.

Stdlib only; this leaf imports no catalog or router.
"""
from __future__ import annotations

import re
import unicodedata

from .reference_regions import executable_routing_text
from .request_lead import after_courtesy

SHOW_REEL_PHRASES = (
    "show-reel", "show reel", "showreel", "product showcase video",
    "animated product film", "motion showcase", "animated portfolio",
    "launch film", "product teaser video",
)
_LEAD_VERBS = frozenset({"animate", "build", "create", "cut", "design", "edit", "make", "produce", "write"})
# Definite and possessive determiners are refused: "make the launch film
# shorter" is a causative over an existing film, not a request to author one.
_DETERMINERS = frozenset({"a", "an"})
_MODIFIERS = frozenset({
    "animated", "bilingual", "brief", "cinematic", "multilingual", "new", "quick", "short", "vertical",
    "arabic", "chinese", "english", "french", "german", "hindi", "italian", "japanese", "korean",
    "portuguese", "spanish", "vietnamese",
})
# `-second`/`-minute` durations such as "60-second" or "2-minute".
_DURATION = re.compile(r"\d+-(?:second|sec|minute|min)")
# "and", "in" and "with" are left out: "make the show reel in the README link
# work" and "make the launch film with subtitles the default" continue a
# causative, not a brief.
_FOLLOWERS = frozenset({"about", "covering", "featuring", "for", "from", "of", "showcasing", "showing", "that", "using", "which"})
# A second request chained on in the same sentence ("make a launch film,
# then deploy the landing page", "... and post it to Slack") belongs to
# whichever skill owns the chain, not to this one.
_CHAIN_WORDS = frozenset({"afterward", "afterwards", "also", "then"})
_CHAINED_VERBS = frozenset({
    "add", "build", "create", "deploy", "do", "email", "file", "fix", "make", "open", "post", "publish",
    "review", "schedule", "send", "share", "submit", "translate", "update", "upload", "write",
})
_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)*|[,:]")
_NOUN_PHRASES = tuple(sorted({tuple(_WORD.findall(phrase)) for phrase in SHOW_REEL_PHRASES}, key=len, reverse=True))


def _fold(text: str) -> str:
    # The router's normalized_phrase uses NFKD without combining marks.
    # Apply that same fold here, then recompose Hangul, so the raw plugin
    # input and its router-normalized form read the same.
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return unicodedata.normalize("NFKC", "".join(c for c in decomposed if not unicodedata.combining(c)))


# The ko/ja/zh pack nouns keep their own production word order, anchored at
# the start, so a translation or inspection of the noun is not production.
# Case particles mark the Korean/Japanese adverbials; they cannot consume a
# different object.
_NATIVE_PRODUCTION = re.compile(_fold(
    r"^제품\s*쇼케이스\s*영상(?:을|를)?\s*"
    r"(?:(?!(?:참고|바탕))[a-z0-9가-힣-]{1,24}로\s*){0,4}"
    r"(?:만들어\s*(?:줘|주세요)|제작해\s*(?:줘|주세요))(?=$|[\s.!?。！？,、;])"
    r"|^製品ショーケース動画を\s*(?:(?!参考)[a-z0-9一-龯ァ-ヶー-]{1,24}で\s*){0,4}"
    r"(?:作って|制作して)(?:ください)?(?=$|[\s.!?。！？,、;])"
    r"|^(?:请|請)?\s*(?:帮我|幫我)?\s*(?:制作|製作)\s*"
    r"(?:[一二两兩三四五六七八九十\d]+[个個部段支])?\s*产品展示短片"
    r"(?=$|[\s.!?。！？,、;]|为|爲|来|來|用于|用於)"
))


def _english_production_lead(sentence: str) -> bool:
    words = _WORD.findall(sentence)
    if not words or words[0] not in _LEAD_VERBS:
        return False
    index = 1
    if index < len(words) and words[index] in _DETERMINERS:
        index += 1
    # The noun is tried before each modifier is taken, so "animated" can open
    # "animated product film" as well as qualify "launch film".
    for _ in range(3):
        for phrase in _NOUN_PHRASES:
            if tuple(words[index:index + len(phrase)]) == phrase:
                rest = words[index + len(phrase):]
                if rest and rest[0] not in {",", ":"} and rest[0] not in _FOLLOWERS:
                    return False
                return not any(
                    word in _CHAIN_WORDS or (word == "and" and following in _CHAINED_VERBS)
                    for word, following in zip(rest, [*rest[1:], ""])
                )
        if index >= len(words) or not (words[index] in _MODIFIERS or _DURATION.fullmatch(words[index])):
            return False
        index += 1
    return False


def show_reel_request_admitted(message: str) -> bool:
    """Admit only a request that leads with producing a show reel."""
    folded = _fold(executable_routing_text(message)).strip()
    text = after_courtesy(folded)
    sentence, *ending = re.split(r"([.!?;。！？])(?:\s|$)", text, maxsplit=1)
    # A first sentence asked, not told ("make a show reel? no, just send the
    # old one", "make a launch film, right?") is a question about one, unless
    # a courtesy opener put it as a request ("can you make a show reel?").
    asked = bool(ending) and ending[0] in "?？" and text == folded
    return (not asked and _english_production_lead(sentence)) or bool(_NATIVE_PRODUCTION.search(text))


def show_reel_request_phrases(message: str, phrases: list[str]) -> list[str]:
    """Awareness phrase-reader interface; admission is the evidence, not a noun."""
    del phrases
    return ["request:show_reel_production"] if show_reel_request_admitted(message) else []
