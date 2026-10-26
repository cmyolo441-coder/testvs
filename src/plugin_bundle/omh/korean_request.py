"""Where a Korean request puts its verb, and the ending that makes it a request.

The two safety gates admit an English request by its first words: a lead verb
after the courtesy openers (`request_lead`), with "please", "can you" or the
like marking it as a request. Korean puts the verb last, and the request is
marked by the verb's ending: "-해줘", "-해 주세요", "-해줄래?", or "부탁해"
after the verb's noun. "결제 시스템 설계해줘" ends on 설계 (design); "내 카드로
결제해줘" ends on 결제 (pay). So the Korean mirror of the English lead is the
first sentence's last predicate, read only when a request ending closes it.

`korean_request(text)` returns that predicate, the 하다 verbs of the clauses
joined before it ("마스킹해서 ... 보내줘", "삭제하고 ... 공유해줘"), the
clause's words, and the sentences after it (`later`, for a gate that reads the
whole message), or None when the sentence does not end on a request: a
statement ("결제했어"), a prohibition ("결제하지 마"), a past tense ("송금해
줬어"), or a question that is not itself a request ("결제해줘?"; the asking
forms "-해줄래?" and "-해 줄 수 있어?" are requests). Which predicates admit a
gate is each gate's own closed list (`spend_signals`, `data_share_cues`), and
so is any other verb's joined form ("지우고", "가려서"), which a gate reads
from the words.

What the request is, and what it is not:

- The first sentence after any that hold only a greeting or an address
  ("안녕하세요.", "헤르메스,"), the Korean courtesy openers. A sentence ends at
  ".", "!", "?" or a ":" before a space, so a label's text is not the request
  ("버튼 라벨: 내 카드로 결제해줘" asks for nothing).
- Quoted text is not the request: "이 문장 영어로 번역해줘 \"내 카드로
  결제해줘\"" asks for a translation.
- What comes before a negation the reader knows (`NEGATIONS`: 말고, 않고,
  마시고, 마세요, 마요, 없이) is not the request: in "카드로 결제하지 말고
  견적만 보내줘" the request is "견적만 보내줘" (send only the quote), and in
  "내 카드로 결제 없이 회의실 예약만 해줘" it is "회의실 예약만 해줘". Those
  words are dropped, so no gate reads their verb or their phrases. A negation
  not in the list is read as part of the request.

A predicate is the verb's noun for a 하다 verb ("결제" from "결제해줘", "결제 좀
해줘", "결제 진행해 주세요", "결제 부탁해"), with a particle such as 을/를/까지
taken off, and the word before the ending for any other verb ("보내" from
"보내줘", "사" from "사 주세요"). The light verbs it passed are kept
(`light_verbs`): "결제 처리해줘" processes a payment, which a gate may read
differently from paying.

The text is read composed (NFC): the router folds Hangul into jamo (NFKD) and
the route hint keeps it composed, and both must read one sentence the same
way. `clause` is composed too; a caller that matches its own fold folds it.
Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

# 주다 as the request auxiliary ("do it for me"), in the forms that ask for it.
REQUEST_AUXILIARIES = (
    "줘",
    "줘요",
    "줘라",
    "주세요",
    "주십시오",
    "주라",
    "줄래",
    "줄래요",
    "주실래요",
    "주시겠어요",
    "줄 수 있어",
    "줄 수 있어요",
    "줄 수 있나요",
    "주실 수 있나요",
    "주실 수 있어요",
)
# The forms that ask as a question: "결제해 줄 수 있어?" is a request, while
# "결제해줘?" echoes one. Compared without spaces, as they may be typed.
ASKING_AUXILIARIES = frozenset(
    {"줄래", "줄래요", "주실래요", "주시겠어요", "줄수있어", "줄수있어요", "줄수있나요", "주실수있나요", "주실수있어요"}
)
# 부탁 ("please do") after the verb's noun: "결제 부탁해", "송금 부탁드립니다".
REQUEST_FAVOURS = frozenset({"부탁해", "부탁해요", "부탁할게", "부탁할게요", "부탁합니다", "부탁드려요", "부탁드립니다"})
# Words between a 하다 verb's noun and its ending that change nothing about
# which verb it is: "결제 좀 해줘", "결제 다시 해줘".
SOFTENERS = frozenset({"좀", "다시", "바로", "지금", "먼저", "빨리", "대신", "꼭"})
# 하다 verbs that take their work from the noun before them: "결제 진행해줘"
# pays, "비식별 처리해줘" de-identifies.
LIGHT_VERBS = frozenset({"진행", "처리", "완료"})
# Particles a request leaves on the verb's noun: "결제를 해줘", "결제까지 해줘".
PARTICLES = ("까지", "부터", "을", "를", "도", "만", "은", "는")
# A greeting or an address before the request, the Korean courtesy openers:
# "안녕하세요. 제 카드로 결제해 주세요", "헤르메스, 고객 정보 ...", "저기 혹시 ...".
OPENERS = frozenset(
    {"안녕", "안녕하세요", "안녕하십니까", "하이", "헬로", "헤이", "저기", "저기요", "여보세요", "부탁", "하나만",
     "헤르메스", "헤르메스야", "헤르메스님", "hermes", "omh"}
)
# What negates the clause before it: "결제하지 말고", "결제 말고", "결제하지 않고",
# the honorific "결제하지 마시고", "결제하지 마세요 ...", and "결제 없이" (without
# paying).
NEGATIONS = frozenset({"말고", "말고요", "말고서", "않고", "마시고", "마시구요", "마세요", "마요", "없이"})
# How a 하다 verb's clause joins the next one: "마스킹해서", "삭제하고",
# "처리한 다음에", "마스킹 후"; and how a clause says what comes after it:
# "보내기 전에".
_JOINING_ENDINGS = ("하고서", "하고", "해서")
_AFTER_WORDS = frozenset({"다음", "다음에", "뒤", "뒤에", "후", "후에"})
_BEFORE_WORDS = frozenset({"전", "전에"})

_AUXILIARY = re.compile(
    r"(?:^|\s)(?P<head>\S*?)\s?(?P<aux>"
    + "|".join(
        re.escape(aux).replace(r"\ ", r"\s?") for aux in sorted(REQUEST_AUXILIARIES, key=len, reverse=True)
    )
    + r")$"
)
# Quoted text, which names words rather than asking for them. A single-quoted
# span counts only when it holds Hangul and its marks are no English
# apostrophes: a `'` after a Latin letter or digit opens none ("acme's",
# "it's"), and one before a Latin letter closes none, while a Korean particle
# may follow the closing mark ("'고객 정보'를").
# It is matched on the composed text, so the router's decomposed fold reads
# the same spans as the hint.
_QUOTED = re.compile(
    r"\"[^\"\n]*\"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』|(?<![A-Za-z0-9])'[^'\n]*[가-힣][^'\n]*'(?![A-Za-z])"
)
# A sentence ends at ".", "!", "?" or ":" followed by a space or the end;
# "0.1개", "v2.3" and "10:30" do not end one.
_SENTENCE_END = re.compile(r"[.!?。！？:]+(?=\s|$)")
# What may trail a request without changing it: "결제해줘~", "결제해줘 ㅎㅎ",
# "결제해줘요 :)".
_TRAILING = re.compile(r"[\s~^;:()ㄱ-ㆎ☀-➿\U0001f000-\U0001faff]+$")
# Punctuation inside the sentence that only separates words.
_SEPARATORS = re.compile(r"[,;:\"'“”‘’()\[\]<>]")


@dataclass(frozen=True)
class KoreanRequest:
    """The request clause of a Korean message, read from its end."""

    # The request clause, composed (NFC) and lower-cased, its words joined by
    # single spaces: what a gate reads its phrases in.
    clause: str
    # The clause's words.
    words: tuple[str, ...]
    # The requested verb, and the index of the word that holds it.
    predicate: str
    predicate_index: int
    # The light verbs between the verb's noun and its ending ("처리", "진행").
    light_verbs: tuple[str, ...]
    # The 하다 verbs of the clauses joined before it, with their word index.
    joined: tuple[tuple[str, int], ...]
    # The verbs of "-기 전에" clauses: "보내" from "보내기 전에", "업로드" from
    # "업로드하기 전에".
    before: tuple[str, ...]
    # The sentences after the request's, quoted text removed: what a gate
    # that reads the whole message reads next ("... 보내줘. 급해요.").
    later: tuple[str, ...] = ()


def composed(text: str) -> str:
    """`text` composed (NFC) and lower-cased: how the gates read Korean phrases."""
    return unicodedata.normalize("NFC", text).casefold()


def without_particle(word: str) -> str:
    """`word` without one particle a request leaves on a verb's noun."""
    for particle in PARTICLES:
        if word.endswith(particle) and len(word) > len(particle):
            return word[: -len(particle)]
    return word


def _words(text: str) -> list[str]:
    return _TRAILING.sub("", _SEPARATORS.sub(" ", composed(text))).split()


def _request_sentence(text: str) -> tuple[list[str], bool, tuple[str, ...]]:
    """The words of the first sentence that is more than openers, whether a question mark ends it, and the sentences after it."""
    text = _QUOTED.sub(" ", unicodedata.normalize("NFC", text.strip()))
    start = 0
    while True:
        end = _SENTENCE_END.search(text, start)
        words = _words(text[start : end.start() if end else len(text)])
        while words and words[0] in OPENERS:
            words = words[1:]
        if words or end is None:
            later = tuple(part.strip() for part in _SENTENCE_END.split(text[end.end() :] if end else "") if part.strip())
            return words, bool(end) and ("?" in end.group() or "？" in end.group()), later
        start = end.end()


def _after_negation(words: list[str]) -> list[str]:
    """`words` after the last negated clause: "결제하지 말고 견적만 보내줘" -> 견적만 보내줘."""
    for index in range(len(words) - 1, -1, -1):
        if words[index] in NEGATIONS:
            return words[index + 1 :]
    return words


def _noun_at(words: list[str], index: int, lights: tuple[str, ...] = ()) -> tuple[str, int, tuple[str, ...]]:
    """The 하다 verb's noun at or before `index`, past softeners and light verbs."""
    while index >= 0 and words[index] in SOFTENERS:
        index -= 1
    if index < 0:
        return "", -1, lights
    noun = without_particle(words[index])
    if noun in LIGHT_VERBS and index > 0:
        return _noun_at(words, index - 1, (*lights, noun))
    return noun, index, lights


def _hada_noun(words: list[str], index: int, stem: str) -> tuple[str, int, tuple[str, ...]]:
    """The noun of a 하다 verb whose word at `index` is `stem` plus its ending."""
    if stem:
        return _noun_at([*words[:index], stem], index)
    return _noun_at(words, index - 1)


def _final_predicate(words: list[str]) -> tuple[str, int, tuple[str, ...], str] | None:
    """The requested verb, its word index, its light verbs and its ending; None without a request ending."""
    if words[-1] in REQUEST_FAVOURS:
        noun, index, lights = _noun_at(words, len(words) - 2)
        return (noun, index, lights, words[-1]) if noun else None
    match = _AUXILIARY.search(" ".join(words))
    if match is None or not match.group("head"):
        return None
    head = match.group("head")
    index = len(" ".join(words)[: match.start("head")].split())
    if head.endswith("해"):
        noun, index, lights = _hada_noun(words, index, head[:-1])
        return (noun, index, lights, match.group("aux")) if noun else None
    return head, index, (), match.group("aux")


def _joined_clauses(words: list[str], end: int) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    joined: list[tuple[str, int]] = []
    before: list[str] = []
    for index in range(end):
        word = words[index]
        following = words[index + 1] if index + 1 < len(words) else ""
        if following in _BEFORE_WORDS and word.endswith("기") and len(word) > 1:
            stem = word[:-1]
            before.append(_hada_noun(words, index, stem[:-1])[0] if stem.endswith("하") else stem)
            continue
        ending = next((ending for ending in _JOINING_ENDINGS if word.endswith(ending)), "")
        if not ending and word.endswith("한") and following in _AFTER_WORDS:
            ending = "한"
        if not ending and following in _AFTER_WORDS:
            # A verb's noun before "후", "뒤" or "다음": "마스킹 후", "비식별 처리 후".
            noun, noun_index, _lights = _noun_at(words, index)
            if noun:
                joined.append((noun, noun_index))
            continue
        if ending:
            noun, noun_index, _lights = _hada_noun(words, index, word[: -len(ending)])
            if noun:
                joined.append((noun, noun_index))
    return tuple(joined), tuple(before)


def korean_request(text: str, *, require_ending: bool = True) -> KoreanRequest | None:
    """The request clause's verb when a request ending closes it, else None.

    With `require_ending=False` (a loop's prompt, which the loop itself marks
    as an instruction) a clause that ends on a bare 하다 verb's noun or on
    "-해" is read too, as English waives the request marker there.
    """
    words, question, later = _request_sentence(text)
    words = _after_negation(words)
    if not words:
        return None
    final = _final_predicate(words)
    if final is None and not require_ending:
        last = words[-1][:-1] if words[-1].endswith("해") else words[-1]
        final = (without_particle(last), len(words) - 1, (), "") if last else None
    if final is None:
        return None
    predicate, index, lights, ending = final
    if question and ending.replace(" ", "") not in ASKING_AUXILIARIES:
        return None
    joined, before = _joined_clauses(words, index)
    return KoreanRequest(
        clause=" ".join(words),
        words=tuple(words),
        predicate=predicate,
        predicate_index=index,
        light_verbs=lights,
        joined=joined,
        before=before,
        later=later,
    )
