"""Whether a message asks the agent to spend the person's own money: one rule for the router and the route hint.

`spend-gate` phrases also name the TOPIC, the SUBJECT or a LABEL of a sentence
that asks for something else: a bug report ("pay with my card in safari is
broken"), a product idea ("pay with my card would be nice to have in v2"), a
comparison ("pay with my card or paypal"), button copy ("buy it for me and
wishlist are the two CTAs"), an article, a ticket, an idiom ("pay a visit",
"buy time"), someone else's payment, and the person's own story. Four review
rounds found each of those dispatching the skill or producing its route hint,
and every list of forbidden verbs, nouns or frames was outrun by the next one.

So the rule is closed-world and deliberately conservative: it requires what a
request has, rather than forbidding what a statement has.
`spend_request_clause(text, phrases)` returns the request clause -- the
sentence's first words after filler and courtesy, read Unicode-aware, up to
the first comma, colon, question mark, "and", "if", "when", "which", "about"
and the like -- only when all of this holds, and "" otherwise:
- The message is marked as a request outside the gate's own phrase: "please",
  "pls", "plz", "kindly", "can you", "could you", "would you", "will you", "go
  ahead (and)", "i need you to", "i want you to" or "hermes" among the
  courtesy openers before the lead (`request_lead.COURTESY_OPENERS`, which
  the other readers of a request's opening strip too); "please" in the
  clause; or an amount or "for me" as the spend's object, outside the
  gate's phrases and before "at", "for", "from", "in", "on" or "to"
  ("transfer 40 dollars", "pay with my card $40 for the refill", "buy the
  cheapest one for me", "pay for me with my card") -- not "pay with my card
  for $0 orders is broken" or "book the flight with my card for me never
  shows up". "for me" right after one of the gate's phrases, or inside one
  ("buy it for me"), is not a marker (`_has_request_marker`).
  An unmarked imperative ("pay with my card for the hotel") routes as before
  this skill existed -- an accepted recall cost. Two clauses are marked
  without one: naming the gate as a command ("use spend gate") asks for it
  outright, and the prompt of a loop on a timer
  (`hermes_session_commands.native_loop_prompt`, read when the sentence's own
  clause is not admitted) is an instruction the loop runs ("/loop 1h buy the
  tickets with my card if they drop under $50").
- The clause opens with a spend addressed to the agent ("buy", "pay",
  "order", "book", "renew", "place the order", "complete the checkout", "use
  my card", ...), not an idiom ("pay attention to the spend cap", "pay a
  visit"). "send" and "transfer" also need money, an amount, or the person's
  own card or wallet in the sentence; a reconcile question ("did the payment
  go through") needs the agent to be the one who paid ("after you clicked
  pay"); naming the gate as a command ("use spend gate") opens a clause too.
- One of the given phrases occurs in the clause, and every occurrence is
  followed only by what may follow a request: the end of the clause (but not
  "or", which makes a comparison); "for", "on", "at", "from", "to" or "in";
  "with" followed by the person's own card, wallet or account or by an
  amount; an amount; "please", "now", or "right away".
- A cap or a wallet phrase said with no word of paying or buying anywhere in
  the message is its everyday sense, not a spend ("can you check out the
  spending cap for the city council budget" asks to look at a budget):
  `SPEND_EVERYDAY_SENSE_PHRASES` and `SPEND_EVERYDAY_SENSE_ANCHORS`, the same
  pair the router's `EVERYDAY_SENSE_PHRASES` entry for the gate reads. A
  loop's prompt and naming the gate as a command are exempt, as they are
  from the marker (`spend_gate_named_as_command`, which the router reads too).

A Korean request puts its verb last, so its half of the rule
(`_korean_admitted`) reads the request clause from its end
(`korean_request`, which also skips greetings, labels, quotes and a negated
clause): it is admitted when a request ending ("-해줘", "-해 주세요",
"-해줄래?", "부탁해") closes a last predicate that spends -- "결제" (pay),
"송금" (remit), "주문" (order), "사줘" (buy it for me) and the rest of
`KOREAN_SPEND_NOUNS`, but not processing one ("결제 처리해줘") -- or a
booking, a top-up, a subscription, a renewal or a donation of something
one pays for, with a price, money, the person's wallet or account, or a
card as the means in its own clause, or a send whose object is money --
and the clause says one of `SPEND_GATE_KOREAN_PHRASES`. "결제 시스템 설계해줘"
(design a payment system) and "결제 버그 고쳐줘" (fix a payment bug) end on
design and fix, "결제했어" (I paid) and "결제하지 마" (do not pay) end on no
request, "식당 예약해줘" (book a restaurant) books with no money in it, and
"결제 금액 10,000원 영수증 이메일로 보내줘" sends a receipt.

Everything else returns to the router's base behaviour: the router withdraws
the skill (so no phrase or word of it counts) and the route hint and the
awareness markers stay silent. Nothing here lets the gate take a request
another skill's guard holds; an admitted clause only lets the skill's own
phrases score as usual -- except an admitted loop prompt, for which the chat
router dispatches the gate before any scoring (`SPENDING_LOOP_REASON` in
`omh.routing.chat`): a loop that would spend is never answered with `/loop`.

One phrase table, `SPEND_GATE_PHRASES`, its Korean phrases included, is the
skill's trigger list: the router (`omh.routing.recommend`) passes it to
`spend_request_clause`, and the awareness route hint and its markers
(`awareness.py`) pass it to `spend_request_phrases`, so a request one of them
admits the others admit too.
The bundle may not import `omh.routing`, so the everyday-sense pair lives here
and the router imports it. Stdlib only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .boundary_phrase import contains_boundary_phrase
from .hermes_session_commands import native_loop_prompt
from .korean_request import SOFTENERS, KoreanRequest, composed, korean_request
from .request_lead import courtesy_opener_length

# The Korean half of the clause rule (`_korean_admitted`). A Korean request
# ends on its verb, so it is read from the end (`korean_request`): the request
# clause's last predicate, closed by a request ending ("-해줘", "-해 주세요",
# "-해줄래?", "부탁해"), which marks it as a request the way "please" does in
# English. "결제 시스템 설계해줘" and "결제 버그 고쳐줘" end on design and fix;
# "내 카드로 결제해줘" ends on pay.
#
# 하다 verbs that move the person's money whatever their object: pay, remit,
# transfer, deposit, pay out, pay a bill, purchase (twice), order, book a
# ticket. A compound that ends in one is the same verb: "선결제" (prepay),
# "재주문" (reorder), "해외송금" (remit abroad). Processing one ("결제
# 처리해줘", "주문 처리해줘") is not paying, as "process the order" is no spend
# in English.
KOREAN_SPEND_NOUNS = ("결제", "송금", "이체", "입금", "지불", "납부", "구매", "구입", "주문", "예매")
KOREAN_SPEND_NON_LIGHT_VERBS = frozenset({"처리"})
# 하다 verbs whose object is often not money -- a phone is charged, a
# restaurant booked, a channel subscribed to, a token renewed -- which count
# only when the person's money pays for what they book, top up, subscribe to,
# renew, sponsor or donate (`_korean_pays`).
KOREAN_MONEY_CUED_NOUNS = ("충전", "예약", "구독", "갱신", "후원", "기부")
# Other verbs, as the word before the request ending: buy ("사줘"); and send,
# plain or honorific ("보내줘", "보내드려 줘"), which counts only when what it
# sends is money (`_korean_sends_money`).
KOREAN_SPEND_FORMS = frozenset({"사"})
KOREAN_MONEY_CUED_FORMS = frozenset({"보내", "보내드려"})
# An amount: digits with a currency, as a whole word or with what may follow
# an amount ("3만원짜리", "5만원어치", "50만원을", "20만원으로"). "3원칙" is a
# rule, and "1억 건" a count.
_KOREAN_AMOUNT = re.compile(
    r"(?:[$€£¥₩]\s?\d[\d,.]*|\d[\d,.]*\s?(?:만|천|백|억)?\s?(?:원|달러|불|엔|유로|위안|파운드|usd|krw|eur|jpy))"
    r"(?:짜리|어치|씩|을|를|이|가|은|는|도|만|으로|로|에|까지|정도|쯤|가량)?"
)
# An amount that bounds, scales or converts something is no price: "1만원
# 이하 상품", "1000원 단위로", "환율 1400원 기준으로".
_KOREAN_NOT_A_PRICE = frozenset(
    {"이하", "이상", "미만", "초과", "넘는", "넘으면", "넘게", "단위", "단위로", "기준", "기준으로", "할인", "환율"}
)
_KOREAN_MONEY_WORDS = frozenset({"돈", "돈을", "돈도", "돈만", "돈좀"})
# The person's own money as a source: "내 지갑에서", "내 계좌에서".
_KOREAN_MONEY_SOURCES = ("지갑에서", "계좌에서")
# A card is money only as the means: a word ending "카드로" or "카드에서"
# ("내 카드로", "제 법인카드로") in the verb's own clause. "제 카드 정보", "내
# 카드 목록" and "제휴카드 혜택" name a card, not a payment; "결제 내역 카드로
# 정리해서 보내줘" puts the card on organizing.
_KOREAN_CARD_MEANS = ("카드로", "카드에서", "카드로만")
# What a send's object is when it is not money, inside any word ("거래내역"):
# "영수증 이메일로 보내줘", "3만원짜리 주문 목록 보내줘", "내 계좌에서 거래내역
# 보내줘".
_KOREAN_SENT_DOCUMENTS = (
    "견적", "영수증", "보고서", "리포트", "알림", "내역", "목록", "문서", "메일", "이메일", "파일", "자료",
    "링크", "사진", "메시지", "문자", "공지", "초안", "데이터", "명세서", "청구서", "계약서", "주문서",
)
# What a booking, a renewal, a subscription or a top-up names when nobody
# pays for it, inside any word: "5달러 이상 후원자 명단 갱신해줘", "가격 5000원으로
# 갱신해줘", "내 카드 실적 리포트 구독해줘", "우리 카드 게임 대회 예약해줘".
_KOREAN_UNPAID_OBJECTS = (
    *_KOREAN_SENT_DOCUMENTS,
    "명단", "가격", "정보", "혜택", "실적", "게임", "대회", "상담", "일정", "설정", "단위", "기준", "환율",
)
# What stands between a verb and its object without being the object: whom
# it goes to and how ("엄마한테", "계좌로", "카카오페이로"), a determiner, and
# the softeners.
_KOREAN_SEND_PASSING = ("한테", "에게", "께", "에", "으로", "로", "까지")
_KOREAN_DETERMINERS = frozenset({"이", "그", "저", "다", "전부", "모두"})
# The gate's Korean phrases: each verb above, 사 as said with its request
# ending (one syllable is no phrase), and the Korean "with my card", "from my
# wallet" and "from my account". Book (예약) and send (보내) are
# automation-blueprint's phrases too, a sharing declared in
# `omh.skills.trigger_collisions`: here, like every phrase of the gate, they
# score only inside a request this rule admitted. An admitted Korean request
# always says one of these phrases (`_korean_admitted`), so the router scores
# every request this rule admits. They live here, beside the English table,
# and not in `routing/trigger_packs/ko.json`, for the reason the English
# table does: the rule reads them, and the bundle cannot read a pack.
SPEND_GATE_KOREAN_PHRASES = (
    "결제",
    "송금",
    "이체",
    "입금",
    "지불",
    "납부",
    "구매",
    "구입",
    "주문",
    "예매",
    "사줘",
    "사 줘",
    "사주세요",
    "사 주세요",
    "충전",
    "예약",
    "구독",
    "갱신",
    "후원",
    "기부",
    "보내",
    "카드로",
    "지갑에서",
    "계좌에서",
)

# `spend-gate`'s trigger list, word for word (the catalog reads it from here).
SPEND_GATE_PHRASES = (
    "spend-gate",
    "spend gate",
    "pre-spend",
    "pre-spend card",
    "spend cap",
    "spending cap",
    "spend limit",
    "spending limit",
    "spend request",
    "spend approval",
    "approve the spend",
    "approve this purchase",
    "approve the purchase",
    "complete the purchase",
    "complete the checkout",
    "buy this for me",
    "buy it for me",
    "place the order",
    "place an order",
    "pay with my card",
    "pay with my saved card",
    "with my saved card",
    "with my card",
    "with my credit card",
    "use my card",
    "use my credit card",
    "send money",
    "pay again",
    "click pay again",
    "http 402",
    "402 payment required",
    "402 challenge",
    "pay per request",
    "pay-per-request",
    "paid endpoint",
    "paid data endpoint",
    "wallet transfer",
    "from my wallet",
    "money on my behalf",
    "spend on my behalf",
    "pay on my behalf",
    "buy on my behalf",
    "retry the payment",
    "retry the purchase",
    "retry the charge",
    "did the payment go through",
    "did the charge go through",
    "pay for the booking",
    *SPEND_GATE_KOREAN_PHRASES,
)

_CURRENCY_SYMBOLS = "$€£¥₩"
# Unicode-aware words with apostrophes kept, amounts with their currency
# symbol, and clause punctuation.
_TOKEN = re.compile(rf"[{_CURRENCY_SYMBOLS}]\d[\d.,]*|\w+(?:'\w+)*|[,;:.!?]")

# A request is marked as one, outside the gate's own phrase: one of these in the
# words before the lead, "please" in the clause, or an amount or "for me" as
# the spend's object (`_has_request_marker`). "Pay with my card in safari is
# broken" has the phrase and none of these. Every one is a courtesy opener
# (`request_lead.COURTESY_OPENERS`). The other openers -- a greeting ("hey",
# "ok"), "so", "just", "now", "let's", "we need to", "can we", "omh" -- are
# stripped before the lead but mark nothing: "so buy it for me" is unmarked.
SPEND_REQUEST_MARKERS = (
    ("please",),
    ("pls",),
    ("plz",),
    ("kindly",),
    ("can", "you"),
    ("could", "you"),
    ("would", "you"),
    ("will", "you"),
    ("go", "ahead"),
    ("i", "need", "you", "to"),
    ("i", "want", "you", "to"),
    ("hermes",),
)
_FILLER_PUNCTUATION = frozenset({",", ";", ":", ".", "!", "?"})
# Where the request clause ends. The gate's phrases count only before it, and
# one may follow a phrase ("please buy it for me when it is back in stock").
_CLAUSE_BOUNDARIES = frozenset(
    {
        ",", ";", ":", ".", "!", "?",
        "about", "after", "and", "because", "before", "but", "if", "once", "so", "then", "unless", "until",
        "when", "which", "while",
    }
)
# Leads that ask the agent to spend.
SPEND_LEADS = (
    ("buy",),
    ("pay",),
    ("purchase",),
    ("order",),
    ("book",),
    ("top", "up"),
    ("renew",),
    ("subscribe",),
    ("tip",),
    ("donate",),
    ("check", "out"),
    ("checkout",),
    ("place", "the", "order"),
    ("place", "an", "order"),
    ("complete", "the", "purchase"),
    ("complete", "the", "checkout"),
    ("approve", "the", "spend"),
    ("approve", "the", "purchase"),
    ("approve", "this", "purchase"),
    ("use", "my", "card"),
    ("use", "my", "credit", "card"),
    ("use", "my", "debit", "card"),
    ("click", "pay"),
)
# Leads whose object is often not money; they also need a money cue.
SPEND_MONEY_CUED_LEADS = (("send",), ("transfer",))
_MONEY_CUES = (
    ("money",),
    ("my", "card"),
    ("my", "credit", "card"),
    ("my", "wallet"),
)
_CURRENCY_WORDS = frozenset(
    {"bucks", "cents", "dollar", "dollars", "eur", "euro", "euros", "gbp", "jpy", "krw", "pound", "pounds", "usd", "won", "yen"}
)
# A reconcile question is the gate's own scenario only when the agent paid.
SPEND_RECONCILE_LEADS = (
    ("did", "the", "payment", "go", "through"),
    ("did", "the", "charge", "go", "through"),
)
_AGENT_PAID_CUES = (
    ("you", "clicked", "pay"),
    ("you", "paid"),
    ("you", "bought"),
    ("you", "ordered"),
    ("you", "booked"),
    ("you", "charged"),
)
# Naming the gate as a command is asking for it; its artifact's name is not
# ("what is a pre-spend card" asks a question).
SPEND_GATE_NAME_LEADS = (("use", "spend", "gate"), ("run", "spend", "gate"))
# "pay" used as an idiom, not a request: "please pay attention to the spend cap".
_PAY_IDIOMS = (("attention",), ("close", "attention"), ("a", "visit"))

# What may follow a gate phrase inside the request clause.
_ALLOWED_PREPOSITIONS = frozenset({"at", "for", "from", "in", "on", "to"})
# Where the spend's object ends for an amount or "for me" to mark a request
# (`_has_request_marker`). The same six words today, but its own list: what may
# follow a phrase and where an object ends are separate rules. "with" and "by"
# do not end it: "pay the 402 challenge with $5 of my credits", "top up my
# openai credits by $50".
_MARKER_OBJECT_STOPS = frozenset({"at", "for", "from", "in", "on", "to"})
_ALLOWED_ADVERBS = (("please",), ("now",), ("right", "away"))
_OWN_MONEY_AFTER_WITH = (
    ("my", "card"),
    ("my", "saved", "card"),
    ("my", "credit", "card"),
    ("my", "debit", "card"),
    ("my", "wallet"),
    ("my", "account"),
)

# "congress raised the spending cap again", "the speed and spending limit
# rules in this city" and "the photo fell out from my wallet" move no money
# for the person; each phrase keeps the gate only beside a word of paying or
# buying. Six verbs that open a spend the clause rule admits are anchors too:
# "book", "donate", "renew", "subscribe", "tip" and "top" (up), so "please
# renew the domain from my wallet" keeps its route. The other leads' verbs
# are not: "can you check out the spending cap for the city council budget"
# asks to look at a budget. Nor is the gate's own word "gate", which counted
# anywhere in the message: "could you check out the spending limit at the
# airport gate" dispatched the gate. Naming the gate as a command ("run spend
# gate on my spending limit") is exempt from the check instead
# (`spend_gate_named_as_command`). "agent" and "hermes" are not anchors: the
# router's tokenizer drops both. The router reads this pair as the gate's
# `EVERYDAY_SENSE_PHRASES` entry (`omh.routing.policy`).
SPEND_EVERYDAY_SENSE_PHRASES = ("spend cap", "spending cap", "spend limit", "spending limit", "from my wallet")
SPEND_EVERYDAY_SENSE_ANCHORS = frozenset(
    {
        "402", "api", "approve", "book", "bot", "buy", "buying", "buys", "card", "charge", "charged",
        "checkout", "credits", "dollars", "donate", "endpoint", "euros", "funds", "money", "order",
        "orders", "pay", "paying", "payment", "payments", "purchase", "purchases", "renew", "send",
        "subscribe", "subscription", "tip", "top", "transfer", "usd",
    }
)
# Every anchor is one run of letters and digits, so this finds each one the
# router's tokens hold, around an apostrophe, a hyphen or an underscore alike.
_LETTER_AND_DIGIT_RUN = re.compile(r"[^\W_]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _without_filler(tokens: list[str]) -> list[str]:
    """`tokens` after the courtesy openers and punctuation at their start."""
    rest = list(tokens)
    while rest:
        if rest[0] in _FILLER_PUNCTUATION:
            rest = rest[1:]
            continue
        opener = courtesy_opener_length(rest)
        if not opener:
            break
        rest = rest[opener:]
    return rest


def _clause(tokens: list[str]) -> list[str]:
    clause: list[str] = []
    for token in tokens:
        if token in _CLAUSE_BOUNDARIES:
            break
        clause.append(token)
    return clause


def _says(tokens: list[str], cues: Iterable[tuple[str, ...]]) -> bool:
    return any(_phrase_ends(tokens, cue) for cue in cues)


def _starts_with(tokens: list[str], words: tuple[str, ...]) -> bool:
    return tuple(tokens[: len(words)]) == words


def _is_amount(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if tokens[0][0] in _CURRENCY_SYMBOLS:
        return True
    return tokens[0][0].isdigit() and len(tokens) > 1 and tokens[1] in _CURRENCY_WORDS


def _names_money(tokens: list[str]) -> bool:
    if _says(tokens, _MONEY_CUES):
        return True
    return any(_is_amount(tokens[index:]) for index in range(len(tokens)))


def _spend_lead(tokens: list[str]) -> bool:
    for lead in (*SPEND_LEADS, *SPEND_MONEY_CUED_LEADS):
        if _starts_with(tokens, lead):
            if lead == ("pay",) and any(_starts_with(tokens[1:], idiom) for idiom in _PAY_IDIOMS):
                return False
            if lead in SPEND_MONEY_CUED_LEADS:
                return _names_money(tokens)
            return True
    if any(_starts_with(tokens, lead) for lead in SPEND_RECONCILE_LEADS):
        return _says(tokens, _AGENT_PAID_CUES)
    return any(_starts_with(tokens, lead) for lead in SPEND_GATE_NAME_LEADS)


def _everyday_sense(folded: str) -> bool:
    """A cap or wallet phrase with no word of paying anywhere in the message.

    The router asks `everyday_sense_phrase_unanchored` the same question of
    the same pair. It reads a phrase as said where its first word starts a
    word even when its last runs on ("spending capacity"), and this reads it
    only where it ends a word too (`contains_boundary_phrase`), so it can find
    a phrase the router finds and never one the router does not: where the
    two differ, this one admits. The router's other condition, a word of the
    message's own beside the phrase, always holds by the time this is asked:
    every lead it is asked about but "check out" holds an anchor, and "check"
    is such a word. (A loop's prompt and the gate named as a command are not
    asked about.)
    """
    if not contains_boundary_phrase(folded, SPEND_EVERYDAY_SENSE_PHRASES):
        return False
    return not set(_LETTER_AND_DIGIT_RUN.findall(folded)) & SPEND_EVERYDAY_SENSE_ANCHORS


def _continuation_allowed(rest: list[str]) -> bool:
    # Anything else after a phrase means it is not the request: "or" ("pay
    # with my card or paypal") makes a comparison, and a verb or a noun makes
    # the phrase a subject or a label.
    if not rest:
        return True
    if rest[0] in _CLAUSE_BOUNDARIES or rest[0] in _ALLOWED_PREPOSITIONS or _is_amount(rest):
        return True
    if rest[0] == "with":
        return _is_amount(rest[1:]) or any(_starts_with(rest[1:], own) for own in _OWN_MONEY_AFTER_WITH)
    return any(_starts_with(rest, adverb) for adverb in _ALLOWED_ADVERBS)


def _phrase_ends(clause: list[str], phrase: tuple[str, ...]) -> list[int]:
    return [
        index + len(phrase)
        for index in range(len(clause) - len(phrase) + 1)
        if tuple(clause[index : index + len(phrase)]) == phrase
    ]


def _has_request_marker(prefix: list[str], clause: list[str], spans: list[tuple[int, int]]) -> bool:
    if any(marker == tuple(prefix[index : index + len(marker)]) for marker in SPEND_REQUEST_MARKERS for index in range(len(prefix))):
        return True
    if "please" in clause:
        return True
    # An amount or "for me" marks a request only as the spend's object: read
    # outside the gate's phrases and before `_MARKER_OBJECT_STOPS` ("transfer
    # 40 dollars from my wallet", "pay with my card $40 for the refill", "buy
    # the cheapest one for me with my card") -- not "pay with my card for $0
    # orders is broken" or "pay with my card at checkout for me is the
    # default". "for me" counts only after a word of its own since the last
    # phrase, the lead ("pay for me with my card") or an object ("buy the
    # cheapest one for me"). Right after a phrase it does not, whatever came
    # before the phrase ("book the flight with my card for me never shows
    # up", "use my credit card for me"); inside a phrase ("buy it for me") it is
    # the phrase's own words.
    own_words = 0
    for index in range(len(clause)):
        if any(start <= index < end for start, end in spans):
            own_words = 0
            continue
        if clause[index : index + 2] == ["for", "me"]:
            return own_words > 0
        if clause[index] in _MARKER_OBJECT_STOPS:
            return False
        if _is_amount(clause[index:]):
            return True
        own_words += 1
    return False


def _request(text: str, phrases: Iterable[str]) -> tuple[str, list[str]]:
    phrases = tuple(phrases)
    admitted = _admitted(text, phrases, by_loop=False)
    if admitted[0]:
        return admitted
    # A loop's prompt is marked by the loop itself. That waiver also admits a
    # loop that only watches a fault ("/loop 1h pay with my card on staging
    # returns 500, ping me when it is fixed"), which then reaches the gate:
    # a disclosed residual, accepted because the other direction would let a
    # loop that spends skip the gate, while this one moves no money. No list
    # of status words is kept against it; earlier rounds rejected that. The
    # everyday-sense check is waived on the same terms: "loop every hour and
    # check out from my wallet when the price drops" may be a purchase. So is
    # it for the gate named as a command, which asks for the gate whatever
    # cap it names ("run spend gate on my spending limit").
    korean = _korean_admitted(text, phrases, by_loop=False)
    if korean[0]:
        return korean
    prompt = native_loop_prompt(text)
    if not prompt:
        return "", []
    admitted = _admitted(prompt, phrases, by_loop=True)
    return admitted if admitted[0] else _korean_admitted(prompt, phrases, by_loop=True)


def _korean_admitted(text: str, phrases: tuple[str, ...], *, by_loop: bool) -> tuple[str, list[str]]:
    """The English rule's Korean mirror: the request clause when its last predicate is a spend.

    The request ending is the marker (a loop's prompt is marked by the loop,
    as in English); the predicate is a spend verb, or a money-cued one beside
    the person's money; and one of `phrases` is said in the clause -- always
    so for an admitted request when `phrases` holds
    `SPEND_GATE_KOREAN_PHRASES`. The clause is returned composed (NFC); a
    caller that matches another fold folds it.
    """
    request = korean_request(text, require_ending=not by_loop)
    if request is None:
        return "", []
    predicate = request.predicate
    if predicate in KOREAN_MONEY_CUED_FORMS:
        if not _korean_sends_money(request.words, request.predicate_index):
            return "", []
    elif predicate.endswith(KOREAN_MONEY_CUED_NOUNS):
        if not _korean_pays(request):
            return "", []
    elif not (predicate in KOREAN_SPEND_FORMS or predicate.endswith(KOREAN_SPEND_NOUNS)):
        return "", []
    if KOREAN_SPEND_NON_LIGHT_VERBS & set(request.light_verbs):
        return "", []
    found = [phrase for phrase in phrases if composed(phrase) in request.clause]
    if not found:
        return "", []
    return request.clause, list(dict.fromkeys(found))


def _korean_amount(words: tuple[str, ...], index: int) -> bool:
    """Whether the word at `index`, or it and the next ("5만 원"), is an amount."""
    pair = "".join(words[index : index + 2])
    return bool(_KOREAN_AMOUNT.fullmatch(words[index]) or (index + 1 < len(words) and _KOREAN_AMOUNT.fullmatch(pair)))


def _korean_price(words: tuple[str, ...], index: int) -> bool:
    """Whether the word at `index` is an amount that is a price, not a bound, a unit or a rate."""
    if not _korean_amount(words, index):
        return False
    after = words[index + 1 : index + 3]
    return not (set(after) & _KOREAN_NOT_A_PRICE)


def _korean_object(words: tuple[str, ...], predicate_index: int) -> int:
    """The index of the verb's object: the first word before it that is not whom, how, a determiner or a softener."""
    index = predicate_index - 1
    while index >= 0 and (
        words[index] in SOFTENERS
        or words[index] in _KOREAN_DETERMINERS
        or (words[index].endswith(_KOREAN_SEND_PASSING) and not words[index].endswith(_KOREAN_MONEY_SOURCES))
    ):
        index -= 1
    return index


def _korean_pays(request: KoreanRequest) -> bool:
    """Whether the person's money pays for what a booking, top-up, subscription, renewal or donation names.

    What it names, its object, must be something one pays for, not a list, a
    price, a report or an alert (`_KOREAN_UNPAID_OBJECTS`); and its own clause
    must hold the money: a price ("3만원 충전해줘", "20만원으로 호텔 예약해줘"),
    "돈", the person's wallet or account, or a card as the means ("카드로 호텔
    예약해줘").
    """
    words = request.words
    target = _korean_object(words, request.predicate_index)
    if target >= 0 and any(unpaid in words[target] for unpaid in _KOREAN_UNPAID_OBJECTS):
        return False
    start = max((index + 1 for _noun, index in request.joined if index < request.predicate_index), default=0)
    return any(
        _korean_price(words, index)
        or words[index] in _KOREAN_MONEY_WORDS
        or words[index].endswith(_KOREAN_MONEY_SOURCES)
        or words[index].endswith(_KOREAN_CARD_MEANS)
        for index in range(start, request.predicate_index + 1)
    )


def _korean_sends_money(words: tuple[str, ...], predicate_index: int) -> bool:
    """Whether a send's object is money: a price or "돈", or anything from the person's wallet or account.

    "엄마한테 10만원 보내줘" and "축의금 10만원 카카오페이로 보내줘" send money;
    "결제 금액 10,000원 영수증 이메일로 보내줘" sends a receipt and "매일 아침
    리포트 보내줘" a report.
    """
    index = _korean_object(words, predicate_index)
    if index < 0 or any(document in words[index] for document in _KOREAN_SENT_DOCUMENTS):
        return False
    if _korean_price(words, index) or words[index] in _KOREAN_MONEY_WORDS:
        return True
    return any(word.endswith(_KOREAN_MONEY_SOURCES) for word in words[:predicate_index])


def _admitted(text: str, phrases: Iterable[str], *, by_loop: bool) -> tuple[str, list[str]]:
    folded = text.casefold().strip()
    every = _tokens(folded)
    tokens = _without_filler(every)
    prefix = every[: len(every) - len(tokens)]
    clause = _clause(tokens)
    if not _spend_lead(tokens):
        return "", []
    marked = by_loop or any(_starts_with(tokens, lead) for lead in SPEND_GATE_NAME_LEADS)
    # Phrases are found in the clause; what follows each is read past the
    # clause's end, so "or" right after a phrase is still seen.
    found = [
        (phrase, end - len(words), end)
        for phrase in phrases
        for words in (tuple(_tokens(phrase)),)
        for end in _phrase_ends(clause, words)
    ]
    if not found or not all(_continuation_allowed(tokens[end:]) for _phrase, _start, end in found):
        return "", []
    if not marked and not _has_request_marker(prefix, clause, [(start, end) for _phrase, start, end in found]):
        return "", []
    if not marked and _everyday_sense(folded):
        return "", []
    return " ".join(clause), list(dict.fromkeys(phrase for phrase, _start, _end in found))


def spend_gate_named_as_command(text: str) -> bool:
    """Whether the sentence, after its courtesy openers, opens by naming the gate as a command.

    "run spend gate on my spending limit" asks for the gate whatever cap it
    names, so the router's everyday-sense check for the gate skips it, as the
    clause rule here does.
    """
    tokens = _without_filler(_tokens(text.casefold().strip()))
    return any(_starts_with(tokens, lead) for lead in SPEND_GATE_NAME_LEADS)


def spend_phrase_continues_a_request(text: str, phrase: str) -> bool:
    """Whether `phrase` occurs in `text` followed only by what may follow a request.

    The rule every gate phrase is held to (`_continuation_allowed`): "pay with
    my card for the report", not "a share card with my card collection".
    """
    tokens = _tokens(text.casefold())
    return any(_continuation_allowed(tokens[end:]) for end in _phrase_ends(tokens, tuple(_tokens(phrase))))


def spend_request_clause(text: str, phrases: Iterable[str]) -> str:
    """The request clause when it asks the agent to spend the person's money through one of `phrases`, else ""."""
    return _request(text, phrases)[0]


def spend_request_phrases(text: str, phrases: Iterable[str]) -> list[str]:
    """Which of `phrases` the admitted request says; [] when it is not admitted.

    An English clause says a phrase as whole words, an admitted Korean
    sentence anywhere in it (its particles and endings attach to the word).
    """
    return _request(text, phrases)[1]
