"""When a sentence asks to redact or pseudonymize a file that is leaving the company.

The owner decided (2026-10-16) that personal data sent to a team inside the
same company stays internal: the review is for data leaving for an outside
party (a vendor, provider, partner, contractor, third party, or an external
service). The gate fails safe: a place is inside only when it is proven
inside. Every word of it must come from four closed tables, read by both
languages -- anchors (a colleague, staging, QA, internal, ...:
`DATA_SHARE_INTERNAL_DESTINATIONS`), units that need an owner (a team, a
department: `DATA_SHARE_INTERNAL_UNITS`), owners (our, my, another, an
internal function such as data, dev, marketing, finance:
`DATA_SHARE_INTERNAL_OWNERS`) and modifiers (the, a; a server or a wiki:
`DATA_SHARE_INTERNAL_MODIFIERS`) -- and it must say an anchor or an owned
unit. Any other word -- a company, a client, a hospital, a regulator, a name
-- keeps the place outside, so "the vendor's team", "the Acme team", "the
data team at Acme" and "외부 연구팀에" keep the review. A caller whose masker
removed quoted, code, link or relayed text from the request says so
(`shielded`); what the removed text named is unread, so no place is then
proven inside.

`data-share-review` takes nothing another skill holds: it has no boost and
stands down no guard. This narrow gate decides only whether the skill may be
scored at all, and whether the awareness route hint may name it. The skill's
phrases ("redact pii", "pseudonymize", ...) also name code, Hermes settings,
legal questions, issue triage, messages, posts and videos, and an everyday
verb ("remove", "strip", "take ... out") as often removes something that is
not personal data at all, so all of these must hold:

- the sentence itself (no later clause) begins, after the courtesy openers
  every reader of a request's opening strips ("hey", "hi,", "pls", "would
  you", "hermes,", ...: `request_lead.COURTESY_OPENERS`), with an inherently
  protective verb: redact, pseudonymize, anonymize, or de-identify. Words are
  read Unicode-aware, so a sentence that opens in Hangul or another script
  has no Latin lead;
- the verb is a command: it is not defined ("de-identify means ...",
  "de-identify is the term ...") and the sentence is not a question;
- that verb's object does not end in a word for something other than data (a
  label, a flag, a script, a pull request, slides, a header row, ...);
- a file-shaped carrier is leaving: an export, dataset, csv, file, logs,
  records, extract, transcript excerpt, corpus, dump or spreadsheet, beside a
  recipient outside the company (a vendor, contractor, agency, researcher,
  the public, a provider, a training set), or beside a verb that sends it
  unless all of these hold (`_sends_stay_inside`): only closed words stand
  from the lead verb to the first send, its object included ("the pii in
  the export before we"; not "the export acme will get", "the csv the bank
  wants", "before acme sees it"), and a word that names whose records the
  data holds (customer, user, patient, client, member, ...) only right
  before a data container or a field name, not possessive and not after
  "for", "to" or "from" ("the customer export", "the patient csv"; not "the
  customer's export", "the export for the customer"); every verb
  that sends it on -- send, share, upload, attach, post, forward, email, cc,
  transfer -- names after its "to" or "with" a place proven inside ("send it
  to the dev team", "share it with a colleague", "upload it to staging"),
  with only the data sent, its particle and remarks between the two ("send
  them over to"; not "send it where acme wants it, to"); a place ends with
  its sentence (".", "!", "?" or ";" before a space), every sentence is read
  for sends, and the sentences after a place hold only remarks up to the
  next send: thanks, please and urgency ("... to the dev team. it's
  urgent."; not "... to the dev team. acme will review it.");
  and "give", "hand" or "pass" with no "to" either goes to the person
  ("give me a summary", "hand it back") or has a first object proven inside
  ("give the dev team a copy"; not "give acme a copy", "give them a copy").
  A send that names no place is leaving, and so is one whose place holds
  any other word ("the data team and Acme", "the data team at Acme", "a
  colleague who now works at Acme"). A request that is `shielded`, or that
  still has quotation marks, has no inside place;
- no word after the openers names a legal question, Hermes (a setting of it,
  or Hermes doing the job: "before hermes uploads it to the vendor"), or a
  storage-wide scan. "hermes," as the address before the request is an
  opener, not a mention: "hermes, redact pii in the export before we send
  it to the vendor" is admitted like "hey, ...".

A Korean request puts its verb last, so its half of the rule
(`_korean_admitted`) reads the first sentence from its end
(`korean_request`), and the request ending ("-해줘", "-해 주세요",
"부탁해") is what marks it as a request. All of this must hold:

- the sentence ends on a request to protect the data or to send it: its last
  predicate is a protective verb ("고객 정보 마스킹해줘") or an outbound one
  ("... 마스킹해서 벤더한테 보내줘");
- the data is protected: an inherently protective verb (익명화, 가명화,
  가명 처리, 비식별화), or a verb that removes something (마스킹, 삭제, 제거,
  블러, 지우-, 가리-, 빼-) with personal data among the three words before
  it ("개인정보 지우고", "고객 정보 마스킹해서"), as the last predicate or in a
  clause joined to it; "사진 배경 마스킹해서 보내줘" removes a background;
- it leaves: a recipient outside the company named as where it goes, in any
  sentence ("벤더한테", "외부에", "협력업체에"; not "업체 목록에서", "외부 로그"
  or "업체별로"), or a verb that sends it, in any sentence (a predicate, a
  "-기 전에" clause, "보낼 파일") unless a place proven inside the company or
  a file or folder of the person's own is named ("스테이징에", "개발팀에",
  "동료한테", "엑셀로") and every place the message names is one
  (`_korean_stays_inside`). A place's words are read back, however many,
  to a step or a joined form of a verb the rule knows ("마스킹해서",
  "지우고", "보내기 전에"); a noun that merely ends in 서 or 고 is read on
  ("경찰서 보안팀에"). Any other word keeps it outside ("외부 팀에", "은행 담당
  부서에", "은행 IT 보안 개발 팀에", "다른 회사 동료에게", "acme qa에"), as
  does an English recipient anywhere ("vendor team에") and a `shielded`
  request. Every word outside the places and before the requested verb is
  a closed word too (`_korean_closed`), the Korean mirror of the English
  span before a send: data containers and field names, a verb the rule
  knows, an inside place word, a determiner or a softener, and a word that
  says whose records the data holds (고객, 회원, 환자, ...) only bare and
  right before a data word ("고객 엑셀에서", "회원 명단"). So a party named
  before the place keeps the review: one acting ("보험사가 원하는"), one
  owning ("보험사의", "고객의"), a purpose ("협력사용", "보험사를 위한"), or
  any other noun ("삼성 고객 파일"). A later sentence of remarks only says
  nothing ("... 개발팀에 보내줘. 급해요."); a later request is read like the
  first, and a later sentence that asks nothing keeps the review when it
  names a place outside or a party acting ("... 보내줘. 보험사가 볼 거야.");
- no word names a legal question, a Hermes setting, a storage-wide scan, or
  something other than data (코드, 함수, 기능, 버그, ...): "개인정보 마스킹
  기능 만들어서 벤더한테 보내줘" ships a feature;
- one of the skill's Korean phrases is said.

Two readers hold this one function: the router (`omh.routing.policy`) and the
awareness route hint here in the plugin bundle. Stdlib only: this ships in the
plugin bundle, which may not import `omh`. Callers pass lower-cased text,
folded the same way as the phrases.
"""
from __future__ import annotations

from collections.abc import Iterable
import re

from .boundary_phrase import contains_boundary_phrase
from .korean_request import LIGHT_VERBS, SOFTENERS, KoreanRequest, composed, korean_request
from .request_lead import after_courtesy

# The Korean half of the rule (`_korean_admitted`). Personal data a Korean
# request names: personal, customer and sensitive information, identifying
# information and identifiers, a resident registration number, a phone
# number, contact details, an account or card number, a real name, an email
# address. These are phrases the router scores.
DATA_SHARE_KOREAN_PERSONAL_DATA = (
    "개인정보",
    "개인 정보",
    "고객 정보",
    "고객정보",
    "민감정보",
    "민감 정보",
    "식별 정보",
    "식별정보",
    "식별자",
    "주민번호",
    "주민등록번호",
    "전화번호",
    "연락처",
    "계좌번호",
    "카드번호",
    "실명",
    "이메일 주소",
)
# Field names that are personal data only in this frame -- removed, and the
# data then leaving -- and everyday words elsewhere: a name, an email, an
# address, a birth date, a user id, a nickname, an employee or student
# number, an IP, "PII" said in English. They admit a request but are no
# phrase of the skill, so one of its phrases must still be said
# (`_korean_admitted`).
DATA_SHARE_KOREAN_FIELD_NAMES = ("이름", "이메일", "주소", "생년월일", "아이디", "닉네임", "사번", "학번", "ip", "pii")
# Verbs that protect data whatever their object: anonymize, pseudonymize,
# de-identify, also as "<noun> 처리" ("익명 처리", "가명 처리", "비식별 처리").
DATA_SHARE_KOREAN_PROTECTIVE_VERBS = frozenset(
    {"익명화", "익명처리", "익명", "가명화", "가명처리", "가명", "비식별화", "비식별처리", "비식별"}
)
# Verbs that protect data only when it is personal data they remove: mask,
# delete, remove, blur, hash, star out ("별표 처리"), and erase, cover and take
# out ("지워줘", "가려줘", "빼줘"), with their joined forms ("지우고", "지운
# 다음에").
DATA_SHARE_KOREAN_REMOVAL_VERBS = frozenset({"마스킹", "삭제", "제거", "블러", "해시", "별표", "지워", "가려", "빼"})
DATA_SHARE_KOREAN_REMOVAL_JOINED = frozenset({"지우고", "지워서", "가리고", "가려서", "빼고", "빼서"})
DATA_SHARE_KOREAN_REMOVAL_BEFORE_AFTER = frozenset({"지운", "가린", "뺀"})
# Verbs that send it: share, forward, upload, provide, take out, attach,
# publish, submit, send, hand over, post, export; with their joined,
# "-기 전에" and future ("보낼 파일", "공유할 거니까") forms.
DATA_SHARE_KOREAN_OUTBOUND_VERBS = frozenset(
    {"공유", "전달", "업로드", "제공", "반출", "첨부", "공개", "제출", "보내", "보내드려", "넘겨", "올려", "내보내", "넘기", "올리"}
)
DATA_SHARE_KOREAN_OUTBOUND_JOINED = frozenset({"보내고", "보내서", "넘기고", "넘겨서", "올리고", "올려서", "내보내고", "내보내서"})
DATA_SHARE_KOREAN_OUTBOUND_FUTURE = frozenset({"보낼", "넘길", "올릴", "내보낼"})
# Who receives it outside the team: a vendor, the outside, a company, a
# partner, a client firm, a third party, an outsourcer, an agency, a
# researcher. One counts as where the data goes only with a particle that
# names a destination ("벤더한테", "외부로", "협력업체에"), or right before a
# verb that sends ("외부 공유용으로"): "업체 목록에서 전화번호 삭제해줘" deletes
# from a list, and "외부 로그" is a kind of log (`_korean_recipient`).
DATA_SHARE_KOREAN_RECIPIENTS = (
    "벤더",
    "외부",
    "업체",
    "협력사",
    "파트너사",
    "거래처",
    "제3자",
    "외주",
    "대행사",
    "에이전시",
    "연구자",
    "연구기관",
)
_DESTINATION_PARTICLES = ("에게", "한테", "께", "쪽에", "측에", "쪽으로", "으로", "에", "로")
_MEANS_PARTICLES = frozenset({"쪽으로", "으로", "로"})
# A place inside the company is proven, never assumed: every word in it must
# come from the four tables below, in both languages, and it must say an
# anchor, or a unit with an owner. Any other word -- the vendor's, an external,
# a partner, a client, a customer, Acme, a hospital, the Department of Health,
# a name -- keeps the place outside and the review firing, which is the safe
# direction: "the customer success team" and "고객지원팀" keep it too. "테스트
# 데이터에서 고객 정보 지우고 스테이징에 올려줘" and "redact pii in the export
# before we send it to the dev team" keep the data inside.
#
# Anchors: places inside by themselves -- a teammate, a colleague or coworker,
# the company's internal or in-house systems, QA, staging, a test or dev
# server. A Korean reader takes a Korean anchor as a word's start ("개발서버",
# "동료들한테"); an English reader takes an entry as whole words.
DATA_SHARE_INTERNAL_DESTINATIONS = (
    "사내", "내부", "스테이징", "개발서버", "테스트", "로컬", "동료",
    "staging", "qa", "teammate", "teammates", "colleague", "colleagues", "coworker", "coworkers", "co-worker",
    "co-workers", "internal", "in-house", "test server", "test environment", "dev server",
)
# Units: a team or a department, inside only with an owner ("our team", "the
# data team", "another department", "우리 팀에", "인사 부서에"), or as a
# compound whose every part is an owner or an anchor ("개발팀", "QA팀"); bare,
# it may be anyone's ("the team", "팀 전체에"). A Korean word that starts with
# a unit and goes on is a member of it ("팀원들한테", "부서원들한테").
DATA_SHARE_INTERNAL_UNITS = ("팀", "부서", "team", "teams", "department", "departments")
# Owners: whose unit it is -- ours, mine, another of ours -- or an internal
# function.
DATA_SHARE_INTERNAL_OWNERS = (
    "우리", "내", "제", "저희", "다른", "타",
    "our", "my", "another",
    "개발", "데이터", "사이언스", "분석", "마케팅", "영업", "재무", "인사", "보안", "플랫폼", "지원", "운영", "제품",
    "기획", "디자인", "연구", "백엔드", "프론트엔드", "인프라", "엔지니어링", "품질",
    "data", "dev", "development", "engineering", "test", "testing", "marketing", "sales", "finance", "hr", "security",
    "platform", "analytics", "support", "ops", "operations", "product", "design", "research", "science", "ml", "ai",
    "bi", "backend", "frontend", "infra", "infrastructure", "it", "mobile", "web", "growth",
)
# Modifiers: words that may stand beside an anchor or an owned unit and prove
# nothing alone -- a determiner ("the data team", "우리 팀 전체에", "같은 팀
# 동료한테"), and a system of the company's (the internal wiki, the staging
# server; 사내 위키, 테스트 서버).
DATA_SHARE_INTERNAL_MODIFIERS = (
    "같은", "전체", "모든",
    "the", "a", "an", "both",
    "서버", "환경", "위키", "대시보드", "툴", "채널", "슬랙", "메신저",
    "servers", "wiki", "dashboard", "tool", "channel", "workspace", "slack",
)
# Korean only: a file or folder the data is exported to ("개인정보 지운 다음
# 엑셀로 내보내줘" writes a spreadsheet). English reads no file or folder as
# inside the company.
_KOREAN_EXPORT_DESTINATIONS = ("엑셀", "csv", "pdf", "json", "xlsx", "txt", "텍스트", "파일", "폴더", "백업")


def _place_kind(word: str) -> str:
    """What one whole word of a place is: "anchor", "unit", "owner", "modifier", or "" for anything else."""
    if word in DATA_SHARE_INTERNAL_DESTINATIONS:
        return "anchor"
    if word in DATA_SHARE_INTERNAL_UNITS:
        return "unit"
    if word in DATA_SHARE_INTERNAL_OWNERS:
        return "owner"
    return "modifier" if word in DATA_SHARE_INTERNAL_MODIFIERS else ""


def _proven_inside(kinds: list[str]) -> bool:
    """Every word is an inside word, and an anchor, or a unit with its owner, is said."""
    return bool(kinds) and all(kinds) and ("anchor" in kinds or ("unit" in kinds and "owner" in kinds))


# Words that name another job, as a word's start: a legal question, a Hermes
# setting, a storage-wide scan, or something other than data. Hermes named
# alone is no withdrawal: "헤르메스," before a request is its address
# (`korean_request.OPENERS`), as "hermes," is in English.
DATA_SHARE_KOREAN_WITHDRAWAL_STEMS = (
    # legal
    "법률", "법적", "법무", "보호법", "개인정보보호법", "처리방침", "규정", "규제", "컴플라이언스",
    # a Hermes setting
    "설정", "게이트웨이",
    # storage-wide scans
    "버킷",
    # not data
    "코드", "함수", "스크립트", "기능", "로직", "정규식", "모듈", "버그", "이슈", "라벨", "슬라이드", "티켓",
)
# The skill's Korean phrases: the personal data, the protective verbs, and the
# recipients. An admitted Korean request always says one of them, so the
# router scores every request this rule admits. They live here, beside the
# English phrases and not in `routing/trigger_packs/ko.json`, for the reason
# the English ones do: the rule reads them, and the bundle cannot read a pack.
_KOREAN_PROTECTIVE_PHRASES = ("익명화", "익명 처리", "익명처리", "가명화", "가명 처리", "가명처리", "비식별")
DATA_SHARE_REVIEW_KOREAN_PHRASES = (
    *DATA_SHARE_KOREAN_PERSONAL_DATA,
    *_KOREAN_PROTECTIVE_PHRASES,
    *DATA_SHARE_KOREAN_RECIPIENTS,
)

# Each phrase names taking personal data out of something. They are also the
# skill's trigger list, word for word (`tests/test_data_share_review.py`).
DATA_SHARE_REVIEW_PHRASES = (
    "data-share-review",
    "data share review",
    "pii redaction",
    "redact pii",
    "pseudonymize",
    "de-identify",
    "anonymize the dataset",
    "residual pii",
    "redact the pii",
    "redact personal data",
    "strip pii",
    "strip the pii",
    "remove pii",
    "remove the pii",
    "scrub pii",
    "scrub the pii",
    "mask pii",
    "mask personal data",
    "remove personal data",
    "scan for pii",
    "pseudonymise",
    "pseudonymizing",
    "pseudonymization",
    "pseudonymisation",
    "de-identifying",
    "deidentify",
    "quasi-identifier",
    "quasi-identifiers",
    "direct identifiers",
    "anonymize this dataset",
    "anonymize the data",
    "anonymize the logs",
    "redact the logs",
    "redact these logs",
    *DATA_SHARE_REVIEW_KOREAN_PHRASES,
)

# Verbs that protect data whatever their object.
DATA_SHARE_LEAD_VERBS = frozenset(
    {"anonymise", "anonymize", "de-identify", "deidentify", "pseudonymise", "pseudonymize", "redact"}
)
# The object runs from the verb up to the first of these words.
DATA_SHARE_OBJECT_ENDS = frozenset({"and", "before", "for", "from", "in", "of"})
# A lead verb followed by one of these is being defined, not asked for:
# "de-identify means what exactly", "de-identify is the term we use".
DATA_SHARE_DEFINITION_WORDS = frozenset({"is", "means", "refers", "stands"})
# An object ending in one of these is not data: "redact the pii redaction
# label", "pseudonymize the export script".
DATA_SHARE_NON_DATA_NOUNS = frozenset(
    {
        "bug", "bugs", "flag", "flags", "function", "functions", "header", "headers", "html", "issue", "issues",
        "label", "labels", "marker", "markers", "notebook", "notebooks", "overhead", "pr", "request", "requests",
        "row", "rows", "script", "scripts", "slides", "step", "steps", "ticket", "tickets",
    }
)
# The carrier: something file-shaped that can leave. "data" alone is not one,
# and neither is a "sample", which is as often a sample message.
DATA_SHARE_CARRIER_TOKENS = frozenset(
    {
        "corpus", "csv", "csvs", "dataset", "datasets", "dump", "dumps", "excerpt", "excerpts", "export",
        "exports", "extract", "extracts", "file", "files", "log", "logs", "records", "spreadsheet", "spreadsheets",
    }
)
# The outbound cue: who receives it outside the company, or a verb that sends
# it. "hand" counts only as handing something over, and "give" only with a
# recipient (`_gives_it_away`).
DATA_SHARE_RECIPIENT_TOKENS = frozenset(
    {
        "agency", "contractor", "contractors", "external", "provider", "public", "researcher", "researchers",
        "third-party", "vendor", "vendors",
    }
)
DATA_SHARE_RECIPIENT_PHRASES = (
    "third party",
    "fine-tune corpus",
    "fine-tune set",
    "fine-tuning set",
    "training corpus",
    "training set",
)
DATA_SHARE_SEND_TOKENS = frozenset(
    {"attach", "attaching", "post", "posting", "send", "sending", "share", "sharing", "upload", "uploading"}
)
DATA_SHARE_HAND_PHRASES = ("hand over", "hand it", "hand them", "handing it", "handing them", "handing over")
# Every verb that sends the data on, for reading where it goes. Only the first
# set admits a request (`_leaving`); the rest are read so that a later send
# outside is seen ("... so they can forward it to acme", "cc the auditor").
# "give", "hand" and "pass" with no place send nothing ("give me a summary").
_ONWARD_SENDS = DATA_SHARE_SEND_TOKENS | {
    "cc", "email", "emailing", "forward", "forwarding", "give", "giving", "hand", "handing", "pass", "passing",
    "transfer", "transferring",
}
_SENDS_ONLY_WITH_A_PLACE = frozenset({"give", "giving", "hand", "handing", "pass", "passing"})
# "give", "hand" and "pass" with no "to" send nothing only to the person alone
# ("give me a summary", "hand it back"): the person, then the thing given, a
# determiner or nothing. Otherwise their first object is the place ("give acme
# a copy", "give me and acme a copy", "give acme back the file") and the thing
# given is not read as part of it.
_FIRST_PERSON = frozenset({"me", "us", "myself", "ourselves"})
_HANDED_THINGS = DATA_SHARE_CARRIER_TOKENS | {"access", "copies", "copy", "link", "links", "version"}
_AFTER_THE_PERSON = _HANDED_THINGS | {"a", "an", "some", "the", "this", "that"}
_HANDED_BACK = (["it", "back"], ["them", "back"])
# Data containers and the names of personal-data fields. Neither can name a
# party.
_DATA_CONTAINERS = DATA_SHARE_CARRIER_TOKENS | {
    "archive", "archives", "backup", "backups", "column", "columns", "copies", "copy", "data", "database",
    "databases", "db", "document", "documents", "field", "fields", "json", "list", "lists", "report", "reports",
    "row", "rows", "sample", "samples", "sheet", "sheets", "table", "tables", "transcript", "transcripts", "tsv",
    "xlsx",
}
_DATA_FIELDS = frozenset(
    {
        "address", "addresses", "birth", "date", "dates", "details", "dob", "email", "emails", "id", "identifier",
        "identifiers", "ids", "info", "information", "ip", "ips", "name", "names", "number", "numbers", "personal",
        "phone", "phones", "pii", "sensitive", "ssn", "ssns",
    }
)
# The only words that may stand between the lead verb and the first send, its
# object included: where the data is, joining words, the speaker ("we", "us",
# "i", "you", "it"), a determiner, data containers and field names ("the pii
# in the export before we", "the email column from this csv before"). Any other
# word keeps the review: "the export acme will get", "the csv the bank wants",
# "before acme sees it", "for them".
_BEFORE_SEND_WORDS = DATA_SHARE_OBJECT_ENDS | _DATA_CONTAINERS | _DATA_FIELDS | {
    "a", "after", "an", "first", "i", "it", "my", "once", "our", "please", "that", "the", "then", "these", "this",
    "those", "us", "we", "you",
}
# Whose records the data holds. One of these is allowed in that span only as a
# compound modifier: right before a data container or a field name, not
# possessive, and not after "for", "to" or "from" ("the customer export", "the
# patient csv", "the user logs", "the customers table"). Anywhere else it may
# name a recipient and keeps the review: "the customer's export", "the export
# for the customer", "the export the customer wants". A word not on this list
# never qualifies: "the acme export" keeps the review.
DATA_SHARE_PARTY_MODIFIERS = frozenset(
    {
        "account", "accounts", "applicant", "applicants", "client", "clients", "customer", "customers", "donor",
        "donors", "employee", "employees", "member", "members", "patient", "patients", "staff", "student",
        "students", "subscriber", "subscribers", "user", "users",
    }
)
_MODIFIER_BLOCKERS = frozenset({"for", "from", "to"})
_SPAN_DETERMINERS = frozenset({"a", "an", "my", "our", "that", "the", "these", "this", "those"})
# A send's place follows one of these ("send it to the dev team", "share it
# with a colleague") and runs to the next send or the end of the send's
# sentence.
_DESTINATION_PREPOSITIONS = frozenset({"into", "onto", "to", "with"})
# A sentence ends at ".", "!", "?" or ";" followed by a space. Only a send's
# place stops there: sends are read in every sentence. A period inside a word
# ("v1.2", "u.s.") ends nothing; one before a space does ("acme inc. today",
# "e.g. kim"), which can only cut a place short, and what follows is read as
# a remark (`_REMARK_WORDS`). A line break alone ends nothing: the router reads
# the message with its line breaks folded to spaces, the hint without.
_SENTENCE_BREAK = re.compile(r"[.!?;]\S*\s")
# What may follow a send's place in the sentences after it, up to the next
# send, in either language: the closed words read before a send
# (`_BEFORE_SEND_WORDS`), thanks, please and urgency ("it's urgent",
# "thanks!", "no rush", "asap please", "급해요", "감사합니다"). Any other word
# keeps the review: "acme will review it", "thanks, kim", "we need it today",
# "보험사가 볼 거야".
_REMARK_WORDS = frozenset(
    {
        "also", "asap", "cheers", "hurry", "is", "no", "now", "ok", "okay", "please", "pls", "plz", "quickly", "rush",
        "s", "so", "soon", "thank", "thanks", "today", "tomorrow", "tonight", "too", "urgent", "urgently", "very",
        "you",
        "감사", "감사드립니다", "감사합니다", "감사해요", "고마워", "고마워요", "고맙습니다", "급합니다", "급해", "급해요", "꼭",
        "내일", "부탁드려요", "부탁드립니다", "부탁합니다", "부탁해", "부탁해요", "바로", "빨리", "오늘", "좀", "지금",
    }
)
_REMARKS = _BEFORE_SEND_WORDS | _REMARK_WORDS
# English words that may join the words of an inside place: "the data team and
# the qa team", "the data team as well as the qa team", "a coworker in
# finance", "the dev team today, please". Each word they join must still be an
# inside word: "the data team and acme" and "our team in seoul" are outside,
# and so is any other joiner ("the data team at acme", "the dev team for
# review").
_PLACE_JOINERS = frozenset({"and", "as", "in", "or", "please", "plus", "today", "well"})
# A route or a party named between a send and its place ("upload it via
# acme's portal to the data team", "send it for acme to the dev team") keeps
# the data leaving.
_SEND_ROUTE_WORDS = frozenset({"at", "by", "for", "from", "on", "through", "using", "via"})
# The only words between a send and its "to" or "with", read as the span
# before the first send is (`_closed_before_send`, so "send the customer
# export to" passes): the data sent, also described as protected ("the
# anonymized copy", "the redacted logs", "the clean export"), counted ("the
# whole export", "both files", "all of it") or as the version the redaction
# made ("the updated file", "the final csv", "the new version", "the result",
# "the output"); the verb's particle ("upload them to", "hand it over to",
# "send it back to"); how it goes ("send it straight to", "directly"); and
# remarks ("send it asap to"). Each of these says what is sent or how, never
# who receives it. Any other word keeps the review: "send it where acme wants
# it, to the dev team", "send it, as acme asked, to the dev team", "send the
# acme copy to the dev team".
_SENT_DATA_WORDS = frozenset(
    {
        # protected
        "anonymised", "anonymized", "clean", "cleaned", "de-identified", "deidentified", "masked", "pseudonymised",
        "pseudonymized", "redacted", "sanitised", "sanitized",
        # counted
        "all", "both", "whole",
        # the version the redaction made
        "final", "new", "output", "outputs", "result", "results", "updated", "version", "versions",
        # how it goes
        "directly", "straight",
    }
)
_SENT_THING_WORDS = _REMARKS | _SENT_DATA_WORDS | {"back", "over", "them"}
# The multi-word English anchors ("test server"): their words are allowed in a
# place ("the staging server"), but only the whole phrase is an anchor.
_ENGLISH_PHRASE_ANCHORS = tuple(entry for entry in DATA_SHARE_INTERNAL_DESTINATIONS if " " in entry)
_ENGLISH_ANCHOR_PARTS = frozenset(word for entry in _ENGLISH_PHRASE_ANCHORS for word in entry.split())
# Quote marks. A `'` or `’` between two word characters is an apostrophe
# ("it's", "it’s"), not a quote: smart punctuation types the second.
_QUOTES = re.compile(r"[\"“”‘`]|(?<!\w)['’]|['’](?!\w)")
# Words that name another job: a legal question (`legal-compliance-review`),
# Hermes (a setting of it, or Hermes doing the job), or a scan across storage.
DATA_SHARE_WITHDRAWAL_TOKENS = frozenset(
    {
        # legal
        "ccpa", "compliance", "compliant", "contract", "contracts", "dpa", "gdpr", "hipaa", "lawful", "legal",
        "legally", "regulation", "regulations", "regulatory",
        # Hermes
        "config", "configure", "enable", "gateway", "hermes", "setting", "settings",
        # storage-wide scans
        "bucket", "buckets", "drive", "s3",
    }
)
DATA_SHARE_WITHDRAWAL_PHRASES = ("turn on", "turn off", "across our")

# Unicode-aware, so Hangul and other scripts are words too.
_WORD = re.compile(r"\w[\w'-]*")


def _words(text: str) -> list[str]:
    return [word.split("'")[0] for word in _WORD.findall(text)]


def _protective_lead(words: list[str]) -> bool:
    if not words or words[0] not in DATA_SHARE_LEAD_VERBS:
        return False
    if words[1:2] and words[1] in DATA_SHARE_DEFINITION_WORDS:
        return False
    obj: list[str] = []
    for word in words[1:]:
        if word in DATA_SHARE_OBJECT_ENDS:
            break
        obj.append(word)
    return not (obj and obj[-1] in DATA_SHARE_NON_DATA_NOUNS)


def _gives_it_away(words: list[str]) -> bool:
    """"give it to the contractor", never "give me" or "give it back"."""
    for index, word in enumerate(words):
        if word not in {"give", "giving"}:
            continue
        following = words[index + 1 : index + 5]
        if following[:1] in (["me"], ["us"]) or "back" in following:
            continue
        if "to" in following:
            return True
    return False


def _inside(place: list[str]) -> bool:
    """A place proven inside the company: "the dev team", "a coworker in finance", "the test server"; not "the team".

    Hangul in an English place is no inside word, on either surface's fold.
    """
    if not all(word.isascii() for word in place):
        return False
    kinds = [
        _place_kind(word) or ("modifier" if word in _PLACE_JOINERS or word in _ENGLISH_ANCHOR_PARTS else "")
        for word in place
    ]
    if contains_boundary_phrase(" ".join(place), _ENGLISH_PHRASE_ANCHORS):
        kinds.append("anchor")
    return _proven_inside(kinds)


def _closed_before_send(span: list[str], raw_span: list[str], allowed: frozenset[str] = _BEFORE_SEND_WORDS) -> bool:
    """Whether every word of a span is a closed word (`allowed`, read composed) or a compound party modifier.

    The span from the lead verb to the first send reads `_BEFORE_SEND_WORDS`;
    the span between a later send and its "to" reads `_SENT_THING_WORDS`.
    """
    for position, word in enumerate(span):
        if composed(word) in allowed:
            continue
        following = span[position + 1] if position + 1 < len(span) else ""
        earlier = next((before for before in reversed(span[:position]) if before not in _SPAN_DETERMINERS), "")
        if (
            word in DATA_SHARE_PARTY_MODIFIERS
            and "'" not in raw_span[position]
            and (following in _DATA_CONTAINERS or following in _DATA_FIELDS)
            and earlier not in _MODIFIER_BLOCKERS
        ):
            continue
        return False
    return True


def _sentence_starts(text: str) -> list[int]:
    """The indices of the words (`_WORD`) in `text` that begin a sentence after the first."""
    starts: list[int] = []
    previous = 0
    for index, match in enumerate(_WORD.finditer(text)):
        if index and _SENTENCE_BREAK.search(text, previous, match.start()):
            starts.append(index)
        previous = match.end()
    return starts


def _sends_stay_inside(words: list[str], raw_words: list[str], sentence_starts: list[int]) -> bool:
    """Whether every verb that sends the data on names a place proven inside the company.

    Sends are read after the lead verb's object, which is the data ("redact
    the email column" sends nothing). From the lead verb to the first send,
    its object included, only closed words may stand (`_BEFORE_SEND_WORDS`),
    and a party modifier right before a container or a field
    (`_closed_before_send`). Each send's place runs from its "to" or
    "with" to the next send or the end of its sentence, so a second place
    joined to the first is read too ("the data team and acme", "the data team
    or the auditor"). Every sentence is read for sends; the words after a
    place's sentence, up to the next send, must be remarks (`_REMARK_WORDS`):
    "... to the dev team. it's urgent." stays inside, "... to the dev team.
    acme will review it." leaves. "give", "hand" and "pass" with no "to" take
    their first object as the place, unless it is the person or the data
    comes back.
    """
    start = next((index for index, word in enumerate(words) if index and word in DATA_SHARE_OBJECT_ENDS), len(words))
    sends = [index for index in range(start, len(words)) if words[index] in _ONWARD_SENDS]
    if sends and not _closed_before_send(words[1 : sends[0]], raw_words[1 : sends[0]]):
        return False
    inside = False
    for number, index in enumerate(sends):
        following = sends[number + 1] if number + 1 < len(sends) else len(words)
        stop = next((sentence for sentence in sentence_starts if index < sentence < following), following)
        # Composed: the router reads Hangul decomposed ("... to the dev team. 감사합니다.").
        if not {composed(word) for word in words[stop:following]} <= _REMARKS:
            return False
        reach = words[index + 1 : stop]
        preposition = next((position for position, word in enumerate(reach) if word in _DESTINATION_PREPOSITIONS), None)
        if preposition is None and words[index] in _SENDS_ONLY_WITH_A_PLACE:
            to_the_person = reach[:1] and reach[0] in _FIRST_PERSON and (reach[1:2] == [] or reach[1] in _AFTER_THE_PERSON)
            if to_the_person or reach[:2] in _HANDED_BACK:
                continue
            place = [word for word in reach if word not in _HANDED_THINGS]
        elif preposition is None:
            return False
        elif set(reach[:preposition]) & _SEND_ROUTE_WORDS or not _closed_before_send(
            reach[:preposition], raw_words[index + 1 : index + 1 + preposition], _SENT_THING_WORDS
        ):
            return False
        else:
            place = reach[preposition + 1 :]
        if not _inside(place):
            return False
        inside = True
    return inside


def _leaving(text: str, tokens: set[str], words: list[str], shielded: bool) -> bool:
    raw_words = _WORD.findall(text)
    if not tokens & DATA_SHARE_CARRIER_TOKENS:
        return False
    if tokens & DATA_SHARE_RECIPIENT_TOKENS or contains_boundary_phrase(text, DATA_SHARE_RECIPIENT_PHRASES):
        return True
    sends = (
        bool(tokens & DATA_SHARE_SEND_TOKENS)
        or contains_boundary_phrase(text, DATA_SHARE_HAND_PHRASES)
        or _gives_it_away(words)
    )
    return sends and (
        shielded or bool(_QUOTES.search(text)) or not _sends_stay_inside(words, raw_words, _sentence_starts(text))
    )


def data_share_request_admitted(text: str, *, shielded: bool = False) -> bool:
    """A request to redact, pseudonymize, anonymize or de-identify data leaving the company, in English or Korean.

    `shielded`: the caller removed a quoted, code or link region from `text`
    before passing it (`reference_regions`). What was there is unread, so no
    place is then proven inside: 'send it to the "acme" data team' arrives as
    "send it to the data team".
    """
    request = after_courtesy(text.strip())
    words = _words(request)
    tokens = set(words) | {part for word in words for part in word.split("-") if part}
    if tokens & DATA_SHARE_WITHDRAWAL_TOKENS or contains_boundary_phrase(request, DATA_SHARE_WITHDRAWAL_PHRASES):
        return False
    if _korean_admitted(request, tokens, shielded):
        return True
    # A question is not a request to clean a file.
    if text.rstrip().endswith("?"):
        return False
    return _protective_lead(words) and _leaving(request, tokens, words, shielded)


def data_share_phrases_said(text: str, phrases: Iterable[str] = DATA_SHARE_REVIEW_PHRASES) -> list[str]:
    """Which of `phrases` `text` says: an English one as whole words, a Korean one anywhere.

    The router's reading of a trigger (`phrase_is_spoken` checks Latin edges
    only): Korean writes compounds without a space, so "업체" in "협력업체"
    and "개인정보" in "고객개인정보" start mid-word, and a whole-word reader
    found neither in "고객개인정보 지우고 협력업체에 공유해줘", which the router
    dispatches here.
    """
    folded = composed(text)
    return [
        phrase
        for phrase in phrases
        if (contains_boundary_phrase(text, (phrase,)) if phrase.isascii() else composed(phrase) in folded)
    ]


def _korean_says(words: tuple[str, ...], phrases: Iterable[str]) -> bool:
    said = " ".join(words)
    return any(phrase in said for phrase in phrases)


def _korean_protective(form: str, index: int, words: tuple[str, ...]) -> bool:
    if form in DATA_SHARE_KOREAN_PROTECTIVE_VERBS:
        return True
    removes = form in DATA_SHARE_KOREAN_REMOVAL_VERBS | DATA_SHARE_KOREAN_REMOVAL_JOINED | DATA_SHARE_KOREAN_REMOVAL_BEFORE_AFTER
    window = words[max(0, index - 3) : index + 1]
    return removes and (
        _korean_says(window, DATA_SHARE_KOREAN_PERSONAL_DATA)
        or any(word.startswith(DATA_SHARE_KOREAN_FIELD_NAMES) for word in window)
    )


def _korean_sends(word: str) -> bool:
    """A word that sends the data on later: "보낼", "공유할" ("업체에 보낼 파일", "공유할 거니까")."""
    return word in DATA_SHARE_KOREAN_OUTBOUND_FUTURE or (word.endswith("할") and word[:-1] in DATA_SHARE_KOREAN_OUTBOUND_VERBS)


def _korean_recipient(words: tuple[str, ...]) -> bool:
    """Whether a recipient outside the team is said as where the data goes.

    The word is the recipient, the recipient with 측 or 쪽, or a compound it
    ends ("협력업체"), with a destination particle: "벤더한테", "외부에",
    "협력업체에", "제3자에게". 로/으로 names a destination only right before a
    verb that sends ("외부로 보내줘"); before anything else it names a means or
    a basis ("업체별로", "업체명으로", "외부키로"). A recipient without a
    particle counts only right before a word that sends ("외부 공유해줘",
    "외부 공유용으로"). Not a source ("업체 목록에서"), a kind ("외부 로그"),
    or text inside another word ("해외부서").
    """
    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else ""
        sends_next = following.startswith(tuple(DATA_SHARE_KOREAN_OUTBOUND_VERBS))
        particle = next((particle for particle in _DESTINATION_PARTICLES if word.endswith(particle)), "")
        stem = word[: -len(particle)] if particle else word
        names_one = (
            stem in DATA_SHARE_KOREAN_RECIPIENTS
            or stem.endswith(DATA_SHARE_KOREAN_RECIPIENTS)
            or any(stem in (recipient + "측", recipient + "쪽") for recipient in DATA_SHARE_KOREAN_RECIPIENTS)
        )
        if names_one and particle and (particle not in _MEANS_PARTICLES or sends_next):
            return True
        if word in DATA_SHARE_KOREAN_RECIPIENTS and sends_next:
            return True
    return False


# Particles that make a Korean word a place, with 도 ("also") too: "벤더에도
# 보내줘" is a second destination.
_KOREAN_PLACE_PARTICLES = ("에게도", "한테도", "께도", "으로도", "로도", "에도", *_DESTINATION_PARTICLES)
_KOREAN_MEANS_PARTICLES = _MEANS_PARTICLES | {"으로도", "로도"}
# Not places: a step before 에 ("지운 다음에", "보내기 전에") and a channel
# before 로 ("이메일로 보내줘").
_KOREAN_TIMES = frozenset({"다음", "전", "후", "뒤"})
_KOREAN_CHANNELS = frozenset({"이메일", "메일", "슬랙", "메신저"})
# English words in a Korean sentence, also where a particle is glued to them
# ("vendor가").
_LATIN_RUN = re.compile(r"[a-z][a-z-]*")
# Where the words of a place stop, read back from it: a joined verb
# ("지우고", "마스킹해서") or a step ("지운 다음", "보내기 전에"). Any other
# word before a place is read as part of it.
_KOREAN_PLACE_EDGES = frozenset({"다음", "다음에", "전", "전에", "후", "후에", "뒤", "뒤에"})
# A joined 하다 verb ends the clause only when the rule knows the verb: one
# that protects, removes or sends, or 처리 ("가명 처리해서").
_KOREAN_JOINED_ENDINGS = ("하고서", "하고", "해서", "한")
_KOREAN_EDGE_VERBS = frozenset(
    {*DATA_SHARE_KOREAN_PROTECTIVE_VERBS, *DATA_SHARE_KOREAN_REMOVAL_VERBS, *DATA_SHARE_KOREAN_OUTBOUND_VERBS, "처리"}
)
# The parts a unit's compound prefix may be built from: owners and anchors
# ("데이터사이언스팀", "QA팀"), never the English determiners, which would split
# a name ("ana" as "an" + "a").
_KOREAN_COMPOUND_PARTS = tuple(
    sorted(
        {*DATA_SHARE_INTERNAL_OWNERS, *DATA_SHARE_INTERNAL_DESTINATIONS} - {"my", "our", "another"},
        key=len,
        reverse=True,
    )
)
_KOREAN_ANCHORS = tuple(entry for entry in DATA_SHARE_INTERNAL_DESTINATIONS if not entry.isascii())
_KOREAN_UNITS = tuple(entry for entry in DATA_SHARE_INTERNAL_UNITS if not entry.isascii())
# The Korean mirror of `_BEFORE_SEND_WORDS`: the only words that may stand
# outside every place and before the requested verb. A Korean request names
# its data, and any party, before the place ("보험사가 원하는 고객 엑셀 ...
# 개발팀에 보내줘"), so each such word must be closed: data containers and
# personal-data field names (in either script), a verb this rule knows (as a
# noun, joined, "-기 전에" or a later send: "보내기", "보낼"), an inside place
# word, and the determiners, connectives and softeners below. Any other word
# -- a company, an institution, a clause about one ("보험사가", "원하는",
# "규제기관이 요청한", "보험사를 위한") -- keeps the review.
_KOREAN_DATA_CONTAINERS = frozenset(
    {
        "기록", "내역", "데이터", "데이터셋", "덤프", "레코드", "로그", "리스트", "리포트", "명단", "목록", "문서", "보고서",
        "사본", "샘플", "스프레드시트", "시트", "열", "원본", "자료", "추출본", "컬럼", "테이블", "표", "필드", "행",
        *_KOREAN_EXPORT_DESTINATIONS,
    }
)
_KOREAN_DATA_FIELDS = frozenset(
    {*DATA_SHARE_KOREAN_FIELD_NAMES, *(word for phrase in DATA_SHARE_KOREAN_PERSONAL_DATA for word in phrase.split())}
    - {"고객"}
)
_KOREAN_DATA_WORDS = _KOREAN_DATA_CONTAINERS | _KOREAN_DATA_FIELDS | _DATA_CONTAINERS | _DATA_FIELDS
# Whose records the data holds, as `DATA_SHARE_PARTY_MODIFIERS` in English:
# allowed only bare and right before a data word ("고객 정보", "회원 명단",
# "환자 데이터", "고객데이터"). With a particle it names a party ("고객이 받을",
# "고객의", "고객용"), as "the customer's" does.
_KOREAN_PARTY_MODIFIERS = frozenset(
    {
        "가입자", "계정", "고객", "구독자", "기부자", "멤버", "사용자", "사원", "유저", "이용자", "임직원", "지원자",
        "직원", "클라이언트", "학생", "환자", "회원", "후원자", *DATA_SHARE_PARTY_MODIFIERS,
    }
)
_KOREAN_COMPOUND_DATA_PARTS = tuple(sorted(_KOREAN_DATA_WORDS | _KOREAN_PARTY_MODIFIERS, key=len, reverse=True))
# Particles a closed word may carry: object, topic, source, "and", "or",
# "of", "also", "only". Not 이/가, which make it the one who acts ("보험사가
# 원하는"), nor 에/에게/한테, which make it a place.
_KOREAN_SPAN_PARTICLES = tuple(
    sorted(
        ("에서", "으로", "이랑", "하고", "이나", "까지", "부터", "을", "를", "은", "는", "도", "만", "의", "과", "와", "랑", "나", "로"),
        key=len,
        reverse=True,
    )
)
_KOREAN_SPAN_WORDS = frozenset(
    {"이", "그", "저", "이번", "해당", "위", "아래", "각", "및", "그리고", "또", "또는", "혹은", *SOFTENERS}
)
_KOREAN_SPAN_VERBS = _KOREAN_EDGE_VERBS | LIGHT_VERBS


def _korean_data_word(word: str) -> bool:
    """A data container or field name, or a compound of them led by a party modifier: "고객데이터", "로그파일", "회원db"."""
    if word in _KOREAN_DATA_WORDS:
        return True
    return any(
        word.startswith(part) and len(word) > len(part) and _korean_data_word(word[len(part) :])
        for part in _KOREAN_COMPOUND_DATA_PARTS
    )


def _korean_unparticled(word: str) -> tuple[str, str]:
    """`word` without one particle a closed word may carry, and that particle."""
    particle = next((particle for particle in _KOREAN_SPAN_PARTICLES if word.endswith(particle) and word != particle), "")
    return (word[: -len(particle)], particle) if particle else (word, "")


def _korean_closed(word: str, following: str) -> bool:
    """Whether `word`, outside every place and before the requested verb, is a closed word (`_KOREAN_DATA_WORDS` and the rest)."""
    if word in _KOREAN_SPAN_WORDS or _korean_edge(word) or _korean_sends(word):
        return True
    # A "-기 전에" clause's verb: "보내기", "올리기", "공유하기".
    if word.endswith("기") and (word[:-1] in _KOREAN_SPAN_VERBS or (word.endswith("하기") and word[:-2] in _KOREAN_SPAN_VERBS)):
        return True
    stem, particle = _korean_unparticled(word)
    if _korean_data_word(stem) or stem in _KOREAN_SPAN_VERBS or stem in _KOREAN_CHANNELS or _korean_place_word(stem):
        return True
    # For an inside place: "내부용", "테스트용".
    if stem.endswith("용") and len(stem) > 1 and _korean_place_word(stem[:-1]):
        return True
    return not particle and stem in _KOREAN_PARTY_MODIFIERS and _korean_data_word(_korean_unparticled(following)[0])


def _korean_compound_inside(prefix: str) -> bool:
    """Whether a unit's compound prefix is all owners and anchors ("데이터사이언스", not "외부", "고객지원", "acme")."""
    while prefix:
        part = next((word for word in _KOREAN_COMPOUND_PARTS if prefix.startswith(word)), "")
        if not part:
            return False
        prefix = prefix[len(part) :]
    return True


def _korean_place_word(word: str) -> str:
    """A Korean place word's kind, as `_place_kind` gives it, reading Korean anchors and units inside compounds."""
    if word.startswith(_KOREAN_ANCHORS) or word.startswith(_KOREAN_EXPORT_DESTINATIONS):
        return "anchor"
    unit = next((unit for unit in _KOREAN_UNITS if word.endswith(unit)), "")
    if unit and word != unit and _korean_compound_inside(word[: -len(unit)]):
        return "anchor"
    if word.startswith(_KOREAN_UNITS) and word not in _KOREAN_UNITS:
        return "anchor"
    return _place_kind(word)


def _korean_edge(word: str) -> bool:
    """Whether `word` ends the clause before a place: a step, or a joined form of a verb this rule knows.

    Only a verb it knows: "마스킹해서", "처리하고", "지우고", "삭제한 (다음)".
    A word that merely ends in 서 or 고 ("경찰서", "물류창고", "삼성하고") is
    read as part of the place, and an unknown one keeps it outside.
    """
    if (
        word in _KOREAN_PLACE_EDGES
        or word in DATA_SHARE_KOREAN_REMOVAL_JOINED
        or word in DATA_SHARE_KOREAN_OUTBOUND_JOINED
        or word in DATA_SHARE_KOREAN_REMOVAL_BEFORE_AFTER
    ):
        return True
    ending = next((ending for ending in _KOREAN_JOINED_ENDINGS if word.endswith(ending) and word != ending), "")
    return bool(ending) and word[: -len(ending)] in _KOREAN_EDGE_VERBS


def _korean_place_start(words: tuple[str, ...], index: int) -> int:
    """Where the words of the place ending at `index` start: read back, however many, to an edge.

    No window: "은행 IT 보안 개발 팀에" reads 은행 too.
    """
    start = index
    while start and (_korean_place_word(words[start - 1]) or not _korean_edge(words[start - 1])):
        start -= 1
    return start


def _korean_place_inside(words: tuple[str, ...], index: int, stem: str) -> bool:
    """Whether the place ending at `index` is proven inside: its stem and every word before it back to an edge are inside words, one an anchor."""
    start = _korean_place_start(words, index)
    return _proven_inside([_korean_place_word(stem), *(_korean_place_word(word) for word in words[start:index])])


def _korean_places(words: tuple[str, ...]) -> set[int] | None:
    """The indices of the words in every place, when each place is proven inside; None when one is not.

    Fails safe: any place with another word ("외부 팀에", "은행 담당 부서에",
    "다른 회사 동료에게", "acme qa에") or a second place outside ("QA팀에
    보내고 캐글에 올려줘") keeps the data leaving.
    """
    in_a_place: set[int] = set()
    for index, word in enumerate(words):
        particle = next((particle for particle in _KOREAN_PLACE_PARTICLES if word.endswith(particle) and word != particle), "")
        if not particle:
            continue
        stem = word[: -len(particle)]
        if stem in _KOREAN_TIMES:
            continue
        if particle in _KOREAN_MEANS_PARTICLES:
            sends_next = index + 1 < len(words) and words[index + 1].startswith(tuple(DATA_SHARE_KOREAN_OUTBOUND_VERBS))
            if not sends_next or stem in _KOREAN_CHANNELS:
                continue
        if not _korean_place_inside(words, index, stem):
            return None
        in_a_place.update(range(_korean_place_start(words, index), index + 1))
    return in_a_place


def _korean_read_inside(request: KoreanRequest) -> set[int] | None:
    """The indices of the words in a Korean request's places, when every place is proven inside and every other word before its verb is closed; None otherwise.

    A word outside every place that is not closed (`_korean_closed`) keeps
    the review: "보험사가 원하는 고객 엑셀 개인정보 마스킹해서 개발팀에 보내줘"
    names a party before the place.
    """
    words = request.words
    in_a_place = _korean_places(words)
    if in_a_place is None:
        return None
    closed = all(
        index in in_a_place or _korean_closed(word, words[index + 1] if index + 1 < len(words) else "")
        for index, word in enumerate(words[: request.predicate_index])
    )
    return in_a_place if closed else None


def _korean_acts(word: str) -> bool:
    """Whether `word` is one who acts that is no closed word: 이/가 after a party ("보험사가", "acme가"), not "제가" or "데이터가"."""
    return word.endswith(("이", "가")) and len(word) > 1 and not _korean_closed(word[:-1], "")


def _korean_stays_inside(request: KoreanRequest, text: str, shielded: bool) -> bool:
    """Whether every place the data goes, in every sentence, is proven inside the company ("개발팀에", "사내 위키에", "엑셀로").

    The request's sentence, and each one after it: a later request is read
    like the first (`_korean_read_inside`: "... 개발팀에 보내줘. 협력사에도
    공유해줘." leaves), and a later sentence that asks nothing keeps the review
    when it names a place outside or a party acting ("... 개발팀에 보내줘.
    보험사가 볼 거야."); a remark ("급해요.", "감사합니다.") names neither. So do
    an English recipient anywhere ("vendor team에") and quoted text.
    """
    if shielded or set(_LATIN_RUN.findall(text)) & DATA_SHARE_RECIPIENT_TOKENS or _QUOTES.search(text):
        return False
    places = _korean_read_inside(request)
    if places is None:
        return False
    named = bool(places)
    for sentence in request.later:
        words = tuple(_words(composed(sentence)))
        later = korean_request(sentence)
        later_places = _korean_read_inside(later) if later is not None else _korean_places(words)
        if later_places is None or (later is None and any(_korean_acts(word) for word in words)):
            return False
        named = named or bool(later_places)
    return named


def _korean_clauses(request: KoreanRequest) -> list[tuple[str, int]]:
    """The request's verb and the verbs of the clauses joined before it, with their word index."""
    clauses = [(request.predicate, request.predicate_index), *request.joined]
    for index, word in enumerate(request.words[: request.predicate_index]):
        if word in DATA_SHARE_KOREAN_REMOVAL_JOINED or word in DATA_SHARE_KOREAN_OUTBOUND_JOINED:
            clauses.append((word, index))
    return clauses


def _korean_request_sends(request: KoreanRequest) -> bool:
    """A verb that sends the data on: the request's or a joined one ("보내줘", "보내고"), a "-기 전에" one, or a later send ("보낼")."""
    return (
        any(form in DATA_SHARE_KOREAN_OUTBOUND_VERBS or form in DATA_SHARE_KOREAN_OUTBOUND_JOINED for form, _ in _korean_clauses(request))
        or any(verb in DATA_SHARE_KOREAN_OUTBOUND_VERBS for verb in request.before)
        or any(_korean_sends(word) for word in request.words)
    )


def _korean_admitted(text: str, latin_tokens: set[str], shielded: bool) -> bool:
    """The Korean half: a request ending on protecting the data or sending it, with both said.

    One of the skill's Korean phrases must be said too, so the router scores
    every request this admits: personal data or a protective verb anywhere
    (Korean writes them into compounds, "고객개인정보"), or a recipient as
    where the data goes. A field name alone ("이름 지우고 공유해줘") admits
    nothing. A recipient or a send in a later sentence counts too ("...
    개발팀에 보내줘. 벤더한테도 보내줘.").
    """
    request = korean_request(text)
    if request is None:
        return False
    words = request.words
    if any(word.startswith(DATA_SHARE_KOREAN_WITHDRAWAL_STEMS) for word in words):
        return False
    if latin_tokens & DATA_SHARE_NON_DATA_NOUNS:
        return False
    final = request.predicate
    protective = any(_korean_protective(form, index, words) for form, index in _korean_clauses(request))
    later_words = [tuple(_words(composed(sentence))) for sentence in request.later]
    later_requests = [later for later in map(korean_request, request.later) if later is not None]
    recipient = _korean_recipient(words) or any(_korean_recipient(sentence) for sentence in later_words)
    sends = (
        _korean_request_sends(request)
        or any(_korean_request_sends(later) for later in later_requests)
        or any(_korean_sends(word) for sentence in later_words for word in sentence)
    )
    outbound = recipient or (sends and not _korean_stays_inside(request, text, shielded))
    said = recipient or _korean_says(words, DATA_SHARE_KOREAN_PERSONAL_DATA + _KOREAN_PROTECTIVE_PHRASES)
    asks_for_it = final in DATA_SHARE_KOREAN_OUTBOUND_VERBS or _korean_protective(final, request.predicate_index, words)
    return asks_for_it and protective and outbound and said
