"""When a skill's name opens a message to report on the skill or ask about it, not to run it.

The router reads a skill's bare name at the head of a message as a request to run it ("ultrawork the auth refactor",
"dag ulw 결제 모듈 리팩터링"), and the route hint says so. A complaint or a question opens the same way: "ultrawork
broke my build", "dag ulw failed again", "ralplan 때문에 빌드 깨졌어", "code-review 결과가 이상해", "ultrawork 이거 어떻게
써". There the name is what the sentence is about, and another run of the skill is not what it asks for.

`reports_on_the_name(rest)` reads the words after the name. They report on the skill when they ask for nothing
(`asks_for_something`: a courtesy, a clause that opens on a request verb, a Korean request ending) and:

- the name, or a word for the skill's own run or output, is the subject of a finite verb. In English the verb comes
  next, after adverbs: a form of be, do or have, a negative modal, a verb like seems or keeps, a past tense that is
  only a past tense ("broke my build", "took 20 minutes"), or a past tense or third-person verb that a determiner,
  a pronoun, a preposition, an adverb, a number or the end follows ("failed again", "crashed, logs attached",
  "hangs on the second node"). A bare noun after it makes it an adjective and the words a request's object
  ("ultrawork failed tests", "dag ulw stuck jobs"); "has to" or "needs to" makes it an instruction. In Korean, which
  puts the verb last, the predicate comes after adverbs, counts, clauses the name is also the subject of ("돌다가")
  and the skill's own words, and it is a complaint ("이상해", "먹통이야", "또 터져"), negated ("안 돼") or said with a
  degree adverb ("너무 느려");
- the predicate is a Korean past tense, and no noun of its own comes before it: one with a subject particle is the
  subject ("dag ulw 빌드가 깨졌어" is a problem for the skill to work on), and so may be a bare one ("빌드 깨졌어"),
  unless 먹다 or 버리다 says the name did it ("브랜치 날려먹었어"). An object, a place or a time leaves the name the
  subject ("테스트를 다 지웠어", "결제 모듈에서 또 실패했어"). A past tense asked as a question asks how the work went,
  whatever comes before it ("결제 모듈 리팩터링 끝났어?");
- the name is the cause ("때문에 빌드 깨졌어", "탓에 다 꼬였어"), even with a request after it: "dag ulw 때문에 생긴
  버그 고쳐줘" asks for a fix, not for the skill that caused the bug. So is a use of the skill before what happened
  ("쓰니까 빌드 깨졌어", "돌렸더니 망가졌어") when nothing is asked for and what follows is a past, a complaint or a
  negation; "하고 나서 배포해" asks for the run first. Credit (덕분에) reports only when nothing is asked for: "덕분에 빨리
  끝났어, 결제 모듈도 이렇게 해줘" asks for more of the same.
A stop is no request, so "dag ulw hung, kill it" still reports. A clause that only asks again is one ("…, one more
time", "…, 다시"), and so is the name said again ("ultrawork failed again, ultrawork it again").

`stops_the_name(rest)` reads a stop word with nothing else, said once or twice ("ultrawork stop", "loop 계속 멈춰",
"ultrawork 멈춰 멈춰"), or followed by clauses that ask for nothing ("ultrawork stop, too expensive"): about the run in
progress, which a new run of the skill would not stop.

`asks_after_the_name(rest)` reads a question: a question word that opens a question rather than a clause of its own
("why is the build failing", not "why the build is failing"; "왜 깨졌어", not "왜 깨지는지"), with nothing asked for.
After most skills only a question about the skill itself counts ("what is it", "how does it work", "이거 어떻게 써",
"결과 어디 있어"), since a skill may be asked a question as its work ("research what are the best vector databases").
After a skill that only runs work (`RUN_ONLY_SKILLS`) any question that opens with a question word does, and after
another coined name too unless it deliberates the work ("ralplan 결제 api 뭐야" is answered, "ralplan how should we
split the billing refactor" plans). After a coined name a question that ends in a question mark and asks for nothing
is answered whatever it opens with ("ultrawork still running?"). After a skill whose own work is answering what is
running, wrong or left (`QUESTION_IS_THE_WORK`), such a question is its work ("app-debugging what's wrong").

`about_the_skill(name, rest)` is what the callers read: a stop, a question, or a report after a name that can only
mean the skill (`COINED_NAMES`, or a hyphenated id written with its hyphens). A name that is also an ordinary word
may open a sentence about the person's own thing ("frontend is broken on safari", "loop is stuck at 99% cpu"), so a
report after it keeps today's route.

A determiner after the name starts a clause of its own, so the name before it stays the request's ("ultrawork the
build is broken", "dag ulw the build that broke"), and so does a Korean noun of its own before a present-tense
predicate ("dag ulw 결제 모듈이 느려"). A colon or comma after the name addresses the skill ("ultrawork: the build is
broken"). A sigil, ``use``, ``run``, ``please`` or a polite opener before the name is the caller's to read: those
stay invocations. Callers pass the executable text after the name. Stdlib only: the plugin bundle loads this file
with Hermes' interpreter (#1623).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# ── English ──────────────────────────────────────────────────────────────

_EN_ADVERBS: Final = frozenset(
    {"just", "again", "still", "also", "always", "never", "totally", "completely", "really", "actually", "suddenly",
     "sometimes", "often", "literally", "apparently", "somehow", "already", "even", "constantly", "randomly",
     "silently", "badly", "entirely", "basically", "seriously", "probably", "clearly", "only", "once", "twice",
     "finally", "immediately", "instantly", "repeatedly"}
)
# A word for the skill's own run, output or behaviour, which the name before it owns ("dag ulw output was wrong",
# "ulw dag ordering looks wrong").
_EN_OWN_WORDS: Final = frozenset(
    {"output", "outputs", "result", "results", "run", "runs", "plan", "plans", "review", "reviews", "report",
     "reports", "answer", "answers", "response", "responses", "ordering", "graph", "speed", "performance", "log",
     "logs", "summary", "diff", "patch", "changes", "agents", "workers", "nodes", "session", "cost", "bill", "verdict",
     "score", "suggestions", "advice"}
)
# Always a finite verb right after a subject: be, do and have, the negative modals, the linking verbs, and the past
# tenses that are never adjectives.
_EN_FINITE: Final = frozenset(
    {"is", "are", "was", "were", "isn't", "aren't", "wasn't", "weren't", "isnt", "arent", "wasnt", "werent", "does",
     "did", "doesn't", "didn't", "doesnt", "didnt", "has", "had", "hasn't", "hadn't", "hasnt", "hadnt", "haven't",
     "can't", "cant", "cannot", "couldn't", "couldnt", "won't", "wont", "wouldn't", "wouldnt", "shouldn't",
     "shouldnt", "seems", "seemed", "looks", "looked", "feels", "sounds", "gets", "got", "goes", "went", "keeps",
     "kept", "became", "becomes", "stays", "stayed", "remains", "remained", "turned", "takes", "took", "broke",
     "ran", "ate", "threw", "blew", "froze", "fell", "forgot", "began", "came", "gave", "undid", "wrote", "overwrote",
     "rewrote", "stole", "tore", "woke", "chose", "spoke", "swore", "sank", "saw", "flew", "grew", "knew", "drew",
     "shook", "bit", "hid", "fought", "thought", "brought", "caught", "bought", "taught", "refuses", "refused"}
)
# A finite verb only when what follows it is not a bare noun: past tenses that are also adjectives ("hung jobs",
# "stuck jobs", "lost updates") and third-person verbs that are also plural nouns ("loops", "breaks", "costs").
_EN_FINITE_BEFORE_A_NON_NOUN: Final = frozenset(
    {"made", "hung", "left", "lost", "sent", "spent", "stuck", "built", "found", "held", "led", "meant", "said",
     "sold", "told", "understood", "felt", "dealt", "slid", "spun", "dug", "swept", "bent", "burnt", "spilt",
     "fails", "crashes", "breaks", "hangs", "freezes", "stalls", "spawns", "ignores", "skips", "loops", "stops",
     "deletes", "overwrites", "drops", "loses", "returns", "throws", "times", "costs", "eats", "burns", "needs",
     "makes", "uses", "runs", "spends", "creates", "opens", "leaves", "writes", "produces", "generates", "misses",
     "forgets", "picks", "says", "shows", "gives", "hits", "chokes", "dies", "errors", "explodes", "starts"}
)
# "has to", "needs to": what the work must do, an instruction rather than a report.
_EN_OBLIGATION: Final = frozenset({"has", "have", "had", "needs", "needed", "got", "is", "are", "was"})
# A word ending in -ed that is no past tense.
_EN_NOT_PAST: Final = frozenset(
    {"embed", "imbed", "shed", "shred", "bed", "red", "wed", "sled", "infrared", "sacred", "hundred", "kindred",
     "naked", "wicked", "rugged", "ragged", "jagged", "crooked", "beloved", "seed", "speed", "need", "feed", "proceed",
     "exceed", "succeed", "bleed", "breed", "heed", "weed", "reed", "deed"}
)
# What may follow a past tense or a third-person verb and leave it the verb: a determiner, a pronoun, a preposition,
# a conjunction, or an adverb of time, degree or manner.
_EN_AFTER_A_VERB: Final = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "my", "our", "your", "their", "his", "her", "its", "it",
     "them", "me", "us", "him", "everything", "everyone", "all", "both", "some", "any", "no", "every", "each", "half",
     "on", "in", "at", "for", "with", "without", "from", "to", "into", "onto", "out", "up", "down", "off", "over",
     "under", "after", "before", "during", "while", "when", "because", "since", "until", "till", "and", "but", "or",
     "so", "then", "again", "twice", "once", "forever", "halfway", "midway", "anyway", "too", "very", "way", "really",
     "completely", "totally", "today", "yesterday", "tonight", "overnight", "last", "early", "late", "hard", "badly",
     "fast", "slowly", "immediately", "instantly", "silently", "repeatedly", "constantly", "now", "here", "there",
     "everywhere", "somewhere", "like", "as", "than", "about", "around", "through", "across", "against"}
)
_EN_WORD: Final = re.compile(r"[a-z0-9][a-z0-9'’]*")
_EN_NUMBER: Final = re.compile(r"^(?:\d[\d.,:%$]*\w*|one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?)$")
# A request in the words: a courtesy or an ask ("please", "can you", "let's"), or a clause that opens on a request
# verb ("…, fix it", "… and rerun it", "…; just ship it"). A stop is not one ("…, kill it").
_EN_ASK: Final = re.compile(
    r"\b(?:please|pls|plz|can\s+you|could\s+you|would\s+you|will\s+you|let'?s|let\s+us|need\s+you\s+to|"
    r"want\s+you\s+to|i\s+(?:want|need)\s+(?:you|it|this|that)\s+to)\b"
)
_EN_CLAUSE_BREAK: Final = re.compile(r"[,;.!?:]+|\s(?:and|so|then|but|now)\s")
_EN_CLAUSE_FILLERS: Final = frozenset({"just", "now", "then", "so", "and", "also", "ok", "okay", "but", "maybe", "pls"})
_EN_REQUEST_VERBS: Final = frozenset(
    {"fix", "retry", "rerun", "re-run", "run", "restart", "resume", "continue", "finish", "redo", "debug",
     "investigate", "check", "look", "find", "figure", "help", "make", "try", "start", "do", "ship", "merge",
     "deploy", "revert", "undo", "roll", "rollback", "clean", "write", "add", "update", "open", "file", "implement",
     "build", "go", "use", "give", "show", "tell", "explain", "dig", "diagnose", "triage", "repair", "patch",
     "resolve", "handle", "sort", "rewrite", "split", "review", "test", "retest", "rebuild", "reset", "remove",
     "delete", "get", "take", "complete", "salvage", "recover", "rescue", "reopen", "close", "refactor", "migrate",
     "execute", "apply", "proceed", "carry", "follow", "begin", "kick", "launch", "push", "land", "publish", "release",
     "commit", "approve", "keep", "pick"}
)
# A clause that only asks for the run again ("…, one more time", "…. again.", "…, another try").
_EN_AGAIN_ONLY: Final = frozenset({"again", "one", "more", "time", "once", "pass", "another", "try", "go", "shot", "round",
                                   "two", "attempt"})
_KO_AGAIN_ONLY: Final = frozenset({"다시", "한", "번", "한번", "더", "재실행", "재시도", "이어서", "마저", "또", "계속"})

# ── Korean ───────────────────────────────────────────────────────────────

# The name as the cause of what happened.
_KO_CAUSE: Final = re.compile(r"^(?:때문에|때문|땜에|땜시|때매|때메|탓에|탓|(?:으)?로\s*인(?:해|한))(?![가-힣])")
_KO_CREDIT: Final = re.compile(r"^(?:덕분에|덕에)(?![가-힣])")
# A use of the skill before what happened: "쓰니까", "돌렸더니", "돌리다가", "실행했는데".
_KO_USE_CLAUSE: Final = re.compile(
    r"^(?:을|를)?\s*(?:쓰|써|썼|돌리|돌렸|실행하|실행했|했|하)(?:니까|니|더니|는데|다가|고\s*나서|고\s*나니)(?![가-힣])"
)
# Words that may sit between the name and a present-tense predicate without starting a clause of their own: adverbs,
# and a word for the skill's own run or output, with or without a particle.
_KO_ADVERBS: Final = frozenset(
    {"또", "또다시", "계속", "계속해서", "자꾸", "너무", "넘", "진짜", "정말", "참", "엄청", "되게", "완전", "왜", "왜케",
     "이렇게", "저렇게", "그렇게", "좀", "아직", "아직도", "여전히", "이번에도", "결국", "그냥", "방금", "아까", "갑자기",
     "다시", "오늘", "어제", "오늘도", "매번", "항상", "맨날", "늘", "하나도", "전혀", "제대로", "잘", "영", "도무지",
     "도저히", "벌써", "역시", "안", "못", "오래", "많이", "자주", "느리게", "늦게", "한참", "계속계속", "지금", "이제",
     "다", "전부", "모두", "싹", "몽땅", "이상하게", "엉뚱하게"}
)
# The adverbs that make any present-tense predicate after them a report: degree, a "why" that complains, and the
# negative ones.
_KO_DEGREE: Final = frozenset(
    {"너무", "넘", "진짜", "정말", "참", "엄청", "되게", "완전", "왜", "왜케", "이렇게", "자꾸", "맨날", "매번", "여전히",
     "아직도", "하나도", "전혀", "영", "도무지", "도저히", "안", "못"}
)
_KO_OWN_WORD: Final = re.compile(
    r"^(?:결과|결과물|출력|아웃풋|로그|속도|성능|동작|작동|실행|런|플랜|계획|리뷰|리포트|보고서|답|답변|응답|그래프|순서"
    r"|에이전트|비용|요금|돈|토큰|진행|작업|품질|퀄리티|판정|점수|제안|요약)(?:이|가|은|는|도|이랑|만|까지|조차|마저)?$"
)
# A particle that makes a noun the subject or topic, so the predicate is that noun's and not the name's ("dag ulw
# 빌드가 깨졌어" is a problem for the skill to work on); and one that leaves the name the subject: an object, a place,
# a time, or a clause the name is also the subject of ("dag ulw 브랜치를 날렸어", "dag ulw 결제 모듈에서 또 실패했어",
# "dag ulw 돌다가 죽었어").
_KO_SUBJECT_CASE: Final = re.compile(r"\S(?:이|가|은|는|도)$")
_KO_OTHER_CASE: Final = re.compile(r"\S(?:을|를|에서|에게|한테|에|으로|로|까지|부터|동안)$")
_KO_CONNECTIVE: Final = re.compile(r"\S(?:다가|는데|더니|면서|니까|어서|아서|고서|고)$")
# 먹다 or 버리다 after a past verb says the act was done, and done wrong, by its subject, so a bare noun before it is
# the object ("dag ulw 브랜치 날려먹었어", "dag ulw 테스트 다 지워버렸어").
_KO_DONE_TO_IT: Final = re.compile(r"(?:먹었|버렸)")
_KO_COUNT: Final = re.compile(r"^(?:\d+\S*|한|두|세|네|다섯|여러|몇|번|번이나|번째|시간|분|초|시간이나|분이나)$")
# A complaint said as a present-tense predicate (slow, odd, failing, broken, not working, a mess), and a complaint noun
# a predicate after it reports on ("무한루프 돌아", "에러가 나"). A complaint noun that ends the words is a task's
# object, not a report ("dag ulw 타임아웃 에러").
_KO_COMPLAINT: Final = re.compile(
    r"느려|느리|느림|느린데|이상해|이상하|이상함|이상한데|별로|구려|구림|구린데|엉망|먹통|고장났|최악|"
    r"실패해|실패하|실패함|터져|터지|죽어|죽음|꼬여|꼬임|깨져|깨짐|버벅|답답|비싸|무거워|무겁|틀려|틀림|엉뚱|안\s*돼|안\s*됨|"
    r"안돼|안됨"
)
_KO_COMPLAINT_NOUN: Final = re.compile(r"^(?:에러|오류|무한\s*루프|무한루프|렉|버그|실패|고장|타임아웃)(?:이|가|도|만)?$")
# The vowels of a casual -어/-아 predicate (느려, 걸려, 돌아, 해, 돼) and the other endings a sentence closes on.
_KO_PREDICATE_VOWELS: Final = frozenset({0, 1, 4, 6, 9, 10, 14})  # ㅏ ㅐ ㅓ ㅕ ㅘ ㅙ ㅝ
_KO_PREDICATE_ENDINGS: Final = "다네요지냐니래대데군음함됨임듯"
_KO_NEGATED: Final = re.compile(r"(?:^|\s)(?:안|못)\s*\S+$|지\s*(?:않|못)\S*$|안\s*(?:돼|됨|되)")
# A request: 주다 as the request auxiliary, 부탁, a suggestion or command ending, or a casual imperative of a verb of
# work at the end of a clause. A stop is not one ("꺼줘", "취소해줘").
_KO_REQUEST: Final = re.compile(
    r"(?:줘|줘요|줘라|주세요|주십시오|주라|줄래|줄래요|주실래요|주시겠어요|주세여|줘여|부탁\S*|하자|합시다|해라|하세요|해봐|해봐요|하셈"
    r"|줄\s*수\s*있\S*|주실\s*수\s*있\S*)$"
)
_KO_CASUAL_REQUEST: Final = frozenset(
    {"돌려", "고쳐", "해", "다시해", "만들어", "바꿔", "봐", "끝내", "수정해", "진행해", "실행해", "처리해", "구현해", "확인해",
     "정리해", "재시도해", "재실행해", "시작해", "마무리해", "고쳐봐", "찾아", "찾아봐", "알려", "보여", "살려", "되돌려", "복구해"}
)
_KO_STOP: Final = re.compile(r"취소|중지|중단|정지|종료|멈춰|멈추|그만|꺼")
# A stop word and nothing else after the name ("ultrawork stop", "loop 멈춰", "ultraqa 그만해", "team cancel it"): about
# the run in progress, which another run of the skill would not stop.
_STOP_ALONE: Final = re.compile(
    r"^(?:please\s+)?(?:stop|cancel|abort|halt|kill)(?:\s+(?:it|this|that|now|please|everything|all|the\s+(?:run|job|"
    r"loop|work|task)))*\s*[.!?]*$"
    r"|^(?:(?:지금|바로|당장|그냥|일단|좀|빨리)\s*)*(?:멈춰|멈춰요|멈춰줘|멈춰주세요|그만|그만해|그만해요|그만해줘|꺼|꺼줘|꺼주세요"
    r"|(?:취소|중단|중지|정지)(?:\s*(?:해|해요|해줘|해주세요|하자|좀))?)\s*[.!?~]*$"
)

# Skills that only run work, and the retired engine names that resolve to one: a question after one of these names
# is never its task, whatever it asks about ("ultrawork 결제 api 뭐야"). Any other skill may be asked a question as its
# work ("research what are the best vector databases", "ralplan how should we split the billing refactor"), so only
# a question about the skill itself is answered there.
RUN_ONLY_SKILLS: Final = frozenset(
    {"dag-ulw", "ultrawork", "ulw", "maestro", "team", "ralph", "ulr", "ultragoal", "ulg", "ultraprocess", "ulp"}
)
# Names that are no ordinary word, so a sentence they open is about the skill. A name that is also an ordinary word
# ("plan", "loop", "research", "wiki", "frontend", "context", "team") may open a sentence about the person's own plan,
# loop or frontend ("frontend is broken on safari"), and so may a hyphenated name written with spaces ("code review
# takes forever"): a report after one of those keeps today's route. A hyphenated name written with its hyphens
# ("code-review", "build-failure-triage") is the skill's id. A stop or a question about it does not depend on the word.
COINED_NAMES: Final = frozenset(
    {"dag ulw", "ulw dag", "ultrawork", "ulw", "ultraqa", "ultraperf", "ralplan", "maestro", "ralph", "ulr",
     "ultragoal", "ulg", "ultraprocess", "ulp"}
)

# ── a question about the skill ───────────────────────────────────────────

# A question word, not its "any" form ("뭐든", "어떻게든", "언제든", "어디든", "누구든", "뭐나").
_KO_WH: Final = re.compile(r"^(?:뭐|뭔|무엇|무슨|어떻게|왜|언제|어디|누가|누구|몇|얼마)(?!든|나|라도|라든)")
_EN_WH: Final = frozenset({"what", "what's", "whats", "how", "how's", "why", "when", "where", "where's", "who", "which"})
# A question word with a subject of its own after it opens a clause, not a question: what it names is the work ("dag
# ulw why the build is failing", "ultrawork how to split the billing refactor"). A question puts the verb first ("why
# is the build failing").
_EN_EMBEDDED_SUBJECT: Final = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "my", "our", "your", "their", "his", "her", "its", "it", "we",
     "i", "you", "they", "he", "she", "to"}
)
# The same clause in Korean ends on -는지 or -ㄴ지 ("왜 빌드가 깨지는지", "뭐가 바뀌었는지").
_KO_EMBEDDED_QUESTION: Final = re.compile(r"[는은던]지(?:를|도)?$|[가-힣]지를$")
# The words a question about the skill itself is made of, besides its question word, adverbs and the skill's own
# words: pronouns for the skill or its run, the person asking, and general verbs ("what is it", "how does it work",
# "how do I cancel it", "what now", "이거 어떻게 써", "이제 뭐 해", "결과 어디 있어"). Any other word names a subject of its
# own, and a question about that may be the skill's work ("research what are the best vector databases").
_EN_ABOUT_IT: Final = frozenset(
    {"it", "this", "that", "i", "we", "you", "me", "us", "my", "our", "its", "is", "are", "was", "were", "does", "do",
     "did", "can", "could", "should", "will", "would", "has", "have", "had", "'s", "s", "the", "a", "to", "of", "for",
     "with", "about", "up", "there", "here", "now", "next", "work", "works", "working", "use", "using", "cost",
     "costs", "mean", "means", "doing", "done", "take", "takes", "cancel", "stop", "finish", "finished", "running",
     "left", "happen", "happened", "happening", "going", "go", "get", "got", "help", "wrong", "much", "long", "status",
     "then", "anyway", "exactly", "even", "supposed", "be", "need", "start", "started", "started?", "on"}
)
_KO_ABOUT_IT: Final = re.compile(
    r"^(?:이거|이건|이게|그거|그건|그게|저거|얘|쟤|나|내가|우리|이제|지금|다|써|쓰는|쓰면|써야|쓰는\s*거|사용|사용해|사용하는|돌아|돌아가|"
    r"돌아가는|돌려|끝나|끝나는|끝날|해|해야|해야\s*돼|돼|되|이래|하는|하는\s*거|거|거야|건데|건지|문제|문제야|문제가|상태|상태가|있어|남았어|"
    r"걸려|걸리|개|된|된\s*거|되는|되는\s*거|하면|쓰지)\S{0,2}$"
)
_HANGUL_PAST_FINAL: Final = 20  # ㅆ as a syllable's final consonant: 었, 았, 했, 됐, 졌


def _folded(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _has_past(word: str) -> bool:
    """Whether a Korean word holds a past tense (a syllable closed by ㅆ), not the future 겠 or 있 (there is)."""
    for character in word:
        code = ord(character) - 0xAC00
        if 0 <= code < 11172 and code % 28 == _HANGUL_PAST_FINAL and character not in "겠있":
            return True
    return False


def _words(text: str) -> list[str]:
    return [word.strip(".,!?~…;:\"'()[]") for word in text.split() if word.strip(".,!?~…;:\"'()[]")]


def asks_for_something(rest: str) -> bool:
    """Whether ``rest`` asks for something other than a stop: an English courtesy or a clause opening on a request
    verb, or a Korean request ending."""
    folded = _folded(rest)
    if _EN_ASK.search(folded):
        return True
    # The first clause is the one the name opens, read by the callers: "run" there is the skill's run ("dag ulw run
    # finished"), not a request.
    for clause in _EN_CLAUSE_BREAK.split(f" {folded} ")[1:]:
        words = [word for word in _EN_WORD.findall(clause) if word not in _EN_CLAUSE_FILLERS]
        if words and words[0] in _EN_REQUEST_VERBS:
            return True
        if words and set(words) <= _EN_AGAIN_ONLY and set(words) & {"again", "more", "another", "round"}:
            return True
    for clause in re.split(r"[,;.!?~]+", folded):
        words = _words(clause)
        if not words or _KO_STOP.search(words[-1]):
            continue
        if set(words) <= _KO_AGAIN_ONLY:
            return True
        if _KO_REQUEST.search(words[-1]):
            return True
        # A casual 해 or 돌려 is a request unless a negation or a question word comes right before it ("작동 안 해",
        # "이제 뭐 해", "어떻게 해").
        if words[-1] in _KO_CASUAL_REQUEST and not (
            len(words) > 1 and (words[-2] in {"안", "못"} or _KO_WH.match(words[-2]))
        ):
            return True
        if len(words) > 1 and _KO_REQUEST.search(f"{words[-2]} {words[-1]}"):
            return True
    return False


def _english_reports(words: list[str]) -> bool:
    """Whether the name before ``words``, or the skill's own word after it, is the subject of a finite verb."""
    index = 0
    while index < len(words) and words[index] in _EN_ADVERBS:
        index += 1
    if index < len(words) and words[index] in _EN_OWN_WORDS:
        index += 1
        while index < len(words) and words[index] in _EN_ADVERBS:
            index += 1
    if index >= len(words):
        return False
    verb = words[index]
    following = words[index + 1] if index + 1 < len(words) else ""
    if following == "to" and verb in _EN_OBLIGATION:
        return False  # "ultrawork has to migrate billing": an instruction, not a report
    if verb in _EN_FINITE:
        return True
    if verb == "not":
        return following.endswith(("ing", "ed")) or following in {"again", "done", "finished"}
    past = verb.endswith("ed") and len(verb) > 4 and verb not in _EN_NOT_PAST and not verb.endswith("eed")
    if not past and verb not in _EN_FINITE_BEFORE_A_NON_NOUN:
        return False
    return (
        not following
        or following in _EN_AFTER_A_VERB
        or following.endswith("ly")
        or bool(_EN_NUMBER.match(following))
    )


def _predicate_like(word: str) -> bool:
    """Whether a Korean word closes like a predicate rather than a noun (see `_KO_PREDICATE_VOWELS`)."""
    code = ord(word[-1]) - 0xAC00
    if not 0 <= code < 11172:
        return False
    return (code % 28 == 0 and code // 28 % 21 in _KO_PREDICATE_VOWELS) or word[-1] in _KO_PREDICATE_ENDINGS


def _adnominal(word: str) -> bool:
    """Whether a Korean word ends like a verb before a noun it describes ("느린 쿼리", "깨진 테스트")."""
    code = ord(word[-1]) - 0xAC00
    return 0 <= code < 11172 and code % 28 in (4, 8) and word not in _KO_ADVERBS  # ㄴ, ㄹ


def _korean_reports(words: list[str], question: bool) -> bool:
    """Whether the Korean words after the name end on a report with the name as its subject (see the docstring)."""
    predicate, before = words[-1], words[:-1]
    past = _has_past(predicate)
    if past and question:
        return True  # "ultrawork 결제 모듈 리팩터링 끝났어?" asks how the work went
    if before and _adnominal(before[-1]) and not past:
        return False  # the last word is a noun a verb describes: "dag ulw 느린 쿼리"
    bare = False  # a noun with no particle right before the verb may be its subject ("빌드 깨졌어")
    for word in before:
        if (
            word in _KO_ADVERBS
            or _KO_OWN_WORD.match(word)
            or _KO_COUNT.match(word)
            or _KO_COMPLAINT.search(word)
            or _KO_COMPLAINT_NOUN.match(word)
        ):
            continue
        if _KO_CONNECTIVE.search(word) or (past and _KO_OTHER_CASE.search(word)):
            bare = False
            continue
        if not past or _KO_SUBJECT_CASE.search(word):
            return False  # a subject of its own: "결제 모듈이 느려", "빌드가 깨졌어"
        bare = True
    if past:
        return not bare or bool(_KO_DONE_TO_IT.search(predicate))
    if _KO_COMPLAINT.search(predicate):
        return True
    if not _predicate_like(predicate) or _KO_COMPLAINT_NOUN.match(predicate):
        return False
    return bool(
        any(word in _KO_DEGREE for word in before)
        or _KO_NEGATED.search(" ".join(words))
        or any(
            (_KO_COMPLAINT_NOUN.match(word) or _KO_COMPLAINT.search(word)) and not word.endswith("면") for word in before
        )
    )


# Skills whose own work is answering a question about what is running, what is wrong or what is left: after one of
# these only a question about the skill itself counts ("app-debugging what's wrong" asks for its work).
QUESTION_IS_THE_WORK: Final = frozenset(
    {"running-work-board", "harness-session-inventory", "skill-health", "doctor", "cancel", "live-incident-response",
     "verification-gate", "app-debugging", "build-failure-triage", "native-debugging", "failure-signal-audit",
     "production-audit", "workspace-audit", "security-event-response", "deploy-and-monitor", "run-efficiency"}
)
_EN_RUN_STATE: Final = frozenset(
    {"up", "there", "here", "now", "next", "doing", "done", "finished", "finish", "running", "left", "happen",
     "happened", "happening", "going", "go", "get", "got", "wrong", "status", "then", "anyway", "exactly", "even",
     "start", "started", "started?", "on"}
)
_KO_RUN_STATE: Final = re.compile(
    r"^(?:이제|지금|다|돌아|돌아가|돌아가는|돌려|끝나|끝나는|끝날|문제|문제야|문제가|상태|상태가|있어|남았어|걸려|걸리|개|된|된\s*거|되는|되는\s*거)\S{0,2}$"
)


def asks_after_the_name(rest: str, *, about_anything: bool = False, run_state: bool = True) -> bool:
    """Whether ``rest``, the words after a skill's bare name at the head of a message, ask a question and nothing
    else: about the skill itself, or with ``about_anything`` about anything ("dag ulw 결제 api 뭐야")."""
    folded = _folded(rest)
    if not folded or folded[0] in ":," or asks_for_something(folded):
        return False
    words = _words(folded)
    if re.search(r"[가-힣]", folded):
        if not any(_KO_WH.match(word) for word in words) or _KO_EMBEDDED_QUESTION.search(words[-1]):
            return False
        return about_anything or all(
            _KO_WH.match(word) or (run_state and (word in _KO_ADVERBS or _KO_OWN_WORD.match(word) or _has_past(word)))
            or (_KO_ABOUT_IT.match(word) and (run_state or not _KO_RUN_STATE.match(word)))
            for word in words
        )
    words = _EN_WORD.findall(folded)
    if not ({"now what", "what now", "what next"} & {" ".join(pair) for pair in zip(words, words[1:])}) and not (
        words and words[0] in _EN_WH and not (len(words) > 1 and words[1] in _EN_EMBEDDED_SUBJECT)
    ):
        return False
    about_it = _EN_ABOUT_IT if run_state else _EN_ABOUT_IT - _EN_RUN_STATE
    return about_anything or all(
        word in _EN_WH or word in about_it or (run_state and (word in _EN_ADVERBS or word in _EN_OWN_WORDS))
        for word in words
    )


def stops_the_name(rest: str) -> bool:
    """Whether ``rest``, the words after a skill's bare name, are a stop word and nothing else (`_STOP_ALONE`), or a
    Korean stop word after adverbs ("계속 멈춰", "자꾸 꺼져"): a stop, or a complaint about stopping, a run either way
    it is not."""
    folded = _folded(rest)
    if _STOP_ALONE.match(folded):
        return True
    words = _words(folded)
    if words and all(re.fullmatch(r"stop|cancel|abort|halt|kill|멈춰|그만|그만해|취소|중단|중지|정지", word) for word in words):
        return True
    if bool(words) and bool(_KO_STOP.match(words[-1])) and all(word in _KO_ADVERBS for word in words[:-1]):
        return True
    parts = re.split(r"\s*[,;.!?~]+\s*", folded, maxsplit=1)
    return len(parts) == 2 and bool(parts[1]) and stops_the_name(parts[0]) and not asks_for_something(", " + parts[1])


def about_the_skill(name: str, rest: str) -> bool:
    """Whether ``rest``, the words after a skill's bare name as the message writes it (``name``), stop its run, ask
    about it, or report on it, rather than ask to run it. A report counts only after a name that can mean nothing
    but the skill (`COINED_NAMES`, or a name with its hyphens)."""
    written = _folded(name)
    key = written.replace(" ", "-")
    run_only = key in RUN_ONLY_SKILLS
    deliberates = re.search(
        r"\b(?:should|shall|best\s+way|how\s+(?:do|can|could|would)\s+(?:we|i))\b|어떻게\s*(?:\S+\s+)?(?:나눠|나누|해야|하면|할까|진행)",
        _folded(rest),
    )
    if stops_the_name(rest) or asks_after_the_name(
        rest,
        about_anything=run_only or (written in COINED_NAMES and not deliberates),
        run_state=key not in QUESTION_IS_THE_WORK,
    ):
        return True
    # A1: a yes-or-no question after an engine's name asks about it, as one after dag-ulw's does ("ultrawork safe
    # to use on prod?", "ultrawork still running?", "ultrawork 써도 돼?").
    if (run_only or written in COINED_NAMES) and _folded(rest).endswith(("?", "？")) and not asks_for_something(rest):
        return True
    if re.search(rf"(?<![a-z0-9_-]){re.escape(written)}(?![a-z0-9_-])", _folded(rest)):
        return False  # the name again: "ultrawork failed again, ultrawork it again"
    return (("-" in written and " " not in written) or written in COINED_NAMES) and reports_on_the_name(rest)


def reports_on_the_name(rest: str) -> bool:
    """Whether ``rest``, the words after a skill's bare name at the head of a message, report on that skill rather
    than ask to run it (see the module docstring)."""
    folded = _folded(rest)
    if not folded or folded[0] in ":,":
        return False  # "ultrawork: the build is broken" addresses the skill
    if _KO_CAUSE.match(folded):
        return True
    if asks_for_something(folded):
        return False
    if _KO_CREDIT.match(folded):
        return True
    use = _KO_USE_CLAUSE.match(folded)
    if use:
        tail = _words(folded[use.end():])
        if not tail or _has_past(tail[-1]) or _KO_COMPLAINT.search(tail[-1]) or _KO_NEGATED.search(" ".join(tail)):
            return True
    first_clause = re.split(r"[,;.!?]+", folded, maxsplit=1)[0]
    words = _words(first_clause)
    if not words:
        return False
    if re.search(r"[가-힣]", first_clause):
        return _korean_reports(words, question=folded.endswith(("?", "？")) and first_clause == folded[:-1].strip())
    return _english_reports(_EN_WORD.findall(first_clause))
