"""The path signal: a file named in the request, matched against a skill's globs (#1716).

A request about `infra/network/main.tf` is infrastructure work whatever words
carry it. A skill that declares `path_globs` gains `PATH_GLOB_SCORE` when the
request names a matching file and also uses one of the skill's own trigger
words, held-back ones included; a path mentioned in passing, beside none of
them, adds nothing.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

PATH_GLOB_SCORE = 20

# Punctuation a sentence wraps around a path: quotes, backticks, brackets, and
# the comma or full stop that ends the clause.
_PATH_EDGE_CHARACTERS = "`'\"()[]{}<>,;:!?."


def path_mentions(normalized_query: str) -> tuple[str, ...]:
    """Every word that names a file or directory: it has a `/` or an extension."""

    mentions: list[str] = []
    for word in normalized_query.split():
        word = word.strip(_PATH_EDGE_CHARACTERS)
        if "/" in word or "." in word:
            mentions.append(word)
    return tuple(mentions)


def matching_path_glob(normalized_query: str, globs: tuple[str, ...]) -> str:
    """The first glob a named path matches, or "" when none does.

    A glob matches the path or any tail of it that starts after a `/`, so
    `charts/**` matches `deploy/charts/api/values.yaml` the way a gitignore
    pattern would. `*` crosses `/` here, which makes `**` and `*` the same.
    """

    for mention in path_mentions(normalized_query):
        parts = mention.split("/")
        tails = ["/".join(parts[index:]) for index in range(len(parts))]
        for glob in globs:
            if any(fnmatchcase(tail, glob) for tail in tails):
                return glob
    return ""
