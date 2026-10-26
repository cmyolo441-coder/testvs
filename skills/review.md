---
name: review
description: Review a diff like a senior engineer, with evidence
---
You are doing a code review. Work from the actual diff, not from memory.

1. Find the change (`git diff`, `git diff --staged`, or the files named by the user).
2. Read the surrounding code for every hunk you review.
3. Look for: correctness bugs, unhandled errors, races, security issues,
   missing input validation, API misuse, dead code, and tests that assert
   nothing.
4. Run the narrowest test or command that can confirm each suspicion.

Report only findings you can point at, using this shape:

- **file:line** — severity (blocker / major / minor) — what is wrong, why it
  matters, and the smallest fix.

End with a one-line verdict: `ship`, `ship with nits`, or `changes needed`.
Never invent line numbers and never report style opinions as blockers.
