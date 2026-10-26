---
name: refactor
description: Small, behaviour-preserving cleanups with tests as the guard rail
---
You are refactoring. Behaviour must not change.

1. Read the target module and every caller (`grep` for the public names).
2. Find the tests that cover it; run them before touching anything.
3. Make one structural change at a time: extract a function, rename a
   variable, remove duplication, split a long branch.
4. After each step, run the same tests. Keep the diff small enough to read.
5. Do not reformat untouched code, do not change public APIs without saying so,
   and do not add a dependency.

Report the before/after structure in two or three lines, then the test command
and its result.
