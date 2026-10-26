---
name: test-first
description: Write the failing test first, then make it pass
---
You are adding behaviour test-first.

1. Restate the requirement as one assertable sentence.
2. Find the closest existing test file and mirror its style and fixtures.
3. Write the smallest test that fails for the right reason. Run it and paste
   the failure in your reasoning.
4. Implement the minimum code that makes it pass. Run the test again.
5. Run the neighbouring tests to catch regressions.

Do not write implementation before the failing test exists. If the suite has no
runner, use `python -m unittest` or the command in the project README. Report
the test name, the failing run, and the passing run.
