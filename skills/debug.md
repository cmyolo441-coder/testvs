---
name: debug
description: Reproduce, isolate and prove a fix for a bug
---
You are debugging. Do not change code until you can reproduce the problem.

1. Restate the symptom in one sentence and list the exact reproduction command.
2. Run it. Capture the real output.
3. Narrow the failure: add a focused test, or run the smallest slice that still
   fails. Read the code path from entry point to failure.
4. State the root cause in one sentence, then write the minimal fix.
5. Re-run the reproduction and the nearest test suite.

Report: `symptom`, `root cause`, `fix`, `evidence` (command + result).
If you cannot reproduce it, say so and list what you tried. Do not call a fix
verified without a passing command in the transcript.
