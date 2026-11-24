"""Author instructions used before the original PhoenixRepair submission."""

SUMMARY_GUIDANCE = (
    '\nCandidate patches are proposals, not correctness proofs. The next author produces the final PhoenixRepair patch using the remaining shared generation budget. '
    'Compare actual diffs and observed checks against the original issue. Preserve uncertainty and '
    'identify missing requirements; do not recommend blindly merging incompatible mechanisms.'
)

IMPLEMENTATION_GUIDANCE = """
The localization/summary is an UNVERIFIED proposal. It can omit cooperating edits,
misread a requirement, contain contradictions, or describe an incorrect API.
The ORIGINAL issue description and requirements take precedence over this proposal
and over reviewer claims. Before editing, inspect the relevant current source and
its callers/consumers; briefly identify which decisions you keep or correct and why.
Do not repeat the entire localization or blindly implement every proposed edit.
You may correct the mechanism and expand the file set when source/issue evidence
requires it. Preserve sound parts and implement one complete solution to the issue.
Identify the assumptions most likely to invalidate the solution, without restating
the entire plan. Validate required new behavior and compatibility of existing
consumers separately, using the repository's actual language and toolchain.
Choose focused checks that could falsify the implementation, not only confirm its
happy path. For changed contracts, exercise the real caller-to-callee or
producer-to-consumer path; a helper passing alone does not validate its consumers.
Use actual repository types and execution paths where practical. State what a
simplified mock does not cover. Passing existing tests supports only their exercised
paths. If a relevant check cannot run, record the blocker and use another meaningful
check when feasible. Build failures, filtered tests and unavailable environments
are not evidence of correctness. Never describe an unrun check as passed.
Keep temporary reproductions out of the submitted diff and leave test files unchanged.
Before submitting, inspect the ACTUAL diff against the original requirements:
each required behavior needs an implemented change or verified existing support;
plan prose, TODOs, and a review approval are not implementations or test results.
Check that the code implements the decisions retained from the plan, includes
necessary cooperating edits, and adds no unsupported behavior. Prioritize checks
for the highest-risk changes within the existing stage budget.
This run ends at the first complete PhoenixRepair submission. All 200 model calls are shared
across this task's candidates, summary and final generation. No calls are reserved
for a later review, repair or recheck stage. The stage budget will be stated at
runtime. Submit the complete patch before that allowance ends. Unused earlier-stage
allowances may be transferred to the final stage; no charged call is reset.
When requirements share state or event ordering, verify them together in a concrete
scenario before submission. A change fixing one requirement must not contradict a
related one. Check interacting changes together before submitting the PhoenixRepair patch.
Tests must execute actual repository code and report real status; log formatting
must not mask a failed or unrun check. Unknown validation remains explicitly unknown.
"""
