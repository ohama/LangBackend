---
phase: 06-pattern-matching-data-structures
plan: 01
subsystem: documentation
tags: [pattern-matching, decision-tree, maranget-algorithm, theory, tutorial]

# Dependency graph
requires:
  - phase: 05-custom-mlir-dialect
    provides: Custom MLIR dialect infrastructure (FunLang dialect, lowering passes)
provides:
  - Pattern matching theory foundation (decision tree algorithm)
  - Pattern matrix representation methodology
  - Specialization and defaulting operations theory
  - Exhaustiveness checking concepts
  - Theoretical foundation for Chapter 18-19 MLIR implementation
affects: [06-02-list-operations, 06-03-match-compilation, documentation, pattern-matching]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern matrix representation (rows = clauses, columns = scrutinees)"
    - "Decision tree compilation algorithm (Maranget 2008)"
    - "Specialization operation for constructor decomposition"
    - "Defaulting operation for wildcard filtering"
    - "Exhaustiveness checking via empty matrix detection"

key-files:
  created:
    - tutorial/17-pattern-matching-theory.md
  modified:
    - tutorial/SUMMARY.md

key-decisions:
  - "Theory-first approach: Algorithm understanding before MLIR implementation"
  - "Left-to-right column selection heuristic for FunLang (simple and predictable)"
  - "Complete constructor set optimization (no default branch needed)"
  - "Simple exhaustiveness error messages for Phase 6 (detailed analysis deferred)"

patterns-established:
  - "Pattern matrix notation with occurrence vectors"
  - "Recursive compile(P, π) → DecisionTree algorithm"
  - "Base cases: empty (failure), irrefutable (success)"
  - "Specialization: S(c, i, P) filters and expands constructor patterns"
  - "Defaulting: D(i, P) keeps wildcard rows, removes column"

# Metrics
duration: 7min
completed: 2026-02-11
---

# Phase 06 Plan 01: Pattern Matching Theory Summary

**Decision tree compilation algorithm with pattern matrix representation, specialization/defaulting operations, and integrated exhaustiveness checking**

## Performance

- **Duration:** 7 min 21 sec
- **Started:** 2026-02-11T00:58:26Z
- **Completed:** 2026-02-11T01:05:47Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Complete pattern matching theory chapter (2578 lines) covering Maranget's decision tree algorithm
- Pattern matrix representation with detailed examples (single/multiple scrutinees, nested patterns)
- Specialization and defaulting operations with pseudocode and occurrence vector updates
- Complete compilation algorithm with base cases, recursion, and column selection heuristics
- Exhaustiveness checking theory with empty matrix detection and complete constructor sets
- Practice questions and preview of Chapter 18-20 implementation

## Task Commits

Each task was committed atomically:

1. **Task 1: Write Chapter 17 Part 1 - Introduction and Pattern Matrix** - `f23f6a5` (feat)
   - Phase 6 overview, pattern matching importance, problem statement
   - Pattern matrix representation (rows, columns, occurrences)
   - Decision tree concept and comparison with naive if-else
   - Specialization operation with examples
   - Defaulting operation with examples
   - 1531 lines

2. **Task 2: Write Chapter 17 Part 2 - Complete Algorithm and Exhaustiveness** - `84b9bd8` (feat)
   - Complete compilation algorithm (compile function)
   - Base cases: empty matrix (failure), irrefutable row (success)
   - Recursive case: column selection, specialization, defaulting
   - Column selection heuristics (left-to-right, needed-by-most, optimal)
   - Occurrence vector updates (specialization, defaulting)
   - Complete examples: list length, nested patterns
   - Exhaustiveness checking: empty matrix detection, complete constructor sets
   - Summary, practice questions, Chapter 18-20 preview
   - SUMMARY.md updated
   - 1047 lines added

## Files Created/Modified

- `tutorial/17-pattern-matching-theory.md` - Pattern matching theory chapter (2578 lines)
  - Introduction: Phase 6 overview, pattern matching importance
  - Pattern matching problem: FunLang syntax, compilation challenge
  - Pattern matrix representation: rows/columns/occurrences
  - Decision tree concept: structure, benefits, comparison
  - Specialization operation: definition, examples, pseudocode
  - Defaulting operation: definition, examples, pseudocode
  - Complete compilation algorithm: base cases, recursion, heuristics
  - Exhaustiveness checking: empty matrix, complete sets, error reporting
  - Summary: key concepts, workflow, MLIR connection
- `tutorial/SUMMARY.md` - Added Chapter 17 entry

## Decisions Made

**1. Theory-first approach (Chapter 17 before Chapter 18-19)**

Decision tree algorithm is language-agnostic. Understanding it first makes MLIR implementation clearer. Pattern matrix notation becomes the foundation for `funlang.match` operation design.

**2. Left-to-right column selection heuristic for FunLang**

Simpler heuristics (needed-by-most, optimal) have higher implementation cost. For FunLang's typical simple patterns, left-to-right is sufficient and predictable.

**3. Complete constructor set optimization**

List has complete constructor set {Nil, Cons}. When all constructors are tested, default branch is unreachable. This optimization reduces generated code and clarifies exhaustiveness semantics.

**4. Simple exhaustiveness error messages**

Phase 6 provides basic "non-exhaustive match" errors. Detailed missing pattern reconstruction (e.g., "Missing case: Cons(_, Cons(_, _))") is deferred to later phase or bonus section. Keeps implementation focused.

## Deviations from Plan

None - plan executed exactly as written.

All sections covered as specified:
- Part 1: Introduction, problem statement, pattern matrix, decision tree concept
- Part 2: Specialization, defaulting, complete algorithm, exhaustiveness checking

Line count exceeded target (2578 vs 1200+ required). Coverage comprehensive with multiple examples, pseudocode, and MLIR connection.

## Issues Encountered

None. Theory chapter required no code execution, only explanation and examples.

## Next Phase Readiness

**Ready for Chapter 18 (List Operations):**

- Pattern matrix theory established
- Decision tree algorithm understood
- Specialization/defaulting operations defined
- Readers have foundation for `funlang.nil`, `funlang.cons` implementation

**Chapter 18 will implement:**

- `!funlang.list<T>` parameterized type
- `funlang.nil` operation (empty list)
- `funlang.cons` operation (prepend element)
- LLVM representation: `!llvm.struct<(i32, ptr)>` with tag
- Heap allocation with GC_malloc

**Chapter 19 will implement:**

- `funlang.match` operation (region-based pattern matching)
- Lowering to SCF dialect (`scf.index_switch`)
- OpConversionPattern with region handling
- Decision tree compilation in MLIR context

**No blockers.** Theory foundation complete. Ready to build data structures.

---

## Key Concepts Established

**Pattern Matrix Representation:**

```
Matrix P: rows = pattern clauses, columns = scrutinees
Occurrence vector π: access paths to scrutinee values
Cells: patterns (wildcard, constructor, literal)
Actions: code to execute when pattern matches
```

**Specialization Operation:**

```
S(c, i, P) = specialized matrix
- Filter rows compatible with constructor c at column i
- Expand constructor patterns to subpatterns
- Update occurrence vector with subpaths
```

**Defaulting Operation:**

```
D(i, P) = default matrix
- Keep only wildcard rows
- Remove column i (tested)
- Empty result → non-exhaustive match
```

**Compilation Algorithm:**

```python
def compile(P, π):
    if not P:
        return Failure()  # Non-exhaustive
    if is_irrefutable(P[0]):
        return Success(P[0].action)
    column = select_column(P, π)
    constructors = get_constructors(P, column)
    branches = {c: compile(specialize(P, column, c), ...) for c in constructors}
    default = compile(default(P, column), ...)
    return Switch(π[column], branches, default)
```

**Exhaustiveness Checking:**

- Empty pattern matrix → no patterns match → non-exhaustive error
- Complete constructor sets {Nil, Cons} → default unreachable
- Detection integrated in compilation (not separate pass)

**Workflow:**

```
FunLang match expression
  ↓
Pattern matrix + occurrence vector
  ↓
Recursive compilation (specialize/default)
  ↓
Decision tree
  ↓
MLIR IR (scf.index_switch)
  ↓
LLVM IR (switch, br)
```

---

## Connection to MLIR Implementation

**Pattern matrix → `funlang.match` operation:**

Chapter 19 will define `funlang.match` with regions for each pattern case. Region block arguments represent pattern variables (from specialization).

**Decision tree → SCF dialect lowering:**

`funlang.match` lowers to `scf.index_switch` (constructor tag dispatch) and `scf.if` (guards, nested patterns).

**Occurrence vectors → Extraction code:**

Occurrence paths like `list.head`, `list.tail` generate `llvm.extractvalue` and `llvm.load` operations to access subterms.

**Exhaustiveness → Verification:**

`funlang.match` operation will have custom verifier checking that all constructor cases are present or wildcard exists.

---

## Progress Tracking

**Phase 6 - Pattern Matching & Data Structures:**

- [x] Plan 01: Pattern Matching Theory (2578 lines, 7 min)
- [ ] Plan 02: List Operations
- [ ] Plan 03: Match Compilation
- [ ] Plan 04: Functional Programs (map, filter, fold)

**Phase 5 complete. Phase 6 plan 01 complete. Total tutorial: ~34K lines.**

---
*Phase: 06-pattern-matching-data-structures*
*Completed: 2026-02-11*
