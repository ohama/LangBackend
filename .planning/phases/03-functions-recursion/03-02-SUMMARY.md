---
phase: 03-functions-recursion
plan: 02
subsystem: compiler
tags: [mlir, recursion, mutual-recursion, tail-call-optimization, stack-frames]

# Dependency graph
requires:
  - phase: 03-01
    provides: Function definitions, calls, and func dialect operations
  - phase: 02-core-language-basics
    provides: Expression compilation with scf.if for recursive patterns
provides:
  - Recursive function compilation (self-referential calls)
  - Mutual recursion support (functions calling each other)
  - Stack frame management understanding
  - Tail call optimization concepts (accumulator pattern)
affects: [04-closures, tutorial-continuation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Recursive calls via symbol references (func.call @self)"
    - "Mutual recursion via flat symbol namespace (no forward declarations)"
    - "Accumulator pattern for tail-recursive transformations"
    - "Stack frame visualization for recursion depth analysis"

key-files:
  created:
    - tutorial/11-recursion.md
  modified: []

key-decisions:
  - "Recursive calls work naturally via module-level symbol table (no special handling)"
  - "Mutual recursion supported via lazy verification (order-independent)"
  - "TCO not guaranteed in Phase 3 (educational focus, LLVM may optimize)"
  - "Stack overflow prevention via accumulator pattern (tail recursion)"
  - "Phase 7 will add explicit TCO support (tailcc convention)"

patterns-established:
  - "Self-reference: func.call @factorial inside @factorial definition"
  - "Mutual recursion: is_even ⇄ is_odd via symbol cross-references"
  - "Tail recursion: factorial_tail n acc pattern with accumulator"
  - "Stack frame analysis: depth N → N frames without TCO"

# Metrics
duration: 9min
completed: 2026-02-06
---

# Phase 3 Plan 02: Recursion and Tail Call Optimization Summary

**Complete Chapter 11 covering recursive functions (factorial, fibonacci), mutual recursion (is_even/is_odd), stack frame management, and tail call optimization with accumulator pattern (2513 lines, 13 MLIR examples)**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-06T01:29:58Z
- **Completed:** 2026-02-06T01:38:43Z
- **Tasks:** 2 (combined into single chapter)
- **Files modified:** 1

## Accomplishments

- Chapter 11 (2513 lines) with comprehensive recursion coverage
- Factorial and fibonacci as recursive examples (single and double recursion)
- Mutual recursion pattern (is_even/is_odd) demonstrating cyclic call graphs
- Stack frame management explanation (creation, depth, overflow prevention)
- Tail call optimization concepts (tail position, TCO, accumulator pattern)
- 13 MLIR IR examples showing recursive patterns
- Common errors section (5 error types with solutions)
- Phase 3 completion summary and Phase 4 preview

## Task Commits

Task was committed atomically:

1. **Tasks 1 + 2: Write Chapter 11 (combined)** - `37e6095` (feat)
   - Part 1: Recursion fundamentals, factorial, fibonacci, stack frames
   - Part 2: Mutual recursion, tail calls, code generation, complete examples

**Plan metadata:** (will be committed after SUMMARY.md and STATE.md updates)

## Files Created/Modified

- `tutorial/11-recursion.md` - Complete chapter on recursion with factorial/fibonacci examples, mutual recursion (is_even/is_odd), stack frame management, tail call optimization, and Phase 3 completion

## Decisions Made

1. **Recursive calls via symbol references**: No special compilation needed. `func.call @factorial` inside `@factorial` works naturally via module-level symbol table. Self-reference resolved at verification time.

2. **Mutual recursion via lazy verification**: Functions can reference each other regardless of definition order. MLIR verifies symbol existence after all functions compiled. Enables is_even ⇄ is_odd pattern without forward declarations.

3. **TCO not guaranteed in Phase 3**: Educational focus on recursion concepts. LLVM may optimize with -O2/-O3 but not guaranteed with C calling convention. Phase 7 will add explicit TCO support (tailcc convention, -tailcallopt flag).

4. **Stack overflow prevention**: Accumulator pattern transforms non-tail recursion to tail recursion. Example: factorial → factorial_tail n acc. Enables optimization but doesn't guarantee TCO in current phase.

5. **Code generation reuses Chapter 10**: No special handling for recursion. compileFuncDef and App case handle recursive calls identically to regular calls. Symbol table makes it transparent.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - chapter written smoothly with all requirements met (factorial, fibonacci, mutual recursion, stack frames, tail calls, 10+ MLIR examples, common errors, phase summary).

## Next Phase Readiness

**Phase 3 Complete!**
- All requirements fulfilled: functions (Chapter 10), recursion (Chapter 11)
- Module-level symbol table enables self-reference and mutual recursion
- Stack-based execution model established (heap in Phase 4)
- Calling convention foundation (C convention, LLVM prologue/epilogue)

**Ready for Phase 4 (Closures):**
- Function infrastructure complete (parameters, calls, returns)
- Recursion patterns understood (stack frames, TCO concepts)
- Environment management pattern established (will extend with closure capture)
- Chapter 9 (Boehm GC) ready for heap allocation in closures

**Phase 4 scope:**
- Lambda expressions (fun x -> x + 1)
- Environment capture (free variables)
- Closures as (function pointer, environment pointer) pairs
- Heap allocation for closure environments (GC_malloc)
- Higher-order functions (map, filter, fold)

---
*Phase: 03-functions-recursion*
*Completed: 2026-02-06*
