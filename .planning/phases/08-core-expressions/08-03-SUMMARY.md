---
phase: 08-core-expressions
plan: 03
subsystem: codegen
tags: [scf, if-else, mlir, structured-control-flow, cf-dialect]

# Dependency graph
requires:
  - phase: 08-01
    provides: comparison operators and boolean literals
  - phase: 08-02
    provides: let bindings and variable references
provides:
  - If-then-else expression compilation using scf.if
  - scf-to-cf lowering pass pipeline
  - cf dialect registration
affects: [08-04, functions, recursion]

# Tech tracking
tech-stack:
  added: [cf dialect]
  patterns: [region-based control flow, scf.yield terminators]

key-files:
  created: []
  modified:
    - src/FunLang.Compiler/CodeGen.fs
    - src/FunLang.Compiler/MlirWrapper.fs
    - src/FunLang.Compiler/MlirBindings.fs
    - tests/FunLang.Compiler.Tests/MlirBindingsTests.fs

key-decisions:
  - "Result type fixed to i32 (FunLang well-typed assumption)"
  - "Non-short-circuit evaluation continues from 08-01"
  - "cf dialect required for scf-to-cf lowering"

patterns-established:
  - "Region compilation: create region+block, compile body, add terminator"
  - "scf.if structure: condition operand, then region, else region"
  - "Pass pipeline order: scf->cf, arith->llvm, cf->llvm, func->llvm"

# Metrics
duration: 4min
completed: 2026-02-12
---

# Phase 8 Plan 03: If-Then-Else Expressions Summary

**If-then-else expressions compile to scf.if with region-based control flow and scf.yield terminators**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-12
- **Completed:** 2026-02-12
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- If expressions compile to scf.if with then/else regions
- Both regions properly terminated with scf.yield
- Pass pipeline extended with scf-to-cf and cf-to-llvm conversions
- cf dialect added to standard dialects
- 11 unit tests covering all if-else scenarios

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement If expression compilation** - `ae103ae` (feat)
2. **Task 2: Add lowering pass for scf dialect** - `4a11f79` (feat)
3. **Task 3: Add unit tests for if-else expressions** - `b75bf8d` (test)

## Files Created/Modified

- `src/FunLang.Compiler/CodeGen.fs` - Added If case to compileExpr using scf.if
- `src/FunLang.Compiler/MlirWrapper.fs` - Added cf dialect to LoadStandardDialects
- `src/FunLang.Compiler/MlirBindings.fs` - Added mlirGetDialectHandle__cf__ P/Invoke
- `tests/FunLang.Compiler.Tests/MlirBindingsTests.fs` - 11 if-else test cases

## Decisions Made

1. **Result type i32** - Since FunLang is well-typed, we assume branches return i32 (integer results). Type inference can be added later when needed for boolean-returning if-else.

2. **cf dialect required** - The scf-to-cf pass lowers scf.if to cf.cond_br/cf.br, which then need cf-to-llvm to reach LLVM dialect.

3. **Pass pipeline ordering** - scf-to-cf must come first, then arith-to-llvm, cf-to-llvm, func-to-llvm, and finally reconcile-unrealized-casts.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - the implementation followed the plan precisely. The scf.if region structure with scf.yield terminators worked as documented in MLIR.

## Next Phase Readiness

- If-then-else expressions complete, enabling control flow in FunLang
- Ready for 08-04: Function Support (Lambda and Application)
- All infrastructure (dialects, pass pipeline) supports structured control flow

---
*Phase: 08-core-expressions*
*Completed: 2026-02-12*
