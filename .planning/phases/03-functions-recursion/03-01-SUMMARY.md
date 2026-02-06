---
phase: 03-functions-recursion
plan: 01
subsystem: compiler
tags: [mlir, func-dialect, functions, calling-convention, block-arguments]

# Dependency graph
requires:
  - phase: 02-core-language-basics
    provides: Expression compilation (arith, scf, let bindings, control flow)
  - phase: 01-foundation-interop
    provides: MLIR C API bindings and OpBuilder infrastructure
provides:
  - Top-level named function definitions and calls
  - Function parameters as block arguments
  - Module-level symbol table for function resolution
  - Calling convention foundation (C calling convention via LLVM)
  - Multi-function program compilation
affects: [03-02-recursion, 04-closures, tutorial-continuation]

# Tech tracking
tech-stack:
  added: [mlir-func-dialect]
  patterns:
    - "Function parameters as block arguments (not variables)"
    - "Flat namespace for module-level symbols (no forward declarations needed)"
    - "Environment passing pattern extended to include function parameters"
    - "Program structure: functions + main expression"

key-files:
  created:
    - tutorial/10-functions.md
  modified: []

key-decisions:
  - "Function parameters are block arguments, not let bindings (SSA values from entry block)"
  - "All functions at module level with flat namespace (no forward declarations needed)"
  - "C calling convention (System V ABI) for LLVM lowering"
  - "Phase 3 scope: Top-level named functions only (no closures until Phase 4)"
  - "funlang_main as program entry point (called by runtime.c main)"

patterns-established:
  - "compileFuncDef: name → func.func with block arguments for parameters"
  - "compileProgram: compile all function definitions, then main expression"
  - "Environment includes function parameters (from GetFunctionBlockArg)"
  - "App case in compileExpr: compile args then CreateFuncCall"

# Metrics
duration: 8min
completed: 2026-02-06
---

# Phase 3 Plan 01: Functions and func Dialect Summary

**Complete Chapter 10 covering func dialect operations (func.func, func.call, func.return), function parameters as block arguments, code generation for multi-function programs, and C calling convention explanation**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-06T01:20:01Z
- **Completed:** 2026-02-06T01:28:18Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Chapter 10 (2590 lines) with comprehensive func dialect coverage
- 37 MLIR IR examples demonstrating function compilation patterns
- Complete P/Invoke bindings for func dialect (function types, symbol references, block arguments)
- OpBuilder extensions (CreateFuncOp, CreateFuncCall, CreateFuncReturn)
- Code generation (compileFuncDef, App case, compileProgram)
- Calling convention explanation (System V ABI, LLVM automatic handling)
- Common errors section with 5 error types and solutions

## Task Commits

Each task was committed atomically:

1. **Task 1 + 2: Write Chapter 10 (combined)** - `bc943eb` (feat)
   - Part 1: Concepts, AST extension, P/Invoke bindings (func dialect operations)
   - Part 2: Code generation, calling conventions, complete examples

**Plan metadata:** (will be committed after SUMMARY.md and STATE.md updates)

## Files Created/Modified

- `tutorial/10-functions.md` - Complete chapter on functions with func dialect, block arguments, code generation, and calling conventions

## Decisions Made

1. **Function parameters as block arguments**: Not variables or let bindings, but SSA values from entry block arguments. Requires GetFunctionBlockArg to retrieve and add to environment.

2. **Flat namespace for functions**: MLIR module symbol table allows all functions to reference each other regardless of definition order. No forward declarations needed. Enables mutual recursion naturally.

3. **C calling convention (System V ABI)**: LLVM handles calling convention automatically (register allocation, stack management, platform differences). Phase 3 doesn't implement custom calling convention.

4. **Phase 3 scope boundaries**: Top-level named functions only. No closures, no lambda expressions, no higher-order functions (deferred to Phase 4). Establishes function foundation before adding environment capture.

5. **funlang_main entry point**: Main expression compiled into @funlang_main function (called by runtime.c main). Allows multi-function programs with single entry point.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - chapter written smoothly with all requirements met (func dialect coverage, AST extension, P/Invoke bindings, code generation examples, calling conventions, common errors).

## Next Phase Readiness

**Ready for Chapter 11 (Recursion):**
- Function definition and call infrastructure complete
- Module-level symbol table established (supports recursive calls)
- Calling convention foundation in place
- Block arguments pattern established (will extend to tail call optimization)

**Foundation established:**
- compileFuncDef pattern (create func.func, build environment from parameters, compile body)
- compileProgram pattern (compile all functions, then main)
- Environment pattern extended (let bindings + function parameters)

**Next chapter scope:**
- Recursive function compilation (self-referential calls)
- Mutual recursion (two functions calling each other)
- Tail call optimization (LLVM attributes, stack frame reuse)
- Performance analysis (stack overflow vs optimized recursion)

---
*Phase: 03-functions-recursion*
*Completed: 2026-02-06*
