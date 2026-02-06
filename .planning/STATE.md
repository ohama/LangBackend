# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Each chapter produces a working compiler for all features covered so far
**Current focus:** Phase 1 Complete → Phase 2 - Core Language Basics

## Current Position

Phase: 3 of 7 (Functions & Recursion)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-02-06 — Completed 03-01-PLAN.md (Functions and func Dialect)

Progress: [████████░░] 80% (8/10 estimated total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 5 min
- Total execution time: 0.67 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 - Foundation & Interop | 3/3 | 16 min | 5 min |
| 2 - Core Language Basics | 4/4 | 19 min | 5 min |
| 3 - Functions & Recursion | 1/2 | 8 min | 8 min |

**Recent Trend:**
- Last plan: 03-01 (8 min)
- Previous: 02-04 (6 min)
- Previous: 02-03 (5 min)
- Previous: 02-02 (4 min)
- Previous: 02-01 (4 min)
- Trend: Chapter length correlates with duration (2590 lines = 8 min)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: F# implementation using P/Invoke to MLIR-C API
- [Roadmap]: Custom MLIR Dialect approach for FunLang-specific operations
- [Roadmap]: Incremental tutorial structure with working compiler at each chapter
- [Roadmap]: Boehm GC for memory management (integrated in Phase 2 before closures)
- [01-01]: LLVM 19.x release branch for stable API compatibility
- [01-01]: .NET 8.0 LTS for long-term support (until Nov 2026)
- [01-01]: WSL2 recommended for Windows users instead of native MSVC build
- [01-01]: Progressive lowering philosophy: FunLang AST → High-Level MLIR → LLVM Dialect → LLVM IR → Native
- [01-01]: Tutorial chapters use copy-pasteable commands and annotated examples
- [01-02]: CallingConvention.Cdecl for all MLIR-C API P/Invoke declarations
- [01-02]: Opaque handle types as F# structs with single nativeint field
- [01-02]: MlirStringRef for string marshalling with FromString/ToString/Free methods
- [01-02]: Organized bindings by functional area (context, module, type, operation, region, block)
- [01-02]: MlirHelpers module for common patterns (operationToString, createContextWithDialects)
- [01-02]: Cross-platform library loading via "MLIR-C" name without extension
- [01-03]: Context/Module/OpBuilder implement IDisposable for automatic cleanup via 'use' keyword
- [01-03]: Module stores reference to parent Context to prevent premature garbage collection
- [01-03]: OpBuilder fluent API hides operation state complexity with convenience methods
- [01-03]: 7-stage compiler pipeline: parse → AST → MLIR IR → verify → lower → LLVM IR → object → link
- [01-03]: Custom dialect requires C++ wrapper with extern C shim due to C API limitations
- [02-01]: Separate Operator and CompareOp types to distinguish i32 vs i1 return types
- [02-01]: Implement unary negation as 0 - expr (no dedicated arith.negate operation)
- [02-01]: Helper methods (CreateArithBinaryOp, CreateArithCompare) for cleaner code generation
- [02-01]: Boolean results extended to i32 for main function return (arith.extui)
- [02-01]: Printf requires declaration + format string global + print_int helper abstraction
- [02-01]: Format strings must include null terminator (\0) for C compatibility
- [02-01]: Recursive compileExpr pattern maintains SSA form automatically
- [02-02]: Environment as Map<string, MlirValue> for immutable scope management
- [02-02]: Environment passing pattern: compileExpr receives env, Let extends it, others pass through
- [02-02]: Var case returns existing SSA value from environment (no new MLIR operations)
- [02-02]: Shadowing creates new SSA values (%x, %x_0) rather than mutation
- [02-02]: SSA form explained before implementation to establish conceptual foundation
- [02-03]: Block arguments replace PHI nodes (push vs pull, unified semantics, no lost copy problem)
- [02-03]: scf.if with scf.yield for conditional expressions (high-level structured control flow)
- [02-03]: Boolean type as i1 (1-bit integer), true=1, false=0
- [02-03]: Comparison operations return i1 directly (no extension for if conditions)
- [02-03]: Region-based compilation: separate blocks for then/else with environment passing
- [02-03]: SCF→CF lowering pass first in pipeline (before arith/func conversion)
- [02-03]: Progressive lowering: scf.if → cf.cond_br + block arguments → llvm
- [02-04]: Stack allocation for function-local values (automatic, fast, LIFO)
- [02-04]: Heap allocation for escaping values (closures, data structures, flexible lifetime)
- [02-04]: Boehm GC for automatic memory management (conservative, battle-tested, minimal compiler complexity)
- [02-04]: Phase 2 uses SSA registers only (no memory operations), heap begins in Phase 3
- [02-04]: GC_INIT() before GC_malloc(), runtime.c provides funlang_init wrapper
- [02-04]: Link with -lgc, RPATH preferred over LD_LIBRARY_PATH
- [02-04]: memref dialect for future heap allocation (alloca, alloc, load, store)
- [Project]: Tutorial 본문은 한글로 작성 (코드, API명, 기술 용어는 원문 유지)
- [Project]: Plain Korean style (~이다/~한다) not polite style (~입니다/~합니다) for tutorial text
- [03-01]: Function parameters as block arguments (not variables or let bindings, SSA values from entry block)
- [03-01]: Flat namespace for module-level functions (no forward declarations needed, enables mutual recursion)
- [03-01]: C calling convention (System V ABI) handled automatically by LLVM
- [03-01]: Phase 3 scope: Top-level named functions only (no closures/lambdas until Phase 4)
- [03-01]: funlang_main entry point called by runtime.c main

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 - Foundation & Interop:**
- ~~MLIR C API completeness for custom dialects is unverified~~ **RESOLVED**: Confirmed C API cannot define custom dialects; C++ wrapper pattern established in Appendix (01-03)

**Phase 2 - Core Language Basics:**
- PHASE COMPLETE! All 4 plans finished (arithmetic, let bindings, control flow, memory management)

**Phase 3 - Functions & Recursion:**
- Plan 03-01 complete: Function definitions, calls, func dialect, calling conventions
- Next: Plan 03-02 (Recursion and tail call optimization)

## Session Continuity

Last session: 2026-02-06T01:28:18Z
Stopped at: Completed 03-01-PLAN.md (Functions and func Dialect)
Resume file: None
Next: Plan 03-02 (Recursion and Tail Call Optimization)
