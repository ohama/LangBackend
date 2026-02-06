# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Each chapter produces a working compiler for all features covered so far
**Current focus:** Phase 1 Complete → Phase 2 - Core Language Basics

## Current Position

Phase: 2 of 7 (Core Language Basics)
Plan: 1 of 4 in current phase
Status: In progress
Last activity: 2026-02-06 — Completed 02-01-PLAN.md (Arithmetic Expressions with operators, comparisons, and printf)

Progress: [████░░░░░░] 40% (4/10 estimated total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 5 min
- Total execution time: 0.33 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 - Foundation & Interop | 3/3 | 16 min | 5 min |
| 2 - Core Language Basics | 1/4 | 4 min | 4 min |

**Recent Trend:**
- Last plan: 02-01 (4 min)
- Previous: 01-03 (6 min)
- Previous: 01-02 (6 min)
- Previous: 01-01 (4 min)
- Trend: Fast execution pace maintained

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
- [Project]: Tutorial 본문은 한글로 작성 (코드, API명, 기술 용어는 원문 유지)
- [Project]: Plain Korean style (~이다/~한다) not polite style (~입니다/~합니다) for tutorial text

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 - Foundation & Interop:**
- ~~MLIR C API completeness for custom dialects is unverified~~ **RESOLVED**: Confirmed C API cannot define custom dialects; C++ wrapper pattern established in Appendix (01-03)

**Phase 2 - Core Language Basics:**
- None identified yet; Phase 1 foundation is complete and ready

## Session Continuity

Last session: 2026-02-06T00:22:31Z
Stopped at: Completed 02-01-PLAN.md (Arithmetic Expressions)
Resume file: None
Next: Plan 02-02 - Let Bindings (symbol table, scoping, variable shadowing)
