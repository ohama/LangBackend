# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-11)

**Core value:** Implement working FunLang -> MLIR compiler based on tutorial
**Current focus:** v2.0 Compiler Implementation - Phase 8 (Core Expressions)

## Current Position

Phase: 8 of 11 (Core Expressions)
Plan: 2 of 4 complete
Status: In progress
Last activity: 2026-02-12 - Completed 08-02 (Let Bindings and Variables)

Progress: [#########░░] 72% (26/36 plans complete)

## Performance Metrics

**v1.0 Tutorial Documentation:**
- Total plans completed: 20
- Average duration: 8 min
- Total execution time: 2.4 hours
- Status: COMPLETE (Phases 1-6 shipped 2026-02-11)

**By Phase (v1.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 - Foundation & Interop | 3/3 | 16 min | 5 min |
| 2 - Core Language Basics | 4/4 | 19 min | 5 min |
| 3 - Functions & Recursion | 2/2 | 17 min | 9 min |
| 4 - Closures & Higher-Order | 2/2 | 11 min | 6 min |
| 5 - Custom MLIR Dialect | 3/3 | 29 min | 10 min |
| 6 - Pattern Matching | 6/6 | 54 min | 9 min |

**Recent Trend (v1.0):**
- Last plan: 06-06 (8 min, tuple gap closure)
- Trend: Consistent 8-12min for documentation plans

**v2.0 Compiler Implementation:**
- Status: Phase 8 in progress
- Estimated phases: 5 (Phases 7-11)
- Plans completed: 5/16

**By Phase (v2.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 7 - Foundation Infrastructure | 4/4 | 12 min | 3 min |
| 8 - Core Expressions | 2/4 | 10 min | 5 min |

**Recent Trend (v2.0):**
- Last plan: 08-02 (4 min, let bindings and variables)
- Build passing, MLIR-C library required for test execution

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting v2.0 work:

- [v1.0]: F# implementation using P/Invoke to MLIR-C API
- [v1.0]: Custom MLIR Dialect approach for FunLang-specific operations
- [v1.0]: Progressive lowering: FunLang AST -> High-Level MLIR -> LLVM Dialect -> LLVM IR -> Native
- [v1.0]: Boehm GC for memory management (integrated in Phase 2)
- [v1.0]: CallingConvention.Cdecl for all MLIR-C API P/Invoke declarations
- [v1.0]: Context/Module/OpBuilder implement IDisposable for automatic cleanup
- [Project]: Tutorial 본문은 한글로 작성 (v1.0), implementation is F# code (v2.0)
- [07-01]: UTF-8 encoding for all MLIR string marshalling (not Ansi)
- [07-01]: Handle struct pattern for MLIR opaque pointers
- [07-01]: MlirStringRef.WithString pattern for automatic allocation/cleanup
- [07-02]: MlirOperationState uses nativeint for array pointers (flexible marshalling)
- [07-02]: All MlirOperationState fields are mutable (required for MLIR operation building pattern)
- [07-03]: Location is discriminated union without IDisposable (value type owned by Context)
- [07-03]: Module stores contextRef field to prevent premature Context GC
- [07-03]: NativePtr.toNativeInt for converting fixed array pointers in P/Invoke calls
- [08-01]: Comparison predicates use i64 type attribute (MLIR ArithOps.td requirement)
- [08-01]: Boolean operations use non-short-circuit evaluation (arith.andi/ori)
- [08-02]: Immutable F# Map for environment - Map.Add handles shadowing naturally
- [08-02]: Let compilation: compile binding, extend env, compile body with extended env

### Pending Todos

None yet.

### Blockers/Concerns

**v2.0 Implementation:**
- RESOLVED: src/ directory structure established (07-01)
- RESOLVED: Build system determined (dotnet CLI with fsproj, 07-01)
- ACTIVE: MLIR-C library not available - tests require libMLIR-C.so to be built/installed
- Tutorial documentation provides theory; implementation validating practicality

## Session Continuity

Last session: 2026-02-12
Stopped at: Completed 08-02-PLAN.md
Resume file: None
Next: Execute Phase 8 Plan 03 (If-Then-Else Expressions)
