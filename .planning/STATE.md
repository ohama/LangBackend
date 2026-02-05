# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Each chapter produces a working compiler for all features covered so far
**Current focus:** Phase 1 - Foundation & Interop

## Current Position

Phase: 1 of 7 (Foundation & Interop)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-02-05 — Completed 01-02-PLAN.md (Hello MLIR and P/Invoke Bindings)

Progress: [██░░░░░░░░] 20% (2/10 estimated total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 5 min
- Total execution time: 0.17 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 - Foundation & Interop | 2/3 | 10 min | 5 min |

**Recent Trend:**
- Last plan: 01-02 (6 min)
- Previous: 01-01 (4 min)
- Trend: Consistent execution pace

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
- [Project]: Tutorial 본문은 한글로 작성 (코드, API명, 기술 용어는 원문 유지)

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 - Foundation & Interop:**
- MLIR C API completeness for custom dialects is unverified; may need C++ wrapper if API gaps discovered during Phase 1 planning/execution

## Session Continuity

Last session: 2026-02-05T07:09:04Z
Stopped at: Completed 01-02-PLAN.md - Hello MLIR and P/Invoke Bindings chapters written
Resume file: None
Next: Execute 01-03-PLAN.md - AST Representation and First Code Generator
