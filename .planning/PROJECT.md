# LangBackend

## What This Is

A documentation-only tutorial series (10-15 markdown chapters) that teaches how to build an MLIR-based compiler backend for FunLang. The reader already knows FunLang's AST and type system from LangTutorial. Each chapter incrementally adds a language feature to compile, progressing from simple arithmetic to a full native binary via MLIR/LLVM.

## Core Value

Each chapter produces a working compiler for all features covered so far — the reader can compile and run at every step, not just at the end.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Tutorial chapters stored as markdown in `tutorial/`
- [ ] Progression: arithmetic → let bindings → if/else → functions → closures → pattern matching → custom MLIR dialect → native binary
- [ ] Each chapter is an incremental checkpoint (working compiler for features so far)
- [ ] Implementation language is F# with MLIR/LLVM bindings (binding approach TBD — needs research)
- [ ] Pipeline: Typed AST → Custom MLIR Dialect → MLIR Lowering Passes → LLVM Dialect → LLVM IR → Native binary
- [ ] Uses structured MLIR builders/APIs, not string generation
- [ ] Custom MLIR Dialect defined for FunLang-specific operations
- [ ] Target audience: readers who completed LangTutorial (understand FunLang AST, type system, evaluator)
- [ ] 10-15 chapters total, simple grammar to complex
- [ ] Code snippets in each chapter are self-contained and buildable

### Out of Scope

- Reimplementing the FunLang parser/lexer/type checker — reader already has these from LangTutorial
- Runtime/interpreter improvements — this is strictly the compilation backend
- IDE tooling or language server
- JIT compilation — AOT compilation only
- Building a production-grade compiler — this is pedagogical

## Context

- FunLang is an F#-like functional language (v6.0) with: first-class functions, closures, pattern matching, Hindley-Milner type inference, bidirectional type checking, lists, tuples, strings
- LangTutorial has 619 tests, 11 tutorial chapters, and a self-hosted Prelude stdlib
- FunLang currently uses a tree-walking interpreter — no compilation backend exists
- The MLIR binding approach from F# is an open question (LLVM-C API via P/Invoke, or other bindings) — research is needed
- MLIR pipeline follows standard practice: custom dialect → lowering passes → LLVM dialect → LLVM IR → native

## Constraints

- **Language**: Implementation must be in F# to stay consistent with LangTutorial
- **MLIR**: Must use structured builder APIs, not string-based MLIR generation
- **Format**: Documentation-only markdown files in `tutorial/`, no separate buildable project in this repo
- **Dependency**: Assumes reader has FunLang source from LangTutorial as the starting point

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| F# for implementation | Consistency with LangTutorial, reader familiarity | — Pending |
| Custom MLIR Dialect first | Standard MLIR practice: model domain semantics before lowering | — Pending |
| Incremental chapters | Each step compilable — better learning experience than big-bang | — Pending |
| MLIR binding approach | TBD — needs research into F#/MLIR interop options | — Pending |

---
*Last updated: 2026-02-05 after initialization*
