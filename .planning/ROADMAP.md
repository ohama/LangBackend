# Roadmap: LangBackend

## Overview

LangBackend is a 10-15 chapter tutorial series teaching MLIR-based compiler backend development for FunLang. The roadmap progresses from foundational MLIR interop through increasingly complex language features, culminating in a complete compilation pipeline from typed AST to native binary. Each phase produces tutorial chapters that enable readers to build working compilers incrementally, following the core value: every chapter must produce a compilable, runnable result.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation & Interop** - MLIR build setup, F# bindings, and architecture validation
- [x] **Phase 2: Core Language Basics** - Arithmetic, let bindings, control flow, and memory strategy
- [x] **Phase 3: Functions & Recursion** - Function definitions, calls, and recursive patterns
- [x] **Phase 4: Closures & Higher-Order Functions** - Environment capture and closure compilation
- [x] **Phase 5: Custom MLIR Dialect** - FunLang dialect design and progressive lowering
- [ ] **Phase 6: Pattern Matching & Data Structures** - Lists, tuples, and decision tree compilation
- [ ] **Phase 7: Optimization & Polish** - Optimization passes, quality improvements, and completion

## Phase Details

### Phase 1: Foundation & Interop
**Goal**: Reader can build MLIR with C API, create F# P/Invoke bindings, and compile a "hello world" program
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, QUAL-03
**Success Criteria** (what must be TRUE):
  1. Reader can build LLVM 19.x with MLIR-C shared library on their platform
  2. Reader can call MLIR-C API functions from F# using P/Invoke with working examples
  3. Reader has idiomatic F# wrapper types (Context, Module, Builder) that manage MLIR object lifetimes
  4. Reader understands MLIR primer concepts (dialect, operation, region, block, SSA form)
  5. Reader can register a custom MLIR dialect from F# via C API
  6. Reader can compile a trivial FunLang program (integer literal) to native binary and execute it
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Prerequisites (LLVM/MLIR build) and MLIR Primer (concepts)
- [x] 01-02-PLAN.md — Hello MLIR from F# and complete P/Invoke bindings module
- [x] 01-03-PLAN.md — F# wrapper layer, arithmetic compiler, and custom dialect appendix

### Phase 2: Core Language Basics
**Goal**: Reader can compile arithmetic expressions, let bindings, and if/else control flow with working memory management
**Depends on**: Phase 1
**Requirements**: EXPR-01, EXPR-02, EXPR-03, EXPR-04, LET-01, LET-02, LET-03, CTRL-01, CTRL-02, CTRL-03, MEM-01, MEM-02, QUAL-01, QUAL-02, QUAL-04
**Success Criteria** (what must be TRUE):
  1. Reader can compile programs with integer arithmetic (+, -, *, /) and comparison operators (<, >, =, etc.)
  2. Reader can compile let bindings with proper scoping and understands SSA value mapping
  3. Reader can compile if/then/else expressions with block arguments (phi nodes)
  4. Reader can compile programs that print results to stdout
  5. Reader understands memory management strategy (stack vs heap) and has Boehm GC integrated
  6. Each chapter includes expected MLIR IR output and "Common Errors" debugging section
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md — Chapter 06: Arithmetic expressions (binary ops, comparisons, negation, print)
- [x] 02-02-PLAN.md — Chapter 07: Let bindings (SSA form, environment passing, scoping)
- [x] 02-03-PLAN.md — Chapter 08: Control flow (scf.if, block arguments, boolean expressions)
- [x] 02-04-PLAN.md — Chapter 09: Memory management (stack vs heap, Boehm GC integration)

### Phase 3: Functions & Recursion
**Goal**: Reader can compile function definitions, calls, and recursive functions including mutual recursion
**Depends on**: Phase 2
**Requirements**: FUNC-01, FUNC-02, FUNC-03, FUNC-04, FUNC-05
**Success Criteria** (what must be TRUE):
  1. Reader can compile simple function definitions with arguments and return values
  2. Reader can compile function calls and understands calling conventions in MLIR/LLVM
  3. Reader can compile recursive functions (e.g., factorial, fibonacci)
  4. Reader can compile mutually recursive functions with correct forward references
  5. Reader understands stack frame management and function lowering to LLVM dialect
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md — Chapter 10: Functions (definitions, calls, func dialect, calling conventions)
- [x] 03-02-PLAN.md — Chapter 11: Recursion (factorial, fibonacci, mutual recursion, stack frames, tail calls)

### Phase 4: Closures & Higher-Order Functions
**Goal**: Reader can compile closures with captured variables and higher-order functions
**Depends on**: Phase 3
**Requirements**: CLOS-01, CLOS-02, CLOS-03, CLOS-04, CLOS-05, MEM-03
**Success Criteria** (what must be TRUE):
  1. Reader understands closure theory including environment analysis and capture semantics
  2. Reader can compile closures with environment capture (free variables)
  3. Reader can compile higher-order functions (functions as arguments and return values)
  4. Reader understands closure conversion strategy (environment passing, heap allocation)
  5. Reader can compile programs with closures and lists without memory leaks using GC
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Chapter 12: Closures (theory, free variable analysis, closure conversion, heap allocation)
- [x] 04-02-PLAN.md — Chapter 13: Higher-Order Functions (functions as arguments, return values, indirect calls)

### Phase 5: Custom MLIR Dialect
**Goal**: Reader can define a custom FunLang dialect with operations, types, and lowering passes
**Depends on**: Phase 4
**Requirements**: DIAL-01, DIAL-02, DIAL-03, DIAL-04, DIAL-05
**Success Criteria** (what must be TRUE):
  1. Reader understands custom dialect design principles (operations, types, attributes)
  2. Reader can define custom operations (e.g., funlang.closure, funlang.apply, funlang.match)
  3. Reader understands progressive lowering philosophy (FunLang dialect -> SCF/MemRef -> LLVM)
  4. Reader can implement lowering passes from FunLang dialect to standard dialects
  5. Reader can implement pattern-based rewrites using MLIR's rewrite infrastructure
  6. Reader can refactor earlier chapters to use custom dialect operations
**Plans**: 3 plans

Plans:
- [x] 05-01-PLAN.md — Chapter 14: Custom Dialect Design (theory, TableGen ODS, C API shim pattern, progressive lowering)
- [x] 05-02-PLAN.md — Chapter 15: Custom Operations (funlang.closure, funlang.apply, funlang.match with F# integration)
- [x] 05-03-PLAN.md — Chapter 16: Lowering Passes (ConversionPattern, TypeConverter, DRR patterns, complete pipeline)

### Phase 6: Pattern Matching & Data Structures
**Goal**: Reader can compile pattern matching on lists and tuples using decision tree compilation
**Depends on**: Phase 5
**Requirements**: PMTC-01, PMTC-02, PMTC-03, PMTC-04, PMTC-05
**Success Criteria** (what must be TRUE):
  1. Reader understands decision tree compilation strategy for pattern matching
  2. Reader can compile match expressions on literal values and wildcard patterns
  3. Reader can compile cons (::) pattern matching on lists with proper list representation
  4. Reader can compile tuple pattern matching
  5. Reader understands how pattern matching lowers to SCF control flow (switch/if)
  6. Reader can compile realistic functional programs using lists and pattern matching
**Plans**: 4 plans

Plans:
- [ ] 06-01-PLAN.md — Chapter 17: Pattern Matching Theory (decision tree algorithm, pattern matrix, specialization/defaulting)
- [ ] 06-02-PLAN.md — Chapter 18: List Operations (funlang.nil, funlang.cons, list representation, TypeConverter)
- [ ] 06-03-PLAN.md — Chapter 19: Match Compilation (funlang.match operation, SCF lowering, region handling)
- [ ] 06-04-PLAN.md — Chapter 20: Functional Programs (map, filter, fold, complete examples)

### Phase 7: Optimization & Polish
**Goal**: Reader can implement optimization passes and has complete, polished tutorial series
**Depends on**: Phase 6
**Requirements**: OPT-01, OPT-02, OPT-03, OPT-04
**Success Criteria** (what must be TRUE):
  1. Reader can implement constant folding and dead code elimination passes
  2. Reader can implement tail call optimization (critical for functional languages)
  3. Reader sees before/after IR comparisons demonstrating each optimization
  4. Reader has complete tutorial series covering FunLang compilation from AST to native binary
  5. All tutorial chapters are self-contained, incrementally buildable, with working examples
**Plans**: TBD

Plans:
- [ ] 07-01: TBD during planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Interop | 3/3 | Complete | 2026-02-06 |
| 2. Core Language Basics | 4/4 | Complete | 2026-02-06 |
| 3. Functions & Recursion | 2/2 | Complete | 2026-02-06 |
| 4. Closures & Higher-Order Functions | 2/2 | Complete | 2026-02-06 |
| 5. Custom MLIR Dialect | 3/3 | Complete | 2026-02-06 |
| 6. Pattern Matching & Data Structures | 0/4 | Planned | - |
| 7. Optimization & Polish | 0/TBD | Not started | - |
