# Requirements: LangBackend

**Defined:** 2026-02-05
**Core Value:** Each chapter produces a working compiler for all features covered so far

---

## v2.0 Implementation Requirements

Implementation of actual FunLang → MLIR compiler based on tutorial documentation.

### IMPL-INFRA: 컴파일 인프라 구현

- [ ] **IMPL-INFRA-01**: P/Invoke bindings for MLIR-C API (context, module, type, operation, region, block)
- [ ] **IMPL-INFRA-02**: OpBuilder wrapper class with fluent API for MLIR operation creation
- [ ] **IMPL-INFRA-03**: Lowering pipeline (MLIR → LLVM dialect → LLVM IR → Object → Binary)
- [ ] **IMPL-INFRA-04**: CLI integration with --emit-mlir option in FunLang

### IMPL-LANG: 언어 기능 구현

- [ ] **IMPL-LANG-01**: Arithmetic expressions compilation (add, sub, mul, div, negate)
- [ ] **IMPL-LANG-02**: Comparison operators compilation (<, >, <=, >=, ==, <>)
- [ ] **IMPL-LANG-03**: Boolean literals and logical operators (&&, ||)
- [ ] **IMPL-LANG-04**: Let bindings with shadowing support (let x = e1 in e2)
- [ ] **IMPL-LANG-05**: If-then-else expressions (scf.if based)
- [ ] **IMPL-LANG-06**: Named function definitions (func.func)
- [ ] **IMPL-LANG-07**: Function calls and recursion (func.call)
- [ ] **IMPL-LANG-08**: Lambda expressions (fun x -> body)
- [ ] **IMPL-LANG-09**: Free variable capture (closure environment)
- [ ] **IMPL-LANG-10**: Higher-order functions (functions as arguments/return values)
- [ ] **IMPL-LANG-11**: Currying (multi-argument as nested single-argument closures)

### IMPL-TEST: 테스트 구현

- [ ] **IMPL-TEST-01**: E2E fslit tests (FunLang source → compile → execute → verify)
- [ ] **IMPL-TEST-02**: Unit tests for codegen modules (F# Expecto)
- [ ] **IMPL-TEST-03**: MLIR IR tests (FileCheck verification)

---

## v1 Requirements (Tutorial Documentation - COMPLETE)

Requirements for initial release. Each maps to roadmap phases.

### Foundation & Interop

- [ ] **FOUND-01**: Tutorial explains MLIR build setup (LLVM 19.x with MLIR-C shared library)
- [ ] **FOUND-02**: Tutorial explains F# P/Invoke bindings to MLIR-C API with working examples
- [ ] **FOUND-03**: Tutorial includes idiomatic F# wrapper layer over raw P/Invoke (Context, Module, Builder)
- [ ] **FOUND-04**: Tutorial includes basic compiler driver (read FunLang source, emit native binary)
- [ ] **FOUND-05**: Tutorial explains custom MLIR dialect registration from F# via C API

### Arithmetic & Expressions

- [ ] **EXPR-01**: Reader can compile integer literals and arithmetic (+, -, *, /)
- [ ] **EXPR-02**: Reader can compile comparison operators (<, >, <=, >=, =, <>)
- [ ] **EXPR-03**: Reader can compile unary negation
- [ ] **EXPR-04**: Compiled program prints result to stdout

### Let Bindings

- [ ] **LET-01**: Reader can compile let bindings (let x = expr in body)
- [ ] **LET-02**: Reader can compile nested let bindings with correct scoping
- [ ] **LET-03**: Tutorial explains SSA form and how let bindings map to MLIR values

### Control Flow

- [ ] **CTRL-01**: Reader can compile if/then/else expressions
- [ ] **CTRL-02**: Tutorial explains MLIR block arguments (phi nodes equivalent)
- [ ] **CTRL-03**: Reader can compile boolean expressions in conditions

### Functions

- [x] **FUNC-01**: Reader can compile simple function definitions
- [x] **FUNC-02**: Reader can compile function calls with arguments and return values
- [x] **FUNC-03**: Reader can compile recursive functions (e.g., factorial)
- [x] **FUNC-04**: Reader can compile mutually recursive functions
- [x] **FUNC-05**: Tutorial explains calling conventions and stack frames in MLIR/LLVM

### Closures & Higher-Order Functions

- [x] **CLOS-01**: Reader can compile closures with captured variables
- [x] **CLOS-02**: Tutorial explains closure conversion (environment passing strategy)
- [x] **CLOS-03**: Reader can compile higher-order functions (functions as arguments)
- [x] **CLOS-04**: Reader can compile functions as return values
- [x] **CLOS-05**: Tutorial explains heap allocation for closure environments

### Pattern Matching

- [ ] **PMTC-01**: Reader can compile match expressions on literal values
- [ ] **PMTC-02**: Reader can compile wildcard patterns
- [ ] **PMTC-03**: Reader can compile cons (::) pattern matching on lists
- [ ] **PMTC-04**: Tutorial explains decision tree compilation strategy
- [ ] **PMTC-05**: Reader can compile tuple pattern matching

### Custom MLIR Dialect

- [x] **DIAL-01**: Tutorial explains custom FunLang dialect design (operations, types, attributes)
- [x] **DIAL-02**: Reader can define custom operations (e.g., funlang.closure, funlang.apply)
- [x] **DIAL-03**: Tutorial explains progressive lowering (FunLang dialect → SCF → LLVM)
- [x] **DIAL-04**: Reader can implement lowering passes from FunLang dialect to standard dialects
- [x] **DIAL-05**: Tutorial explains MLIR pattern-based rewrites

### Optimization

- [ ] **OPT-01**: Reader can implement constant folding pass
- [ ] **OPT-02**: Reader can implement dead code elimination pass
- [ ] **OPT-03**: Reader can implement tail call optimization
- [ ] **OPT-04**: Tutorial shows before/after IR comparison for each optimization

### Memory Management

- [ ] **MEM-01**: Tutorial explains memory management strategy (stack vs heap allocation)
- [ ] **MEM-02**: Tutorial integrates Boehm GC for heap-allocated values
- [x] **MEM-03**: Reader can compile programs with closures and lists without memory leaks

### Tutorial Quality

- [ ] **QUAL-01**: Each chapter includes expected MLIR IR output for examples
- [ ] **QUAL-02**: Each chapter is self-contained and incrementally buildable
- [ ] **QUAL-03**: Tutorial includes MLIR primer (dialect, operation, region, block, SSA concepts)
- [ ] **QUAL-04**: Each chapter includes "Common Errors" section with debugging tips

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Data Types

- **ADT-01**: Reader can compile algebraic data types (discriminated unions)
- **ADT-02**: Reader can compile pattern matching on ADTs
- **ADT-03**: Tutorial explains tagged union representation in MLIR

### Advanced Optimizations

- **ADVOPT-01**: Reader can implement function inlining pass
- **ADVOPT-02**: Reader can implement escape analysis (stack vs heap decision)
- **ADVOPT-03**: Tutorial shows performance benchmarks vs interpreted FunLang

### Runtime

- **RUN-01**: Tutorial covers standard library function integration (map, filter, fold)
- **RUN-02**: Tutorial covers string compilation
- **RUN-03**: Reader can implement REPL with MLIR JIT execution

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full garbage collector (precise GC) | Semester-long topic; Boehm GC sufficient for tutorial |
| Module system / multi-file compilation | Adds complexity beyond backend tutorial scope |
| IDE integration / language server | Tutorial is about backend, not tooling |
| Parallelism / concurrency | Runtime complexity explosion; sequential execution only |
| Type classes / effect systems | Advanced type theory beyond backend focus |
| JIT compilation | AOT only; JIT is separate concern |
| All MLIR dialects coverage | Focus on 5-6 key dialects in lowering path |
| Production-grade error recovery | Basic error handling sufficient for tutorial |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Pending |
| FOUND-04 | Phase 1 | Pending |
| FOUND-05 | Phase 1 | Pending |
| EXPR-01 | Phase 2 | Pending |
| EXPR-02 | Phase 2 | Pending |
| EXPR-03 | Phase 2 | Pending |
| EXPR-04 | Phase 2 | Pending |
| LET-01 | Phase 2 | Pending |
| LET-02 | Phase 2 | Pending |
| LET-03 | Phase 2 | Pending |
| CTRL-01 | Phase 2 | Pending |
| CTRL-02 | Phase 2 | Pending |
| CTRL-03 | Phase 2 | Pending |
| FUNC-01 | Phase 3 | Complete |
| FUNC-02 | Phase 3 | Complete |
| FUNC-03 | Phase 3 | Complete |
| FUNC-04 | Phase 3 | Complete |
| FUNC-05 | Phase 3 | Complete |
| CLOS-01 | Phase 4 | Complete |
| CLOS-02 | Phase 4 | Complete |
| CLOS-03 | Phase 4 | Complete |
| CLOS-04 | Phase 4 | Complete |
| CLOS-05 | Phase 4 | Complete |
| PMTC-01 | Phase 6 | Pending |
| PMTC-02 | Phase 6 | Pending |
| PMTC-03 | Phase 6 | Pending |
| PMTC-04 | Phase 6 | Pending |
| PMTC-05 | Phase 6 | Pending |
| DIAL-01 | Phase 5 | Complete |
| DIAL-02 | Phase 5 | Complete |
| DIAL-03 | Phase 5 | Complete |
| DIAL-04 | Phase 5 | Complete |
| DIAL-05 | Phase 5 | Complete |
| OPT-01 | Phase 7 | Pending |
| OPT-02 | Phase 7 | Pending |
| OPT-03 | Phase 7 | Pending |
| OPT-04 | Phase 7 | Pending |
| MEM-01 | Phase 2 | Pending |
| MEM-02 | Phase 2 | Pending |
| MEM-03 | Phase 4 | Complete |
| QUAL-01 | Phase 2 | Pending |
| QUAL-02 | Phase 2 | Pending |
| QUAL-03 | Phase 1 | Pending |
| QUAL-04 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 44 total
- Mapped to phases: 44
- Unmapped: 0

**Phase Distribution:**
- Phase 1 (Foundation & Interop): 6 requirements
- Phase 2 (Core Language Basics): 14 requirements
- Phase 3 (Functions & Recursion): 5 requirements
- Phase 4 (Closures & Higher-Order Functions): 6 requirements
- Phase 5 (Custom MLIR Dialect): 5 requirements
- Phase 6 (Pattern Matching & Data Structures): 5 requirements
- Phase 7 (Optimization & Polish): 4 requirements

---

## v2.0 Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| IMPL-INFRA-01 | TBD | Pending |
| IMPL-INFRA-02 | TBD | Pending |
| IMPL-INFRA-03 | TBD | Pending |
| IMPL-INFRA-04 | TBD | Pending |
| IMPL-LANG-01 | TBD | Pending |
| IMPL-LANG-02 | TBD | Pending |
| IMPL-LANG-03 | TBD | Pending |
| IMPL-LANG-04 | TBD | Pending |
| IMPL-LANG-05 | TBD | Pending |
| IMPL-LANG-06 | TBD | Pending |
| IMPL-LANG-07 | TBD | Pending |
| IMPL-LANG-08 | TBD | Pending |
| IMPL-LANG-09 | TBD | Pending |
| IMPL-LANG-10 | TBD | Pending |
| IMPL-LANG-11 | TBD | Pending |
| IMPL-TEST-01 | TBD | Pending |
| IMPL-TEST-02 | TBD | Pending |
| IMPL-TEST-03 | TBD | Pending |

---
*Requirements defined: 2026-02-05*
*Last updated: 2026-02-11 — v2.0 Implementation requirements added*
