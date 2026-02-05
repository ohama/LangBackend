# Requirements: LangBackend

**Defined:** 2026-02-05
**Core Value:** Each chapter produces a working compiler for all features covered so far

## v1 Requirements

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

- [ ] **FUNC-01**: Reader can compile simple function definitions
- [ ] **FUNC-02**: Reader can compile function calls with arguments and return values
- [ ] **FUNC-03**: Reader can compile recursive functions (e.g., factorial)
- [ ] **FUNC-04**: Reader can compile mutually recursive functions
- [ ] **FUNC-05**: Tutorial explains calling conventions and stack frames in MLIR/LLVM

### Closures & Higher-Order Functions

- [ ] **CLOS-01**: Reader can compile closures with captured variables
- [ ] **CLOS-02**: Tutorial explains closure conversion (environment passing strategy)
- [ ] **CLOS-03**: Reader can compile higher-order functions (functions as arguments)
- [ ] **CLOS-04**: Reader can compile functions as return values
- [ ] **CLOS-05**: Tutorial explains heap allocation for closure environments

### Pattern Matching

- [ ] **PMTC-01**: Reader can compile match expressions on literal values
- [ ] **PMTC-02**: Reader can compile wildcard patterns
- [ ] **PMTC-03**: Reader can compile cons (::) pattern matching on lists
- [ ] **PMTC-04**: Tutorial explains decision tree compilation strategy
- [ ] **PMTC-05**: Reader can compile tuple pattern matching

### Custom MLIR Dialect

- [ ] **DIAL-01**: Tutorial explains custom FunLang dialect design (operations, types, attributes)
- [ ] **DIAL-02**: Reader can define custom operations (e.g., funlang.closure, funlang.apply)
- [ ] **DIAL-03**: Tutorial explains progressive lowering (FunLang dialect → SCF → LLVM)
- [ ] **DIAL-04**: Reader can implement lowering passes from FunLang dialect to standard dialects
- [ ] **DIAL-05**: Tutorial explains MLIR pattern-based rewrites

### Optimization

- [ ] **OPT-01**: Reader can implement constant folding pass
- [ ] **OPT-02**: Reader can implement dead code elimination pass
- [ ] **OPT-03**: Reader can implement tail call optimization
- [ ] **OPT-04**: Tutorial shows before/after IR comparison for each optimization

### Memory Management

- [ ] **MEM-01**: Tutorial explains memory management strategy (stack vs heap allocation)
- [ ] **MEM-02**: Tutorial integrates Boehm GC for heap-allocated values
- [ ] **MEM-03**: Reader can compile programs with closures and lists without memory leaks

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
| FOUND-01 | TBD | Pending |
| FOUND-02 | TBD | Pending |
| FOUND-03 | TBD | Pending |
| FOUND-04 | TBD | Pending |
| FOUND-05 | TBD | Pending |
| EXPR-01 | TBD | Pending |
| EXPR-02 | TBD | Pending |
| EXPR-03 | TBD | Pending |
| EXPR-04 | TBD | Pending |
| LET-01 | TBD | Pending |
| LET-02 | TBD | Pending |
| LET-03 | TBD | Pending |
| CTRL-01 | TBD | Pending |
| CTRL-02 | TBD | Pending |
| CTRL-03 | TBD | Pending |
| FUNC-01 | TBD | Pending |
| FUNC-02 | TBD | Pending |
| FUNC-03 | TBD | Pending |
| FUNC-04 | TBD | Pending |
| FUNC-05 | TBD | Pending |
| CLOS-01 | TBD | Pending |
| CLOS-02 | TBD | Pending |
| CLOS-03 | TBD | Pending |
| CLOS-04 | TBD | Pending |
| CLOS-05 | TBD | Pending |
| PMTC-01 | TBD | Pending |
| PMTC-02 | TBD | Pending |
| PMTC-03 | TBD | Pending |
| PMTC-04 | TBD | Pending |
| PMTC-05 | TBD | Pending |
| DIAL-01 | TBD | Pending |
| DIAL-02 | TBD | Pending |
| DIAL-03 | TBD | Pending |
| DIAL-04 | TBD | Pending |
| DIAL-05 | TBD | Pending |
| OPT-01 | TBD | Pending |
| OPT-02 | TBD | Pending |
| OPT-03 | TBD | Pending |
| OPT-04 | TBD | Pending |
| MEM-01 | TBD | Pending |
| MEM-02 | TBD | Pending |
| MEM-03 | TBD | Pending |
| QUAL-01 | TBD | Pending |
| QUAL-02 | TBD | Pending |
| QUAL-03 | TBD | Pending |
| QUAL-04 | TBD | Pending |

**Coverage:**
- v1 requirements: 44 total
- Mapped to phases: 0
- Unmapped: 44 (pending roadmap creation)

---
*Requirements defined: 2026-02-05*
*Last updated: 2026-02-05 after initial definition*
