# Roadmap: LangBackend

## Milestones

- ✅ **v1.0 Tutorial Documentation** - Phases 1-6 (shipped 2026-02-11)
- 🚧 **v2.0 Compiler Implementation** - Phases 7-11 (in progress)

## Phases

<details>
<summary>✅ v1.0 Tutorial Documentation (Phases 1-6) - SHIPPED 2026-02-11</summary>

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
**Plans**: 6 plans (4 core + 2 gap closure)

Plans:
- [x] 06-01-PLAN.md — Chapter 17: Pattern Matching Theory (decision tree algorithm, pattern matrix, specialization/defaulting)
- [x] 06-02-PLAN.md — Chapter 18: List Operations (funlang.nil, funlang.cons, list representation, TypeConverter)
- [x] 06-03-PLAN.md — Chapter 19: Match Compilation (funlang.match operation, SCF lowering, region handling)
- [x] 06-04-PLAN.md — Chapter 20: Functional Programs (map, filter, fold, complete examples)
- [x] 06-05-PLAN.md — Gap Closure: Literal and wildcard pattern compilation (PMTC-01, PMTC-02)
- [x] 06-06-PLAN.md — Gap Closure: Tuple type and pattern matching (PMTC-05)

</details>

### v2.0 Compiler Implementation (In Progress)

**Milestone Goal:** Implement working FunLang → MLIR compiler supporting arithmetic, let bindings, if-else, functions, and closures

#### Phase 7: Foundation Infrastructure
**Goal**: Developer has working P/Invoke bindings to MLIR-C API and fluent F# OpBuilder wrapper
**Depends on**: Phase 6 (v1.0 complete)
**Requirements**: IMPL-INFRA-01, IMPL-INFRA-02
**Success Criteria** (what must be TRUE):
  1. Developer can call MLIR-C API functions from F# to create contexts, modules, types, operations, regions, and blocks
  2. Developer has OpBuilder wrapper class with fluent API for MLIR operation creation
  3. Developer can create simple MLIR modules programmatically using F# wrappers
  4. All wrapper types implement IDisposable for automatic resource cleanup
  5. Bindings work cross-platform (Linux, macOS, Windows via WSL)
**Plans**: 4 plans

Plans:
- [x] 07-01-PLAN.md — Project setup, handle types, string marshalling, core P/Invoke declarations
- [x] 07-02-PLAN.md — Complete P/Invoke declarations (types, operations, regions, blocks)
- [x] 07-03-PLAN.md — IDisposable wrappers (Context, Module, Location) and fluent OpBuilder
- [x] 07-04-PLAN.md — Smoke tests and end-to-end validation

#### Phase 8: Core Expressions
**Goal**: Developer can compile arithmetic expressions, comparisons, booleans, let bindings, and if-else to MLIR
**Depends on**: Phase 7
**Requirements**: IMPL-LANG-01, IMPL-LANG-02, IMPL-LANG-03, IMPL-LANG-04, IMPL-LANG-05
**Success Criteria** (what must be TRUE):
  1. Developer can compile arithmetic expressions (add, sub, mul, div, negate) to arith dialect operations
  2. Developer can compile comparison operators (<, >, <=, >=, ==, <>) to arith.cmpi operations
  3. Developer can compile boolean literals (true, false) and logical operators (&&, ||) to i1 operations
  4. Developer can compile let bindings with shadowing support, mapping to SSA values
  5. Developer can compile if-then-else expressions to scf.if operations with block arguments
  6. Compiled programs can execute simple expressions and print results
**Plans**: 4 plans

Plans:
- [ ] 08-01-PLAN.md — Comparison operators and boolean expressions (arith.cmpi, arith.andi/ori)
- [ ] 08-02-PLAN.md — Let bindings with environment tracking (Env, Var, Let)
- [ ] 08-03-PLAN.md — If-else expressions (scf.if with regions and scf.yield)
- [ ] 08-04-PLAN.md — Comprehensive E2E tests for all Phase 8 features

#### Phase 9: Functions
**Goal**: Developer can compile named function definitions, calls, and recursive functions to MLIR
**Depends on**: Phase 8
**Requirements**: IMPL-LANG-06, IMPL-LANG-07
**Success Criteria** (what must be TRUE):
  1. Developer can compile named function definitions to func.func operations
  2. Developer can compile function calls to func.call operations with correct argument passing
  3. Developer can compile recursive functions that execute correctly (factorial, fibonacci)
  4. Developer can compile mutually recursive functions without forward declaration issues
  5. Compiled programs with functions can execute and return correct results
**Plans**: TBD

Plans:
- [ ] 09-01: TBD during planning

#### Phase 10: Closures
**Goal**: Developer can compile lambda expressions with free variable capture and higher-order functions
**Depends on**: Phase 9
**Requirements**: IMPL-LANG-08, IMPL-LANG-09, IMPL-LANG-10, IMPL-LANG-11
**Success Criteria** (what must be TRUE):
  1. Developer can compile lambda expressions (fun x -> body) to closure structures
  2. Developer can perform free variable analysis and capture environment correctly
  3. Developer can compile higher-order functions (functions as arguments and return values)
  4. Developer can compile currying (multi-argument functions as nested single-argument closures)
  5. Compiled closures execute correctly with proper environment access
  6. Closure-heavy programs run without memory leaks (GC integration working)
**Plans**: TBD

Plans:
- [ ] 10-01: TBD during planning

#### Phase 11: Pipeline & Testing
**Goal**: Developer has complete compilation pipeline with CLI integration and comprehensive test suite
**Depends on**: Phase 10
**Requirements**: IMPL-INFRA-03, IMPL-INFRA-04, IMPL-TEST-01, IMPL-TEST-02, IMPL-TEST-03
**Success Criteria** (what must be TRUE):
  1. Developer has lowering pipeline that converts MLIR → LLVM dialect → LLVM IR → Object → Binary
  2. Developer can invoke compiler from CLI with --emit-mlir option to inspect intermediate IR
  3. Developer has E2E fslit tests that compile FunLang source, execute binary, and verify output
  4. Developer has unit tests (F# Expecto) for codegen modules with good coverage
  5. Developer has MLIR IR tests using FileCheck for verification of IR transformations
  6. Full compiler pipeline works end-to-end: FunLang source → native binary execution
**Plans**: TBD

Plans:
- [ ] 11-01: TBD during planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

**v1.0 Tutorial Documentation (COMPLETE):**

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Interop | 3/3 | Complete | 2026-02-06 |
| 2. Core Language Basics | 4/4 | Complete | 2026-02-06 |
| 3. Functions & Recursion | 2/2 | Complete | 2026-02-06 |
| 4. Closures & Higher-Order Functions | 2/2 | Complete | 2026-02-06 |
| 5. Custom MLIR Dialect | 3/3 | Complete | 2026-02-06 |
| 6. Pattern Matching & Data Structures | 6/6 | Complete | 2026-02-11 |

**v2.0 Compiler Implementation (IN PROGRESS):**

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 7. Foundation Infrastructure | 4/4 | Complete | 2026-02-12 |
| 8. Core Expressions | 0/4 | Not started | - |
| 9. Functions | 0/TBD | Not started | - |
| 10. Closures | 0/TBD | Not started | - |
| 11. Pipeline & Testing | 0/TBD | Not started | - |
