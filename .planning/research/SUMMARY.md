# Project Research Summary

**Project:** MLIR Compiler Backend Tutorial for FunLang
**Domain:** Compiler Infrastructure / Educational Tutorial
**Researched:** 2026-02-05
**Confidence:** HIGH

## Executive Summary

This project is a compiler backend tutorial teaching MLIR-based compilation for functional languages, specifically targeting FunLang (from the companion LangTutorial). Building an MLIR backend from F# faces a critical technical challenge: the F#-to-MLIR interop layer. The MLIR C API is the only viable boundary (P/Invoke approach), but it's incomplete compared to the C++ API. This creates the project's highest technical risk. Beyond interop, the architecture follows well-established patterns: custom MLIR dialect for high-level functional constructs (closures, pattern matching), progressive lowering through standard dialects, and LLVM backend for native code generation.

The recommended approach is to use raw P/Invoke to MLIR's C API wrapped in safe F# types, start with Boehm conservative GC to avoid memory management complexity, and structure the tutorial incrementally with each chapter adding exactly one concept to a stable pipeline architecture. The key insight from research is that the full pipeline architecture (Typed AST → Custom Dialect → Lowering Passes → LLVM Dialect → Native Binary) must be designed and validated with an end-to-end prototype BEFORE any tutorial content is written. Incremental tutorials that paint into architectural corners are a common failure mode.

Critical risks include: (1) discovering mid-tutorial that MLIR C API lacks needed features for custom dialects, (2) implementing naive closure compilation without proper environment analysis, (3) deferring GC integration until after closures are introduced, and (4) using naive pattern match compilation that generates exponentially large code. Each has clear mitigation strategies documented in the pitfalls research.

## Key Findings

### Recommended Stack

The stack centers on MLIR 19.x (from LLVM 19) as the compiler middle-end, with .NET 8/9 SDK for F# implementation. The critical unknown is whether mature .NET/MLIR bindings exist (confidence: LOW). Research assumes they don't and recommends raw P/Invoke to MLIR's C API, which is stable but requires significant boilerplate (estimated 500-1000 lines of FFI wrapper code before any compiler work begins).

**Core technologies:**
- **MLIR 19.x / LLVM 19.x**: Industry-standard IR framework with structured dialects; avoids brittle string-based IR generation
- **.NET 8/9 SDK with F# 8/9**: Modern platform with improved P/Invoke features; F# required for consistency with LangTutorial
- **MLIR C API via P/Invoke**: Only viable F# interop boundary; stable C ABI but incomplete feature coverage compared to C++ API
- **Boehm GC**: Conservative garbage collector for runtime memory management; simple integration (link library, replace malloc)
- **CMake + Ninja + Clang**: Required to build MLIR with C API enabled (`-DMLIR_BUILD_MLIR_C_DYLIB=ON`)

**Critical decision point:** MLIR C API must be prototyped for custom dialect registration BEFORE committing to architecture. If C API lacks needed features, fallback is a thin C++ wrapper DLL exposing additional functions.

### Expected Features

Research identified 12 table stakes features, 14 potential differentiators, and 9 anti-features to avoid. The tutorial must cover core functional language features incrementally.

**Must have (table stakes):**
- Basic arithmetic evaluation (literals, operators)
- Let bindings and variable scoping
- Control flow (if/else, boolean logic)
- Function definition and calls
- Pattern matching on lists and ADTs
- Closures with environment capture
- Type checking integration (map HM types to MLIR)
- LLVM lowering to native binary
- Memory management (GC or explicit strategy)
- Standard library integration (minimal: I/O functions)
- Error handling in compiler
- Incremental testing per chapter

**Should have (competitive advantage):**
- Custom MLIR dialect for FunLang (shows domain-specific ops)
- Multi-stage lowering architecture (progressive lowering philosophy)
- Pattern-based rewrites (MLIR's killer feature)
- Real F# implementation (unique; most tutorials use C++)
- Recursive functions with tail call optimization
- Higher-order functions (first-class functions)
- Gradual complexity ramp (one concept per chapter)
- Optimization passes (inlining, constant folding, DCE)

**Defer (v2+):**
- Full garbage collector implementation (use Boehm GC; full GC is semester-long course)
- Complete standard library (provide 5-10 essential functions)
- Every MLIR dialect (cover 5-6 key dialects in lowering path)
- Multi-file compilation (single-file per chapter)
- Parallelism/concurrency (out of scope)
- Advanced type system features (monads, effect systems)

### Architecture Approach

The architecture follows MLIR's progressive lowering model: high-level functional constructs in a custom FunLang dialect, lowered through intermediate dialects (SCF, MemRef, Arith) to LLVM dialect, then to native code. This is standard for complex source languages and enables optimization at multiple abstraction levels.

**Major components:**
1. **AST-to-MLIR Translator** (F# module) — Pattern matches on typed AST, calls MLIR builder APIs to emit custom dialect operations
2. **Custom FunLang Dialect** (C++ with TableGen definitions, exposed via C API) — Operations: `funlang.closure`, `funlang.match`, `funlang.cons`; Types: `!funlang.closure`, `!funlang.list<T>`
3. **Lowering Pass Pipeline** (MLIR passes in sequence) — Closure conversion → Pattern lowering → Data structure lowering → Standard dialect lowering → LLVM dialect
4. **F# Interop Layer** (P/Invoke wrappers) — Safe F# types with IDisposable wrapping raw MLIR C API handles
5. **Runtime Support Library** (C library) — GC interface, closure allocation/calling, list/tuple helpers, pattern match failure handling
6. **LLVM Backend** (standard) — MLIR-to-LLVM translation, LLVM optimization passes, native code generation

**Key patterns:**
- **Closure conversion**: Transform closures into (function pointer, environment struct) pairs with explicit heap allocation
- **Pattern match compilation**: Decision tree strategy, not naive nested if-else (avoids exponential code growth)
- **Progressive lowering**: FunLang dialect → Standard dialects → LLVM dialect (each step is simple, testable)

### Critical Pitfalls

Research identified 10 critical pitfalls, each with clear prevention strategies. Top 5 by severity:

1. **Early commitment to LLVM-C API without validating MLIR C API coverage** — MLIR C API may lack custom dialect builders; must prototype before Chapter 1. Prevention: Build prototype that creates custom dialect, registers operations, writes lowering pass using only C API.

2. **Naive closure compilation without environment analysis** — "Put everything on heap" creates incorrect semantics (capture by value vs. reference), prevents tail call optimization. Prevention: Dedicate chapter to closure theory and environment analysis before MLIR implementation.

3. **Deferring GC integration until after closures** — Using malloc in early chapters requires rewriting when GC added; functional languages without GC are unusable. Prevention: Use Boehm GC from Chapter 1 (simple dependency, zero MLIR changes).

4. **Incremental chapters that paint into architectural corners** — Starting with simple LLVM lowering, adding custom dialect in Chapter 10 requires rewriting Chapters 1-9. Prevention: Design full pipeline BEFORE writing tutorial; validate with end-to-end prototype compiling complex program.

5. **Pattern matching as naive nested if-else** — Generates exponentially large code for multi-clause patterns. Prevention: Implement decision tree compilation algorithm (reference: Peyton Jones book).

Additional high-impact pitfalls:
- **Assuming MLIR types directly model HM polymorphism** — MLIR is monomorphic; requires monomorphization pass or boxing strategy
- **F# P/Invoke without lifetime management** — MLIR objects have ownership semantics; must use IDisposable wrappers with deterministic cleanup
- **Tutorial code snippets not actually buildable** — Must extract and compile all snippets as part of CI
- **Skipping MLIR verification passes** — Invalid IR compiles silently until later passes crash with cryptic errors
- **Treating MLIR as syntax for LLVM IR** — Misses MLIR's value proposition (progressive lowering through custom dialects)

## Implications for Roadmap

Based on combined research findings, the roadmap should follow this phase structure. The order is driven by three factors: (1) technical dependencies (GC before closures, closures before higher-order functions), (2) pedagogical progression (simple to complex), and (3) pitfall avoidance (validate architecture early).

### Phase 1: Foundation & Interop Validation
**Rationale:** Must resolve F#-to-MLIR interop before any tutorial content. This is the highest technical risk. Also establishes the pipeline architecture that all later chapters will use.

**Delivers:**
- Prototype: F# program that creates MLIR context, custom dialect, and basic operation
- Validated: Can P/Invoke to MLIR C API for all needed operations
- Decision: If C API insufficient, build C++ wrapper DLL
- Architecture: Full pipeline design (Typed AST → Custom Dialect → Lowering → LLVM)
- Build system: MLIR with C API enabled, F# project structure, native library deployment

**Addresses:**
- Pitfall 1 (LLVM-C API commitment)
- Pitfall 4 (architectural corners)
- Pitfall 7 (F# lifetime management)
- Stack: MLIR C API coverage validation

**Research needed:** Deep dive into MLIR C API documentation and experimentation

---

### Phase 2: Memory Management Strategy
**Rationale:** GC must be integrated BEFORE first heap allocation (closures). Deferring creates technical debt that forces rewrite of all earlier chapters.

**Delivers:**
- Boehm GC linked and tested
- Runtime library scaffolding (gc.c with GC_malloc wrappers)
- F# build system copies runtime library to output
- Test: Allocate heap object from F#-generated MLIR, verify GC collects it

**Addresses:**
- Pitfall 3 (deferred GC)
- Features: Memory management basics (table stakes)
- Architecture: Runtime support library component

**Research needed:** None (Boehm GC integration is well-documented)

---

### Phase 3: Arithmetic & Let Bindings (Chapters 1-2)
**Rationale:** First working code. Builds confidence. Establishes verification and testing patterns. Uses only standard dialects (Func, Arith) before custom dialect complexity.

**Delivers:**
- Chapter 1: Arithmetic expressions (literals, +, -, *, /, comparison)
- Chapter 2: Let bindings (SSA form, scoping)
- Direct lowering: Arith dialect → LLVM dialect → native binary
- Test suite pattern established
- Verification after every IR construction

**Addresses:**
- Features: Arithmetic evaluation, let bindings (table stakes)
- Pitfall 9 (skipping verification)
- Pitfall 8 (non-buildable snippets)

**Research needed:** None (standard MLIR patterns)

---

### Phase 4: Control Flow (Chapter 3)
**Rationale:** Natural progression from let bindings. Introduces block arguments (MLIR's phi nodes) and SCF dialect.

**Delivers:**
- Chapter 3: If/else, booleans, comparisons
- SCF dialect usage (scf.if with block arguments)
- Boolean operations (arith.cmpi)

**Addresses:**
- Features: Control flow (table stakes)
- Architecture: SCF dialect in lowering pipeline

**Research needed:** None (standard MLIR patterns)

---

### Phase 5: Custom Dialect Introduction
**Rationale:** Before functions/closures, introduce custom dialect infrastructure. This avoids "works in early chapters, breaks when adding custom dialect" problem.

**Delivers:**
- FunLang dialect scaffolding (TableGen definitions)
- Basic operations: funlang.constant, funlang.add (mirrors Arith)
- Lowering pass: FunLang → Arith → LLVM
- Recompile Chapters 1-3 using custom dialect

**Addresses:**
- Pitfall 4 (architectural corners)
- Pitfall 10 (skipping custom dialect)
- Features: Custom dialect as differentiator

**Research needed:** Moderate (custom dialect registration via C API may need experimentation)

---

### Phase 6: Functions (Chapters 4-5)
**Rationale:** Core abstraction. Requires custom dialect for function representation. Enables recursion.

**Delivers:**
- Chapter 4: Simple functions (definition, calls, arguments, return)
- Chapter 5: Recursive functions (self-reference, stack frames)
- Operations: funlang.func, funlang.call
- Call convention established
- Monomorphization strategy for polymorphic functions

**Addresses:**
- Features: Functions (table stakes)
- Pitfall 6 (HM types vs MLIR types)
- Architecture: Function representation in custom dialect

**Research needed:** Monomorphization implementation strategy

---

### Phase 7: Closures (Chapters 6-7)
**Rationale:** Requires environment analysis theory before MLIR implementation. Depends on GC (Phase 2) and custom dialect (Phase 5).

**Delivers:**
- Chapter 6: Closure theory (environment analysis, capture semantics)
- Chapter 7: Closure implementation in MLIR
- Operations: funlang.closure_create, funlang.closure_call
- Types: !funlang.closure (struct with fn_ptr + env_ptr)
- Lowering pass: Closure conversion (environment allocation, explicit passing)

**Addresses:**
- Pitfall 2 (naive closure compilation)
- Features: Closures (table stakes), higher-order functions foundation
- Architecture: Closure conversion pass

**Research needed:** Environment analysis algorithm details

---

### Phase 8: Data Structures (Chapters 8-9)
**Rationale:** Enables realistic programs. Lists are foundational for pattern matching (next phase).

**Delivers:**
- Chapter 8: Lists (cons, head, tail, isEmpty)
- Chapter 9: Tuples
- Types: !funlang.list<T>, !funlang.tuple<...>
- Operations: funlang.cons, funlang.list_head, funlang.is_empty
- Lowering pass: Lists → heap-allocated linked list structs
- Runtime library: List operations

**Addresses:**
- Features: ADTs preparation
- Architecture: Data structure lowering pass

**Research needed:** None (standard functional language representation)

---

### Phase 9: Pattern Matching (Chapters 10-11)
**Rationale:** Defining feature of functional languages. Requires decision tree compilation to avoid pitfall.

**Delivers:**
- Chapter 10: Pattern matching theory (decision trees)
- Chapter 11: Pattern matching implementation
- Operations: funlang.match (with regions per pattern arm)
- Lowering pass: Decision tree compilation → SCF if/switch
- Exhaustiveness checking (optional for tutorial)

**Addresses:**
- Pitfall 5 (naive pattern matching)
- Features: Pattern matching (table stakes)
- Architecture: Pattern lowering pass

**Research needed:** Decision tree compilation algorithm implementation

---

### Phase 10: Optimization (Chapters 12-13)
**Rationale:** Shows MLIR's optimization capabilities. Optional/advanced content.

**Delivers:**
- Chapter 12: Basic optimizations (constant folding, DCE, inlining)
- Chapter 13: Tail call optimization
- Optimization passes on custom dialect
- Performance benchmarks

**Addresses:**
- Features: Optimization passes, TCO (differentiators)
- Pitfall: TCO is table stakes for functional languages (required for recursion-heavy code)

**Research needed:** TCO transformation strategy in MLIR

---

### Phase Ordering Rationale

**Why this order:**
1. **Foundation first** (Phase 1-2): Validate highest-risk component (interop) and establish architecture before any content
2. **Simple to complex** (Phase 3-4): Start with arithmetic/control flow to build confidence
3. **Custom dialect early** (Phase 5): Introduce before it becomes architectural burden
4. **Functions before closures** (Phase 6-7): Closures are complex functions; need function foundation
5. **Data structures before patterns** (Phase 8-9): Pattern matching needs something to match on (lists)
6. **Optimization last** (Phase 10): Not essential for working compiler; added after core features

**Dependency enforcement:**
- GC (Phase 2) before closures (Phase 7) — heap allocation requirement
- Custom dialect (Phase 5) before closures (Phase 7) — needs custom ops
- Lists (Phase 8) before pattern matching (Phase 9) — pattern targets
- Functions (Phase 6) before closures (Phase 7) — closures are functions with environment

**Pitfall avoidance:**
- Prototype validates interop (Phase 1) — avoids mid-tutorial discovery of C API gaps
- GC integrated early (Phase 2) — avoids rewrite of closure chapters
- Full architecture designed (Phase 1) — avoids incremental painting into corners
- Custom dialect established (Phase 5) — avoids late-stage pipeline changes

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 1 (Foundation):** HIGH priority — MLIR C API custom dialect support is unverified; may need experimentation or fallback to C++ wrapper
- **Phase 5 (Custom Dialect):** MEDIUM priority — TableGen and dialect registration patterns via C API need validation
- **Phase 7 (Closures):** MEDIUM priority — Environment analysis algorithm and representation strategies need detailed design
- **Phase 9 (Pattern Matching):** MEDIUM priority — Decision tree compilation algorithm implementation details
- **Phase 10 (Optimization/TCO):** MEDIUM priority — Tail call transformation in MLIR context needs research

Phases with standard patterns (skip research-phase):

- **Phase 2 (Memory Management):** Boehm GC integration is well-documented
- **Phase 3-4 (Arithmetic, Control Flow):** Standard MLIR dialects (Arith, SCF) with extensive documentation
- **Phase 6 (Functions):** Standard function representation in MLIR
- **Phase 8 (Data Structures):** Standard functional language list/tuple representation

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | HIGH for MLIR/LLVM/F# versions (verifiable); LOW for MLIR C API completeness (requires validation); HIGH for P/Invoke approach (proven technique) |
| Features | HIGH | Well-established compiler curriculum; functional language features are known; MLIR Toy tutorial provides model; implementation complexity estimates depend on interop |
| Architecture | HIGH | Progressive lowering is standard MLIR practice; closure conversion and pattern compilation are well-documented in literature; F#/MLIR interop pattern is clear (even if implementation has unknowns) |
| Pitfalls | HIGH | Grounded in technical constraints (MLIR API design, LLVM semantics, F# type system); observed failure modes from MLIR community and tutorial design experience |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

Areas where research was inconclusive or needs validation during implementation:

- **MLIR C API completeness for custom dialects:** LOW confidence — requires Phase 1 prototype to validate. If insufficient, need C++ wrapper DLL (medium effort, well-understood pattern).

- **F# ecosystem for .NET/MLIR bindings:** LOW confidence — research assumed no mature bindings exist, but web search required to confirm. If bindings exist and are mature, could simplify interop significantly (reevaluate Phase 1).

- **Exact MLIR C API coverage in LLVM 19:** MEDIUM confidence — training data suggests improvements in 18-19, but exact feature set unverified. Phase 1 prototype will reveal gaps.

- **Performance of P/Invoke for high-frequency MLIR builder calls:** MEDIUM confidence — P/Invoke overhead is ~10-50ns per call; likely negligible for compiler (I/O dominates), but no empirical data. Can measure in Phase 3.

- **MLIR TableGen from F#:** LOW confidence — TableGen is Python/C++ toolchain; unclear how F# integrates. Likely: write dialect definitions in TableGen (C++), generate C++ code, expose via C API to F#. Phase 5 will clarify.

- **Monomorphization strategy:** MEDIUM confidence — Standard approach is clear (generate specialized functions per instantiation), but implementation details need design during Phase 6.

- **Decision tree compilation for pattern matching:** MEDIUM confidence — Algorithm is well-documented in literature (Peyton Jones), but MLIR-specific implementation needs design during Phase 9.

- **Tail call optimization in MLIR:** MEDIUM confidence — TCO is well-understood transformation, but MLIR-specific implementation (using which dialects/operations) needs research during Phase 10.

**Mitigation strategy:** Each gap has a clear validation phase. Phase 1 (Foundation) addresses the highest-risk gap (MLIR C API). Other gaps are addressed in their respective feature phases with targeted research-phase invocations if needed.

## Sources

### Primary (HIGH confidence)

**STACK.md:**
- MLIR Documentation (llvm.org/mlir) — C API, dialects, passes
- LLVM Release Notes (llvm.org releases) — MLIR-C API improvements in 17.x-19.x
- .NET P/Invoke Documentation (learn.microsoft.com) — FFI patterns for F#
- F# Language Design (fsharp.org) — F# 8/9 features

**FEATURES.md:**
- MLIR official Toy tutorial — canonical MLIR learning path
- LLVM documentation — target backend
- Functional language compiler literature — closure conversion, pattern compilation
- Tutorial design best practices — incremental complexity

**ARCHITECTURE.md:**
- MLIR Language Reference (mlir.llvm.org/docs/LangRef)
- Toy Tutorial (mlir.llvm.org/docs/Tutorials/Toy) — dialect creation
- Defining Dialects (mlir.llvm.org/docs/DefiningDialects)
- Pass Infrastructure (mlir.llvm.org/docs/PassManagement)
- "Compiling with Continuations" (Appel) — closure conversion techniques
- "Modern Compiler Implementation in ML" (Appel) — pattern match compilation
- MLton compiler (SML) — closure conversion and GC integration strategies
- F# Platform Invoke Documentation (learn.microsoft.com)

**PITFALLS.md:**
- MLIR official documentation and tutorials
- LLVM Garbage Collection documentation (llvm.org/docs/GarbageCollection.html)
- "Modern Compiler Implementation in ML" (Appel) — closure conversion, GC integration
- "The Implementation of Functional Programming Languages" (Peyton Jones) — pattern matching compilation
- F# interop documentation (learn.microsoft.com)

### Secondary (MEDIUM confidence)

- MLIR C API limitations: Known issue from MLIR community discussions, GitHub issues
- Closure compilation complexity: Well-documented in compiler literature
- GC integration difficulty: Common complaint in LLVM-based language implementations (OCaml, MLton)
- Incremental tutorial architecture failures: Observed in "Crafting Interpreters" community

### Tertiary (LOW confidence — requires validation)

- Existence of .NET/MLIR bindings: No confirmed packages in training data; web search required
- MLIR-C API completeness in LLVM 19.x: Training data suggests improvements, but exact coverage unverified
- Current LLVM/MLIR version as of Feb 2026: May be 20.x by now; training data cutoff January 2025

---

*Research completed: 2026-02-05*
*Ready for roadmap: YES*

**Next step:** Roadmapper agent can use this summary to structure phases, with confidence that:
- Phase 1 addresses highest technical risk (interop validation)
- Phase ordering reflects dependencies and pitfall avoidance
- Research flags identify which phases need deeper investigation
- Architecture is designed upfront to avoid incremental painting into corners
