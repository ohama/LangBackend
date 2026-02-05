# Pitfalls Research

**Domain:** MLIR Backend for Functional Languages (F#-based Tutorial)
**Researched:** 2026-02-05
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Early Commitment to LLVM-C API Before Understanding MLIR C++ API Surface

**What goes wrong:**
LLVM's C API (used via P/Invoke from F#) is designed for LLVM IR, not MLIR. MLIR's C API is incomplete and missing critical dialect builders, pattern rewriting infrastructure, and custom dialect registration. Tutorials commit to LLVM-C early, discover it halfway through, and face a complete rewrite or abandon structured builders for string generation.

**Why it happens:**
F# developers naturally reach for P/Invoke against C APIs. LLVM's C API is well-documented for LLVM IR. MLIR's C API existence is misleading — it covers only a small subset of MLIR's functionality (primarily IR inspection, not construction or transformation).

**How to avoid:**
- Research MLIR C API coverage BEFORE chapter 1
- Prototype: Can you create a custom dialect, define operations with regions, and write pattern rewrites using only MLIR C API?
- If no, decision point: (a) use C++ wrapper DLL with F# P/Invoke, (b) use existing bindings like LLVMSharp (check MLIR support), or (c) accept string-based MLIR generation (violates requirements)
- Document the interop architecture in Phase 1 (foundation) before any tutorial content

**Warning signs:**
- Chapter 1 works with simple LLVM IR generation, chapter 5 fails when custom MLIR dialect is needed
- Prototype code uses printf-style string building instead of StructuredBuilder APIs
- No examples found online of F# successfully using MLIR C API for dialect definition
- MLIR C API headers show functions for parsing/printing IR but not for DialectRegistry or RewritePattern

**Phase to address:**
Phase 1 (Foundation & Interop Architecture) — Must resolve before any tutorial chapters written

---

### Pitfall 2: Naive Closure Compilation Without Environment Analysis

**What goes wrong:**
Tutorials implement closures by "just put everything on the heap" without environment analysis. This works for simple examples but creates incorrect semantics (variable capture by reference vs. value), prevents tail-call optimization, and generates bloated IR that confuses later optimization passes. Readers build a compiler that "seems to work" but produces subtly incorrect programs.

**Why it happens:**
Closure conversion is complex. Tutorials skip the theory to "get something working fast." Environment analysis (what to capture, when to capture, capture by value vs. reference) requires non-trivial dataflow analysis that feels orthogonal to "teaching MLIR."

**How to avoid:**
- Dedicate an entire chapter to closure representation BEFORE implementing closures in MLIR
- Explain: (a) flat closure representation (captured vars as struct fields), (b) environment chain (linked closures), (c) hybrid approaches
- Include environment analysis: distinguish between immutable captures (by-value) and mutable captures (by-reference)
- Show examples where naive approach breaks: recursive closures, nested function definitions, mutable variable capture
- Tutorial chapter structure: Chapter N = closure theory + environment analysis; Chapter N+1 = MLIR implementation of analyzed closures

**Warning signs:**
- Closure chapter says "allocate heap object with all free variables" without defining "free variables" algorithm
- No discussion of mutable vs. immutable captures
- No examples of closures-returning-closures or recursive closures
- MLIR code shows every captured variable as a pointer without justification
- Tests only cover simple non-recursive closures

**Phase to address:**
Phase 4 (Closures chapter) — Environment analysis must precede MLIR implementation

---

### Pitfall 3: Deferring GC Integration Until "Later Chapters"

**What goes wrong:**
Tutorial compiles closures and heap-allocated tuples using raw `malloc` in early chapters, promising "we'll add GC in chapter 12." But GC integration requires rewriting allocation sites, adding write barriers, generating stack maps, and changing function calling conventions. Chapters 1-11 must be rewritten. If not rewritten, readers have learned a non-functional compiler architecture.

**Why it happens:**
GC is perceived as "advanced" and "orthogonal to learning MLIR." Authors want early wins with working code, so they use malloc. But functional languages without GC are not functional languages — memory leaks make every closure-heavy program unusable.

**How to avoid:**
- Decide GC strategy in Phase 1 (Foundation), implement in Phase 3 (before first heap allocation)
- Options:
  - (A) Use Boehm conservative GC from Chapter 1 (link external library, zero MLIR changes)
  - (B) Use LLVM's statepoint GC support with precise stack maps (requires LLVM 15+, intrinsics integration)
  - (C) Defer heap allocation features until after GC chapter (rearrange chapter order: GC before closures)
- Recommended: Option A (Boehm GC) for tutorial simplicity — add one C library dependency, replace `malloc` with `GC_malloc`, done
- Document GC strategy in architecture phase; validate it works before closure chapter

**Warning signs:**
- Roadmap shows closures in chapter 5, GC in chapter 12
- Chapter 5 uses `malloc` without explaining how memory is freed
- No prototype demonstrating GC integration with MLIR LLVM dialect
- Assumption that "LLVM handles GC automatically" (it does not — requires explicit statepoint support or conservative GC)

**Phase to address:**
Phase 1 (Foundation) — Decide strategy; Phase 3 (Memory Management) — Implement before closures

---

### Pitfall 4: Incremental Chapter Structure That Paints Into Architectural Corners

**What goes wrong:**
Tutorial starts with simple arithmetic compiled directly to LLVM dialect (no custom dialect). Chapter 5 adds custom dialect, requiring chapters 1-4 to be rewritten. Or: Tutorial uses string-based IR in early chapters for simplicity, chapter 8 needs structured builders for pattern rewrites, requiring full rewrite. Readers following along have non-working code after chapter transitions.

**Why it happens:**
Incremental tutorials optimize for "get chapter 1 working fast" without designing the end-state architecture. Each chapter is written in isolation, discovering incompatibilities only when later chapters are drafted.

**How to avoid:**
- Design the FULL pipeline architecture in Phase 1 (Foundation) before writing any tutorial chapters
- Validate architecture with end-to-end prototype: compile one simple program and one complex program (with closures, pattern matching) through the full pipeline
- Lock the architecture: Typed AST → Custom MLIR Dialect → [Lowering Passes] → LLVM Dialect → LLVM IR → Binary
- Every chapter must use the same architecture; only the subset of features changes
- Chapter 1 should compile `1 + 2` through the FULL pipeline (even if custom dialect has only one operation `arith.constant` initially)
- Incremental = adding operations to existing dialect, not changing the pipeline

**Warning signs:**
- Roadmap shows "introduce custom dialect" as a mid-series chapter (implies early chapters don't use it)
- Early chapters described as "LLVM IR directly" while later chapters described as "custom dialect"
- No end-to-end prototype exists before tutorial writing begins
- Chapter dependencies diagram shows backward edges (chapter N requires rewriting chapter N-3)

**Phase to address:**
Phase 1 (Foundation) — Design pipeline architecture; Phase 2 (Validation) — Build end-to-end prototype before tutorial content

---

### Pitfall 5: Pattern Matching Compiled as Nested If-Else Without Optimization

**What goes wrong:**
Pattern matching compiled naively generates deeply nested if-else trees that are exponentially large for multi-clause patterns. LLVM's optimizer does not magically convert this to efficient decision trees or jump tables. Generated code has poor branch prediction and code size.

**Why it happens:**
Pattern match compilation is a well-studied problem, but tutorials skip the theory to show "working" code fast. Nested if-else is conceptually simple and generates correct (but slow) code.

**How to avoid:**
- Pattern matching chapter must teach decision tree or backtracking automaton compilation
- Reference: "The Implementation of Functional Programming Languages" (Peyton Jones) — decision tree compilation algorithm
- Show example where naive approach generates 2^N branches, optimized approach generates O(N) decision tree
- MLIR implementation can use `scf.if` for decision tree (not nested if-else), or lower to `cf.switch` for integer tag dispatch
- Include performance comparison: naive vs. optimized pattern matching on micro-benchmark

**Warning signs:**
- Pattern matching chapter shows only 2-clause examples (doesn't stress-test algorithm)
- Generated MLIR has deeply nested `scf.if` operations (one per pattern clause)
- No discussion of pattern match compilation algorithms or decision trees
- Tests don't include patterns with overlapping clauses or many guards

**Phase to address:**
Phase 6 (Pattern Matching) — Teach decision tree compilation, not naive if-else chains

---

### Pitfall 6: Assuming MLIR's Type System Directly Models FunLang's HM Types

**What goes wrong:**
Tutorial maps FunLang's polymorphic types (∀a. a → a) directly to MLIR types. But MLIR types are monomorphic and concrete. Compilation requires monomorphization (generating separate function copies per instantiation) or boxing (uniform representation). Without this, generic functions cannot compile.

**Why it happens:**
MLIR has a rich type system (tensors, memrefs, function types), leading to assumption it can model polymorphism. But MLIR types are compilation-target types (like C types), not source-language types. Hindley-Milner polymorphism must be erased before MLIR.

**How to avoid:**
- Chapter on functions must explain: FunLang type inference produces polymorphic types, MLIR compilation requires monomorphic types
- Two strategies:
  - (A) Monomorphization: Generate one MLIR function per instantiation (like C++ templates)
  - (B) Uniform representation: Box all values, use single MLIR function (like OCaml bytecode)
- Recommended: Monomorphization for tutorial (simpler MLIR, better performance)
- Implementation: After type inference, add monomorphization pass that generates specialized AST nodes per call site
- Show example: `id : ∀a. a → a` called with `Int` and `String` generates two MLIR functions: `id_Int` and `id_String`

**Warning signs:**
- Tutorial assumes MLIR function types can have type parameters (they cannot)
- No discussion of monomorphization or boxing strategies
- Generic function examples are missing or only show one instantiation type
- Type system chapter in LangTutorial shows polymorphism, but MLIR tutorial never mentions how to compile it

**Phase to address:**
Phase 5 (Functions) — Explain and implement monomorphization before first generic function example

---

### Pitfall 7: F# P/Invoke Without Lifetime Management for MLIR Objects

**What goes wrong:**
MLIR C++ objects (MLIRContext, Module, Operation) have ownership semantics. Wrapping in F# via P/Invoke without careful lifetime tracking causes double-frees, use-after-free, or memory leaks. Compiler crashes non-deterministically or during cleanup.

**Why it happens:**
F# developers accustomed to GC assume MLIR objects can be finalized like managed objects. But MLIR C++ API uses RAII and explicit ownership. C API (if used) requires manual lifetime management. P/Invoke exposes raw pointers without lifetime tracking.

**How to avoid:**
- Design F# wrapper types with IDisposable for MLIR object lifetimes
- Pattern: `type MLIRContext = { Handle: nativeptr<MLIRContextOpaque> } interface IDisposable`
- Use `use` bindings in F# to ensure deterministic cleanup: `use ctx = MLIRContext.Create()`
- Document ownership rules: Does creating an Operation transfer ownership to the parent Block/Region, or must it be explicitly freed?
- If using C++ wrapper DLL, expose clear ownership semantics in C ABI (e.g., `_create` returns owned pointer, `_destroy` frees it)
- Test with Valgrind/AddressSanitizer to catch memory errors early

**Warning signs:**
- F# code calls `mlirModuleCreate` but never calls `mlirModuleDestroy`
- No IDisposable wrappers around MLIR handles
- Compiler crashes with "munmap_chunk(): invalid pointer" or similar C++ runtime errors
- No documentation of which F# functions own returned handles vs. borrow them

**Phase to address:**
Phase 1 (Foundation) — Design lifetime-safe F# wrappers for MLIR API before any tutorial code uses them

---

### Pitfall 8: Tutorial Code Snippets Are Not Actually Buildable

**What goes wrong:**
Tutorial shows F# code snippets for each chapter, but they cannot compile without missing context: imports, helper functions, or earlier chapter code. Readers copy-paste code, get compile errors, lose trust in tutorial.

**Why it happens:**
Authors write tutorial prose in markdown, extract code snippets, but never validate they compile independently. Snippets assume context that exists in author's head but not in the markdown.

**How to avoid:**
- Every code snippet in every chapter must be extracted and compiled as part of CI/validation
- Use literate programming or code extraction tool: `fsharp` blocks in markdown are extracted to .fs files and compiled
- Each chapter's code should be a standalone .fsx script that can run with `dotnet fsi chapterN.fsx`
- Include minimal imports/helpers at top of each snippet, or link to "shared prelude" file readers should have
- Automated test: Extract all tutorial code snippets, compile them, run basic smoke tests

**Warning signs:**
- Tutorial markdown has code blocks but no corresponding .fs/.fsx files in repo
- No CI job that validates tutorial code compiles
- Code snippets start with `let compile ast = ...` without showing imports or AST definition
- Authors say "readers should have this from LangTutorial" but don't link to specific file/function

**Phase to address:**
Phase 2 (Validation) — Set up code extraction and compilation tests before writing tutorial content

---

### Pitfall 9: Skipping MLIR Verification Passes Until Debugging

**What goes wrong:**
Tutorial generates MLIR IR without calling verification passes after each transformation. IR has invalid nesting (block without terminator, use before definition, type mismatches). These errors only surface when MLIR → LLVM lowering crashes with cryptic errors, making debugging extremely painful.

**Why it happens:**
MLIR verifier is not called by default — it must be explicitly invoked. Tutorials skip it for brevity, assuming "if it generates IR, it's valid." But MLIR is a *typed* IR with strict invariants, and invalid IR compiles silently until later passes fail.

**How to avoid:**
- Call `mlirOperationVerify()` after every IR construction/transformation in tutorial code
- Fail fast: If verification fails, print diagnostic, abort compilation
- Show example in Chapter 1: Build IR, verify it, show what happens if verification fails (intentional error)
- Include verification in every chapter's "compile" function: `buildIR(); verify(); return module;`
- Explain MLIR's verification: SSA dominance, type correctness, terminator requirements, region structure

**Warning signs:**
- Tutorial code builds MLIR Module but never calls verify
- Debugging section says "if you get strange errors later, try adding verification"
- No examples showing what verification failure looks like
- IR construction uses unchecked casts or assumes type correctness without verification

**Phase to address:**
Phase 1 (Foundation) — Establish pattern of verification in first chapter; every later chapter follows it

---

### Pitfall 10: Compiling to LLVM Dialect Without Understanding MLIR's Multi-Level IR

**What goes wrong:**
Tutorial treats MLIR as "syntax for LLVM IR" and immediately lowers to LLVM dialect, skipping custom dialects and incremental lowering. This works for trivial examples but cannot model functional language semantics (closures, pattern matching, tail calls). Readers never learn MLIR's actual value proposition.

**Why it happens:**
LLVM IR is familiar, MLIR is new. Authors want to show results quickly, so they skip custom dialects and use only LLVM dialect. But LLVM dialect is low-level (like LLVM IR) — it cannot directly represent high-level functional constructs.

**How to avoid:**
- Emphasize MLIR's key idea: progressive lowering through multiple dialects, each at different abstraction levels
- Chapter structure: Custom FunLang dialect (high-level) → Lowering passes → LLVM dialect (low-level) → LLVM IR
- Custom dialect operations: `funlang.closure`, `funlang.pattern_match`, `funlang.tailcall`
- Lowering passes map high-level ops to lower-level ops: `funlang.closure` → `llvm.struct` + `llvm.call_indirect`
- Show value: Custom dialect enables pattern-specific optimizations before lowering to LLVM

**Warning signs:**
- Tutorial roadmap shows "LLVM dialect" in chapter 1-2, "custom dialect" in chapter 10+ (backwards)
- Code generates `llvm.func` and `llvm.call` directly from AST without intermediate dialect
- No explanation of why MLIR has multiple dialects vs. just using LLVM IR directly
- Custom dialect is described as "optional advanced topic" rather than core architecture

**Phase to address:**
Phase 1 (Foundation) — Custom dialect is the primary compilation target; LLVM dialect is the final lowering step

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| String-based MLIR generation instead of structured builders | Easy F# integration, no C++ bindings needed | Cannot use pattern rewrites, verification is manual, error messages are poor | Never — violates core requirement |
| `malloc` instead of GC for early chapters | Simpler code, no GC dependency | Chapters must be rewritten when GC added; readers learn broken architecture | Never — use Boehm GC from start |
| Skip monomorphization, box all values | Uniform representation is conceptually simpler | 10-100x performance loss, readers learn unidiomatic approach | Only if tutorial explicitly teaches boxing tradeoffs |
| Naive pattern matching (nested if-else) | Implementation is 20 lines vs. 200 for decision trees | Generated code is exponentially large, poor performance | Acceptable for MVP if chapter includes "Optimization" section explaining decision trees |
| Hard-code function signatures instead of type-driven codegen | Avoids type system complexity | Breaks when polymorphic functions added; not generalizable | Only in Chapter 1-2 (arithmetic only) |
| Skip tail call optimization | Simpler calling convention | Stack overflow on recursive functions; unfunctional functional language | Never — tail calls are table stakes for FP |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| MLIR C API via P/Invoke | Assuming C API has full feature parity with C++ API | Check MLIR C API documentation; prototype custom dialect creation before committing to P/Invoke |
| LLVM statepoint GC | Assuming `gc "statepoint-example"` automatically generates stack maps | Must emit `llvm.experimental.gc.statepoint` intrinsics manually; requires liveness analysis |
| F# NativePtr vs. IntPtr | Using IntPtr for MLIR handles (loses type safety) | Use `nativeptr<MLIRContextOpaque>` with units-of-measure or phantom types for handle safety |
| MLIR Passes from F# | Calling C++ pass pipeline without registering custom dialect | Must register dialect before running passes; use `mlirDialectRegistryInsert` |
| LLVM IR to native binary | Calling `llc` command-line tool with shell exec | Use LLVM C API `LLVMTargetMachineEmitToFile` for programmatic compilation; avoid shell dependencies |
| Boehm GC linking | Forgetting to call `GC_INIT()` before first allocation | Initialize GC in compiler's generated `main()` function; document in tutorial |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Quadratic pattern matching compilation | Compilation time grows as O(N²) in number of pattern clauses | Use decision tree or backtracking automaton algorithm | >10 clauses per match |
| No MLIR pass pipeline caching | Recompile everything on each AST change | Cache MLIR modules or use incremental compilation | >1000 LOC programs |
| Copying MLIR operations instead of in-place rewrite | Memory usage grows unbounded during lowering passes | Use MLIR's pattern rewrite framework (in-place transformation) | >10 lowering passes |
| Boxing all values without unboxing optimization | 10-100x slowdown vs. specialized code | Use monomorphization or add unboxing pass after type inference | Any numeric-heavy code |
| Linear search in symbol tables | Compilation time grows as O(N²) in number of definitions | Use hash map for symbol table | >100 top-level definitions |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Executing arbitrary LLVM IR from tutorial examples without sandboxing | Code injection if tutorial examples are untrusted | Tutorial is documentation-only, no automatic execution; readers build compiler locally (low risk) |
| Not validating MLIR/LLVM version compatibility | Compiler crashes or undefined behavior if LLVM ABI changes | Pin exact LLVM version in tutorial (e.g., LLVM 19.0.0); include version check in F# initialization code |
| Exposing raw pointers from unsafe MLIR operations in safe F# API | Memory corruption if readers misuse low-level API | Wrap all unsafe operations in safe F# types with IDisposable; mark FFI module as private |
| Including LLVM development tools in distributed binaries | Large binary size (200+ MB), potential attack surface | Tutorial produces compiler binary with minimal LLVM libraries (Core, Target); exclude opt, llc tools |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| MLIR error messages shown raw to reader | Cryptic C++ stack traces: "Assertion failed: isa<OperandType>()" | Wrap MLIR errors with context: "Failed to compile pattern matching (line 10): MLIR type error - expected Integer, got String" |
| No intermediate IR dumps for debugging | Reader's code doesn't work, no visibility into what MLIR was generated | Include `-dump-mlir` flag in every chapter; show MLIR output in tutorial examples |
| Tutorial examples only show happy path | Reader's code has type error, compiler crashes instead of error message | Each chapter includes "Common Errors" section with debugging tips |
| Assuming reader knows MLIR/LLVM concepts | Reader lost when tutorial says "lower to LLVM dialect" (what's a dialect?) | Chapter 1 includes "MLIR Primer" section with glossary: dialect, operation, region, block, SSA |
| Code snippets without expected output | Reader runs code, doesn't know if output is correct | Every code example includes "Expected output:" section with MLIR IR dump or binary behavior |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Closures:** Often missing escape analysis (which closures can be stack-allocated vs. heap) — verify heap allocation is actually needed
- [ ] **Pattern matching:** Often missing exhaustiveness checking (does match cover all cases?) — verify compiler errors on non-exhaustive match
- [ ] **Function calls:** Often missing tail call optimization — verify recursive functions don't stack overflow
- [ ] **MLIR lowering:** Often missing verification passes between transformations — verify each lowering pass produces valid IR
- [ ] **Memory management:** Often missing GC integration — verify heap-allocated values are actually freed
- [ ] **Type checking:** Often missing monomorphization for polymorphic functions — verify generic functions compile for multiple types
- [ ] **Interop layer:** Often missing lifetime management for MLIR handles — verify no memory leaks with Valgrind
- [ ] **Tutorial code:** Often missing imports/context to make snippets buildable — verify each code block compiles standalone
- [ ] **Error handling:** Often missing error messages for invalid programs — verify compiler reports errors instead of crashing
- [ ] **End-to-end test:** Often missing complex program compilation — verify compiler handles realistic program with all features

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Committed to LLVM-C API, discovered it lacks MLIR dialect support | HIGH | Build C++ wrapper DLL exposing needed MLIR functions as C ABI; invoke via P/Invoke; requires C++ build setup |
| Used malloc, now need GC in later chapters | MEDIUM | Replace `malloc` with `GC_malloc` (Boehm GC); add GC_INIT call; revise all chapters to link libgc |
| Naive pattern matching generates huge IR | LOW | Add pattern match optimization chapter before later chapters; rewrite pattern compilation algorithm |
| Tutorial pipeline architecture is wrong | HIGH | Redesign pipeline; rewrite all chapters using new architecture (essentially restart tutorial) |
| MLIR IR is invalid but wasn't verified | LOW | Add verification calls; fix IR construction bugs revealed by verifier |
| Code snippets don't compile | MEDIUM | Set up code extraction tool; fix all broken snippets; add CI job to prevent regression |
| Used nested if-else for pattern matching | MEDIUM | Add optimization pass chapter; implement decision tree compilation; show before/after performance |
| Forgot monomorphization | HIGH | Add monomorphization pass before MLIR compilation; rewrite function compilation chapter |
| Memory leaks from MLIR handles | MEDIUM | Add IDisposable wrappers; audit all MLIR API calls for missing cleanup; use FSharp.Core's `use` bindings |
| Generic functions broken | HIGH | Implement monomorphization or boxing strategy; rewrite type compilation chapter |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| LLVM-C API limitations | Phase 1 (Foundation) | Prototype custom dialect creation succeeds |
| Naive closure compilation | Phase 4 (Closures) | Environment analysis implemented, tests cover mutable captures |
| Deferred GC integration | Phase 1 (Foundation) + Phase 3 (Memory) | GC linked and tested before closures chapter |
| Incremental structure paints into corner | Phase 1 (Foundation) + Phase 2 (Validation) | End-to-end prototype compiles complex program with all features |
| Naive pattern matching | Phase 6 (Pattern Matching) | Decision tree algorithm implemented, benchmarks show linear scaling |
| MLIR types vs. HM types confusion | Phase 5 (Functions) | Monomorphization pass implemented, generic functions compile |
| F# P/Invoke lifetime bugs | Phase 1 (Foundation) | IDisposable wrappers exist, Valgrind reports no leaks |
| Non-buildable code snippets | Phase 2 (Validation) | CI job extracts and compiles all tutorial code |
| Skipped MLIR verification | Phase 1 (Foundation) | Every chapter includes verification, intentional error shown in Ch1 |
| Skipped custom dialect | Phase 1 (Foundation) | Custom dialect defined first, LLVM dialect is lowering target only |

---

## Tutorial-Specific Risks

### Risk 1: Chapter Ordering Creates Backward Dependencies

**Problem:** Later chapters require rewriting earlier chapters (e.g., adding GC requires rewriting closures).

**Prevention:**
- Design full chapter dependency graph before writing
- Validate: No chapter N should require changes to chapter <N
- Allow forward references ("we'll explain this in chapter 8") but not backward changes

**Validation:** Dependency graph is a DAG (directed acyclic graph), topologically sortable

---

### Risk 2: Tutorial Teaches Unidiomatic MLIR/LLVM Usage

**Problem:** Tutorial shows working code that violates MLIR best practices, readers propagate anti-patterns.

**Prevention:**
- Include "MLIR Style Guide" section in chapter 1
- Reference official MLIR tutorials and documentation
- Show both naive approach and idiomatic approach for complex features
- Highlight: "This works, but MLIR's preferred way is..."

**Validation:** MLIR developers review tutorial content for idiomaticity

---

### Risk 3: FunLang Feature Set Exceeds Tutorial Scope

**Problem:** LangTutorial's FunLang has 15+ features, tutorial covers only 8 features, readers cannot compile full programs.

**Prevention:**
- Phase 1: Define explicit feature subset for tutorial (arithmetic, let, if, functions, closures, pattern matching, lists, tuples)
- Document out-of-scope features: modules, imports, typeclasses, effects
- Final chapter: "What's Next" section explains missing features

**Validation:** Feature checklist matches PROJECT.md requirements; subset is clearly documented

---

### Risk 4: F#/.NET Runtime Conflicts With Generated Binary

**Problem:** Compiler is F# program, compiled output is native binary; mixing .NET and native code causes runtime conflicts.

**Prevention:**
- Tutorial compiler is pure F# (runs on .NET)
- Compiled FunLang programs are pure native binaries (no .NET dependency)
- No P/Invoke from FunLang code back to .NET runtime
- Document: "Compiler is F#, compiled programs are C-like native binaries"

**Validation:** Compiled FunLang binary runs on system without .NET installed

---

## Sources

### Primary Sources (Domain Expertise)

- MLIR official documentation and tutorials (https://mlir.llvm.org)
- LLVM Garbage Collection documentation (https://llvm.org/docs/GarbageCollection.html)
- "Modern Compiler Implementation in ML" (Appel) — closure conversion, GC integration
- "The Implementation of Functional Programming Languages" (Peyton Jones) — pattern matching compilation
- F# interop documentation (https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/functions/external-functions)

### Known Anti-Patterns

- MLIR C API limitations: Known issue that C API is incomplete (observed in MLIR community discussions, GitHub issues)
- Closure compilation complexity: Well-documented in compiler literature (Appel, Shao)
- GC integration difficulty: Common complaint in LLVM-based language implementations (see OCaml, MLton discussions)
- Incremental tutorial architecture: Observed failure mode in "Crafting Interpreters" community (users report chapter transitions breaking earlier code)

### Verification Methodology

Research based on:
1. Domain expertise in functional language compilation (Hindley-Milner, closures, pattern matching)
2. MLIR/LLVM architecture knowledge (multi-level IR, dialect lowering, verification)
3. F#/.NET interop experience (P/Invoke, lifetime management, marshaling)
4. Incremental tutorial design principles (progressive disclosure, backward compatibility, working examples)

All pitfalls are grounded in technical constraints (MLIR API design, LLVM IR semantics, F# type system) rather than speculation.

---
*Pitfalls research for: MLIR Backend Tutorial (FunLang/F#)*
*Researched: 2026-02-05*
