# Feature Research

**Domain:** MLIR Compiler Backend Tutorial for Functional Languages
**Researched:** 2026-02-05
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = tutorial feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Basic expression evaluation (arithmetic) | Foundation for all compilers; readers need immediate success | LOW | Chapter 1: literals, +, -, *, / operations |
| Variables and let bindings | Can't write useful programs without variable binding | MEDIUM | SSA form introduction, scoping rules |
| Control flow (if/else) | Essential for conditionals; demonstrates branching | MEDIUM | Branch ops, block arguments, phi nodes |
| Functions (definition and calls) | Core abstraction in any language | MEDIUM-HIGH | Function ops, call convention, stack frames |
| Type checking integration | Readers already know FunLang types; must show how types flow through MLIR | MEDIUM | Type attributes, verification passes |
| LLVM lowering | Tutorial promises "native binary" - must demonstrate full compilation | HIGH | Standard dialect → LLVM dialect → machine code |
| Pattern matching | FunLang is functional; pattern matching is non-negotiable | HIGH | Discriminator generation, case dispatch |
| Closures and lambda | Defining feature of functional languages | HIGH | Environment capture, heap allocation |
| Memory management basics | Need to handle heap allocations from closures | MEDIUM-HIGH | malloc/free or basic GC, ownership model |
| Standard library integration | Need I/O for useful programs (print at minimum) | LOW-MEDIUM | External function declarations, libc integration |
| Error handling in compiler | Tutorial code must handle invalid inputs gracefully | LOW | Diagnostic emission, error recovery |
| Incremental testing approach | Each chapter must be verifiable | LOW | Test files per chapter, regression suite |

### Differentiators (Competitive Advantage)

Features that set this tutorial apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Custom MLIR dialect for FunLang | Shows how to design domain-specific ops; most tutorials skip this | HIGH | FunLang dialect with func, apply, closure ops |
| Multi-stage lowering architecture | Demonstrates progressive lowering philosophy | MEDIUM | FunLang → High-level → Affine → SCF → Std → LLVM |
| Pattern-based rewrites | Shows MLIR's killer feature; teaches rewrite patterns | HIGH | DRR (Declarative Rewrite Rules), C++ patterns |
| Optimization passes | Goes beyond "just works" to "works well" | MEDIUM-HIGH | Inlining, constant folding, dead code elimination |
| Interactive REPL integration | Makes learning immediate and satisfying | MEDIUM | Read-eval-print loop with MLIR execution |
| Debugging support | Teaches debug info generation (rare in tutorials) | MEDIUM | LLVM debug info, line numbers, variable inspection |
| Real F# implementation | Using F# (not C++) is unique; shows MLIR is polyglot | MEDIUM | F# bindings to MLIR C API, functional design patterns |
| Recursive functions with TCO | Shows functional language optimization | HIGH | Tail call optimization pass, trampolines |
| ADTs and tagged unions | Functional language staple; demonstrates sum types | HIGH | Discriminated unions, type tags, switch generation |
| Higher-order functions | First-class functions; demonstrates function pointers | HIGH | Function type values, indirect calls |
| Gradual complexity ramp | Each chapter adds exactly one concept | LOW | Pedagogical gold standard but rare in practice |
| Performance comparisons | Shows MLIR output vs other backends | LOW | Benchmark each chapter against OCaml/F# native |
| Visual IR dumps | Helps readers see what's happening | LOW | GraphViz output, IR pretty-printing |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full garbage collector | "Real" functional languages need GC | GC is a semester-long course; tutorial becomes about GC not MLIR | Use reference counting or manual memory with clear documentation that "real implementation needs GC" |
| Complete standard library | Tutorials need lots of built-ins | Library design distracts from MLIR concepts; maintenance burden | Provide 5-10 essential functions (print, +, -, <, etc); show how to add more |
| Every MLIR dialect | MLIR has 50+ dialects; readers want to learn them all | Information overload; most dialects irrelevant to this use case | Cover 5-6 key dialects in lowering path; link to docs for others |
| Advanced optimizations | Readers want LLVM-level performance | Complex opts obscure core concepts; diminishing returns | Show 2-3 meaningful opts (inline, const fold); acknowledge LLVM does heavy lifting |
| Multi-file compilation | Real compilers handle multiple files | Adds complexity of module system, linking, name resolution | Single-file per chapter; mention "production needs modules" |
| Parallelism/concurrency | Functional languages support concurrency | Runtime complexity explodes; needs thread scheduler | Explicitly scope out; focus on sequential execution |
| Package manager | Need dependencies | Out of scope for backend tutorial | Use simple file inclusion if needed |
| IDE integration | Syntax highlighting would be nice | Tutorial is about backend not tooling | Provide basic syntax file; focus on compiler |
| All functional features | Monads, type classes, effect systems | Tutorial explodes in scope; each needs dedicated coverage | Cover core features; acknowledge advanced features in conclusion |

## Feature Dependencies

```
[Arithmetic Evaluation]
    └──requires──> [LLVM Integration (basic)]

[Let Bindings]
    └──requires──> [Arithmetic Evaluation]
    └──requires──> [SSA Form Understanding]

[If/Else]
    └──requires──> [Let Bindings]
    └──requires──> [Block Arguments]

[Functions]
    └──requires──> [If/Else]
    └──requires──> [Call Convention]
    └──enhances──> [Let Bindings (for recursion)]

[Closures]
    └──requires──> [Functions]
    └──requires──> [Heap Allocation]
    └──requires──> [Environment Capture Strategy]

[Pattern Matching]
    └──requires──> [If/Else]
    └──requires──> [ADTs/Tagged Unions]
    └──enhances──> [Functions (for pattern-based dispatch)]

[Custom Dialect]
    └──requires──> [Functions]
    └──enhances──> [All subsequent features (cleaner high-level IR)]

[Higher-Order Functions]
    └──requires──> [Closures]
    └──requires──> [Function Pointers]

[Tail Call Optimization]
    └──requires──> [Recursive Functions]
    └──enhances──> [Functions]

[Optimization Passes]
    └──requires──> [Custom Dialect OR Standard Dialect fluency]
    └──enhances──> [Any compiled feature]
```

### Dependency Notes

- **Arithmetic → LLVM:** Can't test anything without actual execution; need LLVM from chapter 1
- **Let Bindings → SSA Form:** Variables in SSA are fundamentally different; must teach concept explicitly
- **Closures → Heap Allocation:** Environment must outlive function creation; requires heap
- **Pattern Matching → ADTs:** Can't match without tagged unions to discriminate on
- **Custom Dialect enhances everything:** High-level ops make subsequent chapters cleaner, but adds upfront cost
- **Optimization Passes can come anytime:** Once dialect is stable, opts can be incrementally added

## Tutorial Chapter Progression

### Phase 1: Foundation (Chapters 1-3)

**Goal:** Get code executing; build confidence

- **Chapter 1:** Arithmetic expressions
  - Literals (int, float)
  - Binary operators (+, -, *, /, <, >)
  - Direct LLVM lowering (no custom dialect)
  - Print result to stdout
  - **Milestone:** `2 + 3 * 4` compiles and runs

- **Chapter 2:** Let bindings
  - Variable declaration and use
  - SSA form introduction
  - Scoping rules (nested lets)
  - **Milestone:** `let x = 5 in x * x` compiles

- **Chapter 3:** Control flow (if/else)
  - Conditional expressions
  - Block arguments (MLIR's phi nodes)
  - Boolean expressions
  - **Milestone:** `if x > 0 then x else -x` compiles

### Phase 2: Abstraction (Chapters 4-6)

**Goal:** Functions and reusability

- **Chapter 4:** Functions (simple, non-recursive)
  - Function definition
  - Function calls
  - Arguments and return values
  - Call convention
  - **Milestone:** `let square x = x * x in square 5` compiles

- **Chapter 5:** Recursive functions
  - Self-reference in function body
  - Stack frames
  - **Milestone:** Factorial function compiles

- **Chapter 6:** Mutual recursion
  - Forward declarations
  - Multiple functions calling each other
  - **Milestone:** `is_even`/`is_odd` compiles

### Phase 3: Functional Features (Chapters 7-9)

**Goal:** Core functional programming capabilities

- **Chapter 7:** Closures (basic)
  - Environment capture
  - Closure representation
  - Heap allocation
  - **Milestone:** `let make_adder n = (fun x -> x + n) in (make_adder 5) 3` compiles

- **Chapter 8:** Higher-order functions
  - Functions as arguments
  - Functions as return values
  - Function pointers vs closures
  - **Milestone:** `map` function compiles

- **Chapter 9:** Pattern matching (simple)
  - Pattern syntax to IR
  - Match on literals
  - Wildcard patterns
  - **Milestone:** `match x with 0 -> ... | _ -> ...` compiles

### Phase 4: Advanced Types (Chapters 10-11)

**Goal:** Algebraic data types

- **Chapter 10:** ADTs and tagged unions
  - Type definitions
  - Constructor generation
  - Discriminator tags
  - **Milestone:** `type Option = Some of int | None` compiles

- **Chapter 11:** Pattern matching on ADTs
  - Discriminator extraction
  - Constructor pattern matching
  - Nested patterns
  - **Milestone:** Full pattern matching on custom types

### Phase 5: Custom Dialect (Chapters 12-13)

**Goal:** Demonstrate MLIR's extensibility

- **Chapter 12:** FunLang custom dialect
  - Dialect definition
  - Custom operations (func.apply, closure.create)
  - Type system integration
  - **Milestone:** Recompile earlier chapters with custom ops

- **Chapter 13:** Progressive lowering
  - FunLang → SCF dialect
  - SCF → Standard dialect
  - Standard → LLVM
  - Pattern-based rewrites
  - **Milestone:** Multi-stage compilation pipeline

### Phase 6: Optimization (Chapters 14-15)

**Goal:** Make compiled code efficient

- **Chapter 14:** Basic optimizations
  - Constant folding
  - Dead code elimination
  - Inlining (simple)
  - **Milestone:** Measurable performance improvement

- **Chapter 15:** Tail call optimization
  - Tail position detection
  - TCO transformation
  - Benchmarking
  - **Milestone:** Factorial(10000) doesn't stack overflow

## MVP Definition

### Launch With (v1)

Minimum viable tutorial — what's needed to teach MLIR backend compilation.

- [x] Chapters 1-6 (Foundation + Abstraction)
- [x] Basic arithmetic through recursive functions
- [x] Direct LLVM lowering (no custom dialect initially)
- [x] Test suite for each chapter
- [x] Clear error messages
- [x] Working F# implementation
- [x] README with setup instructions

**Rationale:** This teaches the complete compilation pipeline and gets to "real" programs (recursive functions). Reader can claim to have built a working compiler.

### Add After Validation (v1.x)

Features to add once core chapters are validated.

- [ ] Chapter 7-9 (Closures + Higher-order functions)
  - **Trigger:** Reader feedback requests functional features
- [ ] Chapter 10-11 (ADTs + Pattern matching on ADTs)
  - **Trigger:** Core tutorial is solid; ready for complexity
- [ ] Visual IR dumps
  - **Trigger:** Readers struggle to understand transformations
- [ ] Performance benchmarks
  - **Trigger:** Questions about "is this fast?"

### Future Consideration (v2+)

Features to defer until tutorial is established.

- [ ] Chapters 12-13 (Custom dialect)
  - **Why defer:** High complexity; most readers won't build dialects
  - **Trigger:** Advanced readers request extensibility coverage
- [ ] Chapters 14-15 (Optimizations)
  - **Why defer:** Nice-to-have; LLVM handles most opts
  - **Trigger:** Tutorial has traction; readers want "next level"
- [ ] REPL integration
  - **Why defer:** Development effort; tutorial works without it
  - **Trigger:** Teaching/classroom adoption
- [ ] Debugging support
  - **Why defer:** Complex; readers can use print debugging
  - **Trigger:** Production use cases emerge

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Arithmetic evaluation | HIGH | LOW | P1 |
| Let bindings | HIGH | MEDIUM | P1 |
| If/else | HIGH | MEDIUM | P1 |
| Functions (simple) | HIGH | MEDIUM | P1 |
| Recursive functions | HIGH | MEDIUM | P1 |
| Test framework | HIGH | LOW | P1 |
| Error handling | HIGH | LOW | P1 |
| Closures | HIGH | HIGH | P2 |
| Pattern matching (basic) | MEDIUM | MEDIUM | P2 |
| Higher-order functions | MEDIUM | MEDIUM | P2 |
| ADTs | MEDIUM | HIGH | P2 |
| Pattern matching on ADTs | MEDIUM | HIGH | P2 |
| Custom dialect | MEDIUM | HIGH | P3 |
| Optimizations | LOW | MEDIUM | P3 |
| TCO | LOW | HIGH | P3 |
| REPL | LOW | MEDIUM | P3 |
| Debug info | LOW | MEDIUM | P3 |
| Visual IR | LOW | LOW | P2 |
| Benchmarks | LOW | LOW | P2 |

**Priority key:**
- P1: Must have for launch (Chapters 1-6)
- P2: Should have, add when core is solid (Chapters 7-11 + visual aids)
- P3: Nice to have, future consideration (Chapters 12-15 + advanced features)

## Comparison: Incremental vs Big-Bang Tutorial Approaches

| Aspect | Incremental (Recommended) | Big-Bang (Traditional) |
|--------|---------------------------|------------------------|
| Learning curve | Gradual; each chapter adds one concept | Steep; reader must grasp everything |
| Early wins | Chapter 1 compiles real code | Nothing works until end |
| Debugging | Isolated changes = easy debugging | Everything interacts = hard debugging |
| Completion rate | High (reader makes progress) | Low (reader gets stuck) |
| Code complexity | Increases gradually | Complex from start |
| Testing | Each chapter independently testable | Integration testing only |
| Maintenance | Easy to update single chapters | Changes cascade |
| **Best for:** | Learning and tutorials | Reference implementations |

**Our approach:** Incremental. Each chapter is a checkpoint that compiles and runs.

## Functional Language Specific Considerations

### Must Cover (Functional Language Essentials)

| Feature | Why Essential for FunLang | Implementation Notes |
|---------|---------------------------|---------------------|
| Closures | Core abstraction in functional PLs | Chapter 7; heap allocation required |
| Pattern matching | Primary control flow mechanism | Chapters 9 + 11; discriminators + exhaustiveness |
| Higher-order functions | Functions as values | Chapter 8; function pointers + closures |
| Immutability | Default in functional languages | SSA naturally enforces this |
| Tail call optimization | Functional idiom for loops | Chapter 15; critical for recursion-heavy code |
| ADTs | Primary data structuring mechanism | Chapter 10; tagged unions |

### Nice to Have (But Not Defining)

- Type inference (reader already knows FunLang types)
- Lazy evaluation (complex; can mention but not implement)
- Monads (advanced; out of scope)
- Type classes (advanced; out of scope)
- Effect systems (research topic; out of scope)

### Tricky Aspects (Functional → MLIR)

| Challenge | Why Tricky | Solution Approach |
|-----------|------------|-------------------|
| Closure representation | Need to capture environment + code pointer | Struct with fn ptr + captured values |
| Higher-order function dispatch | Don't know call target statically | Indirect calls via function pointers |
| Pattern match exhaustiveness | Need to verify all cases covered | Static analysis pass OR runtime error |
| Immutable data sharing | Functional languages share immutable data | Tutorial uses copying; mention optimization possibilities |
| Tail recursion | LLVM doesn't guarantee TCO | Explicit transformation pass |

## Sources and Confidence

**Sources:**
- MLIR official Toy tutorial (canonical MLIR learning path)
- LLVM documentation (target backend)
- Functional language compiler literature (closure conversion, pattern compilation)
- Tutorial design best practices (incremental complexity)
- Personal knowledge of MLIR infrastructure

**Confidence levels:**
- **Table stakes features:** HIGH (well-established compiler curriculum)
- **Functional language features:** HIGH (FunLang already defined; mapping known)
- **MLIR-specific features:** HIGH (Toy tutorial provides model)
- **Pedagogical progression:** MEDIUM-HIGH (incremental approach proven but sequencing has tradeoffs)
- **Implementation complexity estimates:** MEDIUM (depends on F# interop with MLIR C API)

**Known gaps:**
- F# bindings to MLIR C API may uncover unforeseen complexity
- Chapter sequencing may need adjustment based on teaching experience
- Performance characteristics unknown until implementation (affects benchmarking strategy)

---
*Feature research for: MLIR Compiler Backend Tutorial for Functional Languages*
*Researched: 2026-02-05*
