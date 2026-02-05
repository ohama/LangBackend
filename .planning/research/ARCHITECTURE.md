# Architecture Research

**Domain:** MLIR Compiler Backend for Functional Languages
**Researched:** 2026-02-05
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FunLang Compiler Frontend                 │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Lexer   │→ │ Parser  │→ │ Type     │→ │ Typed    │      │
│  │         │  │         │  │ Checker  │  │ AST      │      │
│  └─────────┘  └─────────┘  └──────────┘  └────┬─────┘      │
│                                                │             │
├────────────────────────────────────────────────┼─────────────┤
│                    MLIR Backend (NEW)          │             │
├────────────────────────────────────────────────┼─────────────┤
│  ┌─────────────────────────────────────────────┴─────────┐   │
│  │            1. AST-to-MLIR Translation                  │   │
│  │  - Pattern match on Typed AST                          │   │
│  │  - Emit FunLang custom dialect operations              │   │
│  │  - Build MLIR IR using structured APIs                 │   │
│  └────────────────────────┬───────────────────────────────┘   │
│                           │                                   │
│  ┌────────────────────────┴───────────────────────────────┐   │
│  │            2. Custom Dialect Layer                      │   │
│  │  Operations: funlang.closure, funlang.match,           │   │
│  │              funlang.cons, funlang.tuple, etc.         │   │
│  │  Types: !funlang.closure, !funlang.list<T>,            │   │
│  │         !funlang.tuple<T1,T2,...>                      │   │
│  └────────────────────────┬───────────────────────────────┘   │
│                           │                                   │
│  ┌────────────────────────┴───────────────────────────────┐   │
│  │            3. Lowering Pass Pipeline                    │   │
│  │  Pass 1: Closure Conversion                            │   │
│  │  Pass 2: Pattern Match Lowering (decision trees)       │   │
│  │  Pass 3: List/Tuple Lowering (to structs/heap alloc)   │   │
│  │  Pass 4: FunLang → Func/SCF/MemRef Dialect            │   │
│  │  Pass 5: Func/SCF/MemRef → LLVM Dialect               │   │
│  └────────────────────────┬───────────────────────────────┘   │
│                           │                                   │
│  ┌────────────────────────┴───────────────────────────────┐   │
│  │            4. LLVM Dialect → LLVM IR                    │   │
│  │  - Standard MLIR-to-LLVM translation                   │   │
│  │  - Link runtime support library                        │   │
│  └────────────────────────┬───────────────────────────────┘   │
│                           │                                   │
├───────────────────────────┼─────────────────────────────────┤
│            Runtime Support (C/F# P/Invoke)      │             │
├───────────────────────────┼─────────────────────────────────┤
│  ┌────────────┐  ┌────────┴──────┐  ┌──────────────┐        │
│  │ GC Runtime │  │ Closure       │  │ Pattern      │        │
│  │ (Boehm GC  │  │ Environment   │  │ Match        │        │
│  │  or malloc)│  │ Allocation    │  │ Helpers      │        │
│  └────────────┘  └───────────────┘  └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                      LLVM Backend                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ LLVM IR  │→ │ LLVM     │→ │ Native   │                   │
│  │ Optimize │  │ CodeGen  │  │ Binary   │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **AST-to-MLIR Translator** | Convert typed AST to custom MLIR dialect | F# module with pattern matching on AST, calls MLIR builder APIs |
| **Custom Dialect Definition** | Define FunLang-specific operations and types | TableGen definitions (.td files) or MLIR C++ API, exposed to F# via P/Invoke |
| **Closure Conversion Pass** | Transform closures into explicit environment structs | MLIR pass that rewrites funlang.closure → funlang.closure_env + funlang.closure_call |
| **Pattern Match Lowering Pass** | Convert pattern matching to decision trees | MLIR pass that transforms funlang.match → SCF if/switch constructs |
| **List/Tuple Lowering Pass** | Lower high-level data structures to memref/struct | MLIR pass converting funlang.list/tuple → heap allocations + struct access |
| **Standard Dialect Lowering** | Translate to Func/SCF/MemRef dialects | MLIR conversion passes following standard MLIR patterns |
| **LLVM Dialect Lowering** | Convert to LLVM dialect | Standard MLIR-to-LLVM conversion passes |
| **Runtime Library** | Provide GC, closure, and helper functions | C library linked at compile time, callable from LLVM IR |

## Recommended Project Structure

```
LangBackend/
├── src/
│   ├── FunLang.Compiler/         # Main compiler implementation
│   │   ├── MLIRBuilder.fs        # F# wrapper around MLIR C API
│   │   ├── AstToMlir.fs          # Typed AST → MLIR translation
│   │   ├── Codegen.fs            # Top-level codegen orchestrator
│   │   └── FunLang.Compiler.fsproj
│   │
│   ├── FunLang.Dialect/          # Custom MLIR dialect definition
│   │   ├── IR/
│   │   │   ├── FunLangOps.td     # TableGen operation definitions
│   │   │   ├── FunLangTypes.td   # TableGen type definitions
│   │   │   └── FunLangDialect.td # Dialect metadata
│   │   ├── Transforms/
│   │   │   ├── ClosureConversion.cpp   # Closure lowering pass
│   │   │   ├── PatternLowering.cpp     # Pattern match lowering
│   │   │   ├── DataStructureLowering.cpp  # List/tuple lowering
│   │   │   └── Passes.td                  # Pass registration
│   │   └── CMakeLists.txt        # C++ build for dialect + passes
│   │
│   ├── FunLang.Runtime/          # Runtime support library
│   │   ├── gc.c                  # Garbage collection interface
│   │   ├── closure.c             # Closure allocation/calling
│   │   ├── data.c                # List/tuple runtime helpers
│   │   ├── pattern.c             # Pattern match failure handling
│   │   └── CMakeLists.txt        # C build for runtime
│   │
│   └── FunLang.Interop/          # F# P/Invoke bindings to MLIR
│       ├── MLIRBindings.fs       # Low-level C API bindings
│       ├── DialectBindings.fs    # Custom dialect bindings
│       └── FunLang.Interop.fsproj
│
├── tutorial/                      # Tutorial markdown chapters
│   ├── 01-arithmetic.md
│   ├── 02-variables.md
│   ├── 03-control-flow.md
│   ├── 04-functions.md
│   ├── 05-closures.md
│   ├── 06-pattern-matching.md
│   ├── 07-custom-dialect.md
│   ├── 08-lowering-passes.md
│   ├── 09-llvm-integration.md
│   └── 10-native-binary.md
│
└── tests/
    ├── integration/              # End-to-end compilation tests
    └── examples/                 # FunLang programs to compile
```

### Structure Rationale

- **FunLang.Compiler/**: Pure F# code that orchestrates the compilation. This is what the tutorial teaches.
- **FunLang.Dialect/**: C++ MLIR code (standard practice for custom dialects). Pre-built and distributed as library.
- **FunLang.Runtime/**: C runtime library (standard for functional language backends). Pre-built and linked.
- **FunLang.Interop/**: F# P/Invoke layer to call MLIR C API. Shields compiler code from unsafe interop.
- **tutorial/**: Incremental chapters that build up from simple (arithmetic) to complex (full pipeline).

## Architectural Patterns

### Pattern 1: Custom Dialect as Intermediate Representation

**What:** Define domain-specific operations in a custom MLIR dialect before lowering to standard dialects.

**When to use:** When source language has high-level constructs (closures, pattern matching, algebraic data types) that don't map directly to LLVM/C semantics.

**Trade-offs:**
- **Pros:** Clean separation of concerns; optimization passes can work at high level; easier to debug and inspect IR
- **Cons:** More upfront work to define dialect; need to maintain lowering passes

**Example:**
```mlir
// High-level FunLang dialect (after AST translation)
func.func @example(%arg0: !funlang.list<i32>) -> i32 {
  %0 = funlang.match %arg0 {
    ^empty:
      %c0 = arith.constant 0 : i32
      funlang.yield %c0 : i32
    ^cons(%head: i32, %tail: !funlang.list<i32>):
      funlang.yield %head : i32
  }
  return %0 : i32
}
```

After pattern lowering pass:
```mlir
func.func @example(%arg0: !funlang.list<i32>) -> i32 {
  %is_empty = funlang.is_empty %arg0 : !funlang.list<i32>
  %result = scf.if %is_empty -> (i32) {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  } else {
    %head = funlang.list_head %arg0 : !funlang.list<i32>
    scf.yield %head : i32
  }
  return %result : i32
}
```

### Pattern 2: Progressive Lowering Pipeline

**What:** Incrementally lower from high-level abstractions through intermediate dialects to LLVM dialect.

**When to use:** Always, for complex source languages. Allows composition of simple transformation passes.

**Trade-offs:**
- **Pros:** Each pass is simple and testable; can reuse standard MLIR dialects; optimization opportunities at each level
- **Cons:** More passes = longer compile time; need to understand multiple dialect semantics

**Example Pipeline:**
```
FunLang Dialect
    ↓ (closure conversion)
FunLang Dialect (closures explicit)
    ↓ (pattern lowering)
FunLang Dialect (no pattern match) + SCF
    ↓ (data structure lowering)
Func + SCF + MemRef + Arith
    ↓ (standard dialect lowering)
LLVM Dialect
    ↓ (MLIR-to-LLVM translation)
LLVM IR
```

### Pattern 3: Closure Conversion via Environment Passing

**What:** Transform closures into pairs of (function pointer, environment struct).

**When to use:** Required for first-class functions and closures in functional languages.

**Trade-offs:**
- **Pros:** Explicit representation enables optimization; compatible with C calling conventions
- **Cons:** Heap allocation overhead; pointer indirection for captured variables

**Example:**
```fsharp
// FunLang source
let x = 42 in
let f = fun y -> x + y in
f 10
```

After closure conversion:
```mlir
// Create closure environment
%env_size = arith.constant 8 : i64  // sizeof(i32) for captured x
%env_raw = call @malloc(%env_size) : (i64) -> !llvm.ptr
%env = llvm.bitcast %env_raw : !llvm.ptr to !llvm.ptr<i32>

// Store captured variable x = 42 into environment
%x = arith.constant 42 : i32
llvm.store %x, %env : i32, !llvm.ptr<i32>

// Create closure struct { fn_ptr, env_ptr }
%closure = funlang.make_closure @f_impl, %env : (!llvm.ptr, !llvm.ptr) -> !funlang.closure

// Call closure
%arg = arith.constant 10 : i32
%result = funlang.call_closure %closure, %arg : (!funlang.closure, i32) -> i32
```

The closure implementation function:
```mlir
func.func @f_impl(%env: !llvm.ptr, %y: i32) -> i32 {
  %env_typed = llvm.bitcast %env : !llvm.ptr to !llvm.ptr<i32>
  %x = llvm.load %env_typed : !llvm.ptr<i32> -> i32
  %result = arith.addi %x, %y : i32
  return %result : i32
}
```

### Pattern 4: Pattern Match as Decision Trees

**What:** Compile pattern matching into efficient decision trees with guard tests.

**When to use:** For languages with algebraic data types and pattern matching (essential for FunLang).

**Trade-offs:**
- **Pros:** Efficient runtime execution; can optimize for common cases; exhaustiveness checked at compile time
- **Cons:** Code size can increase; complex to implement correctly

**Example:**
```fsharp
// FunLang: match on list
match xs with
| [] -> 0
| [x] -> x
| x :: y :: _ -> x + y
```

Decision tree strategy:
1. Check if list is empty → return 0
2. Check if tail is empty → return head
3. Otherwise → extract first two elements, return sum

```mlir
func.func @match_example(%xs: !funlang.list<i32>) -> i32 {
  // Test 1: is empty?
  %is_empty = funlang.is_empty %xs : !funlang.list<i32>
  %result = scf.if %is_empty -> (i32) {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  } else {
    %head = funlang.list_head %xs : !funlang.list<i32>
    %tail = funlang.list_tail %xs : !funlang.list<i32>

    // Test 2: is tail empty?
    %tail_empty = funlang.is_empty %tail : !funlang.list<i32>
    %result2 = scf.if %tail_empty -> (i32) {
      scf.yield %head : i32
    } else {
      %second = funlang.list_head %tail : !funlang.list<i32>
      %sum = arith.addi %head, %second : i32
      scf.yield %sum : i32
    }
    scf.yield %result2 : i32
  }
  return %result : i32
}
```

## Data Flow

### Compilation Pipeline Flow

```
[Typed AST]
    ↓
[AST Pattern Match] → Extract node type, children, type info
    ↓
[MLIR Builder API] → Create operations in FunLang dialect
    ↓
[FunLang MLIR Module] (high-level IR)
    ↓
[Pass Manager] → Run transformation passes
    ↓ Pass 1: Closure Conversion
    ↓   - Find Lambda nodes with free variables
    ↓   - Allocate environment structs
    ↓   - Rewrite to explicit env passing
    ↓
    ↓ Pass 2: Pattern Match Lowering
    ↓   - Detect funlang.match operations
    ↓   - Build decision tree
    ↓   - Emit SCF if/switch constructs
    ↓
    ↓ Pass 3: Data Structure Lowering
    ↓   - Convert funlang.list → heap-allocated linked list
    ↓   - Convert funlang.tuple → stack/heap struct
    ↓   - Insert runtime calls for construction/access
    ↓
[Standard Dialects: Func + SCF + MemRef + Arith]
    ↓
[Standard Lowering Passes]
    ↓   - Func → LLVM functions
    ↓   - SCF → LLVM control flow (br, cond_br)
    ↓   - MemRef → LLVM memory operations
    ↓   - Arith → LLVM arithmetic
    ↓
[LLVM Dialect]
    ↓
[MLIR-to-LLVM Translation]
    ↓
[LLVM IR Module]
    ↓
[Link Runtime Library]
    ↓   - libfunlang_runtime.a (GC, closures, data structures)
    ↓
[LLVM Optimization Passes] (optional)
    ↓
[LLVM CodeGen]
    ↓
[Native Binary]
```

### Key Data Flows

1. **Type Information Propagation:**
   - Typed AST carries HM inferred types
   - AST-to-MLIR translator maps FunLang types to MLIR types
   - MLIR type system enforces correctness during lowering
   - Type information used for runtime decisions (boxed vs unboxed, GC tracing)

2. **Closure Environment Capture:**
   - Free variable analysis during AST-to-MLIR translation
   - Environment allocation in closure conversion pass
   - Environment passing through function calls
   - Environment access lowered to memref loads

3. **Pattern Match Compilation:**
   - Pattern AST analyzed for structure
   - Decision tree constructed (prefer most specific patterns first)
   - Guards compiled to conditional branches
   - Match failure triggers runtime error

## Functional Language Specifics

### Closures

**Challenge:** First-class functions can capture arbitrary variables from enclosing scopes.

**MLIR Solution:**
- Define `!funlang.closure` type as struct `{ fn_ptr: !llvm.ptr, env_ptr: !llvm.ptr }`
- Closure conversion pass:
  1. Analyze free variables in lambda bodies
  2. Generate environment struct type for each closure
  3. Insert allocation code (malloc/GC allocate)
  4. Store captured variables into environment
  5. Generate closure struct with function pointer + env pointer
  6. Rewrite lambda body to take env as first parameter
  7. Rewrite call sites to unpack closure and pass env

**Build Order Implication:** Closures must be working before pattern matching (match closures over functions).

### Pattern Matching

**Challenge:** Exhaustiveness checking, efficient dispatch, complex nested patterns.

**MLIR Solution:**
- Define `funlang.match` operation with regions for each pattern arm
- Pattern lowering pass:
  1. Build decision tree from pattern list
  2. Generate discriminant checks (tag tests for ADTs, isEmpty for lists)
  3. Extract pattern variables (destructuring)
  4. Jump to appropriate arm based on tests
  5. Assert exhaustiveness or insert failure branch

**Build Order Implication:** Pattern matching can be implemented before closures (simpler), but full utility requires closures (functions in match arms).

### Garbage Collection

**Challenge:** FunLang has no manual memory management; need automatic GC.

**MLIR Solution:**
- **Option 1 (Simple):** Use Boehm GC conservative collector
  - Replace malloc/free with GC_malloc
  - No need to track pointers precisely
  - Link `libgc` at compile time
  - **Pros:** Easy to integrate, works immediately
  - **Cons:** Conservative (may retain garbage), slower than precise GC

- **Option 2 (Advanced):** Implement precise GC with stack maps
  - LLVM's `gc.statepoint` intrinsics
  - Generate GC metadata during lowering
  - Requires runtime cooperation
  - **Pros:** Better collection, faster
  - **Cons:** Complex implementation, more work

**Recommendation for Tutorial:** Start with Boehm GC (Option 1), mention Option 2 as advanced topic.

**Build Order Implication:** GC integration needed before lists/tuples (heap allocation). Can use manual malloc for early chapters.

### Memory Management Strategy

```
┌─────────────────────────────────────────┐
│         Stack Allocation (fast)         │
│  - Local variables                      │
│  - Small tuples (if optimization)       │
│  - Primitive values (int, bool)         │
└─────────────────────────────────────────┘
               ↓ (escape analysis)
┌─────────────────────────────────────────┐
│        Heap Allocation (GC managed)     │
│  - Closures (environment)               │
│  - Lists (cons cells)                   │
│  - Large tuples                         │
│  - Values that escape scope             │
└─────────────────────────────────────────┘
```

**MLIR Representation:**
- Stack allocation → `memref.alloca`
- Heap allocation → `call @GC_malloc` (or `@malloc` if no GC yet)
- Lowering pass decides stack vs heap based on escape analysis

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **MLIR C API** | P/Invoke from F# via `FunLang.Interop` | Use `[<DllImport("MLIR-C")>]` for MLIR builder functions |
| **Custom Dialect (C++)** | Build as shared library, P/Invoke | Expose dialect registration and pass functions to C API |
| **LLVM** | MLIR built-in translation | Use `mlir::translateModuleToLLVMIR()` |
| **Runtime Library (C)** | Static linking or dynamic loading | Linked into final binary; functions called from LLVM IR |
| **Boehm GC** | Link `libgc`, replace malloc | Optional: tutorial can start without GC |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **F# Compiler ↔ MLIR Builder** | P/Invoke function calls | F# calls MLIR C API through interop layer |
| **MLIR Module ↔ Pass Manager** | MLIR PassManager API | Passes registered and applied in sequence |
| **Custom Dialect ↔ Standard Dialects** | Lowering patterns | Conversion framework handles type/operation mapping |
| **LLVM Dialect ↔ LLVM IR** | Built-in MLIR translation | Standard MLIR API: `mlir::translateModuleToLLVMIR()` |

## Incremental Build Strategy

To support the tutorial's incremental chapter structure, here's the recommended build order:

### Phase 1: Foundations (Chapters 1-3)
**Goal:** Compile arithmetic and let bindings to native code.

**Components:**
- F# AST-to-MLIR translator (basic operations: add, subtract, multiply, divide, let)
- MLIR builder wrapper in F# (OpBuilder, ModuleOp, FuncOp)
- Use only standard MLIR dialects: `arith`, `func`
- Direct lowering to LLVM dialect (no custom dialect yet)
- Emit LLVM IR, compile with `llc` to native

**No need for:** Custom dialect, closures, GC, pattern matching

### Phase 2: Control Flow (Chapter 4)
**Goal:** Add if/else, booleans, comparisons.

**Components:**
- Extend translator for `If`, `Bool`, comparison operations
- Use `scf.if` (Structured Control Flow dialect)
- Map boolean operations to `arith.cmpi`

**No need for:** Closures, GC, pattern matching

### Phase 3: Functions (Chapters 5-6)
**Goal:** First-class functions and closures.

**Components:**
- Introduce custom FunLang dialect (simple version)
- Define `!funlang.closure` type
- Implement closure conversion pass (F# or C++)
- Add runtime library for closure allocation (`closure_alloc`, `closure_call`)
- Introduce heap allocation (can use simple `malloc` for now)

**No need for:** GC (yet), pattern matching

### Phase 4: Data Structures (Chapters 7-8)
**Goal:** Lists and tuples.

**Components:**
- Extend custom dialect with `!funlang.list<T>`, `!funlang.tuple<...>`
- Operations: `funlang.cons`, `funlang.list_head`, `funlang.is_empty`, etc.
- Runtime library for list operations
- Lowering pass: lists → heap-allocated linked list structs

**Introduce GC:** This is the right time to add Boehm GC (lots of heap allocation).

### Phase 5: Pattern Matching (Chapters 9-10)
**Goal:** Full pattern matching on lists, tuples, constants.

**Components:**
- Extend dialect with `funlang.match` operation
- Pattern lowering pass (decision tree compilation)
- Exhaustiveness checking (can be optional for tutorial)

**This completes the core language.**

### Phase 6: Custom Dialect Deep Dive (Chapters 11-12)
**Goal:** Explain custom dialect implementation in depth.

**Components:**
- TableGen definitions for operations and types
- Dialect verification and optimization passes
- Custom attributes and constraints

**This is pedagogical — teaches MLIR internals.**

### Phase 7: Optimization (Chapters 13-15)
**Goal:** Add optimization passes.

**Components:**
- Constant folding pass on FunLang dialect
- Dead code elimination
- Inline expansion for small functions
- LLVM optimization passes (`-O2`, `-O3`)

**This is optional/advanced.**

## Anti-Patterns

### Anti-Pattern 1: Skipping Custom Dialect

**What people do:** Directly lower from AST to LLVM dialect or LLVM IR strings.

**Why it's wrong:**
- Loses high-level semantic information (can't optimize closures, pattern matches)
- LLVM dialect is too low-level (everything is pointers and control flow)
- String generation is fragile, doesn't type-check
- Harder to debug (IR doesn't match source constructs)

**Do this instead:** Define a custom dialect that mirrors FunLang semantics, then progressively lower through intermediate representations.

### Anti-Pattern 2: Over-Engineering the Runtime

**What people do:** Implement a complex GC, object system, and runtime in first iteration.

**Why it's wrong:**
- Tutorial readers need incremental progress
- Complex runtime obscures the MLIR concepts
- GC integration is orthogonal to MLIR learning

**Do this instead:** Start with manual `malloc` (or no heap at all), add Boehm GC later when lists/closures are introduced. Keep runtime minimal.

### Anti-Pattern 3: Compiling Untyped AST

**What people do:** Directly compile the parsed AST without type checking.

**Why it's wrong:**
- MLIR is strongly typed; need type information to emit correct IR
- Type errors manifest as obscure MLIR verification failures
- Can't implement polymorphism or type-dependent optimizations

**Do this instead:** Always compile the **typed AST** produced by FunLang's type checker. Use type information to guide MLIR type selection.

### Anti-Pattern 4: Monolithic Lowering Pass

**What people do:** Single pass that lowers FunLang dialect all the way to LLVM dialect.

**Why it's wrong:**
- Hard to maintain and debug
- Can't reuse standard MLIR dialects and passes
- Difficult to add new features incrementally

**Do this instead:** Break lowering into phases:
1. FunLang dialect (high-level)
2. FunLang dialect (closures explicit, patterns lowered)
3. Standard dialects (Func, SCF, MemRef, Arith)
4. LLVM dialect

Each transition is a separate pass.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **Toy examples (< 100 LOC)** | Direct translation, no optimizations, Boehm GC, simple malloc |
| **Small programs (< 1000 LOC)** | Add basic optimizations (constant folding, DCE), incremental compilation per function |
| **Medium programs (< 10K LOC)** | Introduce optimization pipeline, consider precise GC, parallel pass execution |
| **Large programs (10K+ LOC)** | Module splitting, link-time optimization (LTO), ahead-of-time compilation cache |

### Scaling Priorities

1. **First bottleneck:** MLIR pass execution time
   - **Fix:** Enable multi-threading in PassManager (MLIR supports this)
   - Cache MLIR modules for unchanged functions

2. **Second bottleneck:** LLVM IR compilation time
   - **Fix:** Use LLVM's parallel codegen (`-parallel-llvm`)
   - Consider JIT for development builds, AOT for release

3. **Third bottleneck:** Runtime performance (GC pauses)
   - **Fix:** Upgrade from Boehm GC to generational GC or reference counting
   - Implement escape analysis to reduce heap allocations

## F# / MLIR Interop Strategy

**Challenge:** MLIR is a C++ library. F# needs to call MLIR APIs.

**Solution:** Use MLIR C API (official, stable) via F# P/Invoke.

### Binding Approach

```fsharp
// FunLang.Interop/MLIRBindings.fs

module MLIRBindings

open System
open System.Runtime.InteropServices

// Opaque handle types (C API pattern)
type MlirContext = nativeint
type MlirModule = nativeint
type MlirOperation = nativeint
type MlirValue = nativeint
type MlirType = nativeint

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirContext mlirContextCreate()

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirContextDestroy(MlirContext ctx)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

// ... more bindings
```

### High-Level F# Wrapper

```fsharp
// FunLang.Compiler/MLIRBuilder.fs

type MLIRBuilder(moduleName: string) =
    let context = MLIRBindings.mlirContextCreate()
    let location = MLIRBindings.mlirLocationUnknownGet(context)
    let mlirModule = MLIRBindings.mlirModuleCreateEmpty(location)

    member _.Context = context
    member _.Module = mlirModule

    member _.CreateFunction(name: string, argTypes: MlirType list, retType: MlirType) =
        // Use MLIR C API to build function
        // ...

    member _.EmitConstant(value: int) : MlirValue =
        // Emit arith.constant
        // ...

    interface IDisposable with
        member _.Dispose() =
            MLIRBindings.mlirModuleDestroy(mlirModule)
            MLIRBindings.mlirContextDestroy(context)
```

### Custom Dialect Exposure

For the custom FunLang dialect (written in C++), expose C wrapper functions:

```cpp
// FunLang.Dialect/CAPI/Dialects.cpp

extern "C" {
    void funlangRegisterDialect(MlirContext ctx) {
        mlir::DialectRegistry registry;
        registry.insert<mlir::funlang::FunLangDialect>();
        mlirContextAppendDialectRegistry(ctx, registry);
    }

    MlirOperation funlangCreateClosureOp(MlirLocation loc,
                                          MlirValue fnPtr,
                                          MlirValue envPtr) {
        // Create funlang.make_closure operation
        // ...
    }
}
```

Then P/Invoke from F#:

```fsharp
[<DllImport("FunLangDialect", CallingConvention = CallingConvention.Cdecl)>]
extern void funlangRegisterDialect(MlirContext ctx)

[<DllImport("FunLangDialect", CallingConvention = CallingConvention.Cdecl)>]
extern MlirOperation funlangCreateClosureOp(MlirLocation loc, MlirValue fnPtr, MlirValue envPtr)
```

## Sources

**MLIR Official Documentation:**
- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- Toy Tutorial (dialect creation): https://mlir.llvm.org/docs/Tutorials/Toy/
- Defining Dialects: https://mlir.llvm.org/docs/DefiningDialects/
- Pass Infrastructure: https://mlir.llvm.org/docs/PassManagement/

**Functional Language Compilation:**
- "Compiling with Continuations" (Appel) — closure conversion techniques
- "Modern Compiler Implementation in ML" (Appel) — pattern match compilation
- MLton compiler (SML) — closure conversion and GC integration strategies
- OCaml compiler (ocamlopt) — efficient functional language compilation

**MLIR for Functional Languages:**
- Koka language MLIR backend research (Microsoft Research)
- Mojo language (uses MLIR, has first-class functions)
- Julia LLVM backend (similar closure challenges)

**F# Interop:**
- LLVM.NET (similar P/Invoke approach for LLVM from .NET)
- F# Platform Invoke Documentation: https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/functions/external-functions

---
*Architecture research for: MLIR Compiler Backend for Functional Languages*
*Researched: 2026-02-05*
