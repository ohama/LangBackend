---
phase: 05-custom-mlir-dialect
plan: 01
subsystem: documentation
tags: [mlir, dialect, tablegen, c-api, custom-dialect, progressive-lowering]

# Dependency graph
requires:
  - "04-02: Higher-order functions implementation (closure infrastructure)"
  - "01-03: Appendix custom dialect (C API shim pattern established)"

provides:
  - "Chapter 14: Custom Dialect Design theory (1500+ lines)"
  - "TableGen ODS fundamentals documentation"
  - "C API shim pattern documentation"
  - "Progressive lowering philosophy"

affects:
  - "05-02: FunLang dialect implementation (will implement designs from Chapter 14)"
  - "Phase 6: Pattern matching (will extend FunLang dialect with match operations)"

# Tech tracking
tech-stack:
  added:
    - "TableGen ODS syntax documentation"
    - "C API shim pattern (wrap/unwrap)"
    - "OpBuilder usage patterns"

  patterns:
    - "Progressive lowering: FunLang → Func/MemRef → LLVM"
    - "Dialect hierarchy (high-level → mid-level → low-level)"
    - "C API memory management (caller-provided buffer, iterator pattern)"

# File tracking
key-files:
  created:
    - tutorial/14-custom-dialect-design.md: "Custom dialect design theory (2682 lines)"

  modified:
    - tutorial/SUMMARY.md: "Added Chapter 14 entry"

# Decisions
decisions:
  - id: progressive-lowering-strategy
    context: "Phase 4 used direct lowering (FunLang AST → LLVM dialect)"
    decision: "Adopt progressive lowering (FunLang → Func/MemRef → LLVM)"
    rationale: "Enables optimization at each level, better error messages, independent verification"

  - id: funlang-dialect-operations
    context: "What operations to include in FunLang dialect"
    decision: "Phase 5: make_closure, apply; Phase 6: match, nil, cons"
    rationale: "Start with closures (immediate Phase 4 improvement), defer pattern matching to Phase 6"

  - id: closure-type-representation
    context: "How to represent closures in custom dialect"
    decision: "!funlang.closure opaque type (no parameters)"
    rationale: "Simpler than parameterized types, internal representation handled in lowering"

  - id: c-api-shim-pattern
    context: "F# needs to call TableGen-generated C++ code"
    decision: "extern C wrapper functions with wrap/unwrap helpers"
    rationale: "Only viable pattern for F# P/Invoke to access C++ dialect API"

# Metrics
duration: 9 min
completed: 2026-02-06
---

# Phase 05 Plan 01: Custom Dialect Design Theory Summary

**One-liner:** Documented custom MLIR dialect design theory: progressive lowering philosophy, TableGen ODS syntax, and C API shim pattern for F# interop

## What Was Built

### Chapter 14: Custom Dialect Design (2682 lines)

Comprehensive theoretical foundation for custom MLIR dialects:

**Part 1: Dialect Design Motivation (1138 lines)**
- Phase 4 problem analysis: GEP patterns, complexity, lost semantics
- Custom dialect benefits: domain semantics preservation, compiler simplification, optimization opportunities
- MLIR Dialect architecture: Operation, Type, Attribute, Region, Symbol Table
- Progressive lowering philosophy: FunLang → Func/MemRef → LLVM
- ConversionTarget and RewritePattern concepts

**Part 2: TableGen ODS and C API Shim (1544 lines)**
- TableGen ODS fundamentals: DSL for code generation
- FunLang dialect definition: name, namespace, dependencies
- Operation definition: make_closure, apply (traits, assemblyFormat)
- Type definition: ClosureType, ListType (preview)
- C API Shim pattern: FunLangCAPI.h/.cpp implementation
- wrap/unwrap helpers: C handle ↔ C++ pointer conversion
- OpBuilder usage: operation creation patterns
- F# P/Invoke bindings: FunLangBindings.fs structure
- FunLang operations preview: make_closure, apply, match, nil/cons
- Common pitfalls: AnyType abuse, missing traits, symbol table misuse, memory management

## Key Concepts Established

### 1. Progressive Lowering Philosophy

**Problem (Phase 4 direct lowering):**
```fsharp
// FunLang AST → LLVM Dialect (single giant transformation)
let compileLambda ... =
    // Manual GEP index calculation
    // Manual environment size computation
    // Lost domain semantics immediately
```

**Solution (Phase 5 progressive lowering):**
```
FunLang Dialect (domain semantics)
    ↓ (FunLangToFunc pass)
Func + MemRef Dialect (mid-level abstraction)
    ↓ (FuncToLLVM pass)
LLVM Dialect (low-level machine model)
```

**Benefits:**
- Optimization at each level (closure inlining at FunLang level, standard opts at Func level)
- Independent verification (type check at each stage)
- Better error messages ("closure type mismatch" vs "pointer type mismatch")
- Simpler compiler code (each pass handles single abstraction level)

### 2. TableGen ODS Syntax

**Before (C++ manual definition):**
```cpp
class MakeClosureOp : public Op<...> {
    // 100+ lines of boilerplate
    static void build(...);
    LogicalResult verify();
    void print(...);
    static ParseResult parse(...);
};
```

**After (TableGen declarative definition):**
```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$funcName,
                       Variadic<AnyType>:$capturedValues);
  let results = (outs FunLang_ClosureType:$result);
  let assemblyFormat = "$funcName `(` $capturedValues `)` attr-dict `:` type($result)";
}
```

**Generated automatically:**
- Parser/Printer (from assemblyFormat)
- Builder (type-safe API)
- Verifier (basic type checks)
- Accessors (getFuncName(), getCapturedValues())

### 3. C API Shim Pattern

**Problem:**
- TableGen generates C++ code (classes, methods)
- F# P/Invoke requires `extern "C"` functions
- Cannot call C++ classes from F#

**Solution:**

```
F# Code (Compiler.fs)
    ↓ P/Invoke
C API Shim (FunLangCAPI.cpp)
    ↓ Call C++ API
C++ Dialect (TableGen generated)
```

**Implementation:**

```cpp
// C API (FunLangCAPI.h)
extern "C" {
    MlirOperation mlirFunLangMakeClosureOpCreate(
        MlirContext ctx, MlirLocation loc,
        MlirAttribute funcName, intptr_t numCaptured,
        MlirValue *capturedValues);
}

// Implementation (FunLangCAPI.cpp)
MlirOperation mlirFunLangMakeClosureOpCreate(...) {
    MLIRContext *context = unwrap(ctx);        // C → C++
    OpBuilder builder(context);
    auto op = builder.create<MakeClosureOp>(...); // C++ API
    return wrap(op.getOperation());             // C++ → C
}
```

**F# Binding:**

```fsharp
[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern MlirOperation mlirFunLangMakeClosureOpCreate(...)

type FunLangOps =
    static member CreateMakeClosure(...) =
        mlirFunLangMakeClosureOpCreate(...)
```

## Before/After Comparison

### Before (Phase 4): Direct LLVM Lowering

**Generated MLIR:**
```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    %env_size = arith.constant 12 : i64
    %env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr
    %n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr
    func.return %env_ptr : !llvm.ptr
}
```

**Problems:**
- GEP patterns repeated everywhere
- Lost "closure" concept (just opaque pointer)
- Manual environment size calculation
- No closure-specific optimizations possible

### After (Phase 5): Progressive Lowering

**Stage 1: FunLang Dialect (domain semantics)**
```mlir
func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.make_closure @lambda_adder(%n) : !funlang.closure
    func.return %closure : !funlang.closure
}
```

**Stage 2: After FunLangToFunc Lowering**
```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    %c3 = arith.constant 3 : index
    %env = memref.alloc(%c3) : memref<?xi32>
    %c1 = arith.constant 1 : index
    memref.store %n, %env[%c1] : memref<?xi32>
    %ptr = memref.cast %env : memref<?xi32> to !llvm.ptr
    func.return %ptr : !llvm.ptr
}
```

**Stage 3: After FuncToLLVM Lowering**
```mlir
llvm.func @make_adder(%n: i32) -> !llvm.ptr {
    %c12 = llvm.mlir.constant(12 : i64) : i64
    %env = llvm.call @GC_malloc(%c12) : (i64) -> !llvm.ptr
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr
    llvm.return %env : !llvm.ptr
}
```

**Benefits:**
- Stage 1: Domain semantics preserved (`!funlang.closure`)
- Optimization possible at Stage 1 (closure inlining, escape analysis)
- Each stage independently verifiable
- Final LLVM IR identical, but path is cleaner

## FunLang Dialect Design Decisions

### Operations (Phase 5)

| Operation | Purpose | Signature |
|-----------|---------|-----------|
| `funlang.make_closure` | Create closure | `@func_name + captured_values → !funlang.closure` |
| `funlang.apply` | Call closure | `!funlang.closure + args → result` |

### Types (Phase 5)

| Type | Purpose | Representation |
|------|---------|----------------|
| `!funlang.closure` | Closure value | Opaque (no parameters) |

### Operations (Phase 6 Preview)

| Operation | Purpose | Signature |
|-----------|---------|-----------|
| `funlang.match` | Pattern match | `scrutinee + regions → result` |
| `funlang.nil` | Empty list | `→ !funlang.list<T>` |
| `funlang.cons` | List cons | `head + tail → !funlang.list<T>` |

### Types (Phase 6 Preview)

| Type | Purpose | Representation |
|------|---------|----------------|
| `!funlang.list<T>` | Immutable list | Parameterized by element type |

## Common Pitfalls Documented

### 1. AnyType Abuse

**Bad:**
```tablegen
let results = (outs AnyType:$result);  // No type safety!
```

**Good:**
```tablegen
let results = (outs FunLang_ClosureType:$result);  // Type-safe!
```

### 2. Missing Traits

**Bad:**
```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure"> {
  // No traits → MLIR assumes side effects → no CSE!
}
```

**Good:**
```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  // Pure trait → MLIR can optimize (CSE, DCE, etc.)
}
```

### 3. String vs SymbolRefAttr

**Bad:**
```tablegen
let arguments = (ins StrAttr:$funcName);  // No compile-time checking!
```

**Good:**
```tablegen
let arguments = (ins FlatSymbolRefAttr:$funcName);  // Symbol table verification!
```

### 4. C API Memory Management

**Bad:**
```cpp
// Returns dangling pointer (stack memory)
MlirValue* getValues() {
    SmallVector<MlirValue, 4> result;
    // ...
    return result.data();  // DANGER!
}
```

**Good:**
```cpp
// Caller provides buffer
intptr_t getValuesInto(MlirValue *buffer, intptr_t size) {
    // Copy into caller's buffer
    // Return count
}
```

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

All verification checks passed:

```bash
# Total line count
wc -l tutorial/14-custom-dialect-design.md
# Output: 2682 (target: 1500+) ✓

# TableGen content
grep -c "TableGen\|def.*Dialect\|def.*Op" tutorial/14-custom-dialect-design.md
# Output: 72 (substantial TableGen coverage) ✓

# C API shim pattern
grep -c "extern.*C\|mlirFunLang\|wrap\|unwrap" tutorial/14-custom-dialect-design.md
# Output: 81 (comprehensive C API documentation) ✓

# Progressive lowering content
grep -c "Progressive Lowering\|점진적 하강" tutorial/14-custom-dialect-design.md
# Output: 6 (dedicated section) ✓

# Phase 4 comparison
grep -c "Phase 4\|getelementptr\|GC_malloc" tutorial/14-custom-dialect-design.md
# Output: 34 (substantial Phase 4 analysis) ✓

# funlang.* references
grep -c "funlang\." tutorial/14-custom-dialect-design.md
# Output: 106 (extensive examples) ✓

# SUMMARY.md updated
grep "14-custom-dialect-design" tutorial/SUMMARY.md
# Output: Chapter 14 link present ✓
```

## Next Phase Readiness

**Phase 5 Plan 02 (Chapter 15: FunLang Dialect Implementation):**

Ready to proceed with:
- TableGen file creation (FunLangOps.td, FunLangTypes.td, FunLangDialect.td)
- C++ dialect implementation (FunLangOps.cpp, FunLangTypes.cpp)
- C API shim implementation (FunLangCAPI.h, FunLangCAPI.cpp)
- F# bindings (FunLangBindings.fs)
- Lowering pass (FunLangToFunc.cpp)
- CMakeLists.txt build configuration

**Theoretical foundation complete.**

All design decisions documented:
- Progressive lowering strategy
- FunLang dialect operations (make_closure, apply)
- C API shim pattern
- Common pitfalls to avoid

**No blockers.** Implementation can proceed immediately.

## Files Changed

**Created:**
- `tutorial/14-custom-dialect-design.md` (2682 lines)
  - Part 1: Dialect design motivation (1138 lines)
  - Part 2: TableGen ODS and C API Shim (1544 lines)

**Modified:**
- `tutorial/SUMMARY.md` (added Chapter 14 entry)

## Commits

1. **6c4f20f** - `feat(05-01): write Chapter 14 Part 1 - dialect design motivation`
   - Introduction: Phase 4 problems analysis
   - Custom dialect benefits
   - MLIR Dialect architecture
   - Progressive lowering philosophy
   - ConversionTarget and RewritePattern concepts
   - Lines: 1138 (Part 1 complete)

2. **c7929ec** - `feat(05-01): write Chapter 14 Part 2 - TableGen ODS and C API Shim`
   - TableGen ODS fundamentals
   - FunLang dialect definition
   - Operation/Type definition structure
   - C API Shim pattern implementation
   - F# P/Invoke bindings
   - FunLang operations preview
   - Common pitfalls
   - Total lines: 2682 (Chapter 14 complete)
   - SUMMARY.md updated

## Lessons Learned

1. **Progressive lowering is essential for complex transformations**
   - Direct AST → LLVM lowering becomes unmaintainable quickly
   - Each intermediate stage enables specific optimizations
   - Independent verification at each level catches errors early

2. **TableGen dramatically reduces boilerplate**
   - Declarative operation definition vs 100+ lines of C++ per operation
   - Automatic parser/printer generation ensures consistency
   - Type safety built into generated code

3. **C API shim pattern is the only viable F# interop strategy**
   - Cannot call C++ classes from F# P/Invoke
   - wrap/unwrap pattern is standard MLIR convention
   - Ownership must be explicit (caller-provided buffer vs MLIR-owned)

4. **Custom types enable domain-specific optimizations**
   - `!funlang.closure` vs `!llvm.ptr` enables closure-specific passes
   - Type system prevents misuse (can't pass integer as closure)
   - IDE support (jump to definition) works with SymbolRefAttr

## Success Criteria

All success criteria met:

- [x] Chapter 14 exists with 1500+ lines (actual: 2682 lines)
- [x] TableGen ODS syntax explained with FunLang dialect definition
- [x] C API shim pattern documented with code examples
- [x] Progressive lowering philosophy covered (FunLang → SCF/MemRef → LLVM)
- [x] Common pitfalls section present (4 pitfalls documented)
- [x] SUMMARY.md updated with Chapter 14 entry
- [x] Korean plain style (~이다/~한다) throughout
- [x] Technical terms (TableGen, Operation, Type) in English

**Plan 05-01 complete. Duration: 9 minutes.**
