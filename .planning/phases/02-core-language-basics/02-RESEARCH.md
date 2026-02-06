# Phase 2: Core Language Basics - Research

**Researched:** 2026-02-06
**Domain:** MLIR SSA Form, Control Flow, Arithmetic Operations, Memory Management, Boehm GC Integration
**Confidence:** HIGH

## Summary

Phase 2 expands the minimal integer-literal compiler from Phase 1 into a functional language compiler supporting arithmetic expressions, let bindings, and if/else control flow with proper memory management. This phase teaches three fundamental compiler concepts: (1) how let bindings map to SSA values, (2) how control flow works via block arguments (MLIR's alternative to PHI nodes), and (3) when to use stack vs heap allocation.

The **critical architectural choice** is MLIR's use of **block arguments** instead of PHI nodes for representing SSA values at control flow merge points. Block arguments provide cleaner semantics (no "lost copy problem"), avoid special cases (PHI nodes must be at block top), and directly map to functional programming constructs. For if/else expressions that produce values, the `scf.if` operation with `scf.yield` terminators naturally expresses conditional value selection without explicit PHI nodes.

For memory management, the standard approach is **stack allocation for local temporaries** (`memref.alloca` with automatic deallocation) and **heap allocation for escaping values** (`memref.alloc` with explicit `memref.dealloc` or GC). Boehm GC integrates as a drop-in replacement for malloc/free, requiring only `GC_malloc()` calls and linking against `libgc`.

**Primary recommendation:** Use `arith` dialect for arithmetic/comparison operations, `scf.if` with block arguments for control flow, stack allocation for all values in Phase 2 (heap/GC deferred until closures in Phase 3+), and explain SSA/block arguments incrementally through worked examples showing MLIR IR output.

## Standard Stack

The established technologies for MLIR-based compiler implementation:

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **MLIR arith dialect** | LLVM 19.x | Integer arithmetic and comparison operations | Built-in dialect providing SSA-form arithmetic ops (addi, subi, muli, divsi, divui, cmpi) that lower cleanly to LLVM dialect |
| **MLIR scf dialect** | LLVM 19.x | Structured control flow (if, for, while) | High-level control flow representation using block arguments; easier to analyze/transform than branch-based CFG before lowering to `cf` dialect |
| **MLIR func dialect** | LLVM 19.x | Function definition and calls | Provides `func.func`, `func.call`, `func.return` with IsolatedFromAbove trait enforcing function scoping |
| **MLIR memref dialect** | LLVM 19.x | Memory allocation and access | Provides `memref.alloca` (stack), `memref.alloc` (heap), `memref.load`, `memref.store` for memory operations |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Boehm GC (bdwgc)** | 8.2+ | Conservative garbage collector for C/C++ | Phase 2+ for heap-allocated values; drop-in malloc replacement with `GC_malloc()` |
| **MLIR cf dialect** | LLVM 19.x | Low-level control flow (branch, cond_branch) | Lowering target for `scf` dialect before LLVM dialect conversion |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **scf.if** | Direct `cf.cond_br` with PHI nodes | `scf.if` is higher-level, uses block arguments (cleaner than PHI), easier to transform; `cf` is lowering target, not user-facing |
| **Boehm GC** | Reference counting | Boehm GC is simpler (no cycle handling needed), conservative (no code generation changes), but potentially higher memory overhead |
| **Boehm GC** | LLVM GC statepoints | Statepoints require safepoint insertion, stack map generation, complex runtime; Boehm GC is drop-in with no compiler changes |
| **Block arguments** | Traditional PHI nodes | Block arguments avoid PHI placement complexity, eliminate "lost copy problem", unify function args with basic block args |

**Installation (Boehm GC):**
```bash
# Clone and build Boehm GC
git clone https://github.com/ivmai/libatomic_ops
git clone https://github.com/ivmai/bdwgc
ln -s $(pwd)/libatomic_ops $(pwd)/bdwgc/libatomic_ops
cd bdwgc
autoreconf -vif
automake --add-missing
./configure --prefix=$HOME/boehm-gc-install
make && make install

# Link in your compiler
# Replace malloc with GC_malloc, remove free calls
```

## Architecture Patterns

### Recommended Project Structure

Phase 2 builds on Phase 1 foundation (Chapters 00-05 + Appendix). New chapters:

```
tutorial/
├── 00-prerequisites.md          # [Phase 1] LLVM build, .NET setup
├── 01-mlir-primer.md           # [Phase 1] MLIR concepts
├── 02-hello-mlir.md            # [Phase 1] First F# MLIR program
├── 03-pinvoke-bindings.md      # [Phase 1] P/Invoke layer
├── 04-wrapper-layer.md         # [Phase 1] Safe F# wrappers
├── 05-arithmetic-compiler.md   # [Phase 1] Integer literals only
├── appendix-custom-dialect.md  # [Phase 1] C++ dialect registration
├── 06-arithmetic-expressions.md # [Phase 2 NEW] Full arithmetic (+, -, *, /, negate, comparisons)
├── 07-let-bindings.md          # [Phase 2 NEW] Let bindings, SSA form, scoping
├── 08-control-flow.md          # [Phase 2 NEW] If/else with block arguments
└── 09-memory-management.md     # [Phase 2 NEW] Stack/heap allocation, Boehm GC integration
```

### Pattern 1: Arithmetic Operations in MLIR arith Dialect

**What:** Compile arithmetic expressions to MLIR `arith` dialect operations maintaining SSA form.

**When to use:** All integer arithmetic and comparison operations.

**Example:**
```fsharp
// AST: Add(Number 10, Multiply(Number 3, Number 4))
// Translates to MLIR:

let compileArithmetic (builder: OpBuilder) (expr: Expr) : MlirValue =
    match expr with
    | Number(n, _) ->
        let i32Type = builder.Context.GetIntegerType(32)
        let attr = builder.Context.GetIntegerAttr(i32Type, int64 n)
        builder.CreateConstantOp(attr).GetResult(0)
    | Add(lhs, rhs, _) ->
        let lhsVal = compileArithmetic builder lhs
        let rhsVal = compileArithmetic builder rhs
        builder.CreateArithAddi(lhsVal, rhsVal).GetResult(0)
    | Multiply(lhs, rhs, _) ->
        let lhsVal = compileArithmetic builder lhs
        let rhsVal = compileArithmetic builder rhs
        builder.CreateArithMuli(lhsVal, rhsVal).GetResult(0)
```

**Generated MLIR IR:**
```mlir
// Source: 10 + 3 * 4
func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %0 = arith.muli %c3, %c4 : i32    // 3 * 4
  %1 = arith.addi %c10, %0 : i32     // 10 + (3 * 4)
  func.return %1 : i32
}
```

### Pattern 2: Let Bindings Map to SSA Values

**What:** Let bindings in functional languages directly map to SSA value definitions; no variable mutation needed.

**When to use:** All let binding compilation.

**Example:**
```fsharp
// AST: Let("x", Number 5, Add(Var "x", Var "x"))
// "let x = 5 in x + x"

let compileLet (builder: OpBuilder) (name: string) (bindExpr: Expr) (bodyExpr: Expr) (env: Map<string, MlirValue>) : MlirValue =
    // Compile binding expression
    let bindVal = compileExpr builder bindExpr env

    // Extend environment with binding
    let env' = env.Add(name, bindVal)

    // Compile body with extended environment
    compileExpr builder bodyExpr env'

// Variable reference just looks up SSA value
let compileVar (name: string) (env: Map<string, MlirValue>) : MlirValue =
    match env.TryFind(name) with
    | Some(value) -> value
    | None -> failwithf "Unbound variable: %s" name
```

**Generated MLIR IR:**
```mlir
// Source: let x = 5 in x + x
func.func @main() -> i32 {
  %c5 = arith.constant 5 : i32        // x = 5 (SSA value %c5)
  %0 = arith.addi %c5, %c5 : i32      // x + x (use %c5 twice)
  func.return %0 : i32
}
```

**Key insight:** SSA form means "let x = expr" creates an immutable binding. No variable update, no phi nodes needed for let bindings—only for control flow merges.

### Pattern 3: scf.if with Block Arguments for Conditional Values

**What:** MLIR's `scf.if` operation uses block arguments to return values from branches, avoiding explicit PHI nodes.

**When to use:** Compiling if/else expressions that produce values.

**Example:**
```fsharp
// AST: If(Bool true, Number 42, Number 0)
// Compile to scf.if with scf.yield

let compileIf (builder: OpBuilder) (condExpr: Expr) (thenExpr: Expr) (elseExpr: Expr) (env: Map<string, MlirValue>) : MlirValue =
    let condVal = compileExpr builder condExpr env  // Must be i1
    let i32Type = builder.Context.GetIntegerType(32)

    // Create scf.if operation with result type
    let ifOp = builder.CreateScfIf(condVal, [| i32Type |])

    // Build 'then' region
    builder.SetInsertionPoint(ifOp.GetThenBlock())
    let thenVal = compileExpr builder thenExpr env
    builder.CreateScfYield([| thenVal |])

    // Build 'else' region
    builder.SetInsertionPoint(ifOp.GetElseBlock())
    let elseVal = compileExpr builder elseExpr env
    builder.CreateScfYield([| elseVal |])

    // scf.if result becomes SSA value
    ifOp.GetResult(0)
```

**Generated MLIR IR:**
```mlir
// Source: if true then 42 else 0
func.func @main() -> i32 {
  %true = arith.constant 1 : i1
  %result = scf.if %true -> (i32) {
    %c42 = arith.constant 42 : i32
    scf.yield %c42 : i32
  } else {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  }
  func.return %result : i32
}
```

### Pattern 4: Stack Allocation for Local Temporaries

**What:** Use `memref.alloca` for stack-allocated local variables with automatic lifetime management.

**When to use:** Phase 2 local values that don't escape function scope. (Heap allocation deferred to Phase 3+ for closures.)

**Example:**
```mlir
// Stack allocation example (for tutorial demonstration)
func.func @example() -> i32 {
  %alloca = memref.alloca() : memref<1xi32>  // Allocate on stack
  %c42 = arith.constant 42 : i32
  memref.store %c42, %alloca[%c0] : memref<1xi32>  // Store value
  %loaded = memref.load %alloca[%c0] : memref<1xi32>  // Load value
  func.return %loaded : i32
  // Stack automatically deallocated on return
}
```

**Key insight:** Phase 2 doesn't actually need memref operations yet (all values are SSA registers). Stack vs heap allocation becomes relevant in Phase 3+ when implementing closures, mutable references, or large data structures.

### Anti-Patterns to Avoid

- **Using cf.cond_br directly instead of scf.if:** Start with high-level `scf.if`, lower to `cf` later. Block arguments in `scf.if` are cleaner than managing PHI nodes manually.
- **Mutating variables in let bindings:** SSA form means immutability. `let x = 5 in let x = 10 in x` creates two distinct SSA values, not mutation.
- **Forgetting scf.yield terminators:** Every `scf.if` branch that produces values MUST end with `scf.yield`. Missing yields cause verification failure.
- **Hand-rolling PHI node insertion:** MLIR block arguments eliminate this complexity. Let the `scf` to `cf` lowering pass handle PHI generation.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| **Value merging at control flow joins** | Manual PHI node insertion logic | MLIR block arguments + `scf.if` | PHI placement is complex (dominance frontiers, minimal SSA). Block arguments handle this automatically. Lowering passes insert PHIs when converting `scf` → `cf` → `llvm`. |
| **Garbage collection** | Custom mark-sweep or reference counting | Boehm GC (conservative collector) | Boehm GC is battle-tested (30+ years), handles cycles, requires zero compiler changes (drop-in malloc replacement). Custom GC requires write barriers, stack maps, safepoints—complex and error-prone. |
| **Comparison operation lowering** | Custom logic for each comparison type | `arith.cmpi` with predicate enum | `arith.cmpi` handles 10 predicates (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge). One operation, works on scalars/vectors/tensors. Don't reimplement. |
| **Boolean expression compilation** | Custom true/false handling | `arith.constant 1 : i1` and `arith.constant 0 : i1` | MLIR booleans are `i1` type. True=1, False=0. Use arith.constant, not custom ops. |

**Key insight:** MLIR's progressive lowering philosophy means you compile to high-level dialects (`arith`, `scf`, `func`) and let built-in passes lower to `cf` → `llvm` → LLVM IR. Don't skip levels or hand-roll lowering logic.

## Common Pitfalls

### Pitfall 1: Incorrect Boolean Type for scf.if Condition

**What goes wrong:** Using `i32` (integer) as condition for `scf.if` instead of `i1` (boolean) causes MLIR verification failure.

**Why it happens:** Beginners think "any non-zero integer is true" (C semantics). MLIR is strongly typed: conditions must be `i1`.

**How to avoid:** Always compile boolean expressions (comparisons, boolean literals) to `i1` type. Use `arith.cmpi` for comparisons (returns `i1`), `arith.constant 1 : i1` for true, `arith.constant 0 : i1` for false.

**Warning signs:** MLIR verifier error: `'scf.if' op operand #0 must be 1-bit signless integer, but got 'i32'`

**Example fix:**
```fsharp
// WRONG: Using i32 as condition
let i32Type = builder.Context.GetIntegerType(32)
let condVal = builder.CreateConstant(1, i32Type)  // i32, not i1!
let ifOp = builder.CreateScfIf(condVal, ...)  // VERIFICATION FAILS

// CORRECT: Use i1 type
let i1Type = builder.Context.GetIntegerType(1)
let condVal = builder.CreateConstant(1, i1Type)  // i1
let ifOp = builder.CreateScfIf(condVal, ...)  // OK
```

### Pitfall 2: Mismatched scf.yield Types Across Branches

**What goes wrong:** `scf.if` branches yield values of different types, causing verification failure.

**Why it happens:** Tutorial authors forget to ensure both `then` and `else` branches produce the same type signature.

**How to avoid:** Determine result type BEFORE creating `scf.if`. Ensure both `scf.yield` operations have matching operand types.

**Warning signs:** MLIR verifier error: `'scf.yield' op types mismatch between then and else regions`

**Example fix:**
```mlir
// WRONG: Type mismatch
%result = scf.if %cond -> (i32) {
  %c42 = arith.constant 42 : i32
  scf.yield %c42 : i32
} else {
  %true = arith.constant 1 : i1   // Yielding i1, not i32!
  scf.yield %true : i1              // TYPE MISMATCH
}

// CORRECT: Both branches yield i32
%result = scf.if %cond -> (i32) {
  %c42 = arith.constant 42 : i32
  scf.yield %c42 : i32
} else {
  %c0 = arith.constant 0 : i32
  scf.yield %c0 : i32
}
```

### Pitfall 3: Forgetting to Lower scf Dialect Before LLVM Translation

**What goes wrong:** Attempting to translate MLIR module containing `scf.if` directly to LLVM IR fails because LLVM doesn't understand `scf` dialect.

**Why it happens:** Beginners don't realize MLIR uses progressive lowering: `scf` → `cf` → `llvm` → LLVM IR.

**How to avoid:** Run lowering passes in correct order:
1. `--convert-scf-to-cf` (structured control flow → branch-based control flow)
2. `--convert-func-to-llvm` (func dialect → llvm dialect)
3. `--convert-arith-to-llvm` (arith dialect → llvm dialect)
4. `--reconcile-unrealized-casts` (cleanup)
5. Then `mlirTranslateModuleToLLVMIR`

**Warning signs:** Error during `mlirTranslateModuleToLLVMIR`: `Unhandled operation 'scf.if'`

**Example fix:**
```fsharp
// WRONG: Skip scf lowering
let passManager = mlirPassManagerCreate(context)
mlirPassManagerAddOwnedPass(passManager, mlirCreateConversionConvertFuncToLLVMPass())
// ... missing scf-to-cf conversion!
mlirPassManagerRun(passManager, module)  // scf.if still present!
mlirTranslateModuleToLLVMIR(module)      // FAILS

// CORRECT: Lower scf first
let passManager = mlirPassManagerCreate(context)
mlirPassManagerAddOwnedPass(passManager, mlirCreateConversionConvertSCFToCFPass())
mlirPassManagerAddOwnedPass(passManager, mlirCreateConversionConvertArithToLLVMPass())
mlirPassManagerAddOwnedPass(passManager, mlirCreateConversionConvertFuncToLLVMPass())
mlirPassManagerAddOwnedPass(passManager, mlirCreateConversionReconcileUnrealizedCastsPass())
mlirPassManagerRun(passManager, module)
mlirTranslateModuleToLLVMIR(module)  // Now succeeds
```

### Pitfall 4: Shadowing Variables Creates New SSA Values, Not Mutation

**What goes wrong:** Tutorial readers expect `let x = 5 in let x = 10 in x` to "overwrite" x, but SSA form creates two distinct values.

**Why it happens:** Intuition from imperative languages (variable mutation) doesn't apply to SSA.

**How to avoid:** Explain explicitly: "let bindings in SSA create new values, not mutations." Show MLIR IR with distinct SSA value names (`%x`, `%x_0`, etc.).

**Warning signs:** Reader confusion: "Why does MLIR have two %x values? I only declared x once!"

**Example clarification:**
```fsharp
// Source: let x = 5 in let x = 10 in x
// Reader expectation (WRONG): x is mutated from 5 to 10

// Actual SSA translation (CORRECT):
func.func @main() -> i32 {
  %x = arith.constant 5 : i32      // First binding: %x = 5
  %x_0 = arith.constant 10 : i32   // Shadowing creates NEW value: %x_0 = 10
  func.return %x_0 : i32            // Body uses %x_0, not %x
}
```

**Tutorial guidance:** "In SSA form, every value is defined exactly once. 'let x = ...' creates a new SSA value. Shadowing is renaming, not mutation."

### Pitfall 5: Boehm GC Requires -lgc Link Flag and GC_INIT() Call

**What goes wrong:** Forgetting to link Boehm GC library (`-lgc`) or call `GC_INIT()` causes runtime errors (segfaults or undefined behavior).

**Why it happens:** Boehm GC is a library, not automatic. Must explicitly initialize and link.

**How to avoid:**
- Call `GC_INIT()` at program startup (before any `GC_malloc` calls)
- Link with `-lgc` or `-L/path/to/boehm-gc/lib -lgc`
- Install Boehm GC headers (`gc.h`) in include path

**Warning signs:**
- Runtime segfault on first `GC_malloc` call
- Linker error: `undefined reference to 'GC_malloc'`

**Example fix:**
```c
// WRONG: No GC_INIT, no -lgc link
#include <gc.h>

int main() {
    int* p = GC_malloc(sizeof(int));  // Segfault! GC not initialized
    return 0;
}
// Compile: gcc program.c -o program  (missing -lgc!)

// CORRECT: Initialize GC and link library
#include <gc.h>

int main() {
    GC_INIT();  // Initialize Boehm GC
    int* p = GC_malloc(sizeof(int));
    *p = 42;
    return 0;
}
// Compile: gcc program.c -o program -lgc
```

## Code Examples

Verified patterns from official sources:

### Example 1: Comparison Operations with arith.cmpi

```mlir
// Source: MLIR official docs - arith dialect
// Comparison returning i1 boolean

%lhs = arith.constant 10 : i32
%rhs = arith.constant 20 : i32

// Less than (signed)
%is_less = arith.cmpi slt, %lhs, %rhs : i32  // returns i1
// Result: %is_less = 1 : i1 (true)

// Equality
%is_equal = arith.cmpi eq, %lhs, %rhs : i32  // returns i1
// Result: %is_equal = 0 : i1 (false)

// Use in scf.if
%result = scf.if %is_less -> (i32) {
  %c1 = arith.constant 1 : i32
  scf.yield %c1 : i32
} else {
  %c0 = arith.constant 0 : i32
  scf.yield %c0 : i32
}
```

### Example 2: Nested Let Bindings with SSA Environment

```fsharp
// Compiler pattern for nested scopes
type Env = Map<string, MlirValue>

let rec compileExpr (builder: OpBuilder) (expr: Expr) (env: Env) : MlirValue =
    match expr with
    | Number(n, _) ->
        let i32Type = builder.Context.GetIntegerType(32)
        let attr = builder.Context.GetIntegerAttr(i32Type, int64 n)
        builder.CreateConstantOp(attr).GetResult(0)

    | Var(name, _) ->
        match env.TryFind(name) with
        | Some(value) -> value
        | None -> failwithf "Unbound variable: %s" name

    | Let(name, bindExpr, bodyExpr, _) ->
        // Compile binding expression in current environment
        let bindVal = compileExpr builder bindExpr env

        // Extend environment with new binding
        let env' = env.Add(name, bindVal)

        // Compile body in extended environment
        compileExpr builder bodyExpr env'

    | Add(lhs, rhs, _) ->
        let lhsVal = compileExpr builder lhs env
        let rhsVal = compileExpr builder rhs env
        builder.CreateArithAddi(lhsVal, rhsVal).GetResult(0)
```

**Generated MLIR for:** `let x = 10 in let y = 20 in x + y`
```mlir
func.func @main() -> i32 {
  %x = arith.constant 10 : i32        // x binding
  %y = arith.constant 20 : i32        // y binding (env extended)
  %result = arith.addi %x, %y : i32   // Use both bindings
  func.return %result : i32
}
```

### Example 3: scf.if Lowering to cf.cond_br with Block Arguments

```mlir
// High-level scf.if (user writes this)
func.func @example(%arg0: i1) -> i32 {
  %result = scf.if %arg0 -> (i32) {
    %c10 = arith.constant 10 : i32
    scf.yield %c10 : i32
  } else {
    %c20 = arith.constant 20 : i32
    scf.yield %c20 : i32
  }
  func.return %result : i32
}

// After --convert-scf-to-cf lowering (automatic)
func.func @example(%arg0: i1) -> i32 {
  cf.cond_br %arg0, ^then, ^else
^then:
  %c10 = arith.constant 10 : i32
  cf.br ^merge(%c10 : i32)
^else:
  %c20 = arith.constant 20 : i32
  cf.br ^merge(%c20 : i32)
^merge(%result: i32):  // Block argument (replaces PHI node)
  func.return %result : i32
}
```

**Key insight:** Block argument `%result` in `^merge` block replaces PHI node. Predecessors pass values via `cf.br` operands.

### Example 4: Boehm GC Integration with MLIR-Compiled Code

```c
// Runtime support for MLIR-compiled FunLang program
#include <stdio.h>
#include <gc.h>  // Boehm GC header

// MLIR-compiled function (generated via LLVM backend)
extern int funlang_main();

int main(int argc, char** argv) {
    // Initialize Boehm GC before any allocations
    GC_INIT();

    // Call MLIR-compiled code
    int result = funlang_main();

    printf("Result: %d\n", result);
    return result;
}
```

**MLIR code generation for heap allocation:**
```mlir
// Allocate heap memory (will use GC_malloc at runtime)
%size = arith.constant 8 : i64
%ptr = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr

// Declare external GC_malloc function
llvm.func @GC_malloc(i64) -> !llvm.ptr attributes { sym_visibility = "private" }
```

**Linking:**
```bash
# Compile MLIR to object file
mlir-opt program.mlir --convert-to-llvm | mlir-translate --mlir-to-llvmir | \
  llc -filetype=obj -o program.o

# Link with Boehm GC and runtime
gcc runtime.c program.o -o program -lgc
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| **PHI nodes for control flow merges** | Block arguments (MLIR, Swift SIL) | MLIR initial design (2019) | Cleaner semantics, no "lost copy problem", unifies function args and basic block args |
| **std dialect for arithmetic** | `arith` dialect | MLIR refactoring (2021) | Standard dialect was too broad; `arith` is focused on arithmetic/comparison, clearer separation of concerns |
| **Manual GC with write barriers** | Conservative GC (Boehm) or precise GC (LLVM statepoints) | Boehm GC: 1990s (stable), LLVM statepoints: 2014+ | Boehm is drop-in with no compiler changes; statepoints require safepoint insertion but enable precise collection |
| **memref.alloc without deallocation** | Explicit `memref.dealloc` or buffer deallocation passes | MLIR design (bufferization) | One-Shot Bufferize doesn't auto-deallocate; must run `buffer-deallocation-pipeline` to insert deallocs and avoid leaks |

**Deprecated/outdated:**
- **std dialect (pre-2021):** Split into `arith`, `func`, `math`, `memref`, etc. Don't reference `std` in modern MLIR code.
- **LLVM PHI nodes as user-facing IR:** MLIR abstracts PHI nodes behind block arguments in high-level dialects. PHI nodes only appear after lowering to `llvm` dialect.
- **Manual phi insertion algorithms:** Not needed with MLIR's progressive lowering. `scf` → `cf` lowering automatically generates block arguments/PHI nodes.

## Open Questions

Things that couldn't be fully resolved:

1. **When to introduce memref operations in tutorial?**
   - What we know: Phase 2 AST (expressions, let bindings, if/else) can compile entirely to SSA registers without memref operations.
   - What's unclear: Should tutorial introduce `memref.alloca` / `memref.load` / `memref.store` in Phase 2 as pedagogical preparation for Phase 3 (closures), or defer until actually needed?
   - Recommendation: **Defer to Phase 3.** Phase 2 should focus on SSA form, block arguments, and control flow. Introduce memref only when closures require capturing environment on heap. Keep Phase 2 chapters focused on registers.

2. **F# P/Invoke bindings for scf dialect operations**
   - What we know: MLIR C API exposes `scf` dialect operations via `mlir-c/Dialect/SCF.h`. Functions exist for creating `scf.if`, `scf.for`, `scf.yield`.
   - What's unclear: Phase 1 P/Invoke bindings (Chapter 03) only covered core IR, arith dialect, and func dialect. Need to verify C API coverage for scf dialect.
   - Recommendation: **Add scf dialect bindings in Chapter 06-07.** Check `mlir-c/Dialect/SCF.h` for `mlirScfIfCreate`, `mlirScfYieldCreate` functions. If C API has gaps, wrap in C++ shim (Appendix pattern).

3. **Optimal lowering pass order for scf + arith + func**
   - What we know: Progressive lowering is `scf` → `cf` → `llvm`. Passes: `--convert-scf-to-cf`, `--convert-arith-to-llvm`, `--convert-func-to-llvm`, `--reconcile-unrealized-casts`.
   - What's unclear: Does pass order matter? Can we run `--convert-arith-to-llvm` before `--convert-scf-to-cf`, or must SCF lower first?
   - Recommendation: **Test and document in Chapter 08.** Empirically verify pass order. Likely safe to run conversions in parallel (scf→cf and arith→llvm independently), but reconcile-unrealized-casts must run last.

4. **Boehm GC thread safety for future parallelism**
   - What we know: Boehm GC supports multithreading with `GC_INIT()` in each thread or `GC_pthread_create()`. Phase 2 is single-threaded.
   - What's unclear: Should tutorial mention thread safety now (future-proofing) or defer to Phase 6 (concurrency)?
   - Recommendation: **Brief mention in Chapter 09, detailed coverage in Phase 6.** Note: "Boehm GC is thread-safe when properly initialized. We'll cover parallel GC in Phase 6 (Concurrency)."

## Sources

### Primary (HIGH confidence)
- [MLIR SCF Dialect - Official Docs](https://mlir.llvm.org/docs/Dialects/SCFDialect/) - scf.if, scf.yield, block arguments
- [MLIR Arith Dialect - Official Docs](https://mlir.llvm.org/docs/Dialects/ArithOps/) - arith.addi, arith.muli, arith.cmpi operations
- [MLIR Func Dialect - Official Docs](https://mlir.llvm.org/docs/Dialects/Func/) - func.func, func.call, IsolatedFromAbove trait
- [MLIR MemRef Dialect - Official Docs](https://mlir.llvm.org/docs/Dialects/MemRef/) - memref.alloca vs memref.alloc, deallocation
- [MLIR Rationale - Block Arguments](https://mlir.llvm.org/docs/Rationale/Rationale/) - Why block arguments vs PHI nodes
- [LLVM Garbage Collection - Official Docs](https://llvm.org/docs/GarbageCollection.html) - GC framework, statepoints, runtime integration
- [Boehm GC - Official Site](https://www.hboehm.info/gc/) - GC_malloc, GC_INIT, integration guide

### Secondary (MEDIUM confidence)
- [Boehm GC GitHub](https://github.com/bdwgc/bdwgc) - Build instructions (verified against official site)
- [Wikipedia - Boehm Garbage Collector](https://en.wikipedia.org/wiki/Boehm_garbage_collector) - Code examples (cross-checked with official docs)
- [Medium - SSA Form Explained](https://medium.com/@mlshark/an-introduction-to-static-single-assignment-ssa-form-in-compiler-design-77d33ee773de) - SSA concepts (verified against CMU lecture notes)
- [CMU Lecture Notes - SSA Form](https://www.cs.cmu.edu/~janh/courses/411/23/lec/12-ssa.pdf) - Let bindings in SSA (academic source)
- [GitHub - tattn/llvm-boehmgc-sample](https://github.com/tattn/llvm-boehmgc-sample) - LLVM IR + Boehm GC example (code verified)

### Tertiary (LOW confidence)
- WebSearch results for "MLIR compiler common mistakes" - Aggregated pitfalls (marked for user validation in tutorials)
- Hacker News discussions on block arguments - Community insights (verify against official docs)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are official MLIR/LLVM dialects with stable APIs in LLVM 19.x
- Architecture patterns: HIGH - Patterns verified against official MLIR documentation and source code
- Pitfalls: MEDIUM - Derived from MLIR documentation + common beginner mistakes (user testing will validate)
- Boehm GC integration: MEDIUM-HIGH - Official docs confirmed, but MLIR-specific integration requires empirical testing

**Research date:** 2026-02-06
**Valid until:** 2026-03-08 (30 days) - LLVM/MLIR is stable; Boehm GC is mature. Revalidate if LLVM 20.x is released.
