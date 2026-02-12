# Phase 8: Core Expressions - Research

**Researched:** 2026-02-12
**Domain:** MLIR arith/scf dialect integration for functional language expressions
**Confidence:** HIGH

## Summary

Phase 8 extends the existing MLIR code generator (Phase 7) to support the full spectrum of core expressions: comparisons, booleans, let bindings, and if-else control flow. This research identifies the specific MLIR operations, attribute patterns, and SSA handling strategies needed to compile these FunLang features.

The standard approach uses MLIR's arith dialect for arithmetic comparisons (arith.cmpi) and boolean operations (arith.andi, arith.ori), combined with the scf dialect for structured control flow (scf.if with scf.yield). Let bindings map naturally to SSA values through a symbol table that tracks variable-to-MlirValue mappings, with shadowing handled by simply rebinding names to new SSA values.

Phase 7 already provides the foundation: MlirWrapper.fs has OpBuilder, MlirBindings.fs has all necessary P/Invoke declarations, and CodeGen.fs demonstrates the pattern for arithmetic operations. This phase extends that pattern to comparisons, booleans, environments, and control flow.

**Primary recommendation:** Use arith.cmpi with integer predicate attributes for comparisons, arith.andi/ori for boolean logic (NOT short-circuit - use scf.if for that), and maintain a simple Map<string, MlirValue> environment for let binding SSA translation.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| MLIR arith dialect | LLVM 19+ | Arithmetic and comparison operations | Official LLVM dialect for integer/float math, produces i1 results for comparisons |
| MLIR scf dialect | LLVM 19+ | Structured control flow (if/for/while) | Standard approach for control flow that lowers cleanly to CF dialect and LLVM IR |
| MLIR C API | LLVM 19+ | P/Invoke bindings from F# | Only stable C API for MLIR, already integrated in Phase 7 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| F# Map | Built-in | Symbol table for variable bindings | Immutable, perfect for functional shadowing semantics |
| mlirIntegerAttrGet | MLIR C API | Create predicate attributes for cmpi | Required for all comparison operations |
| mlirUnitAttrGet | MLIR C API | Create unit attributes | Already used in Phase 7 for llvm.emit_c_interface |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scf.if for short-circuit && and \|\| | arith.andi/ori | scf.if enables true short-circuit (phase requirement unclear), arith ops are simpler |
| CF dialect (cf.br, cf.cond_br) for if-else | scf.if | scf.if is structured (easier to analyze/optimize), lowers to CF automatically |
| Custom FunLang dialect for comparisons | arith.cmpi | Custom dialect adds complexity, arith is battle-tested |

**Installation:**
Already completed in Phase 7. No new dependencies required.

## Architecture Patterns

### Recommended Project Structure
```
src/FunLang.Compiler/
├── MlirBindings.fs        # P/Invoke (Phase 7) - NO CHANGES NEEDED
├── MlirWrapper.fs         # OpBuilder helpers (Phase 7) - ADD: I1Type()
├── CodeGen.fs             # Expression compiler - EXTEND with:
│                          #   - CompileContext.Env: Map<string, MlirValue>
│                          #   - compileExpr cases for: Bool, Equal, LessThan, etc.
│                          #   - compileExpr cases for: And, Or
│                          #   - compileExpr cases for: Let, If
└── Ast.fs (LangTutorial)  # Already has all expression types
```

### Pattern 1: Comparison Operations (arith.cmpi)
**What:** Compile comparison operators to arith.cmpi with predicate attribute
**When to use:** For all FunLang comparison expressions (Equal, LessThan, etc.)
**Example:**
```fsharp
// Source: MLIR ArithOps.td + Phase 7 CodeGen.fs pattern
| Equal(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right

    // Create predicate attribute: 0 = eq
    let i64Type = builder.I64Type()
    let predicateAttr = builder.IntegerAttr(0L, i64Type)

    let i1Type = builder.I1Type()
    let op = emitOp ctx "arith.cmpi" [| i1Type |]
                [| leftVal; rightVal |]
                [| builder.NamedAttr("predicate", predicateAttr) |]
                [||]
    builder.GetResult(op, 0)  // Returns i1 value

// Predicate enum values:
// eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9
```

### Pattern 2: Boolean Operations (arith.andi/ori)
**What:** Compile && and || to arith bitwise operations on i1 type
**When to use:** For non-short-circuit boolean logic (simple && and ||)
**Example:**
```fsharp
// Source: MLIR ArithOps.td
| And(left, right, _) ->
    let leftVal = compileExpr ctx left   // i1 value
    let rightVal = compileExpr ctx right // i1 value
    let i1Type = builder.I1Type()
    let op = emitOp ctx "arith.andi" [| i1Type |]
                [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)

// NOTE: This is NOT short-circuit evaluation.
// For short-circuit: use scf.if pattern (see below).
```

### Pattern 3: Boolean Literals
**What:** Compile true/false to arith.constant with i1 type
**When to use:** For Bool(b, _) expression
**Example:**
```fsharp
// Source: Phase 7 Number pattern + MLIR BuiltinAttributes.h
| Bool(b, _) ->
    let i1Type = builder.I1Type()
    let value = if b then 1L else 0L
    let valueAttr = builder.IntegerAttr(value, i1Type)
    let op = emitOp ctx "arith.constant" [| i1Type |] [||]
                [| builder.NamedAttr("value", valueAttr) |] [||]
    builder.GetResult(op, 0)
```

### Pattern 4: Let Bindings with SSA
**What:** Map let bindings to SSA values via environment (symbol table)
**When to use:** For Let(name, expr1, expr2, _) expression
**Example:**
```fsharp
// Source: "SSA is Functional Programming" + Phase 7 context pattern
type CompileContext = {
    Context: Context
    Builder: OpBuilder
    Location: Location
    Block: MlirBlock
    Env: Map<string, MlirValue>  // NEW: variable -> SSA value mapping
}

| Let(name, expr1, expr2, _) ->
    // Compile binding expression
    let value = compileExpr ctx expr1

    // Shadow: create new environment with binding
    let ctx' = { ctx with Env = ctx.Env.Add(name, value) }

    // Compile body in extended environment
    compileExpr ctx' expr2

| Var(name, _) ->
    // Lookup in environment
    match ctx.Env.TryFind(name) with
    | Some value -> value
    | None -> failwithf "Unbound variable: %s" name
```

### Pattern 5: If-Else Expressions (scf.if)
**What:** Compile if-then-else to scf.if with scf.yield
**When to use:** For If(cond, thenExpr, elseExpr, _) expression
**Example:**
```fsharp
// Source: MLIR SCFOps.td
| If(cond, thenExpr, elseExpr, _) ->
    let condVal = compileExpr ctx cond  // i1 value

    // Determine result type from then branch (assume well-typed)
    // For Phase 8: assume i32 for simplicity, or detect from AST
    let resultType = builder.I32Type()

    // Create then region
    let thenRegion = builder.CreateRegion()
    let thenBlock = builder.CreateBlock([||], ctx.Location)
    builder.AppendBlockToRegion(thenRegion, thenBlock)

    let thenCtx = { ctx with Block = thenBlock }
    let thenVal = compileExpr thenCtx thenExpr

    // scf.yield %thenVal : i32
    let thenYieldOp = builder.CreateOperation(
        "scf.yield", ctx.Location,
        [||], [| thenVal |], [||], [||])
    builder.AppendOperationToBlock(thenBlock, thenYieldOp)

    // Create else region (similar pattern)
    let elseRegion = builder.CreateRegion()
    let elseBlock = builder.CreateBlock([||], ctx.Location)
    builder.AppendBlockToRegion(elseRegion, elseBlock)

    let elseCtx = { ctx with Block = elseBlock }
    let elseVal = compileExpr elseCtx elseExpr

    let elseYieldOp = builder.CreateOperation(
        "scf.yield", ctx.Location,
        [||], [| elseVal |], [||], [||])
    builder.AppendOperationToBlock(elseBlock, elseYieldOp)

    // Create scf.if operation
    let ifOp = builder.CreateOperation(
        "scf.if", ctx.Location,
        [| resultType |],           // result types
        [| condVal |],              // operands (condition)
        [||],                       // attributes
        [| thenRegion; elseRegion |])  // regions
    builder.AppendOperationToBlock(ctx.Block, ifOp)

    builder.GetResult(ifOp, 0)
```

### Pattern 6: Short-Circuit Boolean Evaluation (Optional)
**What:** Implement && and || with true short-circuit semantics using scf.if
**When to use:** If requirement IMPL-LANG-03 demands short-circuit evaluation
**Example:**
```fsharp
// Source: General compiler design + MLIR scf.if
| And(left, right, _) ->
    // if left then right else false
    let leftVal = compileExpr ctx left

    // Then region: evaluate right
    let thenRegion = builder.CreateRegion()
    let thenBlock = builder.CreateBlock([||], ctx.Location)
    builder.AppendBlockToRegion(thenRegion, thenBlock)
    let thenCtx = { ctx with Block = thenBlock }
    let rightVal = compileExpr thenCtx right
    let thenYield = builder.CreateOperation("scf.yield", ctx.Location,
        [||], [| rightVal |], [||], [||])
    builder.AppendOperationToBlock(thenBlock, thenYield)

    // Else region: return false
    let elseRegion = builder.CreateRegion()
    let elseBlock = builder.CreateBlock([||], ctx.Location)
    builder.AppendBlockToRegion(elseRegion, elseBlock)
    let i1Type = builder.I1Type()
    let falseAttr = builder.IntegerAttr(0L, i1Type)
    let falseOp = builder.CreateOperation("arith.constant", ctx.Location,
        [| i1Type |], [||], [| builder.NamedAttr("value", falseAttr) |], [||])
    builder.AppendOperationToBlock(elseBlock, falseOp)
    let falseVal = builder.GetResult(falseOp, 0)
    let elseYield = builder.CreateOperation("scf.yield", ctx.Location,
        [||], [| falseVal |], [||], [||])
    builder.AppendOperationToBlock(elseBlock, elseYield)

    let ifOp = builder.CreateOperation("scf.if", ctx.Location,
        [| i1Type |], [| leftVal |], [||], [| thenRegion; elseRegion |])
    builder.AppendOperationToBlock(ctx.Block, ifOp)
    builder.GetResult(ifOp, 0)

// Similar pattern for Or: if left then true else right
```

### Anti-Patterns to Avoid
- **Creating phi nodes manually:** MLIR uses block arguments and SSA values, not explicit phi nodes
- **Mutating environment Map:** Always use `Map.Add` to create new environments, preserving immutability
- **Mixing i32 and i1 types:** Comparisons return i1, arithmetic returns i32. Don't pass i1 to arith.addi.
- **Forgetting scf.yield:** Both branches of scf.if MUST end with scf.yield when returning values
- **Type mismatches in scf.if:** Both branches must yield same type as declared in scf.if result types

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Integer comparison predicates | String-based predicate names | Integer attributes (0=eq, 2=slt, etc.) | ArithOps.td defines enum as integers, C API expects i64 attribute |
| Short-circuit evaluation | Custom control flow logic | scf.if regions (Pattern 6) | Structured control flow is analyzable, optimizable, and lowers correctly |
| Variable shadowing in SSA | Renaming variables (x, x_1, x_2) | Immutable Map rebinding | F# Map.Add creates new binding, old binding unreachable - natural shadowing |
| Type inference for if-else | Runtime type checking | Type system / AST annotation | MLIR requires result types at operation creation - decide during compilation |
| Boolean to integer conversion | Custom conversion operations | Already handled by arith dialect | arith.cmpi returns i1, arith.andi works on i1 - no conversion needed |

**Key insight:** MLIR's type system is strict and explicit. You must know result types before creating operations. Don't try to defer type resolution to runtime - FunLang's AST should guide type decisions.

## Common Pitfalls

### Pitfall 1: Predicate Attribute Type Mismatch
**What goes wrong:** Creating arith.cmpi predicate with wrong type (e.g., i32 instead of i64)
**Why it happens:** Predicate is an integer attribute, but must be i64 specifically
**How to avoid:** Always use `builder.I64Type()` for predicate attributes, values 0-9
**Warning signs:** MLIR verifier error: "predicate attribute has incorrect type"

### Pitfall 2: Missing scf.yield Terminators
**What goes wrong:** scf.if regions compile but MLIR verification fails
**Why it happens:** Every region in scf.if must end with scf.yield, even if implicit in FunLang
**How to avoid:** Always create explicit scf.yield as last operation in then/else blocks
**Warning signs:** MLIR error: "block must end with terminator", segfault during execution

### Pitfall 3: Type Mismatches in scf.if Branches
**What goes wrong:** Then branch yields i32, else branch yields i1
**Why it happens:** Different code paths return different types (e.g., true vs. 42)
**How to avoid:** FunLang should be well-typed; compiler should validate or infer common type
**Warning signs:** MLIR verifier error: "scf.yield type mismatch with scf.if result type"

### Pitfall 4: Environment Not Threaded Through Recursion
**What goes wrong:** Inner let bindings don't see outer variables
**Why it happens:** Forgot to pass updated `ctx.Env` to recursive `compileExpr` calls
**How to avoid:** Every recursive call to `compileExpr` must receive current context with environment
**Warning signs:** "Unbound variable" error for variables that should be in scope

### Pitfall 5: I1 vs I32 Type Confusion
**What goes wrong:** Passing boolean (i1) result to arithmetic operation expecting i32
**Why it happens:** arith.cmpi returns i1, but arith.addi expects i32
**How to avoid:** Track types explicitly; booleans are i1, numbers are i32, never mix
**Warning signs:** MLIR verifier error: "operand type mismatch"

### Pitfall 6: Block Context Lost in scf.if
**What goes wrong:** Operations inside then/else blocks append to wrong block
**Why it happens:** Forgot to update `ctx.Block` when compiling inside new regions
**How to avoid:** Create new context with updated Block for each region: `{ ctx with Block = thenBlock }`
**Warning signs:** Operations appear outside scf.if in printed IR, or MLIR crashes

## Code Examples

Verified patterns from official sources:

### Example 1: Comparison Operation (from ArithOps.td)
```mlir
// Custom form: signed less than
%x = arith.cmpi slt, %lhs, %rhs : i32

// Generic form (what we build from F#):
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// Predicate values:
// eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9
```

### Example 2: Boolean Constant and AND (from ArithOps.td)
```mlir
%true = arith.constant 1 : i1
%false = arith.constant 0 : i1
%result = arith.andi %true, %false : i1
```

### Example 3: If-Then-Else with scf.if (from SCFOps.td)
```mlir
%x, %y = scf.if %condition -> (f32, f32) {
  %x_true = arith.constant 1.0 : f32
  %y_true = arith.constant 2.0 : f32
  scf.yield %x_true, %y_true : f32, f32
} else {
  %x_false = arith.constant 3.0 : f32
  %y_false = arith.constant 4.0 : f32
  scf.yield %x_false, %y_false : f32, f32
}

// Single result variant:
%result = scf.if %cond -> i32 {
  %val = arith.constant 42 : i32
  scf.yield %val : i32
} else {
  %val = arith.constant 0 : i32
  scf.yield %val : i32
}
```

### Example 4: Let Binding Compilation (conceptual)
```fsharp
// FunLang: let x = 5 in x + 10
// Compiles to:
%c5 = arith.constant 5 : i32     // compile expr1
// [Env = {x -> %c5}]
%c10 = arith.constant 10 : i32   // in expr2, compile 10
%result = arith.addi %c5, %c10 : i32  // compile x + 10, lookup x in Env
```

### Example 5: Nested Let Shadowing (from F# Map semantics)
```fsharp
// FunLang: let x = 1 in let x = 2 in x
// Env evolution:
// {}
// -> {x -> %c1}  (after first let)
// -> {x -> %c2}  (after second let, shadows %c1)
// Lookup x returns %c2

// Map.Add creates new binding, old binding unreachable
let env1 = Map.empty.Add("x", val1)  // {x -> val1}
let env2 = env1.Add("x", val2)       // {x -> val2}, val1 unreachable
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CF dialect (br, cond_br) for all control flow | scf dialect for structured control flow | MLIR introduction (~2019) | scf.if is easier to analyze, preserves structure through lowering |
| Explicit phi nodes for value merging | Block arguments in MLIR | MLIR design | Cleaner SSA representation, no manual phi construction |
| String-based operation names only | Generic operation creation via C API | MLIR C API stabilization | Enables language bindings (F#, Rust, Python) |

**Deprecated/outdated:**
- **std dialect:** Split into arith, math, func dialects (MLIR ~2021). Use arith for comparisons.
- **Manual phi node insertion:** MLIR uses block arguments. Don't look for mlirPhiCreate - it doesn't exist.

## Open Questions

Things that couldn't be fully resolved:

1. **Short-circuit evaluation requirement**
   - What we know: IMPL-LANG-03 says "Boolean literals and logical operators (&&, ||)"
   - What's unclear: Are short-circuit semantics required, or is simple arith.andi/ori sufficient?
   - Recommendation: Start with arith.andi/ori (simpler). If tests require short-circuit, use Pattern 6 (scf.if-based).

2. **Type inference for if-else result**
   - What we know: scf.if requires explicit result type at creation
   - What's unclear: Does FunLang AST carry type annotations, or must we infer?
   - Recommendation: For Phase 8, assume well-typed AST and use i32 as default numeric type. Detect i1 if both branches are boolean expressions.

3. **Mixed-type if-else branches**
   - What we know: MLIR requires same type from both branches
   - What's unclear: Does FunLang type system prevent this, or must compiler error?
   - Recommendation: Add runtime check in compiler: if types differ, fail with clear error message.

## Sources

### Primary (HIGH confidence)
- [MLIR ArithOps.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td) - Comparison predicates, boolean ops, integer attributes
- [MLIR SCFOps.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td) - scf.if structure, scf.yield semantics
- [MLIR BuiltinAttributes.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir-c/BuiltinAttributes.h) - mlirIntegerAttrGet, mlirBoolAttrGet
- Phase 7 implementation - MlirBindings.fs, MlirWrapper.fs, CodeGen.fs (working patterns)

### Secondary (MEDIUM confidence)
- [MLIR Architecture: Dialects and Operations](https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-2-advanced-ml-intermediate-representations/mlir-dialects-operations) - General MLIR concepts verified against official docs
- [MLIR Part 1 - Introduction to MLIR - Stephen Diehl](https://www.stephendiehl.com/posts/mlir_introduction/) - SSA and block arguments
- [MLIR — Dialect Conversion](https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion/) - scf.if type handling

### Tertiary (LOW confidence)
- [Lambda the Ultimate SSA: Optimizing Functional Programs in SSA](https://arxiv.org/pdf/2201.07272) - Theoretical correspondence between SSA and functional languages, not MLIR-specific
- [SSA is Functional Programming](https://www.cs.princeton.edu/~appel/papers/ssafun.pdf) - Academic foundation, not implementation guide

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Phase 7 already uses arith dialect successfully, scf dialect is documented standard
- Architecture: HIGH - Patterns directly from official TableGen definitions and working Phase 7 code
- Pitfalls: MEDIUM - Based on general MLIR knowledge and typical SSA mistakes, not exhaustive error testing

**Research date:** 2026-02-12
**Valid until:** 2026-03-14 (30 days - MLIR C API is stable, dialects rarely change)
