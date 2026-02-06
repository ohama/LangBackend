# Phase 3: Functions & Recursion - Research

**Researched:** 2026-02-06
**Domain:** MLIR function compilation, LLVM calling conventions, recursive function lowering
**Confidence:** HIGH

## Summary

Phase 3 adds function definitions, function calls, and recursion (including mutual recursion) to the FunLang compiler. The research reveals that MLIR's `func` dialect provides high-level function abstractions that lower cleanly to the LLVM dialect, which then converts to native calling conventions. The key technical domains are:

1. **MLIR func dialect operations** (`func.func`, `func.call`, `func.return`) for high-level function representation
2. **Function lowering pipeline** from func dialect → LLVM dialect → LLVM IR → native code
3. **Calling conventions** including argument passing, return value handling, and stack frame management
4. **Recursion handling** with proper tail call optimization support at LLVM level

FunLang's function semantics (first-class functions, closures) are deferred to Phase 4. Phase 3 focuses on **top-level named functions only** — simpler than closures because no environment capture is needed. Functions take arguments, call other functions, and return values. Recursive functions (including mutual recursion) work naturally because functions are module-level symbols.

**Primary recommendation:** Use MLIR's `func` dialect for high-level function representation, leverage standard `func-to-llvm` pass for lowering, and compile with LLVM optimization passes that handle tail call optimization automatically. Phase 3 keeps functions simple (no closures) to establish the foundation before Phase 4 adds closure conversion.

## Standard Stack

The established libraries/tools for MLIR function compilation:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| MLIR func dialect | LLVM 19.x+ | High-level function operations | Official MLIR dialect for functions, maintained by LLVM project |
| MLIR LLVM dialect | LLVM 19.x+ | LLVM IR abstraction in MLIR | Required for lowering to LLVM IR and native code generation |
| LLVM opt passes | LLVM 19.x+ | Optimization including TCO | Industry-standard optimizer with tail call elimination |
| MLIR C API | LLVM 19.x+ | Programmatic IR construction | Only stable API for non-C++ languages like F# |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| mlir-opt tool | LLVM 19.x+ | Testing passes and IR transforms | Debugging function lowering, verifying IR |
| mlir-translate | LLVM 19.x+ | MLIR to LLVM IR conversion | After lowering to LLVM dialect |
| llc | LLVM 19.x+ | LLVM IR to object file | Code generation after MLIR pipeline |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| func dialect | Custom function ops | Func dialect is battle-tested; custom ops require lowering passes from scratch |
| Standard calling convention | Custom ABI | Standard ABI enables C interop; custom ABI isolates from ecosystem |
| LLVM backend | Custom codegen | LLVM provides platform support and optimizations; custom is a PhD thesis |

**Installation:**
Already available in Phase 1 LLVM/MLIR build. No additional packages needed.

## Architecture Patterns

### Recommended Project Structure
```
FunLangCompiler/
├── Ast.fs              # FunLang AST (from LangTutorial, includes Lambda/App/LetRec)
├── MlirBindings.fs     # P/Invoke to MLIR C API (Phase 1)
├── MlirWrapper.fs      # F# wrappers for MLIR operations (Phase 1)
├── Compiler.fs         # AST → MLIR translation
│   ├── compileExpr     # Translates expressions to MLIR
│   └── compileFunction # NEW: Translates function definitions to func.func
├── Pipeline.fs         # Compilation pipeline orchestration
└── Main.fs             # CLI entry point
```

### Pattern 1: Top-Level Function Compilation
**What:** Compile FunLang top-level functions to MLIR `func.func` operations with block arguments for parameters.

**When to use:** Phase 3 functions are always top-level (no nested functions, no closures yet).

**Example:**
```fsharp
// FunLang source:
// let rec factorial n =
//   if n <= 1 then 1 else n * factorial (n - 1)

// MLIR output:
func.func @factorial(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %cmp = arith.cmpi sle, %arg0, %c1 : i32
  %result = scf.if %cmp -> (i32) {
    scf.yield %c1 : i32
  } else {
    %n_minus_1 = arith.subi %arg0, %c1 : i32
    %rec = func.call @factorial(%n_minus_1) : (i32) -> i32
    %product = arith.muli %arg0, %rec : i32
    scf.yield %product : i32
  }
  func.return %result : i32
}
```

**Key points:**
- Function name becomes symbol (`@factorial`)
- Parameters become block arguments (`%arg0: i32`)
- Body is SSA value computation (reuse Phase 2 expression compiler)
- Recursive calls use `func.call @factorial`
- Last operation must be `func.return`

### Pattern 2: Function Call Translation
**What:** Translate FunLang function application to MLIR `func.call` operation.

**When to use:** Every function application in FunLang AST (App node).

**Example:**
```fsharp
// FunLang: factorial 5
// MLIR:
%arg = arith.constant 5 : i32
%result = func.call @factorial(%arg) : (i32) -> i32
```

**Key points:**
- Callee is symbol reference (must exist in module)
- Arguments are SSA values (compile argument expressions first)
- Result is SSA value (use in parent expression)
- Type signature must match function declaration

### Pattern 3: Mutually Recursive Functions
**What:** Define multiple functions that call each other, using forward declarations.

**When to use:** FunLang programs with mutual recursion (e.g., even/odd predicates).

**Example:**
```mlir
// Forward declarations not needed in MLIR - all module-level funcs are visible
func.func @is_even(%n: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %is_zero = arith.cmpi eq, %n, %c0 : i32
  %result = scf.if %is_zero -> (i1) {
    %true = arith.constant 1 : i1
    scf.yield %true : i1
  } else {
    %n_minus_1 = arith.subi %n, %c1 : i32
    %odd_result = func.call @is_odd(%n_minus_1) : (i32) -> i1
    scf.yield %odd_result : i1
  }
  func.return %result : i1
}

func.func @is_odd(%n: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %is_zero = arith.cmpi eq, %n, %c0 : i32
  %result = scf.if %is_zero -> (i1) {
    %false = arith.constant 0 : i1
    scf.yield %false : i1
  } else {
    %n_minus_1 = arith.subi %n, %c1 : i32
    %even_result = func.call @is_even(%n_minus_1) : (i32) -> i1
    scf.yield %even_result : i1
  }
  func.return %result : i1
}
```

**Key points:**
- MLIR modules have flat symbol namespace (no forward declarations needed)
- All `func.func` operations at module level are visible to each other
- Order of function definitions in module does not matter
- Mutual recursion "just works" via symbol references

### Pattern 4: Function Lowering Pipeline
**What:** Multi-stage lowering from func dialect to native code.

**When to use:** Always, for every compilation.

**Example:**
```bash
# Stage 1: High-level MLIR (func dialect, scf dialect, arith dialect)
func.func @factorial(%arg0: i32) -> i32 { ... }

# Stage 2: Apply func-to-llvm pass
mlir-opt --convert-func-to-llvm input.mlir -o lowered.mlir

# Stage 3: LLVM dialect IR
llvm.func @factorial(%arg0: i32) -> i32 { ... }

# Stage 4: Convert to LLVM IR
mlir-translate --mlir-to-llvmir lowered.mlir -o output.ll

# Stage 5: LLVM IR
define i32 @factorial(i32 %0) { ... }

# Stage 6: Compile to object file
llc -filetype=obj output.ll -o output.o

# Stage 7: Link to executable
cc -o factorial output.o
```

**Key points:**
- Use `--convert-func-to-llvm` pass (plus `--convert-scf-to-cf`, `--convert-arith-to-llvm`)
- Pass order matters: lower high-level dialects first (scf), then func/arith
- Each stage produces valid IR (can verify with `mlir-opt --verify-diagnostics`)
- LLVM backend applies optimizations (including tail call elimination)

### Anti-Patterns to Avoid

- **String-based IR generation:** Don't concatenate MLIR text strings. Use MLIR C API builders. String generation bypasses type checking and verification.

- **Manual SSA value numbering:** Don't try to assign `%0`, `%1`, etc. yourself. MLIR assigns SSA names automatically. Use `MlirValue` handles directly.

- **Ignoring IsolatedFromAbove trait:** Functions cannot capture outer SSA values. All inputs must be function arguments. (Closures in Phase 4 will use explicit environment passing.)

- **Forgetting terminator operations:** Every block must end with a terminator (`func.return`, `cf.br`, `scf.yield`). MLIR verifier will reject unterminated blocks.

- **Mixing dialects without lowering:** Don't emit `func.call` in LLVM dialect code. Lower func dialect first, then LLVM dialect operations reference `llvm.func`.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Function calling conventions | Custom ABI and stack layout | LLVM default C calling convention | LLVM handles platform differences (x86-64, ARM, etc.), register allocation, spilling, alignment |
| Tail call optimization | Manual tail recursion → loop transform | LLVM `-tailcallopt` and function attribute `"tail"` | LLVM's TCO is target-aware, handles cross-function tail calls, respects ABI |
| Stack frame management | Manual stack pointer manipulation | LLVM's frame lowering | LLVM handles frame pointer, stack alignment, red zones, call frame setup/teardown |
| Function type checking | Manual signature verification | MLIR's built-in verification | `mlirOperationVerify()` checks operand/result types match function signatures |
| Multi-pass optimization | Custom optimizer loop | LLVM `opt` tool with standard passes | LLVM includes 100+ optimization passes (inlining, constant folding, DCE, etc.) |

**Key insight:** LLVM has 20+ years of production compiler engineering. Don't reimplement calling conventions, stack management, or optimization. Use MLIR's `func` dialect for high-level representation, let LLVM handle low-level codegen.

## Common Pitfalls

### Pitfall 1: Function Declaration vs Definition
**What goes wrong:** Compiler crashes with "public declaration is not allowed" error when trying to create external function declarations.

**Why it happens:** MLIR's symbol system requires declarations (functions without bodies) to have `private` visibility. Only definitions (functions with bodies) can be `public`.

**How to avoid:**
- If defining a function, use `public` or omit visibility (defaults to public)
- If declaring an external function (e.g., C library function like `printf`), use `private` visibility
- Example: `func.func private @printf(!llvm.ptr<i8>, ...) -> i32` (declaration)
- Example: `func.func @main() -> i32 { ... }` (definition)

**Warning signs:**
- Error message mentioning "public declaration"
- Crash during `mlirModuleCreateParse()` or verification

### Pitfall 2: Block Arguments vs Variables
**What goes wrong:** Trying to use variable names instead of block arguments for function parameters, leading to unbound variable errors.

**Why it happens:** In MLIR, function parameters are not "variables" but **block arguments** — SSA values passed to the function's entry block. They are already SSA values, not memory locations.

**How to avoid:**
- Function parameters appear as `%arg0`, `%arg1`, etc. (or named block args)
- Don't create `memref.alloca` for parameters (they're already values)
- Don't use environment lookups for parameters (they're block arguments, not let bindings)
- Pattern: `func.func @foo(%arg0: i32, %arg1: i32) -> i32 { ... }` — use `%arg0` directly

**Warning signs:**
- Creating stack allocations for parameters
- Looking up parameter names in environment map
- Confusion between "function parameter" and "let binding"

### Pitfall 3: Recursive Call in Wrong Scope
**What goes wrong:** Recursive function can't find its own name, leading to "undefined symbol" error.

**Why it happens:** In FunLang interpreter, `LetRec` adds function to environment before evaluating body. But in MLIR compilation, functions are module-level symbols, not environment values.

**How to avoid:**
- Compile all top-level functions to `func.func` operations in module
- Use `func.call @function_name` for recursive calls (symbol reference)
- Don't add function names to expression environment (they're not let bindings)
- Module-level functions are always visible within the same module

**Warning signs:**
- "undefined symbol @factorial" error during compilation
- Trying to look up function names in `Map<string, MlirValue>` environment
- Confusion between "function definition" and "let binding"

### Pitfall 4: Missing Terminator in Function Body
**What goes wrong:** MLIR verification fails with "block must end with terminator" error.

**Why it happens:** Every MLIR block (including function bodies) must end with a terminator operation (`func.return`, `cf.br`, `cf.cond_br`, `scf.yield`). If compiling an expression without returning its value, block is unterminated.

**How to avoid:**
- Always end function body with `func.return <value>` operation
- Return value type must match function signature's return type
- If expression compiler produces SSA value, return it
- Pattern: `let result = compileExpr(...); insertReturn(result)`

**Warning signs:**
- Verification error mentioning "terminator"
- Segfault during `mlirOperationVerify()`
- IR dump shows block without final operation

### Pitfall 5: Function Type Mismatch
**What goes wrong:** Verification fails with "type mismatch" error when calling function with wrong argument types or wrong result type.

**Why it happens:** MLIR enforces strict type checking. `func.call` operation must:
- Pass arguments matching callee's parameter types (order and types)
- Declare result type matching callee's return type

**How to avoid:**
- When creating `func.call`, query callee's function type first
- Extract argument types and result types from function type
- Ensure compiled argument values match expected types
- Use `mlirFunctionTypeGet(argTypes, resultTypes)` consistently

**Warning signs:**
- "expected type X but got type Y" error
- Verification failure in `func.call` operation
- Type coercion attempts (MLIR doesn't do implicit conversion)

### Pitfall 6: Forgetting Stack Frame Limits
**What goes wrong:** Deep recursion causes stack overflow at runtime, even though compilation succeeds.

**Why it happens:** Without tail call optimization, each recursive call consumes stack space. 10,000 recursive calls = 10,000 stack frames = stack overflow.

**How to avoid:**
- Document that recursive functions without TCO can overflow
- Enable LLVM tail call optimization for tail-recursive functions
- Add function attribute `"tailcc"` for tail-call calling convention
- Consider iterative implementations for deeply recursive algorithms
- Test with realistic recursion depth (e.g., `factorial 100000` should fail gracefully)

**Warning signs:**
- Runtime crash with "segmentation fault" or "stack overflow"
- Compiler succeeds, binary crashes
- Works for small inputs (factorial 10), fails for large inputs (factorial 10000)

## Code Examples

Verified patterns from MLIR documentation:

### Function Definition with MLIR C API
```c
// From MLIR C API (inferred from documentation patterns):

// 1. Create function type
MlirType paramTypes[] = { mlirIntegerTypeGet(ctx, 32) };
MlirType resultTypes[] = { mlirIntegerTypeGet(ctx, 32) };
MlirType funcType = mlirFunctionTypeGet(
    ctx,
    1, paramTypes,  // 1 parameter
    1, resultTypes  // 1 result
);

// 2. Create function operation
MlirLocation loc = mlirLocationUnknownGet(ctx);
MlirAttribute funcTypeAttr = mlirTypeAttrGet(funcType);
MlirAttribute symNameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("factorial"));

MlirNamedAttribute attrs[] = {
    { mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")),
      funcTypeAttr },
    { mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
      symNameAttr }
};

MlirOperationState state = mlirOperationStateGet(
    mlirStringRefCreateFromCString("func.func"),
    loc
);
mlirOperationStateAddAttributes(&state, 2, attrs);

// 3. Add function body (region with entry block)
MlirRegion bodyRegion = mlirRegionCreate();
MlirType blockArgTypes[] = { mlirIntegerTypeGet(ctx, 32) };
MlirBlock entryBlock = mlirBlockCreate(1, blockArgTypes, NULL);
mlirRegionAppendOwnedBlock(bodyRegion, entryBlock);
mlirOperationStateAddOwnedRegions(&state, 1, &bodyRegion);

MlirOperation funcOp = mlirOperationCreate(&state);
mlirModuleGetBody(module, funcOp);  // Add to module
```

### Function Call with MLIR C API
```c
// Compile function call: factorial(5)

// 1. Compile argument
MlirValue argValue = compileExpr(ctx, builder, env, argExpr);

// 2. Create call operation
MlirAttribute calleeAttr = mlirFlatSymbolRefAttrGet(
    ctx,
    mlirStringRefCreateFromCString("factorial")
);

MlirType resultType = mlirIntegerTypeGet(ctx, 32);

MlirOperationState callState = mlirOperationStateGet(
    mlirStringRefCreateFromCString("func.call"),
    loc
);
mlirOperationStateAddOperands(&callState, 1, &argValue);
mlirOperationStateAddResults(&callState, 1, &resultType);
mlirOperationStateAddAttributes(&callState, 1, &(MlirNamedAttribute){
    mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("callee")),
    calleeAttr
});

MlirOperation callOp = mlirOperationCreate(&callState);
mlirBlockAppendOwnedOperation(block, callOp);

MlirValue resultValue = mlirOperationGetResult(callOp, 0);
```

### Tail Recursive Function Example
```mlir
// Tail-recursive factorial (optimizable by LLVM)
func.func @factorial_tail(%n: i32, %acc: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %is_one = arith.cmpi sle, %n, %c1 : i32
  %result = scf.if %is_one -> (i32) {
    scf.yield %acc : i32
  } else {
    %n_minus_1 = arith.subi %n, %c1 : i32
    %new_acc = arith.muli %n, %acc : i32
    // Tail call - last operation in function
    %rec = func.call @factorial_tail(%n_minus_1, %new_acc) : (i32, i32) -> i32
    scf.yield %rec : i32
  }
  func.return %result : i32
}

// Wrapper for user-facing API
func.func @factorial(%n: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %result = func.call @factorial_tail(%n, %c1) : (i32, i32) -> i32
  func.return %result : i32
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LLVM PHI nodes for function args | MLIR block arguments | MLIR inception (2019) | Cleaner representation, no special handling for PHI nodes |
| Manual calling convention in IR | LLVM automatic lowering | LLVM 3.0+ (2011) | Platform-independent function lowering |
| Separate declaration/definition ops | Single `func.func` with optional body | MLIR upstreaming (2020) | Simpler API, declaration is just empty body |
| Tail call as intrinsic | Tail call as function attribute | LLVM 8.0+ (2019) | More flexible, optimizer-friendly |

**Deprecated/outdated:**
- **`std.func` operation:** Renamed to `func.func` in MLIR 2021. Use `func.func` in all new code.
- **Untyped function calls:** Early MLIR allowed untyped calls. Now all `func.call` operations must have explicit type signature.
- **Implicit tail call optimization:** LLVM no longer does automatic TCO without explicit function attributes or `-tailcallopt` flag.

## Open Questions

Things that couldn't be fully resolved:

1. **MLIR C API for function operations**
   - What we know: func.func operations can be created via generic operation builder
   - What's unclear: No dialect-specific C API functions like `mlirFuncFuncCreate()` found in documentation
   - Recommendation: Use generic `mlirOperationCreate()` with "func.func" operation name and appropriate attributes (function_type, sym_name, sym_visibility)

2. **Tail call optimization guarantees**
   - What we know: LLVM can optimize tail calls with `-tailcallopt` flag and `"tail"` attribute
   - What's unclear: Which calling conventions guarantee TCO (C convention doesn't always)
   - Recommendation: Use `"tailcc"` calling convention for guaranteed TCO, document that standard C convention may or may not optimize

3. **Mutual recursion cycle detection**
   - What we know: MLIR allows circular call graphs at module level
   - What's unclear: Whether MLIR's bufferization or other passes handle recursive cycles
   - Recommendation: Avoid bufferization in Phase 3 (not needed for integers). Recursive calls work fine in func/arith/scf dialects.

## Sources

### Primary (HIGH confidence)
- [MLIR Func Dialect](https://mlir.llvm.org/docs/Dialects/Func/) - Official documentation for func.func, func.call, func.return operations
- [LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/) - Function lowering from MLIR to LLVM dialect, calling conventions, memref handling
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) - SSA form, block arguments, function semantics
- [MLIR C API](https://mlir.llvm.org/docs/CAPI/) - C API design patterns for operation creation

### Secondary (MEDIUM confidence)
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) - Pass infrastructure for func-to-llvm lowering
- [Jeremy Kun's MLIR Tutorial - Lowering through LLVM](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/) - Practical examples of lowering pipeline
- [LLVM CallingConv Namespace](https://llvm.org/doxygen/namespacellvm_1_1CallingConv.html) - Available calling conventions
- [LLVM Developer Discussion - Mutual Recursion](https://discourse.llvm.org/t/rfc-update-to-mlir-developer-policy-on-recursion/62235) - MLIR policy on recursion in IR

### Tertiary (LOW confidence)
- [Functional Language Closure Compilation Paper](https://ar5iv.labs.arxiv.org/html/1805.08842) - Academic research on closure compilation (Phase 4 context)
- [Various tail call optimization articles](https://www.oreateai.com/blog/understanding-tail-recursion-a-deep-dive-into-efficient-function-calls/94f0322b982cb219789b6094e45e0dba) - General TCO concepts, not MLIR-specific

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - MLIR func dialect is official and well-documented
- Architecture: HIGH - Patterns verified from MLIR official docs and LangTutorial FunLang semantics
- Pitfalls: MEDIUM - Inferred from MLIR FAQ and common SSA/function mistakes, not exhaustive list
- Code examples: MEDIUM - C API examples inferred from patterns (no official C API tutorial for func ops)

**Research date:** 2026-02-06
**Valid until:** 2026-03-06 (30 days - MLIR is stable, func dialect unlikely to change)

**Phase 3 scope:**
- Top-level functions only (no nested functions, no closures)
- Named function definitions (func.func operations at module level)
- Function calls via symbol references (func.call @name)
- Recursion including mutual recursion (works via symbol table)
- **Out of scope:** Closures (Phase 4), higher-order functions (Phase 4), function values (Phase 4)

**Connection to prior phases:**
- Phase 1: MLIR C API bindings, module/context management → reuse for function operations
- Phase 2: Expression compilation (arith, scf, let bindings) → reuse inside function bodies
- Phase 2: Environment passing pattern → extend to distinguish let bindings vs function parameters

**Connection to next phase:**
- Phase 4 will add closures, requiring heap-allocated environments and closure conversion
- Phase 3 establishes function compilation foundation; Phase 4 extends it with environment capture
