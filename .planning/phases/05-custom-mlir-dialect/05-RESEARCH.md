# Phase 5: Custom MLIR Dialect - Research

**Researched:** 2026-02-06
**Domain:** MLIR custom dialect definition and lowering infrastructure
**Confidence:** HIGH

## Summary

Custom MLIR dialects allow defining domain-specific operations and types that progressively lower to standard dialects. For FunLang, a custom dialect will encapsulate high-level language semantics (closures, pattern matching, etc.) and provide a clean separation between language concepts and their low-level implementation.

The standard approach uses **TableGen** (a declarative specification language) to define operations, types, and attributes. TableGen generates C++ code at build time, eliminating boilerplate. Dialects are defined in C++, then exposed to F# via **extern C shim functions** (same pattern established in Phase 1 appendix).

**Progressive lowering** is the key architectural pattern: custom operations lower through multiple dialects (FunLang → SCF/MemRef → LLVM) rather than jumping directly to LLVM. This enables optimization at each level and maintains high-level semantics longer in the compilation pipeline.

**Pattern-based rewrites** using DRR (Declarative Rewrite Rules) or C++ patterns implement lowering passes. The DialectConversion framework provides infrastructure for partial or full conversions with type transformation support.

**Primary recommendation:** Define FunLang dialect using TableGen for operations/types, implement C++ lowering passes using ConversionPattern, expose registration via extern C shim for F# interop.

## Standard Stack

The established libraries/tools for MLIR custom dialects:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| MLIR | 19.x+ | Core MLIR infrastructure | Official LLVM subproject, required for all dialect work |
| TableGen | 19.x+ | Declarative operation definition | LLVM's code generation framework, used by all MLIR dialects |
| MLIR-C API | 19.x+ | C API for MLIR | Enables FFI from F#, same pattern as Phase 1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| OpDefinitionsGen | 19.x+ | TableGen backend for operations | Always - auto-generates operation C++ classes |
| DialectConversion | 19.x+ | Lowering infrastructure | When implementing progressive lowering |
| RewritePattern/DRR | 19.x+ | Pattern rewriting | For optimization and lowering transformations |
| mlir-tblgen | 19.x+ | TableGen tool for MLIR | Build-time code generation from .td files |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TableGen ODS | Manual C++ Op classes | TableGen eliminates ~90% boilerplate, manual approach only for extremely dynamic ops |
| DRR patterns | C++ RewritePattern | DRR is declarative/concise, C++ gives full control for complex rewrites |
| Progressive lowering | Direct LLVM lowering | Progressive approach enables optimization at multiple levels, direct is simpler but less optimizable |

**Installation:**
```bash
# Already installed in Phase 1
# MLIR built from source with C API enabled
```

## Architecture Patterns

### Recommended Project Structure
```
LangBackend/
├── mlir-dialect/               # C++ dialect implementation
│   ├── include/
│   │   └── FunLang/
│   │       ├── FunLangDialect.h       # Dialect class declaration
│   │       ├── FunLangOps.h           # Operations header (generated)
│   │       ├── FunLangOps.td          # TableGen operation definitions
│   │       ├── FunLangTypes.h         # Custom types header (generated)
│   │       └── FunLangTypes.td        # TableGen type definitions
│   ├── lib/
│   │   ├── FunLangDialect.cpp         # Dialect implementation
│   │   ├── FunLangOps.cpp             # Operations implementation
│   │   ├── FunLangTypes.cpp           # Types implementation
│   │   └── Transforms/
│   │       ├── FunLangToSCF.cpp       # Lowering pass: FunLang -> SCF
│   │       └── Passes.td              # Pass definitions
│   └── CMakeLists.txt                 # Build configuration
├── mlir-capi/                  # C API shim for F#
│   ├── FunLangCAPI.h
│   ├── FunLangCAPI.cpp
│   └── CMakeLists.txt
└── tutorial/                   # Tutorial chapters (existing)
```

### Pattern 1: TableGen Operation Definition
**What:** Declarative specification of dialect operations
**When to use:** For all operations - standard MLIR approach
**Example:**
```tablegen
// Source: https://mlir.llvm.org/docs/DefiningDialects/Operations/
def FunLang_ClosureOp : FunLang_Op<"closure", [Pure]> {
  let summary = "Creates a closure with captured environment";
  let description = [{
    The `funlang.closure` operation creates a closure by combining a function
    reference with captured variables. The result is an opaque closure value
    that can be passed to higher-order functions.

    Example:
    ```mlir
    %closure = funlang.closure @lambda_func, %captured_var : (!funlang.closure)
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$captured
  );

  let results = (outs FunLang_ClosureType:$result);

  let assemblyFormat = "$callee `,` $captured attr-dict `:` type($result)";
}
```

### Pattern 2: Progressive Lowering Pass
**What:** Multi-stage lowering from high-level to low-level dialects
**When to use:** Always - maintains semantic clarity and optimization opportunities
**Example:**
```cpp
// Source: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/
// Stage 1: FunLang -> SCF/MemRef/Func
// funlang.closure -> func.func + memref.alloc + memref.store

class FunLangToSCFLowering : public ConversionPattern {
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) override {
    // Lower funlang.closure to:
    // 1. func.func for lambda body
    // 2. memref.alloc for environment
    // 3. memref.store for captured variables
    // 4. Return environment pointer
  }
};

// Stage 2: SCF/MemRef -> LLVM (existing MLIR passes)
// func.func-to-llvm, scf-to-cf, convert-memref-to-llvm
```

### Pattern 3: C API Shim for F# Interop
**What:** extern C functions wrapping C++ dialect operations
**When to use:** For all F# interaction with custom dialect
**Example:**
```cpp
// Source: https://mlir.llvm.org/docs/CAPI/
// FunLangCAPI.cpp
extern "C" {

// Register FunLang dialect in context
void mlirContextRegisterFunLangDialect(MlirContext ctx) {
  mlir::MLIRContext *context = unwrap(ctx);
  context->getOrLoadDialect<mlir::funlang::FunLangDialect>();
}

// Create funlang.closure operation
MlirOperation mlirFunLangClosureOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirAttribute callee,
    intptr_t numCaptured,
    MlirValue *captured) {
  // Build operation using generated builder
  mlir::OpBuilder builder(unwrap(ctx));
  auto closureOp = builder.create<mlir::funlang::ClosureOp>(
      unwrap(loc), unwrap(callee),
      llvm::makeArrayRef(unwrap(captured), numCaptured));
  return wrap(closureOp.getOperation());
}

} // extern "C"
```

### Pattern 4: Declarative Rewrite Rules (DRR)
**What:** TableGen-based pattern matching for rewrites
**When to use:** For simple, declarative transformations
**Example:**
```tablegen
// Source: https://mlir.llvm.org/docs/DeclarativeRewrites/
// Simplify closure with no captures
def SimplifyEmptyClosure : Pat<
  (FunLang_ClosureOp $callee, (ValueList)),
  (FunLang_FuncRefOp $callee)
>;

// Inline closure application when function is known
def InlineKnownClosure : Pat<
  (FunLang_ApplyOp (FunLang_ClosureOp $callee, $captures), $args),
  (FunLang_DirectCallOp $callee, (ConcatValueLists $captures, $args))
>;
```

### Anti-Patterns to Avoid
- **Manual C++ operation classes**: Use TableGen ODS - hand-written ops require 200+ lines vs 20 lines TableGen
- **Single-pass lowering to LLVM**: Progressive lowering through intermediate dialects enables optimization
- **Mixing abstraction levels in one dialect**: Keep high-level ops (closure, match) separate from lowered ops
- **Type-unsafe operation builders**: Use generated type-safe builders from TableGen
- **Ignoring verification**: Always implement verifiers with `hasVerifier = 1` to catch IR invariant violations

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Operation definition boilerplate | Manual C++ Op<> subclasses | TableGen ODS | Generates getters, verifiers, builders, documentation automatically |
| Pattern matching infrastructure | Custom rewrite visitors | RewritePattern/DRR | Framework handles pattern ordering, cost model, application tracking |
| Type conversion during lowering | Manual value remapping | TypeConverter in DialectConversion | Handles type changes, materializations, unrealized_conversion_cast insertion |
| Operation verification | Ad-hoc assertions | `hasVerifier = 1` + verify() | Integrates with MLIR's verification pipeline, proper diagnostic emission |
| Dialect registration | Manual MLIRContext setup | DialectRegistry + loadDialect | Proper dependency tracking, lazy loading |

**Key insight:** MLIR's infrastructure has solved these problems for all existing dialects. Custom solutions will miss edge cases (variadic operands, region handling, symbol tables, etc.) and won't integrate with MLIR tooling (mlir-opt, mlir-translate).

## Common Pitfalls

### Pitfall 1: Incomplete Type System
**What goes wrong:** Defining operations without custom types, relying only on builtin types (i32, !llvm.ptr)
**Why it happens:** Types seem like overhead when operations work with generic pointers
**How to avoid:** Define domain types (FunLang_ClosureType, FunLang_ListType) from the start - enables type checking and optimization
**Warning signs:** Operations accepting AnyType or !llvm.ptr everywhere, unable to distinguish closures from other pointers

### Pitfall 2: Skipping Progressive Lowering
**What goes wrong:** Lowering directly from FunLang operations to LLVM dialect in one pass
**Why it happens:** Seems simpler to avoid intermediate dialects
**How to avoid:** Lower through standard dialects (Func, SCF, MemRef) first - enables existing MLIR optimizations
**Warning signs:** Lowering pass becomes 1000+ lines, reimplementing standard patterns like loop construction

### Pitfall 3: C API Memory Management Confusion
**What goes wrong:** Calling C API functions with dangling pointers, incorrect ownership assumptions
**Why it happens:** MLIR's ownership model (context owns IR) differs from typical C FFI
**How to avoid:** Never store MlirOperation/MlirValue directly - store in MLIR IR, retrieve via traversal. Context outlives all IR.
**Warning signs:** Segfaults when accessing operations after pass runs, use-after-free in F# code

### Pitfall 4: Missing Operation Traits
**What goes wrong:** Operations don't declare traits like Pure, MemoryEffects, leading to incorrect optimization
**Why it happens:** Traits seem optional, operations "work" without them
**How to avoid:** Always declare side-effect traits - MLIR passes rely on them for correctness
**Warning signs:** Dead code elimination removes "pure" operations with visible side effects, or keeps operations that could be eliminated

### Pitfall 5: Ignoring Symbol Table Semantics
**What goes wrong:** Function references in closures break after inlining or other transformations
**Why it happens:** Using strings for function names instead of proper SymbolRefAttr
**How to avoid:** Use FlatSymbolRefAttr for function references - integrates with MLIR symbol table
**Warning signs:** "Symbol not found" errors after optimization passes, broken references after module splitting

## Code Examples

Verified patterns from official sources:

### Dialect Definition (TableGen)
```tablegen
// Source: https://mlir.llvm.org/docs/DefiningDialects/
def FunLang_Dialect : Dialect {
  let name = "funlang";
  let summary = "FunLang functional language dialect";
  let description = [{
    The FunLang dialect represents high-level functional programming
    constructs like closures, pattern matching, and algebraic data types.
    It progressively lowers to SCF, MemRef, and LLVM dialects.
  }];
  let cppNamespace = "::mlir::funlang";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "func::FuncDialect",
    "scf::SCFDialect",
    "LLVM::LLVMDialect"
  ];
}
```

### Custom Type Definition
```tablegen
// Source: https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/
def FunLang_ClosureType : TypeDef<FunLang_Dialect, "Closure"> {
  let summary = "FunLang closure type";
  let description = [{
    Represents a closure value - a function pointer plus captured environment.
    Lowered to !llvm.ptr during conversion to LLVM dialect.
  }];

  let mnemonic = "closure";

  // Parameters: function signature
  let parameters = (ins
    "FunctionType":$funcType
  );

  let assemblyFormat = "`<` $funcType `>`";
}
```

### Lowering Pattern (C++)
```cpp
// Source: https://mlir.llvm.org/docs/DialectConversion/
struct ClosureOpLowering : public OpConversionPattern<funlang::ClosureOp> {
  using OpConversionPattern<funlang::ClosureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::ClosureOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // 1. Get function reference
    auto callee = op.getCalleeAttr();

    // 2. Allocate environment on heap (captured vars)
    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    size_t numCaptured = adaptor.getCaptured().size();
    size_t envSize = 8 + numCaptured * 8; // fn_ptr + captured values

    auto envSizeConst = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(envSize));

    auto gcMalloc = rewriter.create<LLVM::CallOp>(
        loc, ptrType, "GC_malloc", ValueRange{envSizeConst});

    // 3. Store function pointer at env[0]
    auto fnPtrAddr = rewriter.create<LLVM::AddressOfOp>(loc, callee);
    auto gepZero = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, gcMalloc.getResult(),
        ArrayRef<LLVM::GEPArg>{0});
    rewriter.create<LLVM::StoreOp>(loc, fnPtrAddr, gepZero);

    // 4. Store captured values at env[1..]
    for (auto [idx, capturedVal] : llvm::enumerate(adaptor.getCaptured())) {
      auto gepIdx = rewriter.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, gcMalloc.getResult(),
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(idx + 1)});
      rewriter.create<LLVM::StoreOp>(loc, capturedVal, gepIdx);
    }

    // 5. Replace with environment pointer (closure representation)
    rewriter.replaceOp(op, gcMalloc.getResult());
    return success();
  }
};
```

### Pass Definition
```cpp
// Source: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/
struct FunLangToLLVMPass
    : public PassWrapper<FunLangToLLVMPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Legal: LLVM and Func dialects
    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect>();

    // Illegal: FunLang dialect (must be lowered)
    target.addIllegalDialect<funlang::FunLangDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ClosureOpLowering>(&getContext());
    // ... add more patterns

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual C++ Op classes | TableGen ODS | MLIR inception (2019) | 90% less boilerplate, single source of truth |
| Imperative pattern rewrites | DRR (Declarative Rewrite Rules) | MLIR 2020 | Concise patterns, but being superseded by PDLL |
| Direct LLVM lowering | Progressive lowering through dialects | MLIR design principle | More optimization opportunities |
| Opaque types (!llvm.ptr everywhere) | Dialect-specific type systems | MLIR best practice | Type safety, better optimization |
| DRR (TableGen-based) | PDLL (Pattern Description Language) | PDLL added 2022+ | More expressive, but DRR still maintained |

**Deprecated/outdated:**
- **Manual operation verifiers** before TableGen support: Use `hasVerifier = 1` instead
- **String-based function references**: Use SymbolRefAttr for proper symbol table integration
- **Context-free type conversions**: Use TypeConverter for proper materialization handling
- **Full DRR for all patterns**: PDLL is more powerful for complex patterns, though DRR still works for simple cases

**Current best practice (2026):**
- TableGen ODS for operation/type definitions
- ConversionPattern in C++ for complex lowering
- DRR for simple canonicalization/folding
- PDLL for complex pattern matching (though not yet required)
- Progressive lowering through standard dialects

## Open Questions

Things that couldn't be fully resolved:

1. **F# ergonomics for operation builders**
   - What we know: C API provides low-level operation creation
   - What's unclear: Best abstraction level for F# wrappers (mirror TableGen builders vs. domain-specific API)
   - Recommendation: Start with thin wrappers, refactor based on usage patterns in tutorial writing

2. **Performance impact of progressive lowering**
   - What we know: More passes = more compilation time, but better optimization potential
   - What's unclear: Actual compile-time cost for FunLang programs vs. optimization benefits
   - Recommendation: Measure during Phase 7 optimization work, accept higher compile time for educational clarity

3. **PDLL vs DRR adoption timeline**
   - What we know: PDLL is more powerful but DRR is stable and well-documented
   - What's unclear: Whether PDLL should be taught in Phase 5 given its power/complexity tradeoff
   - Recommendation: Use DRR for Phase 5, mention PDLL as "advanced" alternative

4. **Custom type lowering complexity**
   - What we know: TypeConverter handles type transformations, but complex nesting (closure in list in closure) may need careful design
   - What's unclear: Whether to unify all FunLang types to single opaque pointer or maintain type distinctions longer
   - Recommendation: Keep typed representation through SCF lowering, unify to !llvm.ptr only in final LLVM lowering

## Sources

### Primary (HIGH confidence)
- [MLIR Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/) - Official MLIR documentation on dialect definition
- [MLIR Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) - TableGen ODS reference
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) - ConversionPattern and lowering infrastructure
- [MLIR Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/) - DRR pattern syntax and usage
- [MLIR Creating a Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/) - Step-by-step dialect creation
- [MLIR Toy Tutorial Ch.5](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) - Progressive lowering example
- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) - C API design and usage patterns

### Secondary (MEDIUM confidence)
- [Jeremy Kun: MLIR Defining a New Dialect](https://www.jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/) - Practical tutorial (2023, verified against official docs)
- [Jeremy Kun: MLIR Dialect Conversion](https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion/) - Conversion patterns explained
- [Jeremy Kun: MLIR Canonicalizers and DRR](https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/) - Pattern rewrite practical guide
- [Medium: MLIR Custom Dialect Tutorial](https://medium.com/sniper-ai/mlir-tutorial-create-your-custom-dialect-lowering-to-llvm-ir-dialect-system-1-1f125a6a3008) - Community tutorial with code examples

### Tertiary (LOW confidence)
- LLVM Discourse forums on dialect design - community discussions, not authoritative but useful for "gotchas"

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official LLVM/MLIR infrastructure, stable APIs
- Architecture: HIGH - Verified against official tutorials and multiple dialect implementations
- Pitfalls: MEDIUM - Based on official docs + community experience, but context-specific to FunLang use case

**Research date:** 2026-02-06
**Valid until:** 30 days (stable MLIR APIs, but new features may emerge)

**Key research gaps filled:**
- ✅ TableGen ODS syntax for operations and types
- ✅ Progressive lowering pattern (dialect hierarchy)
- ✅ ConversionPattern vs RewritePattern differences
- ✅ C API shim pattern for F# interop (validated against Phase 1 appendix)
- ✅ Common pitfalls in dialect design
- ⚠️ PDLL vs DRR decision (chose DRR for stability/documentation)
- ⚠️ Optimal abstraction level for F# wrappers (deferred to implementation)
