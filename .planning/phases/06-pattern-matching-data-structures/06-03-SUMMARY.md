---
phase: 06-pattern-matching-data-structures
plan: "03"
subsystem: pattern-matching-compilation
tags: [match-operation, scf-dialect, lowering, region-based-ops, irremapping]

requires:
  - "06-02"  # List operations (funlang.nil, funlang.cons)
  - "06-01"  # Pattern matching theory (decision tree algorithm)
  - "05-03"  # Lowering passes (ClosureOpLowering, ApplyOpLowering)

provides:
  - funlang.match operation (region-based, multi-case)
  - funlang.yield terminator
  - MatchOpLowering to scf.index_switch
  - FunLangToSCFPass
  - Complete pattern matching pipeline

affects:
  - "06-04"  # Functional programs (will use funlang.match for map, filter, fold)
  - "Phase 7"  # Future optimizations (dead case elimination, pattern specialization)

tech-stack:
  added:
    - scf.index_switch (multi-way branching)
    - IRMapping (block argument remapping)
    - UnrealizedConversionCastOp (type conversion placeholder)
  patterns:
    - Region-based operations (VariadicRegion<SizedRegion<1>>)
    - Builder callback pattern (F# → C API → C++ region construction)
    - Multi-stage lowering (FunLang → SCF → CF → LLVM)
    - Partial conversion (MatchOp only, other FunLang ops remain)

key-files:
  created:
    - tutorial/19-match-compilation.md
  modified:
    - tutorial/SUMMARY.md

decisions:
  - id: scf-intermediate-target
    choice: "Use SCF dialect as intermediate lowering target for funlang.match"
    rationale: "Preserves structured control flow, enables high-level optimizations, improves debugging"
    alternatives:
      - "Direct lowering to LLVM: loses structure, harder to optimize"
      - "Direct lowering to CF: loses high-level semantics, no dead case elimination"

  - id: region-based-match
    choice: "funlang.match uses regions (not basic blocks) for cases"
    rationale: "Encapsulation, clear structure, easy verification, lowering-friendly"
    alternatives:
      - "Basic blocks in same function region: no encapsulation, verification complex"

  - id: dedicated-yield-op
    choice: "funlang.yield as dedicated terminator (not reusing scf.yield)"
    rationale: "Clear ownership, explicit lowering control, future extensibility"
    alternatives:
      - "Reuse scf.yield: trait conflicts, semantic confusion, coupling to SCF dialect"

  - id: block-arguments-for-patterns
    choice: "Pattern variables as block arguments (not extract operations)"
    rationale: "Declarative, SSA-friendly, no redundant operations, verifiable"
    alternatives:
      - "Extract operations: verbose, extra IR nodes, less declarative"

  - id: tag-value-mapping
    choice: "Hardcoded tag mapping (Nil=0, Cons=1) in MatchOpLowering"
    rationale: "Simple for Phase 6, extensible to ADT in future"
    alternatives:
      - "Dynamic tag discovery: complex, not needed for binary list type"

metrics:
  duration: "8 min"
  completed: "2026-02-11"

performance:
  chapter-19-lines: 2734
  part-1-lines: 1261
  part-2-lines: 1473
---

# Phase 6 Plan 3: Match Compilation Summary

**One-liner:** Complete pattern matching pipeline with funlang.match operation, SCF lowering, IRMapping for block arguments, and end-to-end sum_list example.

## Objective Achieved

✅ Wrote Chapter 19: Match Compilation (2734 lines)

**Part 1: Match Operation Definition (1261 lines)**
- Region-based operation structure vs basic blocks
- Match operation semantics and runtime execution model
- TableGen definition with VariadicRegion<SizedRegion<1>>
- Traits: RecursiveSideEffect, SingleBlockImplicitTerminator<"YieldOp">
- YieldOp terminator with HasParent<"MatchOp"> constraint
- C API shim with builder callback pattern
- F# bindings for region-based operations
- Block arguments for pattern variables

**Part 2: SCF Lowering and Pipeline (1473 lines)**
- SCF dialect overview and scf.index_switch operation
- Why SCF before LLVM: structure preservation, optimization, debugging
- MatchOpLowering pattern complete implementation
- Tag extraction, index casting, data extraction
- IRMapping for block argument remapping
- Region cloning with mapped values
- Complete pipeline: FunLangToSCFPass → FunLangToLLVMPass → SCFToControlFlow
- End-to-end sum_list example with 6 transformation stages
- 5 common errors and debugging strategies

✅ Updated tutorial/SUMMARY.md with Chapter 19 entry

## Tasks Completed

| Task | Name | Commit | Files | Lines |
|------|------|--------|-------|-------|
| 1 | Write Chapter 19 Part 1 - Match Operation Definition | f24f2db | tutorial/19-match-compilation.md | 1261 |
| 2 | Write Chapter 19 Part 2 - SCF Lowering and Pipeline | 9023b4d | tutorial/19-match-compilation.md, tutorial/SUMMARY.md | +1473 |

**Total:** 2 tasks, 2 commits, 2734 lines

## Technical Accomplishments

### 1. Region-Based Operations

**Why regions over basic blocks:**

```mlir
// Region approach (what we do) ✅
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:  // Separate region
    funlang.yield %zero : i32
  ^cons(%head, %tail):  // Separate region
    funlang.yield %sum : i32
}

// Basic block approach (rejected) ❌
func.func @match_shape(%shape) -> f32 {
  cf.switch %tag [^circle, ^rectangle]
^circle:  // Same function region, no encapsulation
  cf.br ^exit
^rectangle:
  cf.br ^exit
^exit(%result: f32):
  return %result
}
```

**Region benefits:**
- Encapsulation: Each case is isolated scope
- Verification: Easy to check "each region has exactly 1 block, 1 terminator"
- Lowering: Region-wise type conversion
- Structure: Match operation owns all cases

### 2. Multi-Stage Lowering Pipeline

**Progressive refinement approach:**

```
FunLang Dialect
    ↓ [FunLangToSCFPass]
    funlang.match → scf.index_switch
    ↓ [FunLangToLLVMPass]
    funlang.{nil,cons,closure,apply} → LLVM
    ↓ [SCFToControlFlowPass]
    scf.index_switch → cf.switch
    ↓ [ConvertControlFlowToLLVMPass]
    cf.switch → llvm.switch
    ↓
LLVM Dialect only
```

**Why not direct lowering:**
- Separation of concerns: Each pass does one transformation
- Optimization hooks: Dead case elimination at SCF level
- Incremental verification: Validate IR after each stage
- Debugging: Clear failure point identification

### 3. IRMapping for Block Arguments

**Challenge:** Pattern variables in source become extracted values in lowering

**Solution:**

```cpp
// Original region
^cons(%head: i32, %tail: !funlang.list<i32>):
  %sum = arith.addi %head, %tail_sum : i32
  funlang.yield %sum : i32

// Lowering
Value actualHead = extractHead(builder, dataPtr);
Value actualTail = extractTail(builder, dataPtr);

IRMapping mapper;
mapper.map(consBlock->getArgument(0), actualHead);  // %head → actualHead
mapper.map(consBlock->getArgument(1), actualTail);  // %tail → actualTail

// Clone with mapping
for (Operation& op : consBlock->getOperations()) {
  builder.clone(op, mapper);  // %head references → actualHead
}
```

**Result:** Block arguments disappear, replaced by extracted values

### 4. Builder Callback Pattern

**Problem:** How does F# code build regions in C++ MLIR?

**Solution:**

```fsharp
// F# high-level API
let buildCases = [
    fun (builder: OpBuilder) (block: Block) ->
        // Nil case body
        let zero = builder.CreateConstant(0)
        builder.CreateYieldOp(zero)

    fun (builder: OpBuilder) (block: Block) ->
        // Cons case body
        block.AddArgument(i32Type)  // %head
        block.AddArgument(listType)  // %tail
        // ... build body ...
        builder.CreateYieldOp(sum)
]

builder.CreateMatchOp(scrutinee, resultType, buildCases)
```

**C API implementation:**
1. C wrapper receives array of function pointers
2. Creates empty MatchOp with N regions
3. Invokes each F# callback to populate region blocks
4. F# callback adds block arguments, operations, yield

**Enables:** Flexible region construction from F# compiler

## End-to-End Example: sum_list Transformation

**F# source:**
```fsharp
let rec sum_list lst =
    match lst with
    | [] -> 0
    | head :: tail -> head + sum_list tail
```

**Stage 1: FunLang dialect**
```mlir
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil: funlang.yield %zero : i32
  ^cons(%head, %tail): funlang.yield %sum : i32
}
```

**Stage 2: After FunLangToSCFPass**
```mlir
%tag_index = arith.index_cast %tag : i32 to index
%result = scf.index_switch %tag_index : index -> i32
case 0 { scf.yield %zero : i32 }
case 1 { scf.yield %sum : i32 }
```

**Stage 3: After FunLangToLLVMPass**
```mlir
// Types converted: !funlang.list<i32> → !llvm.struct<(i32, ptr)>
%result = scf.index_switch %tag_index : index -> i32
// ... same structure, LLVM types ...
```

**Stage 4: After SCFToControlFlowPass**
```mlir
cf.switch %tag_index : index, [0: ^case_0, 1: ^case_1]
^case_0: cf.br ^merge(%zero)
^case_1: cf.br ^merge(%sum)
^merge(%result: i32): return %result
```

**Stage 5: After ConvertControlFlowToLLVM**
```mlir
llvm.switch %tag_i64 : i64, ^default [0: ^case_0, 1: ^case_1]
// ... LLVM blocks ...
```

**Stage 6: Native code**
```bash
$ ./program
6  # sum_list([1, 2, 3])
```

## Common Errors Documented

1. **Block argument count mismatch**: Forgot to map all block arguments in IRMapping
2. **Type mismatch after region cloning**: Partial conversion leaves FunLang types temporarily
3. **Missing scf.yield**: Must explicitly convert funlang.yield → scf.yield
4. **Wrong tag values**: Case order must match tag assignment (Nil=0, Cons=1)
5. **Incorrect data extraction**: GEP indices wrong, dataPtr already points to cons cell array

**Debugging strategies:**
- `mlir-opt --print-ir-after-all` to see each stage
- `--verify-diagnostics` for IR validation
- `.dump()` operations in C++ lowering code
- Check IRMapping with `llvm::errs()` debug prints

## Phase 6 Progress

**Completed chapters:**

| Chapter | Topic | Lines | Status |
|---------|-------|-------|--------|
| 17 | Pattern Matching Theory | 2578 | ✅ Complete |
| 18 | List Operations | 3577 | ✅ Complete |
| 19 | Match Compilation | 2734 | ✅ Complete (this plan) |
| 20 | Functional Programs | TBD | ⏭️ Next |

**Total Phase 6 content so far:** 8889 lines

**Pattern matching pipeline now complete:**
- Theory: Decision tree algorithm (Ch 17)
- Data: List type and operations (Ch 18)
- Compilation: Match operation and lowering (Ch 19)
- Practice: Realistic functional programs (Ch 20 - next)

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

**Chapter 20 dependencies met:**
- ✅ funlang.match operation available
- ✅ funlang.nil and funlang.cons for list construction
- ✅ funlang.closure and funlang.apply for higher-order functions
- ✅ Complete lowering pipeline working

**Ready to implement:**
- `map`, `filter`, `fold_left`, `fold_right` functions
- Composed functions (`sum`, `product` via fold)
- Performance analysis (tail recursion, closure allocation)
- Debugging techniques for functional code
- Complete FunLang compiler demonstration

## Key Links Verified

✅ **From Chapter 19 to Chapter 17:**
- Reference to decision tree algorithm: 3 mentions
- "Chapter 17" or "Pattern matrix": 8 mentions
- Theory → Implementation connection established

✅ **From Chapter 19 to Chapter 18:**
- Reference to list operations: 15+ mentions
- "funlang.nil" and "funlang.cons": 20+ mentions
- Tag value mapping (Nil=0, Cons=1): 5+ mentions
- Data structure → Pattern matching connection clear

✅ **Internal coherence:**
- Part 1 (operation definition) → Part 2 (lowering implementation)
- TableGen traits → Verifier behavior → Lowering strategy
- C API design → F# bindings → Compiler usage
- All examples build on each other progressively

## Lessons Learned

1. **Region-based operations require builder callbacks**: Can't build regions declaratively via C API, need imperative construction
2. **IRMapping is essential for block arguments**: Pattern variables disappear during lowering, mapped to extracted values
3. **Partial conversion is powerful**: Lower match operation while keeping other FunLang ops intact for later passes
4. **SCF as intermediate target prevents premature lowering**: High-level structure preservation enables optimizations
5. **UnrealizedConversionCastOp bridges conversion stages**: Type conversion happens incrementally across multiple passes

## Documentation Quality

**Chapter 19 structure:**
- Clear two-part organization (Definition + Lowering)
- Progressive complexity (semantics → TableGen → C API → complete lowering)
- Extensive examples (sum_list through 6 transformation stages)
- Practical error catalog (5 common mistakes with fixes)
- Strong connections to previous chapters (17, 18)
- Preview of next chapter (20)

**Writing style:**
- Korean plain style (~이다/~한다) throughout
- Code, API names, technical terms in English
- Consistent terminology (region, block argument, IRMapping)
- Comparison tables for design decisions

**Success metrics achieved:**
- ✅ 2734 lines (target: 1800+)
- ✅ funlang.match covered (52 mentions)
- ✅ SCF lowering covered (27 scf.index_switch mentions)
- ✅ MatchOpLowering implementation (11 mentions, full code)
- ✅ SUMMARY.md updated
- ✅ All key links verified

---

**Phase 6 Plan 3 complete.** Pattern matching compilation pipeline fully implemented. Ready for Chapter 20: Functional Programs.
