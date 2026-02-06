---
phase: 05-custom-mlir-dialect
plan: 03
type: summary
subsystem: documentation
tags: [mlir, lowering, dialect-conversion, pattern-rewriting]
dependencies:
  requires:
    - 05-02  # Custom operations (funlang.closure, funlang.apply)
    - 04-02  # Higher-order functions (lowering targets)
  provides:
    - chapter-16  # Lowering passes documentation
    - lowering-patterns  # ConversionPattern implementations
    - complete-phase-5  # Phase 5 fully documented
  affects:
    - 06-01  # Pattern matching will use similar lowering patterns
key-files:
  created:
    - tutorial/16-lowering-passes.md  # 2718 lines - Lowering passes chapter
  modified:
    - tutorial/SUMMARY.md  # Added Chapter 16 entry
tech-stack:
  added: []  # No new dependencies (documentation)
  patterns:
    - DialectConversion framework (ConversionTarget, RewritePatternSet, TypeConverter)
    - OpConversionPattern for custom operations
    - Declarative Rewrite Rules (DRR) for optimization
    - C API shim pattern for pass execution
decisions:
  - id: lowering-01
    title: Direct FunLang -> LLVM lowering (Phase 5)
    rationale: Simple closure operations don't need intermediate SCF dialect
    alternatives: Multi-stage lowering through SCF (reserved for Phase 6 pattern matching)
  - id: lowering-02
    title: OpConversionPattern over DRR for complex lowering
    rationale: ClosureOp/ApplyOp need dynamic logic (for loops, size calculation)
    impact: DRR reserved for simple optimization patterns
  - id: lowering-03
    title: Partial conversion for FunLangToLLVM pass
    rationale: Other dialects (arith, func) lowered by separate passes
    impact: Modular pass pipeline
metrics:
  duration: 457  # seconds (~7.6 min)
  completed: 2026-02-06
---

# Phase 5 Plan 3: Lowering Passes Summary

**One-liner:** Pattern-based lowering from FunLang dialect to LLVM dialect with ClosureOp/ApplyOp ConversionPatterns

## Overview

**Objective:** Complete Phase 5 by documenting progressive lowering from FunLang dialect to LLVM dialect using MLIR's DialectConversion framework

**Outcome:** Chapter 16 (2718 lines) covering:
- DialectConversion framework (ConversionTarget, RewritePatternSet, TypeConverter)
- ClosureOp lowering pattern (funlang.closure → GC_malloc + GEP + store)
- ApplyOp lowering pattern (funlang.apply → GEP + load + indirect call)
- TypeConverter for FunLang types (funlang.closure → !llvm.ptr)
- DRR optimization patterns (SimplifyEmptyClosure, InlineKnownApply)
- Complete FunLangToLLVMPass implementation
- End-to-end makeAdder example through entire pipeline
- Common errors and debugging tips

**Duration:** 7.6 minutes

## What Was Built

### Core Content Sections

**1. Introduction (200 lines)**
- Phase 5 recap (Chapters 14-15)
- Lowering pass concept and necessity
- Progressive lowering strategy (FunLang → LLVM)
- Chapter 16 goals and success criteria

**2. DialectConversion Framework (350 lines)**
- ConversionTarget: legal/illegal operations
- addLegalDialect, addIllegalDialect, addDynamicallyLegalOp
- RewritePatternSet composition
- applyPartialConversion vs applyFullConversion
- TypeConverter role and usage
- Conversion failure handling

**3. ClosureOp Lowering Pattern (450 lines)**
- Chapter 12 closure creation pattern recap
- funlang.closure → GC_malloc + GEP + store
- Complete ClosureOpLowering implementation in C++
- OpAdaptor for converted operands
- ConversionPatternRewriter usage
- Environment size calculation and slot management
- C API shim preview

**4. ApplyOp Lowering Pattern (350 lines)**
- Chapter 13 indirect call pattern recap
- funlang.apply → GEP + load + llvm.call (indirect)
- Complete ApplyOpLowering implementation in C++
- Function pointer extraction from env[0]
- Argument list construction (closure + args)
- Result type conversion
- End-to-end makeAdder example

**5. TypeConverter for FunLang Types (250 lines)**
- funlang.closure → !llvm.ptr conversion
- funlang.list<T> → !llvm.ptr conversion
- Function signature automatic conversion
- Materialization functions (source/target)
- unrealized_conversion_cast handling
- Type conversion chains for multi-stage lowering

**6. Declarative Rewrite Rules (DRR) (300 lines)**
- DRR concept and TableGen syntax
- DRR vs C++ ConversionPattern comparison
- Empty closure optimization pattern (SimplifyEmptyClosure)
- Known closure inlining pattern (InlineKnownApply)
- Constant propagation limitations
- mlir-tblgen compilation workflow
- When to use DRR vs C++

**7. Complete Lowering Pass (250 lines)**
- FunLangToLLVMPass complete implementation
- PassWrapper template and metadata
- getDependentDialects registration
- runOnOperation execution flow
- Pass registration and command-line flag
- C API shim for F# integration
- F# P/Invoke declarations and wrappers
- Full compilation pipeline in F#

**8. End-to-End Example (200 lines)**
- makeAdder source code
- Stage-by-stage IR transformation:
  - AST representation
  - FunLang dialect MLIR (Chapter 15)
  - After FunLangToLLVM pass (Chapter 16)
  - After convert-arith-to-llvm
  - After convert-func-to-llvm
  - LLVM IR output
  - Native code compilation and linking
- Execution flow tracing
- Complete pipeline diagram

**9. Common Errors (100 lines)**
- Error 1: Illegal operation remaining (pattern not registered/matched)
- Error 2: Type conversion failure (missing TypeConverter rule)
- Error 3: Wrong operand types (store type mismatch, GEP usage)
- Error 4: Pass not registered (initialization missing)
- Error 5: Segmentation fault (rewriter misuse, use-after-free)
- Debugging strategies for each error

**10. Summary (50 lines)**
- Phase 5 completion recap (Chapters 14-16)
- Code compression benefits table
- Phase 6 preview (pattern matching with SCF dialect)
- Direct vs multi-stage lowering comparison

## Key Technical Decisions

### 1. Direct FunLang → LLVM Lowering (Phase 5)

**Decision:** Skip intermediate SCF dialect for Phase 5 closure operations

**Rationale:**
- funlang.closure and funlang.apply are simple operations
- No complex control flow needed
- Direct lowering more efficient for straightforward transformations

**Alternatives considered:**
- Multi-stage lowering (FunLang → SCF → LLVM)
  - Rejected: Adds unnecessary complexity for simple operations
  - Reserved for Phase 6 pattern matching (funlang.match needs SCF)

**Impact:**
- Simpler lowering pass implementation
- Faster compilation for closure operations
- Clear separation: Phase 5 (direct) vs Phase 6 (multi-stage)

### 2. OpConversionPattern over DRR for Complex Lowering

**Decision:** Use C++ ConversionPatterns for ClosureOp/ApplyOp lowering

**Rationale:**
- ClosureOp needs dynamic size calculation
- Variable number of captured variables (for loop required)
- ApplyOp needs argument list construction
- DRR cannot express imperative logic

**DRR reserved for:**
- Simple optimization patterns (constant folding)
- Peephole optimizations (empty closure elimination)
- Fixed-size transformations

**Impact:**
- Full control over lowering logic
- Easy debugging with C++ breakpoints
- DRR complements (not replaces) ConversionPatterns

### 3. Partial Conversion for FunLangToLLVM Pass

**Decision:** Use applyPartialConversion instead of applyFullConversion

**Rationale:**
- Modular pass pipeline
- Other dialects (arith, func) lowered separately
- Standard MLIR passes handle standard dialects

**Pipeline structure:**
```
1. FunLangToLLVM pass (custom)
2. convert-arith-to-llvm (standard)
3. convert-func-to-llvm (standard)
4. mlir-translate --mlir-to-llvmir
```

**Impact:**
- Composable passes
- Can insert optimization passes between stages
- Clear separation of concerns

## Verification Results

All verification checks passed:

```bash
# Total line count: 2718 (exceeded 1500 minimum)
wc -l tutorial/16-lowering-passes.md
# Output: 2718

# ApplyOp lowering content: 41 mentions
grep -c "ApplyOpLowering\|funlang.apply" tutorial/16-lowering-passes.md
# Output: 41

# DRR content: 40 mentions
grep -c "DRR\|Declarative.*Rewrite\|Pat<\|SimplifyEmpty" tutorial/16-lowering-passes.md
# Output: 40

# Complete pass content: 34 mentions
grep -c "FunLangToLLVM\|registerFunLangPasses\|runOnOperation" tutorial/16-lowering-passes.md
# Output: 34

# TypeConverter content: 58 mentions
grep -c "TypeConverter\|addConversion\|Materialization" tutorial/16-lowering-passes.md
# Output: 58

# SUMMARY.md updated
grep "16-lowering-passes" tutorial/SUMMARY.md
# Output: - [Chapter 16: Lowering Passes](16-lowering-passes.md)
```

## Content Quality Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Total lines** | 1500+ | 2718 | ✅ Exceeded |
| **ClosureOp lowering** | Complete C++ impl | 450 lines | ✅ Complete |
| **ApplyOp lowering** | Complete C++ impl | 350 lines | ✅ Complete |
| **TypeConverter** | FunLang → LLVM | 250 lines | ✅ Complete |
| **DRR patterns** | 2+ examples | 300 lines (2 patterns) | ✅ Complete |
| **Complete pass** | FunLangToLLVM | 250 lines | ✅ Complete |
| **End-to-end example** | makeAdder pipeline | 200 lines | ✅ Complete |
| **Common errors** | 5+ errors | 100 lines (5 errors) | ✅ Complete |
| **Korean plain style** | ~이다/~한다 | Consistent | ✅ Verified |

## Links to Prior Work

**Chapter 12 (Closures):**
- ClosureOp lowering reuses Chapter 12 closure creation pattern
- GC_malloc + GEP + store sequence for environment allocation
- Environment layout: [fn_ptr, captured_var1, captured_var2, ...]

**Chapter 13 (Higher-Order Functions):**
- ApplyOp lowering reuses Chapter 13 indirect call pattern
- Function pointer extraction from env[0]
- Indirect call with environment as first argument

**Chapter 14 (Custom Dialect Design):**
- DialectConversion framework explained in theory
- C API shim pattern applied to pass registration

**Chapter 15 (Custom Operations):**
- Lowering targets: funlang.closure, funlang.apply
- Type conversion: !funlang.closure → !llvm.ptr
- Complete compiler pipeline context

## Phase 5 Complete

**Three-chapter journey:**

1. **Chapter 14 (2682 lines):** Custom dialect design theory
   - Progressive lowering philosophy
   - TableGen ODS syntax
   - C API shim pattern
   - FunLang dialect architecture

2. **Chapter 15 (3642 lines):** Custom operations implementation
   - funlang.closure operation definition
   - funlang.apply operation definition
   - !funlang.closure custom type
   - Complete F# integration (P/Invoke → OpBuilder extensions)

3. **Chapter 16 (2718 lines):** Lowering passes
   - DialectConversion framework
   - ConversionPattern implementations
   - TypeConverter setup
   - Complete FunLangToLLVMPass

**Total Phase 5 content:** 9042 lines

**Code compression achieved:**

| Metric | Before (Phase 4) | After (Phase 5) |
|--------|-----------------|----------------|
| Closure creation | 12 lines | 1 line (92% reduction) |
| Closure application | 8 lines | 1 line (87% reduction) |
| Compiler code | ~200 lines | ~100 lines (50% reduction) |
| Type safety | !llvm.ptr (opaque) | !funlang.closure (typed) |
| Optimization | Difficult (low-level) | Easy (DRR patterns) |

## Deviations from Plan

**None** - Plan executed exactly as written.

All sections completed:
- ✅ DialectConversion Framework (350+ lines)
- ✅ ClosureOp Lowering Pattern (450+ lines)
- ✅ ApplyOp Lowering Pattern (350+ lines)
- ✅ TypeConverter for FunLang Types (250+ lines)
- ✅ Declarative Rewrite Rules (DRR) (300+ lines)
- ✅ Complete Lowering Pass (250+ lines)
- ✅ End-to-End Example (200+ lines)
- ✅ Common Errors (100+ lines)
- ✅ SUMMARY.md updated

## Next Phase Readiness

**Phase 6: Pattern Matching**

**Ready to proceed:**
- ✅ Custom dialect infrastructure complete (Chapter 14)
- ✅ Operation definition patterns established (Chapter 15)
- ✅ Lowering pass implementation patterns documented (Chapter 16)

**Phase 6 will add:**
- funlang.match operation (complex control flow)
- funlang.nil, funlang.cons (list construction)
- Multi-stage lowering: FunLang → SCF → LLVM
- Region-based pattern matching transformation

**Key difference Phase 5 vs Phase 6:**
- Phase 5: Direct lowering (simple operations)
- Phase 6: Multi-stage lowering (complex control flow through SCF)

**No blockers.** Ready to execute 06-01-PLAN.md.

## Lessons Learned

### Documentation Approach Success

**Effective patterns:**
1. **Theory before code:** DialectConversion framework explained before patterns
2. **Chapter recaps:** Linked to Chapter 12-13 patterns being reused
3. **Complete code blocks:** Full C++ implementations, not snippets
4. **Step-by-step breakdown:** Each section of code explained separately
5. **Before/After comparisons:** FunLang dialect → LLVM dialect transformations
6. **End-to-end example:** makeAdder through entire pipeline (7 stages)
7. **Error handling:** Common errors with causes and solutions

**Reader benefits:**
- Understand "why" behind lowering patterns
- Can implement own ConversionPatterns
- Can debug lowering failures
- See connection between Chapters 12-13 and 16

### Code Reuse Validation

**Phase 4 → Phase 5 continuity:**
- Chapter 12 closure creation pattern = ClosureOp lowering target
- Chapter 13 indirect call pattern = ApplyOp lowering target
- No new concepts, just automation of manual patterns

**This validates the roadmap strategy:**
- Phase 4: Manual low-level patterns (teach fundamentals)
- Phase 5: Automate patterns with custom dialect (teach abstractions)

## Commits

**Task 1 - ClosureOp Lowering (1013 lines):**
```
945c002 feat(05-03): write Chapter 16 Part 1 - Lowering Infrastructure and ClosureOp Lowering
```

**Task 2 - ApplyOp Lowering & DRR (1705 lines):**
```
c70f5d5 feat(05-03): write Chapter 16 Part 2 - ApplyOp Lowering and DRR Patterns
```

**Total:** 2 commits, 2718 lines of documentation

---

**Phase 5 Status:** ✅ COMPLETE

All three plans (05-01, 05-02, 05-03) successfully executed.
Total Phase 5 documentation: 9042 lines (Chapters 14-16).
Custom MLIR dialect journey from design to lowering complete.

