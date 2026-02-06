---
phase: 04-closures-higher-order
plan: 02
subsystem: tutorial-writing
tags: [higher-order-functions, indirect-call, makeAdder, currying, upward-funarg]

dependencies:
  requires: [04-01-closures]
  provides: [higher-order-functions-chapter, phase-4-complete]
  affects: [05-custom-dialect]

tech-stack:
  added: []
  patterns: [indirect-call, function-as-argument, function-as-return-value, currying]

key-files:
  created:
    - tutorial/13-higher-order-functions.md
  modified:
    - tutorial/SUMMARY.md

decisions:
  - name: "Uniform closure representation"
    choice: "All functions represented as closures (fn_ptr, env)"
    rationale: "Simplifies compiler implementation, enables HOF support"
    alternatives: ["Separate representations for top-level vs lambdas"]
    impact: "All functions can be passed as arguments or returned"

  - name: "Indirect call pattern"
    choice: "llvm.call with function pointer extracted from closure"
    rationale: "Runtime function selection for higher-order functions"
    alternatives: ["Direct calls only (no HOF support)"]
    impact: "Enables apply, compose, map and all HOF patterns"

  - name: "Heap allocation for returned closures"
    choice: "All closures heap-allocated via GC_malloc"
    rationale: "Solves upward funarg problem, prevents dangling pointers"
    alternatives: ["Stack allocation with escape analysis"]
    impact: "Safe but slower; optimization deferred to Phase 7"

metrics:
  duration: 6m
  completed: 2026-02-06
  lines-of-code: 1618
  test-coverage: n/a
---

# Phase 04 Plan 02: Higher-Order Functions - Summary

**One-liner:** Indirect call pattern and makeAdder/currying enable complete higher-order function support (apply, compose, map)

## What Was Built

### Tutorial Chapter 13: Higher-Order Functions (1618 lines)

**Part 1: Functions as Arguments (754 lines)**
- Introduction to higher-order functions concept
- Functions as first-class values (closure-based representation)
- Indirect call pattern: `llvm.call %fn_ptr(%closure, %args)`
- F# helper `CallClosure` for indirect calls
- Apply function implementation (simplest HOF)
- Compose function with multiple function arguments

**Part 2: Functions as Return Values (864 lines)**
- Upward funarg problem explanation
- Why heap allocation is mandatory for escaping closures
- MakeAdder implementation (canonical closure-returning function)
- Closure lifecycle management by GC
- Currying pattern (multi-argument → nested closures)
- Partial application examples
- Memory management details (GC tracking, cyclic closures)
- Conceptual map function (full implementation in Phase 6)
- Common Errors section (5 error types with solutions)
- Phase 4 completion summary
- Phase 5 preview (custom MLIR dialect)

**Technical Content:**
- 21+ MLIR function definitions showing indirect calls
- 10+ F# code examples
- Complete apply, compose, makeAdder implementations
- Currying with partial application
- Memory safety demonstrations

**Documentation Updates:**
- Updated `tutorial/SUMMARY.md` with both Chapter 12 and 13
- Phase 4 marked complete in tutorial
- Phase 5 custom dialect preview included

## Key Patterns Established

### 1. Indirect Call Pattern

**Problem:** Higher-order functions don't know which function is passed at compile time

**Solution:** Extract function pointer from closure and call indirectly

```mlir
// Extract function pointer from closure[0]
%fn_ptr_addr = llvm.getelementptr %closure[0, 0] : (!llvm.ptr, i64) -> !llvm.ptr
%fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr

// Call with closure as first argument
%result = llvm.call %fn_ptr(%closure, %args...) : (!llvm.ptr, ...) -> result_type
```

**Impact:** Enables all HOF patterns (apply, compose, map, filter, fold)

### 2. Functions as Arguments (Apply Pattern)

**Signature:** `apply : (a -> b) -> a -> b`

**Implementation:**
- Accept closure as parameter
- Use indirect call pattern to invoke
- Pass closure's environment automatically

**Use cases:** Abstraction, callbacks, event handlers

### 3. Functions as Return Values (MakeAdder Pattern)

**Signature:** `makeAdder : int -> (int -> int)`

**Implementation:**
1. Outer function creates environment for inner closure
2. Heap-allocate environment (survives outer function return)
3. Store captured variables in environment
4. Return closure pointer

**Key insight:** Upward funarg problem solved by heap allocation + GC

### 4. Currying Pattern

**Transformation:** `f : a -> b -> c` becomes `f : a -> (b -> c)`

**Benefits:**
- Partial application: `let add5 = add 5`
- Function composition: easier to chain
- Uniform type system: all functions single-argument

**Implementation:** Nested closures, each capturing previous arguments

## Technical Achievements

### Compiler Capabilities Unlocked

**Phase 4 Complete - Now Compiles:**
```fsharp
// 1. Closures with capture (Chapter 12)
let makeAdder n = fun x -> x + n

// 2. Higher-order functions (Chapter 13)
let apply f x = f x
let compose f g x = f (g x)

// 3. Partial application
let add5 = makeAdder 5
let result = add5 10   // 15

// 4. Function composition
let inc x = x + 1
let double x = x * 2
let incThenDouble = compose double inc
let result2 = incThenDouble 5   // 12

// 5. Currying
let add x y = x + y   // = fun x -> fun y -> x + y
let add10 = add 10
```

**Functional Programming Core Complete:**
- ✅ Closures (environment capture)
- ✅ Higher-order functions (functions as arguments)
- ✅ Functions as return values
- ✅ Partial application
- ✅ Function composition
- ✅ Currying
- ⏸️ Map/filter/fold (Phase 6 with lists)

### Memory Safety Guarantees

**1. No Dangling Pointers:**
- All escaping closures heap-allocated
- GC manages lifetime automatically
- Safe to return closures from functions

**2. Cyclic Closure Handling:**
- Mutual recursion creates cycles
- Boehm GC (tracing) handles correctly
- No memory leaks from cycles

**3. Automatic Cleanup:**
- Programmer never calls `free`
- GC reclaims unused closures
- Environment deallocated when closure unreachable

## Common Errors Documented

1. **Error 1:** Forgetting to pass closure as first argument
   - Symptom: Type mismatch or segfault
   - Fix: `llvm.call %fn_ptr(%closure, %args)`

2. **Error 2:** Direct call to lifted function body
   - Symptom: Wrong environment passed
   - Fix: Use indirect call through closure

3. **Error 3:** Stack allocation for escaping closures
   - Symptom: Dangling pointer after function return
   - Fix: Always use `GC_malloc` for environments

4. **Error 4:** Type mismatch in indirect call
   - Symptom: Verification error or crash
   - Fix: Ensure call signature matches lifted function

5. **Error 5:** Closure identity confusion
   - Symptom: Expecting equal closures with same behavior
   - Fix: Closures compared by pointer, not behavior

## Deviations from Plan

None - plan executed exactly as written.

Three tasks completed atomically:
1. Chapter 13 Part 1 (functions as arguments) - 754 lines
2. Chapter 13 Part 2 (functions as return values) - 864 lines
3. SUMMARY.md update with Phase 4 chapters

All verification criteria passed:
- ✅ 1618 lines (exceeds 1500+ requirement)
- ✅ 13+ "고차 함수" mentions
- ✅ 17+ "llvm.call" examples
- ✅ 69+ apply/compose mentions
- ✅ 47+ makeAdder references
- ✅ 20+ GC_malloc mentions
- ✅ 6+ Error section entries
- ✅ 6+ upward/escape mentions
- ✅ Korean plain style (~이다/~한다)
- ✅ Technical terms in English

## Next Phase Readiness

### Phase 5: Custom MLIR Dialect

**Motivation:** Phase 4 generates low-level LLVM dialect directly (GEP, load, store). Phase 5 abstracts this with FunLang-specific operations.

**Planned Operations:**
- `funlang.closure` - abstract closure creation
- `funlang.closure_call` - abstract indirect call
- `funlang.capture` - abstract environment storage

**Benefits:**
1. **Simplified compiler code** - high-level operations vs manual GEP/load/store
2. **Optimization opportunities** - dialect-specific transformation passes
3. **Type safety** - custom dialect type system

**Preview Comparison:**

```mlir
// Phase 4 (low-level, verbose)
%env = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @func : !llvm.ptr
%slot0 = llvm.getelementptr %env[0, 0] : (!llvm.ptr, i64) -> !llvm.ptr
llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
// ... more stores ...

// Phase 5 (high-level, concise)
%closure = funlang.closure @func, %captured_vars : (i32) -> i32
```

**Phase 5 will NOT add new language features** - it refactors compiler implementation for better maintainability and optimization.

### Prerequisites for Phase 5

- ✅ Phase 4 complete (all closure/HOF patterns established)
- ✅ C++ dialect registration knowledge (Appendix exists)
- ⏸️ Lowering pass design (Phase 5 work)
- ⏸️ Dialect definition in TableGen (Phase 5 work)

No blockers. Ready to proceed.

## Lessons Learned

### What Worked Well

1. **Incremental chapter structure** - Part 1 (arguments) then Part 2 (return values) created natural progression
2. **Concrete examples first** - apply, compose, makeAdder before abstract theory
3. **Error documentation** - Common Errors section anticipates reader problems
4. **MLIR examples** - 21+ function definitions show pattern clearly
5. **Phase completion summary** - Explicit "what we can compile now" checklist

### What Could Be Improved

1. **Map function** - Only conceptual in Phase 4; full implementation requires Phase 6 (lists)
   - Decision: Correct to defer; lists are prerequisite
2. **Performance discussion** - Briefly mentioned but not quantified
   - Future work: Benchmarks in Phase 7 (optimization)
3. **Escape analysis preview** - Mentioned but not detailed
   - Future work: Phase 7 will cover stack allocation optimization

### Technical Insights

1. **Uniform closure representation simplifies compiler**
   - Alternative (separate representations) adds complexity for marginal gains
   - Phase 7 can optimize uniform representation via specialization

2. **GC solves upward funarg elegantly**
   - Alternative (manual memory management) error-prone and unsafe
   - Boehm GC's tracing handles cyclic closures correctly

3. **Indirect calls unavoidable for HOF**
   - Performance cost acceptable for flexibility
   - Future: devirtualization can recover performance in common cases

## Phase 4 Complete

**Tutorial chapters written:** 2 (Chapter 12, Chapter 13)
**Total lines:** 3136 (1518 + 1618)
**Concepts covered:**
- Closures (lexical scoping, free variables, closure conversion)
- Environment structures (heap allocation, GC lifetime)
- Higher-order functions (functions as arguments/return values)
- Indirect calls (function pointers, runtime dispatch)
- Currying (nested closures, partial application)
- Memory safety (upward funarg, GC tracking, cyclic references)

**FunLang now supports:**
- Complete functional programming core
- Safe closure creation and usage
- Higher-order function patterns (apply, compose, map, filter concepts)
- Automatic memory management for closures

**Ready for:** Phase 5 (Custom MLIR Dialect implementation)

---

**Phase 4 Goal Achieved:** FunLang is now a complete functional language with closures and higher-order functions. Phase 5 will refactor the compiler internals for better maintainability, but the language capabilities are feature-complete at the functional programming core level.
