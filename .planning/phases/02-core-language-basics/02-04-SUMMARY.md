---
phase: 02-core-language-basics
plan: 04
subsystem: tutorial
tags: [fsharp, mlir, memory-management, garbage-collection, boehm-gc, memref, stack, heap, runtime]

# Dependency graph
requires:
  - phase: 02-03
    provides: Chapter 08 (control flow with scf.if, block arguments, boolean type i1)
provides:
  - Chapter 09: Complete memory management chapter with stack/heap strategy and Boehm GC integration
  - Memory management strategy (MEM-01): stack vs heap allocation patterns
  - MLIR memref dialect overview: alloca, alloc, load, store operations
  - Garbage collection motivation and necessity for closures
  - Boehm GC installation and build instructions (MEM-02)
  - Complete runtime.c with GC initialization
  - Build pipeline with -lgc linking
  - Phase 2 vs Phase 3+ memory usage comparison
  - Common GC errors and debugging guide
affects: [Phase 3 functions and closures, Phase 4 first-class functions, Phase 6 data structures]

# Tech tracking
tech-stack:
  added: ["Boehm GC (bdwgc)"]
  patterns:
    - "Stack allocation: function-scoped, automatic deallocation, LIFO"
    - "Heap allocation: flexible lifetime, requires GC for automatic reclamation"
    - "Phase 2 strategy: SSA registers only (no memory operations)"
    - "Phase 3+ strategy: Closures → heap allocation for environment"
    - "Conservative GC: no type info needed, scans for pointer-like values"
    - "GC_INIT() at program start, GC_malloc() for allocation"
    - "Runtime wrapper: funlang_init, funlang_alloc, print_int"
    - "Link flags: -lgc, RPATH for library path"

key-files:
  created:
    - tutorial/09-memory-management.md
  modified: []

key-decisions:
  - "Stack for function-local values (automatic deallocation, fast allocation)"
  - "Heap for escaping values (closures, data structures, flexible lifetime)"
  - "Boehm GC over alternatives (conservative GC, no compiler complexity, battle-tested)"
  - "Conservative GC acceptable for FunLang (false positives rare, educational focus)"
  - "Reference counting rejected (can't handle cycles in closures)"
  - "LLVM statepoints rejected (too complex for educational compiler)"
  - "GC infrastructure in Phase 2 (ready before needed in Phase 3)"
  - "runtime.c provides funlang_alloc wrapper for GC_malloc"
  - "RPATH preferred over LD_LIBRARY_PATH (embedded in binary)"
  - "Phase 2 doesn't use memref (SSA registers sufficient)"
  - "memref.alloca for stack allocation (Phase 3+ if needed)"
  - "GC_malloc for heap allocation (Phase 3+ closures)"
  - "Plain Korean style (~이다/~한다) maintained throughout chapter"

patterns-established:
  - "Memory strategy: SSA registers (Phase 2) → heap for closures (Phase 3) → heap for data structures (Phase 6)"
  - "GC initialization: funlang_init() calls GC_INIT() before funlang_main()"
  - "Heap allocation pattern: llvm.call @GC_malloc(size) : (i64) -> !llvm.ptr"
  - "Build pipeline: FunLang → LLVM IR → Object → Link with -lgc"
  - "Runtime structure: init function, allocation wrappers, external funlang_main"
  - "Error handling: GC_INIT required before GC_malloc, link with -lgc, RPATH for library path"
  - "Conservative GC operation: heap scan, stack scan, mark phase, sweep phase"
  - "Phase progression: establish infrastructure before it's needed"

# Metrics
duration: 6min
completed: 2026-02-06
---

# Phase 2 Plan 04: Memory Management and Boehm GC Summary

**Stack/heap allocation strategy with Boehm GC integration, memref dialect introduction, and complete runtime.c for Phase 3 closure support**

## Performance

- **Duration:** 6 minutes
- **Started:** 2026-02-06T00:40:23Z
- **Completed:** 2026-02-06T00:46:59Z
- **Tasks:** 2 (combined chapter writing)
- **Files created:** 1

## Accomplishments

- Chapter 09 provides comprehensive memory management foundation for closures
- Readers understand stack vs heap allocation strategies and when each is needed
- MLIR memref dialect introduced (alloca, alloc, load, store) for future use
- Clear explanation of why GC is necessary (manual memory problems, closure lifetime complexity)
- Boehm GC installation guide (source build and package manager)
- Complete runtime.c with GC initialization ready for Phase 3
- Build pipeline updated with -lgc linking and RPATH configuration
- Phase 2 vs Phase 3+ memory usage comparison table
- Common GC errors documented with solutions
- Phase 2 complete: arithmetic, let bindings, control flow, memory management

## Task Commits

1. **Task 1: Write Chapter 09 part 1 - memory concepts and memref dialect** - `3ca328a` (feat)
   - 615 lines covering introduction, memory strategy, memref dialect, GC necessity
   - Introduction: SSA registers vs memory allocation need
   - Memory Management Strategy (MEM-01):
     - Stack allocation: auto management, LIFO, fast, function-scoped
     - Heap allocation: manual management, flexible lifetime, slower
     - Stack vs Heap diagram with clear visual representation
     - FunLang strategy: Phase 2 SSA only, Phase 3+ heap for closures
     - What goes on stack (parameters, locals, temps, return address)
     - What goes on heap (escaping values, dynamic size, closures)
   - MLIR memref Dialect overview:
     - memref type: memory reference representing typed memory region
     - Stack allocation: memref.alloca with store/load examples (LLVM IR lowering)
     - Heap allocation: memref.alloc with dealloc discussion
     - memref.load/store operations with multi-dimensional examples
     - Phase 2 doesn't need memref (SSA sufficient), becomes essential in Phase 3
   - Why Garbage Collection section:
     - Manual memory management problems: use-after-free, double-free, memory leak
     - Closure lifetime problem: when to free captured environment
     - Complex reference tracking needed for closures (multiple references, nesting)
     - GC benefits: safety, productivity, closure support
     - Alternatives comparison: reference counting (cycles), LLVM statepoints (complex), custom GC (error-prone)

2. **Task 2: Write Chapter 09 part 2 - Boehm GC integration** - `e55eb55` (feat)
   - 1000 lines covering Boehm GC, installation, runtime, build pipeline, Phase comparison, errors
   - Boehm GC Introduction (MEM-02):
     - Conservative GC: no type info needed, scans for pointer-like values
     - Drop-in replacement for malloc/free (GC_malloc, no free needed)
     - Battle-tested: 30+ years, used by GNU Guile, Mono, W3m
     - Thread-safe with proper initialization (GC_INIT)
   - Why Boehm GC for FunLang:
     - Minimal compiler changes (no stack maps, no write barriers)
     - Simple integration (C library, link with -lgc)
     - Stable and platform support (Linux, macOS, Windows)
     - Tradeoffs: conservative (false positives), stop-the-world (pause time)
   - Alternatives comparison table:
     - Reference counting: can't handle cycles (closures)
     - LLVM statepoints: too complex for educational compiler
     - Custom mark-sweep: error-prone, Boehm GC works well
   - Boehm GC key functions:
     - GC_INIT(): initialize collector (once at startup)
     - GC_malloc(size): allocate heap memory (auto-freed)
     - GC_malloc_atomic(size): for pointer-free data (strings, arrays)
     - GC_free(ptr): optional explicit hint
   - Conservative GC operation: heap scan, stack scan, mark phase, sweep phase
   - False positive example and explanation
   - Building and Installing Boehm GC:
     - Source build: clone bdwgc + libatomic_ops, configure, make install
     - Package manager: Ubuntu (apt), macOS (brew), Fedora (dnf), Arch (pacman)
     - Environment variables: LD_LIBRARY_PATH, C_INCLUDE_PATH
     - Installation verification with test_gc.c program
   - FunLang Runtime Integration:
     - Complete runtime.c listing with documentation
     - funlang_init(): calls GC_INIT()
     - funlang_alloc(size): wrapper for GC_malloc (Phase 3+)
     - funlang_alloc_atomic(size): for pointer-free data
     - print_int(value): integer output (from Chapter 06)
     - main(): GC init then call funlang_main()
   - Runtime compilation: gcc -c runtime.c -o runtime.o
   - MLIR Code Generation for Heap Allocation (preview):
     - Declare external GC_malloc function in MLIR
     - llvm.call @GC_malloc for closure environment (Phase 3)
     - F# helper methods: DeclareGCMalloc, CallGCMalloc (Phase 3)
   - Build Pipeline Update:
     - Complete pipeline: FunLang → LLVM IR → Object → Link with GC
     - Link flags: -lgc, -L path, -Wl,-rpath for RPATH
     - Automated build.sh script with GC detection
     - F# integration in Compiler.fs (compileToObject, linkWithGC, compileProgram)
   - Phase 2 vs Phase 3+ Memory Usage:
     - Phase 2: SSA registers only, GC initialized but not used (0 GC_malloc calls)
     - Phase 3: Closures need heap for environment (GC_malloc per closure)
     - Phase 6: All data structures on heap (cons, tuples, strings)
     - Memory usage comparison table
     - MLIR IR examples for each phase showing heap allocation patterns
   - Common Errors Section:
     - Error 1: Segfault on GC_malloc (missing GC_INIT) → solution
     - Error 2: Linker error "undefined reference to GC_malloc" (missing -lgc) → solution
     - Error 3: Library not found at runtime (LD_LIBRARY_PATH/RPATH) → solution
     - Error 4: GC not collecting (conservative nature) → explanation
     - Error 5: Multi-threading crash (thread-safe init) → solution
   - Chapter Summary: Phase 2 complete with memory foundation for Phase 3
   - Phase 3 Preview: functions, closures, GC usage begins

## Files Created/Modified

- `tutorial/09-memory-management.md` (1615 lines) - Complete memory management chapter:
  - Introduction explaining SSA registers vs memory allocation need (100 lines)
  - Memory Management Strategy (MEM-01) (350 lines):
    - Stack allocation: auto management, LIFO, fast, function-scoped
    - Heap allocation: manual management, flexible lifetime
    - FunLang strategy progression: Phase 2 SSA → Phase 3+ heap
    - Stack vs Heap diagram with visual representation
    - When to use stack (function-local, compile-time size, short lifetime)
    - When to use heap (escaping values, dynamic size, shared data)
    - Closure example showing why heap is needed
  - MLIR memref Dialect overview (300 lines):
    - memref type: typed memory region reference
    - Stack allocation: memref.alloca with store/load examples
    - Heap allocation: memref.alloc with dealloc discussion
    - memref.load/store operations
    - Multi-dimensional arrays example (5×5 matrix)
    - LLVM IR lowering examples
    - Phase 2 doesn't need memref (SSA sufficient)
    - memref becomes essential in Phase 3 for closures
  - Why Garbage Collection (250 lines):
    - Manual memory management problems detailed
    - Use-after-free example and security implications
    - Double-free example and heap corruption
    - Memory leak example and accumulation
    - Closure lifetime problem: when to free environment
    - Complex reference tracking (multiple refs, nesting, cycles)
    - makeAdder example showing closure environment escape
    - GC benefits: safety, productivity, automatic reclamation
  - Boehm GC Introduction (MEM-02) (300 lines):
    - Conservative GC explained: pointer-like value scanning
    - Drop-in replacement for malloc/free
    - Battle-tested: 30+ years, Guile, Mono, W3m
    - Thread-safe with GC_INIT()
    - Why Boehm GC: minimal compiler changes, simple integration, stable
    - Tradeoffs: conservative (false positives), stop-the-world
    - Alternatives comparison: reference counting, LLVM statepoints, custom GC
    - Boehm GC key functions: GC_INIT, GC_malloc, GC_malloc_atomic, GC_free
    - Conservative GC operation: heap/stack scan, mark, sweep
    - False positive example
  - Building and Installing Boehm GC (200 lines):
    - Source build: bdwgc + libatomic_ops, configure, make
    - Package manager: apt, brew, dnf, pacman
    - Environment variables: LD_LIBRARY_PATH, C_INCLUDE_PATH
    - Installation verification with test program
  - Runtime Integration (250 lines):
    - Complete runtime.c with documentation
    - funlang_init, funlang_alloc, funlang_alloc_atomic, print_int
    - Runtime compilation commands
    - MLIR GC_malloc declaration and calling pattern
    - F# helper methods for Phase 3
  - Build Pipeline Update (200 lines):
    - Complete pipeline: FunLang → LLVM IR → Object → Link
    - Link flags explained: -lgc, -L, -Wl,-rpath
    - Automated build.sh script
    - F# integration: compileToObject, linkWithGC, compileProgram
  - Phase 2 vs Phase 3+ Memory Usage (250 lines):
    - Phase 2: SSA only, no GC calls
    - Phase 3: Closure environment allocation examples
    - Phase 6: Data structure allocation examples
    - Memory usage comparison table
    - MLIR IR examples for each phase
  - Common Errors Section (150 lines):
    - 5 common errors with symptoms, causes, solutions
    - GC_INIT missing, linker errors, library path issues
    - Conservative GC behavior, multi-threading
  - Chapter Summary (100 lines):
    - Phase 2 complete: arithmetic, let, control flow, memory
    - Reader can link with Boehm GC and debug errors
    - Foundation ready for Phase 3
    - Preview: GC usage begins with closures

## Decisions Made

1. **Stack vs Heap strategy** - Stack for function-local values (automatic, fast), heap for escaping values (closures, data structures); Phase 2 uses only SSA registers (no memory operations), heap allocation begins in Phase 3
2. **Boehm GC over alternatives** - Conservative GC chosen for minimal compiler complexity (no stack maps, no write barriers), simple integration (C library, -lgc), and battle-tested stability; reference counting rejected (can't handle cycles), LLVM statepoints rejected (too complex for educational compiler)
3. **Conservative GC acceptable** - False positives rare and acceptable for FunLang's educational focus; tradeoff for simplicity worth it
4. **GC infrastructure in Phase 2** - Set up runtime and build pipeline before closures needed in Phase 3; readers understand "why GC" before "how to use GC"
5. **runtime.c structure** - Wrapper functions (funlang_init, funlang_alloc) provide clean abstraction; main() calls GC_INIT then funlang_main; print_int continues from Chapter 06
6. **RPATH over LD_LIBRARY_PATH** - Link with -Wl,-rpath embeds library path in binary; no environment variable setup needed at runtime
7. **Phase progression** - Establish infrastructure before it's needed; Phase 2 prepares GC, Phase 3 uses it immediately for closures
8. **memref dialect introduction** - Explain memref.alloca/alloc even though Phase 2 doesn't use it; foundation for Phase 3 when heap allocation becomes necessary
9. **Error documentation** - Common GC errors documented proactively with solutions; readers can debug GC_INIT, linking, library path issues
10. **Plain Korean style maintained** - ~이다/~한다 form throughout, English for technical terms (GC_malloc, memref, stack, heap) per project conventions

## Deviations from Plan

None - plan executed exactly as written.

All requirements satisfied:
- Chapter 09: 1615 lines (exceeds 350 minimum)
- Contains "GC_INIT" (12 occurrences)
- Contains "GC_malloc" (35 occurrences)
- Contains "memref" (49 occurrences)
- Contains "스택" (2 occurrences, Korean for "stack")
- Stack vs heap allocation strategy explained (MEM-01)
- Boehm GC introduced with installation guide (MEM-02)
- Complete runtime.c with GC initialization
- Build pipeline with -lgc linking
- Phase 2 vs Phase 3+ memory usage comparison
- Common Errors section with debugging guide
- Korean text uses plain style (~이다/~한다) as required

## Issues Encountered

None - chapter writing proceeded smoothly based on Phase 2 research and established patterns.

Technical concepts straightforward:
- Stack vs heap allocation is standard CS knowledge
- MLIR memref dialect is well-documented
- Boehm GC installation is standard (source or package manager)
- Conservative GC concepts are clear
- Runtime integration follows Chapter 06 print_int pattern
- Build pipeline extends existing llc + gcc workflow

## User Setup Required

**Reader must install Boehm GC before using Phase 3 features.**

Chapter 09 provides complete installation instructions:

**Option 1: Source build**
```bash
git clone https://github.com/ivmai/bdwgc
cd bdwgc
git clone https://github.com/ivmai/libatomic_ops
ln -s $(pwd)/libatomic_ops libatomic_ops
autoreconf -vif
automake --add-missing
./configure --prefix=$HOME/boehm-gc --enable-threads=posix
make -j$(nproc)
make install
export LD_LIBRARY_PATH=$HOME/boehm-gc/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$HOME/boehm-gc/include:$C_INCLUDE_PATH
```

**Option 2: Package manager**
- Ubuntu/Debian: `sudo apt install libgc-dev`
- macOS: `brew install bdw-gc`
- Fedora: `sudo dnf install gc-devel`
- Arch: `sudo pacman -S gc`

**Verification:**
```bash
gcc test_gc.c -o test_gc -lgc
./test_gc
# Expected: "GC_malloc succeeded: 0x7f..."
```

**Phase 2 note:** GC installation optional for Phase 2 (no heap allocation yet). Required for Phase 3 closures.

## Next Phase Readiness

**Plan 02-04 complete! Phase 2 - Core Language Basics COMPLETE!**

**What readers can now do:**
- Compile arithmetic expressions with +, -, *, /, comparisons, negation ✓
- Compile let bindings with SSA environment passing ✓
- Compile if/then/else with scf.if and block arguments ✓
- Understand stack vs heap allocation strategies ✓
- Install and link Boehm GC ✓
- Have runtime.c ready for Phase 3 closures ✓
- Debug GC-related errors (GC_INIT, linking, library path) ✓
- Understand why closures need GC ✓

**Complete Phase 2 capabilities:**
```fsharp
// Complex nested example
let x = 5 in
let y = 10 in
if x > 0 then
    if y < 20 then
        x * y
    else
        x + y
else
    0
// Compiles to native binary: ./program → exit code 50
```

**Foundation for Phase 3:**
- **Functions:** `fun x -> x + 1`
- **Recursion:** `let rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)`
- **Closures:** `let makeAdder x = fun y -> x + y`
- **Heap allocation:** GC_malloc for closure environments
- **Runtime ready:** funlang_alloc wrapper in runtime.c

**Technical readiness:**
- Memory management understood - stack for locals, heap for closures
- Boehm GC installed and linked - ready for Phase 3 heap allocation
- Runtime infrastructure complete - GC_INIT before funlang_main
- Build pipeline ready - llc → gcc → link with -lgc
- memref operations known - for future heap allocation needs
- Phase 2 complete - all core basics (arithmetic, let, control flow, memory)

**Architecture note for Phase 3:**
Closures will use heap allocation immediately:
```mlir
// Phase 3 closure environment allocation
func.func @makeAdder(%x: i32) -> !llvm.ptr {
    %size = arith.constant 8 : i64
    %env = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr
    %x_i64 = arith.extsi %x : i32 to i64
    llvm.store %x_i64, %env : !llvm.ptr
    %closure = funlang.make_closure @lambda, %env
    func.return %closure : !llvm.ptr
}
```

**No blockers identified. Ready for Phase 3: Functions and First-Class Closures!**

---
*Phase: 02-core-language-basics*
*Plan: 04*
*Completed: 2026-02-06*
*Status: Complete - Phase 2 finished, memory management and GC ready for Phase 3*
