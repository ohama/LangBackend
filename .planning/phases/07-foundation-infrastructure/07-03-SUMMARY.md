---
phase: 07-foundation-infrastructure
plan: 03
subsystem: infra
tags: [mlir, wrapper, idisposable, fsharp, opbuilder, fluent-api]

# Dependency graph
requires:
  - phase: 07-foundation-infrastructure
    plan: 02
    provides: Complete MLIR-C API P/Invoke bindings
provides:
  - IDisposable Context wrapper with dialect loading
  - Location discriminated union for diagnostics
  - IDisposable Module wrapper with parent reference
  - OpBuilder fluent API for operation construction
affects: [07-04, 08-core-expressions, 09-functions, 10-closures, 11-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [IDisposable for automatic resource cleanup, parent reference to prevent premature GC, fluent API for complex operations]

key-files:
  created:
    - src/FunLang.Compiler/MlirWrapper.fs
  modified:
    - src/FunLang.Compiler/FunLang.Compiler.fsproj

key-decisions:
  - "Context implements IDisposable with mutable handle and disposed flag"
  - "Location is discriminated union (does NOT implement IDisposable - value type owned by Context)"
  - "Module stores contextRef field to keep parent Context alive (prevents premature GC)"
  - "OpBuilder uses NativePtr.toNativeInt to convert fixed array pointers to nativeint for P/Invoke"
  - "Warnings about unverifiable IL code are expected and acceptable for P/Invoke interop with fixed pointers"

patterns-established:
  - "IDisposable pattern: mutable handle, disposed flag, check disposed in methods"
  - "Parent reference pattern: child stores reference to prevent parent GC"
  - "Fluent API pattern: type helpers, attribute helpers, generic CreateOperation"
  - "String marshalling pattern: MlirStringRef.WithString for automatic allocation/cleanup"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 7 Plan 03: MLIR Builder Abstraction Summary

**IDisposable wrappers (Context, Module) and fluent OpBuilder API provide safe, convenient MLIR IR construction**

## Performance

- **Duration:** 3 min 0 sec
- **Started:** 2026-02-12T04:13:16Z
- **Completed:** 2026-02-12T04:16:07Z
- **Tasks:** 3
- **Files created:** 1
- **Files modified:** 1

## Accomplishments

### Context and Location (Task 1)
- **Context:** IDisposable wrapper with automatic resource cleanup
  - mlirContextCreate in constructor
  - mlirContextDestroy in Dispose
  - Mutable handle and disposed flag
  - LoadDialect(dialectName) method
  - LoadStandardDialects() convenience method (loads func, arith, scf, llvm)
- **Location:** Discriminated union for diagnostics
  - Unknown(Context) case
  - FileLineCol(Context, filename, line, col) case
  - Handle property returns raw MlirLocation
  - Context property returns associated context
  - No IDisposable (value type owned by Context)

### Module Wrapper (Task 2)
- **Module:** IDisposable wrapper for MLIR modules
  - Constructor accepts Context and Location
  - mlirModuleCreateEmpty in constructor
  - mlirModuleDestroy in Dispose
  - **CRITICAL:** contextRef field keeps parent Context alive
  - Context property returns contextRef
  - Body property returns module body block
  - Operation property returns module operation
  - Print() method marshals IR to string using StringBuilder and callback

### OpBuilder Fluent API (Task 3)
- **Type helpers:**
  - I32Type(), I64Type(), I1Type() - integer types
  - IndexType() - platform-dependent index type
  - PtrType() - LLVM opaque pointer type
  - FunctionType(inputs, results) - function type with array marshalling
- **Attribute helpers:**
  - IntegerAttr(value, type)
  - StringAttr(value) - uses MlirStringRef.WithString
  - SymbolRefAttr(name) - uses MlirStringRef.WithString
  - NamedAttr(name, attr) - uses MlirStringRef.WithString
- **Operation helpers:**
  - GetResult(op, index)
  - CreateBlock(argTypes, location)
  - CreateRegion()
  - AppendBlockToRegion(region, block)
  - AppendOperationToBlock(block, op)
- **Generic operation creation:**
  - CreateOperation(name, location, resultTypes, operands, attributes, regions)
  - Wraps MlirOperationState complexity
  - Uses fixed/NativePtr.toNativeInt for array marshalling

### Overall
- 221 lines of wrapper code
- 4 types: Context (class), Location (discriminated union), Module (class), OpBuilder (class)
- All IDisposable types follow F# dispose pattern
- Module prevents parent Context GC through contextRef field
- OpBuilder provides fluent API for type/attribute/operation construction

## Task Commits

Each task was committed atomically:

1. **Task 1: Context and Location wrappers** - `cba6f0a` (feat)
2. **Task 2: Module wrapper with parent reference** - `c36fef3` (feat)
3. **Task 3: OpBuilder fluent API** - `a923ac1` (feat)

## Files Created/Modified

- `src/FunLang.Compiler/MlirWrapper.fs` - Created (221 lines)
  - Context wrapper (28 lines)
  - Location discriminated union (20 lines)
  - Module wrapper (44 lines)
  - OpBuilder fluent API (115 lines)
- `src/FunLang.Compiler/FunLang.Compiler.fsproj` - Modified
  - Added MlirWrapper.fs to ItemGroup

## Decisions Made

**1. Context IDisposable pattern**
- Rationale: MLIR contexts are heavyweight resources that must be explicitly destroyed. F# IDisposable pattern enables automatic cleanup with `use` keyword.
- Impact: Developers can write `use ctx = new Context()` and context is automatically destroyed when scope exits.

**2. Location as discriminated union (NOT IDisposable)**
- Rationale: MLIR locations are lightweight value types owned by the context. They don't need manual cleanup.
- Impact: Simpler API - no disposal required. Location lifetime managed by parent Context.

**3. Module parent reference (contextRef)**
- Rationale: F# GC could collect Context while Module still uses it. Storing reference prevents premature collection.
- Impact: Module keeps Context alive as long as Module exists. Prevents use-after-free bugs.

**4. NativePtr.toNativeInt for array marshalling**
- Rationale: P/Invoke signatures use nativeint for array pointers. Fixed expressions return nativeptr<T>. Conversion needed.
- Impact: Explicit conversion adds clarity. Warnings about unverifiable IL are expected for P/Invoke.

**5. OpBuilder fluent API design**
- Rationale: MLIR operation construction is verbose (create state, add results, add operands, add attributes, create operation). Fluent API simplifies.
- Impact: Later compiler phases can write `builder.CreateOperation(...)` instead of manual state management.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Issue 1: F# static member tuple syntax**
- Symptom: `MlirStringRef.WithString dialectName (fun ...)` failed to compile
- Root cause: F# static members use tuple syntax for arguments, not curried syntax
- Fix: Changed to `MlirStringRef.WithString(dialectName, fun ...)`

**Issue 2: nativeptr to nativeint conversion**
- Symptom: `fixed inputs` returns `nativeptr<MlirType>`, but P/Invoke expects `nativeint`
- Root cause: Type mismatch between F# fixed expression and P/Invoke signature
- Fix: Added `open FSharp.NativeInterop` and used `NativePtr.toNativeInt`

**Issue 3: FS0009 warnings about unverifiable IL**
- Symptom: 16 warnings about "may result in unverifiable .NET IL code"
- Root cause: Using `fixed` keyword for array pinning generates unverifiable IL
- Resolution: Expected and acceptable - standard for P/Invoke interop with native code

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 7 Plan 04 (MLIR Smoke Tests):**
- Context wrapper enables automatic resource cleanup via `use` keyword
- Module wrapper enables safe module construction with parent reference
- OpBuilder provides fluent API for creating types, attributes, operations
- All wrappers follow F# IDisposable pattern
- Location provides Unknown and FileLineCol diagnostics
- Print() method enables IR inspection for debugging
- Ready to test end-to-end: create context → load dialects → create module → build operations → print IR

**No blockers.**

---
*Phase: 07-foundation-infrastructure*
*Completed: 2026-02-12*
