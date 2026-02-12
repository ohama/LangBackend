---
phase: 07-foundation-infrastructure
plan: 02
subsystem: infra
tags: [mlir, pinvoke, fsharp, dotnet, bindings, types, operations]

# Dependency graph
requires:
  - phase: 07-foundation-infrastructure
    plan: 01
    provides: F# project structure with handle types and basic bindings
provides:
  - Complete MLIR-C API P/Invoke bindings for type system
  - Complete MLIR-C API P/Invoke bindings for operation building
  - Complete MLIR-C API P/Invoke bindings for region/block management
  - Complete MLIR-C API P/Invoke bindings for value and attribute handling
  - MlirOperationState struct with Sequential layout
affects: [07-03, 07-04, 08-core-expressions, 09-functions, 10-closures, 11-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [MlirOperationState with mutable fields, nativeint for array pointers, CallingConvention.Cdecl]

key-files:
  created: []
  modified:
    - src/FunLang.Compiler/MlirBindings.fs

key-decisions:
  - "MlirOperationState uses nativeint for array pointers (Results, Operands, Regions, Successors, Attributes)"
  - "All struct fields in MlirOperationState are mutable for modification before mlirOperationCreate"
  - "Function type creation uses nativeint parameters for flexible array handling"

patterns-established:
  - "Operation building: state creation, mutation, then mlirOperationCreate(&state)"
  - "Region/block ownership: append/insert functions transfer ownership"
  - "All P/Invoke declarations organized by functional category"

# Metrics
duration: 2min
completed: 2026-02-12
---

# Phase 7 Plan 02: Complete MLIR P/Invoke Bindings Summary

**Complete MLIR-C API bindings covering types, operations, regions, blocks, values, and attributes with 60+ function declarations**

## Performance

- **Duration:** 2 min 9 sec
- **Started:** 2026-02-12T03:54:49Z
- **Completed:** 2026-02-12T03:56:58Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

### Type System (Task 1)
- Integer types: mlirIntegerTypeGet, mlirIntegerTypeSignedGet, mlirIntegerTypeUnsignedGet
- Index type: mlirIndexTypeGet
- Float types: mlirF32TypeGet, mlirF64TypeGet
- Function type: mlirFunctionTypeGet (using nativeint for array parameters)
- LLVM pointer type: mlirLLVMPointerTypeGet
- Type queries: mlirTypeIsNull, mlirTypeEqual

### Operation Building (Task 2)
- MlirOperationState struct with StructLayout(LayoutKind.Sequential)
- 13 mutable fields for operation construction
- mlirOperationStateGet
- mlirOperationStateAddResults/Operands/OwnedRegions/Attributes
- mlirOperationCreate (accepts MlirOperationState& parameter)
- mlirOperationDestroy
- mlirOperationGetResult/NumResults/Region/Block

### Region, Block, Value, Attribute (Task 3)
- Region: Create, Destroy, AppendOwnedBlock, GetFirstBlock
- Block: Create, Destroy, AppendOwnedOperation, InsertOwnedOperationBefore, GetArgument, GetNumArguments, GetTerminator
- Value: IsNull, GetType, Equal
- Attribute: IntegerAttrGet, FloatAttrDoubleGet, StringAttrGet, UnitAttrGet, FlatSymbolRefAttrGet, NamedAttributeGet, IsNull

### Overall
- 444 lines of complete P/Invoke bindings
- 60+ DllImport declarations
- All functions use CallingConvention.Cdecl
- 3 structs with StructLayout(Sequential): MlirStringRef, MlirNamedAttribute, MlirOperationState
- Complete coverage of MLIR-C API categories needed for compiler

## Task Commits

Each task was committed atomically:

1. **Task 1: Type system P/Invoke declarations** - `b6999fc` (feat)
2. **Task 2: Operation building P/Invoke declarations** - `08c59af` (feat)
3. **Task 3: Region, block, value, attribute P/Invoke declarations** - `ef1746d` (feat)

## Files Created/Modified

- `src/FunLang.Compiler/MlirBindings.fs` - Added 213 lines (444 total)
  - Type system functions (11 declarations)
  - MlirOperationState struct (13 fields)
  - Operation building functions (10 declarations)
  - Region management functions (4 declarations)
  - Block management functions (7 declarations)
  - Value functions (3 declarations)
  - Attribute functions (7 declarations)

## Decisions Made

**1. nativeint for array pointers in MlirOperationState**
- Rationale: C API uses pointer-to-array parameters (MlirType*, MlirValue*, etc.). Using nativeint allows flexible marshalling - caller can use Marshal.AllocHGlobal for arrays or pin F# arrays.
- Impact: Requires manual memory management but provides maximum flexibility.

**2. mutable fields in MlirOperationState**
- Rationale: MLIR operation building pattern requires constructing state, then modifying it with AddResults/AddOperands/etc., then passing to mlirOperationCreate. Immutable struct would prevent this pattern.
- Impact: Follows MLIR C API design exactly, enables correct operation construction.

**3. MlirFunctionTypeGet uses nativeint for array parameters**
- Rationale: Consistent with operation state design. Allows passing arrays of MlirType without complex marshalling attributes.
- Impact: Caller responsibility to allocate/free, but provides clean interop.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all builds succeeded with 0 warnings, 0 errors.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 7 Plan 03 (MLIR Builder Abstraction):**
- Complete P/Invoke layer provides all needed MLIR-C API access
- Type system functions available for creating i32, i64, f32, f64, function types
- Operation building infrastructure complete (state, create, query)
- Region/block management ready for control flow graphs
- Value and attribute handling ready for SSA and metadata
- All bindings follow CallingConvention.Cdecl standard
- MlirOperationState enables safe operation construction

**No blockers.**

---
*Phase: 07-foundation-infrastructure*
*Completed: 2026-02-12*
