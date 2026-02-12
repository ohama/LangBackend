namespace FunLang.Compiler

open System
open System.Runtime.InteropServices

//=============================================================================
// Handle Types
//=============================================================================

/// MLIR context - manages dialects, types, and global state
[<Struct>]
type MlirContext =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR module - top-level container for functions and global data
[<Struct>]
type MlirModule =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR operation - fundamental IR unit (instructions, functions, etc.)
[<Struct>]
type MlirOperation =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR type - represents value types (i32, f64, pointers, etc.)
[<Struct>]
type MlirType =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR location - source code location for diagnostics
[<Struct>]
type MlirLocation =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR region - contains a list of blocks
[<Struct>]
type MlirRegion =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR block - basic block containing a sequence of operations
[<Struct>]
type MlirBlock =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR value - SSA value produced by an operation
[<Struct>]
type MlirValue =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR attribute - compile-time constant metadata
[<Struct>]
type MlirAttribute =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR dialect handle - opaque handle to a registered dialect
[<Struct>]
type MlirDialectHandle =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR identifier - interned string for operation names, attribute keys, etc.
[<Struct>]
type MlirIdentifier =
    val Handle: nativeint
    new(handle) = { Handle = handle }

//=============================================================================
// String Marshalling
//=============================================================================

/// MLIR string reference - non-owning pointer to string data
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirStringRef =
    val Data: nativeint  // const char*
    val Length: nativeint  // size_t

    new(data, length) = { Data = data; Length = length }

    /// Convert F# string to MlirStringRef (allocates unmanaged memory)
    static member FromString(s: string) =
        if String.IsNullOrEmpty(s) then
            MlirStringRef(nativeint 0, nativeint 0)
        else
            let bytes = System.Text.Encoding.UTF8.GetBytes(s)
            let ptr = Marshal.AllocHGlobal(bytes.Length)
            Marshal.Copy(bytes, 0, ptr, bytes.Length)
            MlirStringRef(ptr, nativeint bytes.Length)

    /// Convert MlirStringRef to F# string
    member this.ToFSharpString() =
        if this.Data = nativeint 0 || this.Length = nativeint 0 then
            String.Empty
        else
            let length = int this.Length
            let bytes = Array.zeroCreate<byte> length
            Marshal.Copy(this.Data, bytes, 0, length)
            System.Text.Encoding.UTF8.GetString(bytes)

    /// Free unmanaged memory (call after passing to MLIR)
    member this.Free() =
        if this.Data <> nativeint 0 then
            Marshal.FreeHGlobal(this.Data)

    /// Create from string, use it, and automatically free
    static member WithString(s: string, f: MlirStringRef -> 'a) =
        let strRef = MlirStringRef.FromString(s)
        try
            f strRef
        finally
            strRef.Free()

/// MLIR named attribute - key-value pair
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirNamedAttribute =
    val Name: MlirIdentifier
    val Attribute: MlirAttribute

//=============================================================================
// Operation State
//=============================================================================

/// MLIR operation state - used to construct operations
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirOperationState =
    val mutable Name: MlirStringRef
    val mutable Location: MlirLocation
    val mutable NumResults: nativeint
    val mutable Results: nativeint      // MlirType*
    val mutable NumOperands: nativeint
    val mutable Operands: nativeint     // MlirValue*
    val mutable NumRegions: nativeint
    val mutable Regions: nativeint      // MlirRegion*
    val mutable NumSuccessors: nativeint
    val mutable Successors: nativeint   // MlirBlock*
    val mutable NumAttributes: nativeint
    val mutable Attributes: nativeint   // MlirNamedAttribute*
    [<MarshalAs(UnmanagedType.U1)>]
    val mutable EnableResultTypeInference: bool

//=============================================================================
// Callback Delegates
//=============================================================================

/// Callback for MLIR IR printing (invoked with chunks of output)
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirStringCallback = delegate of MlirStringRef * nativeint -> unit

//=============================================================================
// P/Invoke Declarations
//=============================================================================

module MlirNative =

    //==========================================================================
    // Context Management
    //==========================================================================

    /// Create an MLIR context
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirContext mlirContextCreate()

    /// Destroy an MLIR context (frees all owned IR)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirContextDestroy(MlirContext ctx)

    /// Check if context handle is null
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirContextIsNull(MlirContext ctx)

    /// Get or load a dialect by name
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirContextGetOrLoadDialect(MlirContext ctx, MlirStringRef name)

    /// Get dialect handle for the 'func' dialect
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__func__()

    /// Get dialect handle for the 'arith' dialect
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__arith__()

    /// Get dialect handle for the 'scf' (structured control flow) dialect
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__scf__()

    /// Get dialect handle for the 'cf' (control flow) dialect
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__cf__()

    /// Get dialect handle for the 'llvm' dialect
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__llvm__()

    /// Register a dialect with a context
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirDialectHandleRegisterDialect(MlirDialectHandle handle, MlirContext ctx)

    /// Load a dialect into the context (must be registered first)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirDialectHandleLoadDialect(MlirDialectHandle handle, MlirContext ctx)

    //==========================================================================
    // Module Management
    //==========================================================================

    /// Create an empty MLIR module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

    /// Destroy a module (frees all owned IR)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirModuleDestroy(MlirModule m)

    /// Get the top-level operation of a module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirModuleGetOperation(MlirModule m)

    /// Get the body (region) of a module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirModuleGetBody(MlirModule m)

    //==========================================================================
    // Location
    //==========================================================================

    /// Create an unknown location (for generated code)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationUnknownGet(MlirContext ctx)

    /// Create a file-line-column location
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationFileLineColGet(MlirContext ctx, MlirStringRef filename, uint32 line, uint32 col)

    //==========================================================================
    // Identifier
    //==========================================================================

    /// Get an identifier from a string
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirIdentifier mlirIdentifierGet(MlirContext ctx, MlirStringRef str)

    //==========================================================================
    // IR Printing
    //==========================================================================

    /// Print an operation to a callback
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationPrint(MlirOperation op, MlirStringCallback callback, nativeint userData)

    /// Print a module to a callback (via module operation)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirModulePrint(MlirModule m, MlirStringCallback callback, nativeint userData)

    //==========================================================================
    // Type System
    //==========================================================================

    /// Create an integer type with specified bit width
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIntegerTypeGet(MlirContext ctx, uint32 bitwidth)

    /// Create a signed integer type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIntegerTypeSignedGet(MlirContext ctx, uint32 bitwidth)

    /// Create an unsigned integer type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIntegerTypeUnsignedGet(MlirContext ctx, uint32 bitwidth)

    /// Create the index type (platform-dependent integer for indexing)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIndexTypeGet(MlirContext ctx)

    /// Create a 32-bit floating-point type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirF32TypeGet(MlirContext ctx)

    /// Create a 64-bit floating-point type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirF64TypeGet(MlirContext ctx)

    /// Create a function type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunctionTypeGet(MlirContext ctx, nativeint numInputs, nativeint inputs, nativeint numResults, nativeint results)

    /// Create an LLVM pointer type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirLLVMPointerTypeGet(MlirContext ctx, uint32 addressSpace)

    /// Check if a type is null
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirTypeIsNull(MlirType typ)

    /// Check if two types are equal
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirTypeEqual(MlirType t1, MlirType t2)

    //==========================================================================
    // Operation Building
    //==========================================================================

    /// Create an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperationState mlirOperationStateGet(MlirStringRef name, MlirLocation loc)

    /// Add results to an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationStateAddResults(MlirOperationState& state, nativeint n, nativeint results)

    /// Add operands to an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationStateAddOperands(MlirOperationState& state, nativeint n, nativeint operands)

    /// Add owned regions to an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationStateAddOwnedRegions(MlirOperationState& state, nativeint n, nativeint regions)

    /// Add attributes to an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationStateAddAttributes(MlirOperationState& state, nativeint n, nativeint attributes)

    /// Create an operation from an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirOperationCreate(MlirOperationState& state)

    /// Destroy an operation (if not owned by a block)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationDestroy(MlirOperation op)

    /// Get a result value from an operation by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirOperationGetResult(MlirOperation op, nativeint pos)

    /// Get the number of results an operation produces
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirOperationGetNumResults(MlirOperation op)

    /// Get a region from an operation by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirOperationGetRegion(MlirOperation op, nativeint pos)

    /// Get the block an operation is in
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirOperationGetBlock(MlirOperation op)

    //==========================================================================
    // Region Management
    //==========================================================================

    /// Create a new region
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirRegionCreate()

    /// Destroy a region (if not owned by an operation)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegionDestroy(MlirRegion region)

    /// Append a block to a region (region takes ownership)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegionAppendOwnedBlock(MlirRegion region, MlirBlock block)

    /// Get the first block in a region
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirRegionGetFirstBlock(MlirRegion region)

    //==========================================================================
    // Block Management
    //==========================================================================

    /// Create a new block with arguments
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirBlockCreate(nativeint numArgs, nativeint argTypes, nativeint argLocs)

    /// Destroy a block (if not owned by a region)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockDestroy(MlirBlock block)

    /// Append an operation to a block (block takes ownership)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation op)

    /// Insert an operation into a block before a reference operation
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockInsertOwnedOperationBefore(MlirBlock block, MlirOperation reference, MlirOperation op)

    /// Get a block argument by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirBlockGetArgument(MlirBlock block, nativeint pos)

    /// Get the number of arguments a block has
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirBlockGetNumArguments(MlirBlock block)

    /// Get the terminator operation of a block
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirBlockGetTerminator(MlirBlock block)

    //==========================================================================
    // Value
    //==========================================================================

    /// Check if a value is null
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirValueIsNull(MlirValue value)

    /// Get the type of a value
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirValueGetType(MlirValue value)

    /// Check if two values are equal
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirValueEqual(MlirValue v1, MlirValue v2)

    //==========================================================================
    // Attribute System
    //==========================================================================

    /// Create an integer attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirIntegerAttrGet(MlirType typ, int64 value)

    /// Create a float attribute from double
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirFloatAttrDoubleGet(MlirContext ctx, MlirType typ, float value)

    /// Create a string attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirStringAttrGet(MlirContext ctx, MlirStringRef str)

    /// Create a unit attribute (singleton attribute with no data)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirUnitAttrGet(MlirContext ctx)

    /// Create a flat symbol reference attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol)

    /// Create a named attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirNamedAttribute mlirNamedAttributeGet(MlirIdentifier name, MlirAttribute attr)

    /// Check if an attribute is null
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirAttributeIsNull(MlirAttribute attr)

    /// Create a type attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirTypeAttrGet(MlirType typ)

    //==========================================================================
    // Pass Manager
    //==========================================================================

    /// Create a pass manager
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirPassManagerCreate(MlirContext ctx)

    /// Destroy a pass manager
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirPassManagerDestroy(nativeint pm)

    /// Run passes on a module (returns LogicalResult: 0 = success)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirPassManagerRunOnOp(nativeint pm, MlirOperation op)

    /// Get nested pass manager for a specific operation name
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirPassManagerGetNestedUnder(nativeint pm, MlirStringRef operationName)

    /// Add a pass to an op pass manager
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOpPassManagerAddOwnedPass(nativeint pm, nativeint pass)

    /// Parse a pass pipeline and add to pass manager
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirParsePassPipeline(nativeint pm, MlirStringRef pipeline, MlirStringCallback callback, nativeint userData)

    //==========================================================================
    // Execution Engine
    //==========================================================================

    /// Create an execution engine from a module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirExecutionEngineCreate(MlirModule m, int32 optLevel, int32 numPaths, nativeint sharedLibPaths, bool enableObjectDump)

    /// Destroy an execution engine
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirExecutionEngineDestroy(nativeint ee)

    /// Lookup a function in the execution engine
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirExecutionEngineLookupPacked(nativeint ee, MlirStringRef name)

    /// Invoke a packed function
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirExecutionEngineInvokePacked(nativeint ee, MlirStringRef name, nativeint arguments)

    //==========================================================================
    // Conversion Passes Registration
    //==========================================================================

    /// Register all MLIR passes
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegisterAllPasses()

    /// Register all LLVM lowering passes
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegisterAllLLVMTranslations(MlirContext ctx)
