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

    /// Get dialect handle for the 'llvm' dialect
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__llvm__()

    /// Register a dialect with a context
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirDialectHandleRegisterDialect(MlirDialectHandle handle, MlirContext ctx)

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
