namespace FunLang.Compiler

open System
open System.Runtime.InteropServices
open FSharp.NativeInterop

//=============================================================================
// Context Wrapper
//=============================================================================

/// MLIR Context wrapper with automatic resource management.
/// Root object that owns all MLIR resources (dialects, types, attributes).
type Context() =
    let mutable handle = MlirNative.mlirContextCreate()
    let mutable disposed = false

    /// Raw MLIR context handle
    member _.Handle = handle

    /// Register and load a dialect handle with this context
    member _.LoadDialect(dialectHandle: MlirDialectHandle) =
        if disposed then raise (ObjectDisposedException("Context"))
        MlirNative.mlirDialectHandleRegisterDialect(dialectHandle, handle)
        MlirNative.mlirDialectHandleLoadDialect(dialectHandle, handle) |> ignore

    /// Load standard dialects needed for FunLang compilation
    member this.LoadStandardDialects() =
        this.LoadDialect(MlirNative.mlirGetDialectHandle__func__())
        this.LoadDialect(MlirNative.mlirGetDialectHandle__arith__())
        this.LoadDialect(MlirNative.mlirGetDialectHandle__scf__())
        this.LoadDialect(MlirNative.mlirGetDialectHandle__cf__())
        this.LoadDialect(MlirNative.mlirGetDialectHandle__llvm__())

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                MlirNative.mlirContextDestroy(handle)
                handle <- Unchecked.defaultof<_>
                disposed <- true

//=============================================================================
// Location
//=============================================================================

/// Source location for MLIR diagnostics.
/// Locations are value types owned by Context, no manual cleanup needed.
type Location =
    | Unknown of Context
    | FileLineCol of Context * filename: string * line: int * col: int

    /// Get the raw MLIR location handle
    member this.Handle =
        match this with
        | Unknown ctx ->
            MlirNative.mlirLocationUnknownGet(ctx.Handle)
        | FileLineCol(ctx, filename, line, col) ->
            MlirStringRef.WithString(filename, fun fnRef ->
                MlirNative.mlirLocationFileLineColGet(ctx.Handle, fnRef, uint32 line, uint32 col))

    /// Get the associated context
    member this.Context =
        match this with
        | Unknown ctx -> ctx
        | FileLineCol(ctx, _, _, _) -> ctx

//=============================================================================
// Module Wrapper
//=============================================================================

/// MLIR Module wrapper with automatic resource management.
/// Modules are the top-level container for functions and operations.
type Module(context: Context, location: Location) =
    let handle = MlirNative.mlirModuleCreateEmpty(location.Handle)
    let contextRef = context  // CRITICAL: Keep parent alive
    let mutable disposed = false

    /// Raw MLIR module handle
    member _.Handle = handle

    /// Associated context (kept alive by this module)
    member _.Context = contextRef

    /// Get the module's body block (where operations are added)
    member _.Body =
        if disposed then raise (ObjectDisposedException("Module"))
        MlirNative.mlirModuleGetBody(handle)

    /// Get the module as an operation
    member _.Operation =
        if disposed then raise (ObjectDisposedException("Module"))
        MlirNative.mlirModuleGetOperation(handle)

    /// Print the module's IR to a string (for debugging)
    member _.Print() =
        if disposed then raise (ObjectDisposedException("Module"))
        let sb = System.Text.StringBuilder()
        let callback = MlirStringCallback(fun strRef _ ->
            let bytes = Array.zeroCreate<byte>(int strRef.Length)
            Marshal.Copy(strRef.Data, bytes, 0, int strRef.Length)
            sb.Append(System.Text.Encoding.UTF8.GetString(bytes)) |> ignore)
        MlirNative.mlirOperationPrint(MlirNative.mlirModuleGetOperation(handle), callback, nativeint 0)
        sb.ToString()

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                MlirNative.mlirModuleDestroy(handle)
                disposed <- true

//=============================================================================
// OpBuilder - Fluent API for Operation Construction
//=============================================================================

/// Fluent builder for MLIR operations.
/// Wraps verbose operation state construction with convenient methods.
type OpBuilder(context: Context) =
    let ctx = context

    // ==================== Type Helpers ====================

    /// Get i32 type
    member _.I32Type() = MlirNative.mlirIntegerTypeGet(ctx.Handle, 32u)

    /// Get i64 type
    member _.I64Type() = MlirNative.mlirIntegerTypeGet(ctx.Handle, 64u)

    /// Get i1 (boolean) type
    member _.I1Type() = MlirNative.mlirIntegerTypeGet(ctx.Handle, 1u)

    /// Get index type
    member _.IndexType() = MlirNative.mlirIndexTypeGet(ctx.Handle)

    /// Get LLVM pointer type (opaque pointer)
    member _.PtrType() = MlirNative.mlirLLVMPointerTypeGet(ctx.Handle, 0u)

    /// Create function type
    member _.FunctionType(inputs: MlirType[], results: MlirType[]) =
        use inputsPin = fixed inputs
        use resultsPin = fixed results
        MlirNative.mlirFunctionTypeGet(
            ctx.Handle,
            nativeint inputs.Length, NativePtr.toNativeInt inputsPin,
            nativeint results.Length, NativePtr.toNativeInt resultsPin)

    // ==================== Attribute Helpers ====================

    /// Create integer attribute
    member _.IntegerAttr(value: int64, typ: MlirType) =
        MlirNative.mlirIntegerAttrGet(typ, value)

    /// Create string attribute
    member _.StringAttr(value: string) =
        MlirStringRef.WithString(value, fun strRef ->
            MlirNative.mlirStringAttrGet(ctx.Handle, strRef))

    /// Create flat symbol reference attribute
    member _.SymbolRefAttr(name: string) =
        MlirStringRef.WithString(name, fun nameRef ->
            MlirNative.mlirFlatSymbolRefAttrGet(ctx.Handle, nameRef))

    /// Create named attribute
    member _.NamedAttr(name: string, attr: MlirAttribute) =
        MlirStringRef.WithString(name, fun nameRef ->
            let id = MlirNative.mlirIdentifierGet(ctx.Handle, nameRef)
            MlirNative.mlirNamedAttributeGet(id, attr))

    // ==================== Operation Helpers ====================

    /// Get result value from operation at given index
    member _.GetResult(op: MlirOperation, index: int) =
        MlirNative.mlirOperationGetResult(op, nativeint index)

    /// Create a block with given argument types
    member _.CreateBlock(argTypes: MlirType[], location: Location) =
        let locs = Array.create argTypes.Length location.Handle
        use typesPin = fixed argTypes
        use locsPin = fixed locs
        MlirNative.mlirBlockCreate(nativeint argTypes.Length, NativePtr.toNativeInt typesPin, NativePtr.toNativeInt locsPin)

    /// Create an empty region
    member _.CreateRegion() =
        MlirNative.mlirRegionCreate()

    /// Append block to region
    member _.AppendBlockToRegion(region: MlirRegion, block: MlirBlock) =
        MlirNative.mlirRegionAppendOwnedBlock(region, block)

    /// Append operation to block
    member _.AppendOperationToBlock(block: MlirBlock, op: MlirOperation) =
        MlirNative.mlirBlockAppendOwnedOperation(block, op)

    // ==================== Generic Operation Creation ====================

    /// Create operation with given name, results, operands, attributes, and regions
    member _.CreateOperation(
        name: string,
        location: Location,
        resultTypes: MlirType[],
        operands: MlirValue[],
        attributes: MlirNamedAttribute[],
        regions: MlirRegion[]) =

        // Allocate string - MUST stay alive until mlirOperationCreate returns
        let nameRef = MlirStringRef.FromString(name)
        try
            let mutable state = MlirNative.mlirOperationStateGet(nameRef, location.Handle)
            state.EnableResultTypeInference <- false  // Explicit: don't infer types

            if resultTypes.Length > 0 then
                use typesPin = fixed resultTypes
                MlirNative.mlirOperationStateAddResults(&state, nativeint resultTypes.Length, NativePtr.toNativeInt typesPin)

            if operands.Length > 0 then
                use operandsPin = fixed operands
                MlirNative.mlirOperationStateAddOperands(&state, nativeint operands.Length, NativePtr.toNativeInt operandsPin)

            if attributes.Length > 0 then
                use attrsPin = fixed attributes
                MlirNative.mlirOperationStateAddAttributes(&state, nativeint attributes.Length, NativePtr.toNativeInt attrsPin)

            if regions.Length > 0 then
                use regionsPin = fixed regions
                MlirNative.mlirOperationStateAddOwnedRegions(&state, nativeint regions.Length, NativePtr.toNativeInt regionsPin)

            MlirNative.mlirOperationCreate(&state)
        finally
            nameRef.Free()

    /// Create type attribute
    member _.TypeAttr(typ: MlirType) =
        MlirNative.mlirTypeAttrGet(typ)

//=============================================================================
// PassManager - Runs optimization and lowering passes
//=============================================================================

/// MLIR Pass Manager for running transformation passes.
type PassManager(context: Context) =
    let handle = MlirNative.mlirPassManagerCreate(context.Handle)
    let contextRef = context
    let mutable disposed = false

    do MlirNative.mlirRegisterAllPasses()

    /// Add a pass pipeline from string (e.g., "builtin.module(convert-func-to-llvm)")
    member _.AddPipeline(pipeline: string) =
        if disposed then raise (ObjectDisposedException("PassManager"))
        let errorMsg = System.Text.StringBuilder()
        let errorCallback = MlirStringCallback(fun strRef _ ->
            let bytes = Array.zeroCreate<byte>(int strRef.Length)
            Marshal.Copy(strRef.Data, bytes, 0, int strRef.Length)
            errorMsg.Append(System.Text.Encoding.UTF8.GetString(bytes)) |> ignore)
        let result = MlirStringRef.WithString(pipeline, fun pipelineRef ->
            MlirNative.mlirParsePassPipeline(handle, pipelineRef, errorCallback, nativeint 0))
        // MlirLogicalResult: 0 = failure, 1 = success
        if result = nativeint 0 then
            failwithf "Failed to parse pipeline '%s': %s" pipeline (errorMsg.ToString())

    /// Run passes on a module
    member _.Run(mlirMod: Module) =
        if disposed then raise (ObjectDisposedException("PassManager"))
        let result = MlirNative.mlirPassManagerRunOnOp(handle, mlirMod.Operation)
        result <> nativeint 0  // MlirLogicalResult: 0 = failure, non-zero = success

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                MlirNative.mlirPassManagerDestroy(handle)
                disposed <- true

//=============================================================================
// ExecutionEngine - JIT compilation and execution
//=============================================================================

/// MLIR Execution Engine for JIT compilation.
type ExecutionEngine(mlirMod: Module, optLevel: int) =
    let handle = MlirNative.mlirExecutionEngineCreate(mlirMod.Handle, optLevel, 0, nativeint 0, false)
    let mutable disposed = false

    do
        if handle = nativeint 0 then
            failwith "Failed to create ExecutionEngine - ensure module is lowered to LLVM dialect"

    /// Invoke a function by name with packed arguments
    member _.InvokePacked(name: string, args: nativeint) =
        if disposed then raise (ObjectDisposedException("ExecutionEngine"))
        MlirStringRef.WithString(name, fun nameRef ->
            MlirNative.mlirExecutionEngineInvokePacked(handle, nameRef, args))

    /// Lookup function pointer by name
    member _.Lookup(name: string) =
        if disposed then raise (ObjectDisposedException("ExecutionEngine"))
        MlirStringRef.WithString(name, fun nameRef ->
            MlirNative.mlirExecutionEngineLookupPacked(handle, nameRef))

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                MlirNative.mlirExecutionEngineDestroy(handle)
                disposed <- true
