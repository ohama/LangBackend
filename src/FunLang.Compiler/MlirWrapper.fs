namespace FunLang.Compiler

open System
open System.Runtime.InteropServices

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

    /// Load a dialect by name (e.g., "func", "arith", "scf", "llvm")
    member _.LoadDialect(dialectName: string) =
        if disposed then raise (ObjectDisposedException("Context"))
        MlirStringRef.WithString(dialectName, fun nameRef ->
            MlirNative.mlirContextGetOrLoadDialect(handle, nameRef) |> ignore)

    /// Load standard dialects needed for FunLang compilation
    member this.LoadStandardDialects() =
        this.LoadDialect("func")
        this.LoadDialect("arith")
        this.LoadDialect("scf")
        this.LoadDialect("llvm")

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
