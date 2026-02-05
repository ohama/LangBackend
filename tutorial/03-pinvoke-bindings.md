# Chapter 03: P/Invoke Bindings

## Introduction

In Chapter 02, you wrote your first F# program that generates MLIR IR. You defined handle types, wrote `DllImport` declarations, and successfully called the MLIR C API to create a simple function. But that code was exploratory and ad-hoc — all the bindings were defined inline in a script.

A real compiler needs organized, reusable bindings. In this chapter, we'll take everything we learned in Chapter 02 and systematize it into a proper F# module: `MlirBindings.fs`. This module will serve as the foundation for all future chapters. You'll learn:

- How to organize MLIR C API bindings by functional area (context, module, type, operation, etc.)
- How to handle string marshalling correctly and safely
- How to work with callbacks for IR printing
- Cross-platform considerations (Linux, macOS, Windows)

By the end of this chapter, you'll have a complete, production-ready binding layer for the MLIR C API.

## Design Philosophy

Our binding layer follows these principles:

1. **Thin wrapper:** Minimal abstraction over the C API. Each F# function maps directly to a C function.
2. **Type safety:** Use F# struct types for MLIR handles to catch type errors at compile time.
3. **Memory safety:** Provide utilities for safe string marshalling and cleanup, but don't hide the need to call destroy functions.
4. **Completeness:** Cover all MLIR C API functions needed for the compiler (context, module, type, operation, region, block, location, attribute, value).
5. **Documentation:** Every function has a comment explaining its purpose and MLIR C API correspondence.

## Project Structure

Before we write code, let's set up a proper F# project. In Chapter 02, we used a script (`.fsx`). Now we'll create a library project:

```bash
cd $HOME/mlir-fsharp-tutorial
dotnet new classlib -lang F# -o MlirBindings
cd MlirBindings
```

This creates a new F# library project with this structure:

```
MlirBindings/
├── MlirBindings.fsproj
└── Library.fs
```

Delete the default `Library.fs`:

```bash
rm Library.fs
```

We'll create `MlirBindings.fs` from scratch.

## Module Organization

Our bindings module will be organized into logical sections:

1. **Handle Types:** F# structs representing MLIR opaque types
2. **String Marshalling:** `MlirStringRef` and helper functions
3. **Callback Delegates:** Function pointer types for MLIR callbacks
4. **Context Management:** Context creation, destruction, dialect loading
5. **Module Management:** Module creation, operations, printing
6. **Location:** Source location utilities
7. **Type System:** Integer types, function types, LLVM types
8. **Operation Building:** Operation state, creation, insertion
9. **Region and Block:** Region and block creation and management
10. **Value and Attribute:** SSA value and attribute handling

Let's build this step by step.

## Handle Types

Create a new file `MlirBindings.fs` in the `MlirBindings` directory:

```bash
touch MlirBindings.fs
```

Add the file to the project by editing `MlirBindings.fsproj`. Replace the contents with:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
  </PropertyGroup>

  <ItemGroup>
    <Compile Include="MlirBindings.fs" />
  </ItemGroup>

</Project>
```

Now open `MlirBindings.fs` and start with the namespace and imports:

```fsharp
namespace MlirBindings

open System
open System.Runtime.InteropServices
```

Define all the handle types we'll need. These are opaque pointers to MLIR internal structures:

```fsharp
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

/// MLIR named attribute - key-value pair (name: attribute)
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirNamedAttribute =
    val Name: MlirStringRef
    val Attribute: MlirAttribute

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
```

Each handle type includes a doc comment explaining its purpose. The `[<Struct>]` attribute ensures these are stack-allocated value types.

## String Marshalling

MLIR uses `MlirStringRef` for passing strings without ownership semantics. Define it with helper utilities:

```fsharp
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
    member this.ToString() =
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
```

The `WithString` helper is particularly useful — it handles allocation and cleanup automatically:

```fsharp
// Instead of:
let strRef = MlirStringRef.FromString("func.func")
let op = createOp strRef
strRef.Free()

// You can write:
MlirStringRef.WithString "func.func" (fun strRef ->
    createOp strRef
)
```

## Callback Delegates

MLIR uses callbacks for printing and string handling. Define the delegate types:

```fsharp
/// Callback for MLIR IR printing (invoked with chunks of output)
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirStringCallback = delegate of MlirStringRef * nativeint -> unit

/// Callback for diagnostic handlers
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirDiagnosticCallback = delegate of MlirDiagnostic * nativeint -> MlirLogicalResult

/// MLIR diagnostic handle
[<Struct>]
type MlirDiagnostic =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR logical result (success/failure)
[<Struct>]
type MlirLogicalResult =
    val Value: int8
    new(value) = { Value = value }
    member this.IsSuccess = this.Value <> 0y
    member this.IsFailure = this.Value = 0y
```

## Operation State

The `MlirOperationState` struct is used to build operations. It's complex because it contains pointers to arrays:

```fsharp
/// MLIR operation state - used to construct operations
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirOperationState =
    val mutable Name: MlirStringRef
    val mutable Location: MlirLocation
    val mutable NumResults: nativeint
    val mutable Results: nativeint  // Pointer to MlirType array
    val mutable NumOperands: nativeint
    val mutable Operands: nativeint  // Pointer to MlirValue array
    val mutable NumRegions: nativeint
    val mutable Regions: nativeint  // Pointer to MlirRegion array
    val mutable NumSuccessors: nativeint
    val mutable Successors: nativeint  // Pointer to MlirBlock array
    val mutable NumAttributes: nativeint
    val mutable Attributes: nativeint  // Pointer to MlirNamedAttribute array
    val mutable EnableResultTypeInference: bool
```

Note: All fields are mutable because we need to modify them before passing to `mlirOperationCreate`.

## P/Invoke Declarations

Now for the main event: the P/Invoke declarations for the MLIR C API. Organize them into a module:

```fsharp
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

    /// Check if two contexts are equal
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirContextEqual(MlirContext ctx1, MlirContext ctx2)

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

    //==========================================================================
    // Module Management
    //==========================================================================

    /// Create an empty MLIR module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

    /// Create an MLIR module from parsing a string
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateParse(MlirContext ctx, MlirStringRef mlir)

    /// Get the top-level operation of a module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirModuleGetOperation(MlirModule m)

    /// Get the body (region) of a module
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirModuleGetBody(MlirModule m)

    /// Destroy a module (frees all owned IR)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirModuleDestroy(MlirModule m)

    //==========================================================================
    // Location
    //==========================================================================

    /// Create an unknown location (for generated code)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationUnknownGet(MlirContext ctx)

    /// Create a file-line-column location
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationFileLineColGet(MlirContext ctx, MlirStringRef filename, uint32 line, uint32 col)

    /// Create a fused location (combination of multiple locations)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationFusedGet(MlirContext ctx, nativeint numLocs, MlirLocation& locs, MlirAttribute metadata)

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

    /// Create a floating-point type (f32, f64, etc.)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirF32TypeGet(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirF64TypeGet(MlirContext ctx)

    /// Create the index type (platform-dependent integer for indexing)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIndexTypeGet(MlirContext ctx)

    /// Create a function type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunctionTypeGet(MlirContext ctx, nativeint numInputs, MlirType& inputs, nativeint numResults, MlirType& results)

    /// Get the number of inputs for a function type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirFunctionTypeGetNumInputs(MlirType funcType)

    /// Get the number of results for a function type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirFunctionTypeGetNumResults(MlirType funcType)

    /// Create an LLVM pointer type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirLLVMPointerTypeGet(MlirContext ctx, uint32 addressSpace)

    /// Create an LLVM void type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirLLVMVoidTypeGet(MlirContext ctx)

    /// Create an LLVM struct type
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirLLVMStructTypeLiteralGet(MlirContext ctx, nativeint numFieldTypes, MlirType& fieldTypes, bool isPacked)

    //==========================================================================
    // Attribute System
    //==========================================================================

    /// Create an integer attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirIntegerAttrGet(MlirType typ, int64 value)

    /// Create a float attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirFloatAttrDoubleGet(MlirContext ctx, MlirType typ, float64 value)

    /// Create a string attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirStringAttrGet(MlirContext ctx, MlirStringRef str)

    /// Create a type attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirTypeAttrGet(MlirType typ)

    /// Create a symbol reference attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol)

    /// Create an array attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirArrayAttrGet(MlirContext ctx, nativeint numElements, MlirAttribute& elements)

    /// Get an identifier from a string
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirIdentifier mlirIdentifierGet(MlirContext ctx, MlirStringRef str)

    /// Create a named attribute
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirNamedAttribute mlirNamedAttributeGet(MlirIdentifier name, MlirAttribute attr)

    //==========================================================================
    // Operation Building
    //==========================================================================

    /// Create an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperationState mlirOperationStateGet(MlirStringRef name, MlirLocation loc)

    /// Create an operation from an operation state
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirOperationCreate(MlirOperationState& state)

    /// Destroy an operation (if not owned by a block)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationDestroy(MlirOperation op)

    /// Get the name of an operation
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirIdentifier mlirOperationGetName(MlirOperation op)

    /// Get the number of regions in an operation
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirOperationGetNumRegions(MlirOperation op)

    /// Get a region from an operation by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirOperationGetRegion(MlirOperation op, nativeint pos)

    /// Get the number of results an operation produces
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirOperationGetNumResults(MlirOperation op)

    /// Get a result value from an operation by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirOperationGetResult(MlirOperation op, nativeint pos)

    /// Get the number of operands an operation takes
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirOperationGetNumOperands(MlirOperation op)

    /// Get an operand value from an operation by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirOperationGetOperand(MlirOperation op, nativeint pos)

    /// Set an operand of an operation
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationSetOperand(MlirOperation op, nativeint pos, MlirValue value)

    /// Print an operation to a callback
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationPrint(MlirOperation op, MlirStringCallback callback, nativeint userData)

    /// Verify an operation (check IR well-formedness)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirOperationVerify(MlirOperation op)

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

    /// Insert a block into a region at position (region takes ownership)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegionInsertOwnedBlock(MlirRegion region, nativeint pos, MlirBlock block)

    /// Get the first block in a region
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirRegionGetFirstBlock(MlirRegion region)

    //==========================================================================
    // Block Management
    //==========================================================================

    /// Create a new block with arguments
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirBlockCreate(nativeint numArgs, MlirType& argTypes, MlirLocation& argLocs)

    /// Destroy a block (if not owned by a region)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockDestroy(MlirBlock block)

    /// Get the number of arguments a block has
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirBlockGetNumArguments(MlirBlock block)

    /// Get a block argument by index
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirBlockGetArgument(MlirBlock block, nativeint pos)

    /// Append an operation to a block (block takes ownership)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation op)

    /// Insert an operation into a block at position (block takes ownership)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockInsertOwnedOperation(MlirBlock block, nativeint pos, MlirOperation op)

    /// Get the first operation in a block
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirBlockGetFirstOperation(MlirBlock block)

    //==========================================================================
    // Value
    //==========================================================================

    /// Get the type of a value
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirValueGetType(MlirValue value)

    /// Print a value
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirValuePrint(MlirValue value, MlirStringCallback callback, nativeint userData)
```

This is a comprehensive binding layer covering all the MLIR C API functions you'll need for building a compiler. Each function is documented with its purpose.

## Cross-Platform Library Loading

One important detail: the library name `"MLIR-C"` works across platforms because .NET automatically appends the correct extension:

- **Linux:** `libMLIR-C.so`
- **macOS:** `libMLIR-C.dylib`
- **Windows:** `MLIR-C.dll`

However, .NET still needs to know where to find the library at runtime. We covered this in Chapter 00 (setting `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`). For a production application, you have several options:

### Option 1: Environment Variable (Development)

Set the library path before running:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet run
```

### Option 2: NativeLibrary.SetDllImportResolver (Runtime)

Use .NET's `NativeLibrary` API to specify custom search paths:

```fsharp
open System.Runtime.InteropServices

module LibraryLoader =
    let initialize() =
        NativeLibrary.SetDllImportResolver(
            typeof<MlirContext>.Assembly,
            fun libraryName assemblyPath searchPath ->
                if libraryName = "MLIR-C" then
                    let customPath = Environment.GetEnvironmentVariable("MLIR_INSTALL_PATH")
                    if not (String.IsNullOrEmpty(customPath)) then
                        let libPath =
                            if RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
                                System.IO.Path.Combine(customPath, "lib", "libMLIR-C.so")
                            elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
                                System.IO.Path.Combine(customPath, "lib", "libMLIR-C.dylib")
                            else
                                System.IO.Path.Combine(customPath, "bin", "MLIR-C.dll")
                        NativeLibrary.Load(libPath)
                    else
                        nativeint 0
                else
                    nativeint 0
        )
```

Call `LibraryLoader.initialize()` before any MLIR functions are invoked.

### Option 3: rpath (Linux/macOS Binaries)

For compiled binaries, embed the library search path in the executable using rpath. This is outside the scope of this tutorial but is the standard solution for distributed applications.

## Helper Utilities

Add some high-level helper functions for common patterns:

```fsharp
module MlirHelpers =
    /// Print an operation to a string
    let operationToString (op: MlirOperation) : string =
        let mutable output = ""
        let callback = MlirStringCallback(fun strRef _ ->
            output <- output + strRef.ToString()
        )
        MlirNative.mlirOperationPrint(op, callback, nativeint 0)
        output

    /// Print a module to a string
    let moduleToString (m: MlirModule) : string =
        let op = MlirNative.mlirModuleGetOperation(m)
        operationToString op

    /// Print a value to a string
    let valueToString (v: MlirValue) : string =
        let mutable output = ""
        let callback = MlirStringCallback(fun strRef _ ->
            output <- output + strRef.ToString()
        )
        MlirNative.mlirValuePrint(v, callback, nativeint 0)
        output

    /// Create a context with common dialects registered
    let createContextWithDialects() : MlirContext =
        let ctx = MlirNative.mlirContextCreate()
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__func__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__arith__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__scf__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__cf__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__llvm__(), ctx)
        ctx

    /// Create a block with no arguments
    let createEmptyBlock(ctx: MlirContext) : MlirBlock =
        let loc = MlirNative.mlirLocationUnknownGet(ctx)
        let mutable dummyType = MlirType()
        let mutable dummyLoc = loc
        MlirNative.mlirBlockCreate(nativeint 0, &dummyType, &dummyLoc)
```

These utilities wrap common operations and reduce boilerplate in user code.

## Complete MlirBindings.fs Listing

Here's the complete `MlirBindings.fs` file with all sections integrated:

```fsharp
namespace MlirBindings

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

/// MLIR diagnostic handle
[<Struct>]
type MlirDiagnostic =
    val Handle: nativeint
    new(handle) = { Handle = handle }

/// MLIR logical result (success/failure)
[<Struct>]
type MlirLogicalResult =
    val Value: int8
    new(value) = { Value = value }
    member this.IsSuccess = this.Value <> 0y
    member this.IsFailure = this.Value = 0y

//=============================================================================
// String Marshalling
//=============================================================================

/// MLIR string reference - non-owning pointer to string data
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirStringRef =
    val Data: nativeint
    val Length: nativeint

    new(data, length) = { Data = data; Length = length }

    static member FromString(s: string) =
        if String.IsNullOrEmpty(s) then
            MlirStringRef(nativeint 0, nativeint 0)
        else
            let bytes = System.Text.Encoding.UTF8.GetBytes(s)
            let ptr = Marshal.AllocHGlobal(bytes.Length)
            Marshal.Copy(bytes, 0, ptr, bytes.Length)
            MlirStringRef(ptr, nativeint bytes.Length)

    member this.ToString() =
        if this.Data = nativeint 0 || this.Length = nativeint 0 then
            String.Empty
        else
            let length = int this.Length
            let bytes = Array.zeroCreate<byte> length
            Marshal.Copy(this.Data, bytes, 0, length)
            System.Text.Encoding.UTF8.GetString(bytes)

    member this.Free() =
        if this.Data <> nativeint 0 then
            Marshal.FreeHGlobal(this.Data)

    static member WithString(s: string, f: MlirStringRef -> 'a) =
        let strRef = MlirStringRef.FromString(s)
        try
            f strRef
        finally
            strRef.Free()

/// MLIR named attribute - key-value pair
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirNamedAttribute =
    val Name: MlirStringRef
    val Attribute: MlirAttribute

//=============================================================================
// Callback Delegates
//=============================================================================

/// Callback for MLIR IR printing
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirStringCallback = delegate of MlirStringRef * nativeint -> unit

/// Callback for diagnostic handlers
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirDiagnosticCallback = delegate of MlirDiagnostic * nativeint -> MlirLogicalResult

//=============================================================================
// Operation State
//=============================================================================

/// MLIR operation state - used to construct operations
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirOperationState =
    val mutable Name: MlirStringRef
    val mutable Location: MlirLocation
    val mutable NumResults: nativeint
    val mutable Results: nativeint
    val mutable NumOperands: nativeint
    val mutable Operands: nativeint
    val mutable NumRegions: nativeint
    val mutable Regions: nativeint
    val mutable NumSuccessors: nativeint
    val mutable Successors: nativeint
    val mutable NumAttributes: nativeint
    val mutable Attributes: nativeint
    val mutable EnableResultTypeInference: bool

//=============================================================================
// P/Invoke Declarations
//=============================================================================

module MlirNative =

    // Context Management
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirContext mlirContextCreate()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirContextDestroy(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__func__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__arith__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__scf__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__cf__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__llvm__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirDialectHandleRegisterDialect(MlirDialectHandle handle, MlirContext ctx)

    // Module Management
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateParse(MlirContext ctx, MlirStringRef mlir)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirModuleGetOperation(MlirModule m)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirModuleGetBody(MlirModule m)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirModuleDestroy(MlirModule m)

    // Location
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationUnknownGet(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationFileLineColGet(MlirContext ctx, MlirStringRef filename, uint32 line, uint32 col)

    // Type System
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIntegerTypeGet(MlirContext ctx, uint32 bitwidth)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirF32TypeGet(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirF64TypeGet(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIndexTypeGet(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunctionTypeGet(MlirContext ctx, nativeint numInputs, MlirType& inputs, nativeint numResults, MlirType& results)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirLLVMPointerTypeGet(MlirContext ctx, uint32 addressSpace)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirLLVMVoidTypeGet(MlirContext ctx)

    // Attributes
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirIntegerAttrGet(MlirType typ, int64 value)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirStringAttrGet(MlirContext ctx, MlirStringRef str)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirTypeAttrGet(MlirType typ)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirIdentifier mlirIdentifierGet(MlirContext ctx, MlirStringRef str)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirNamedAttribute mlirNamedAttributeGet(MlirIdentifier name, MlirAttribute attr)

    // Operation Building
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperationState mlirOperationStateGet(MlirStringRef name, MlirLocation loc)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirOperationCreate(MlirOperationState& state)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationDestroy(MlirOperation op)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirOperationGetRegion(MlirOperation op, nativeint pos)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirOperationGetNumResults(MlirOperation op)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirOperationGetResult(MlirOperation op, nativeint pos)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationSetOperand(MlirOperation op, nativeint pos, MlirValue value)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationPrint(MlirOperation op, MlirStringCallback callback, nativeint userData)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirOperationVerify(MlirOperation op)

    // Region Management
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirRegionCreate()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegionAppendOwnedBlock(MlirRegion region, MlirBlock block)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirRegionGetFirstBlock(MlirRegion region)

    // Block Management
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirBlockCreate(nativeint numArgs, MlirType& argTypes, MlirLocation& argLocs)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirBlockGetNumArguments(MlirBlock block)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirBlockGetArgument(MlirBlock block, nativeint pos)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation op)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockInsertOwnedOperation(MlirBlock block, nativeint pos, MlirOperation op)

    // Value
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirValueGetType(MlirValue value)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirValuePrint(MlirValue value, MlirStringCallback callback, nativeint userData)

//=============================================================================
// Helper Utilities
//=============================================================================

module MlirHelpers =
    let operationToString (op: MlirOperation) : string =
        let mutable output = ""
        let callback = MlirStringCallback(fun strRef _ ->
            output <- output + strRef.ToString()
        )
        MlirNative.mlirOperationPrint(op, callback, nativeint 0)
        output

    let moduleToString (m: MlirModule) : string =
        let op = MlirNative.mlirModuleGetOperation(m)
        operationToString op

    let valueToString (v: MlirValue) : string =
        let mutable output = ""
        let callback = MlirStringCallback(fun strRef _ ->
            output <- output + strRef.ToString()
        )
        MlirNative.mlirValuePrint(v, callback, nativeint 0)
        output

    let createContextWithDialects() : MlirContext =
        let ctx = MlirNative.mlirContextCreate()
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__func__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__arith__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__scf__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__cf__(), ctx)
        MlirNative.mlirDialectHandleRegisterDialect(MlirNative.mlirGetDialectHandle__llvm__(), ctx)
        ctx
```

This is your complete, production-ready MLIR binding layer.

## Building the Library

Build the library project:

```bash
cd $HOME/mlir-fsharp-tutorial/MlirBindings
dotnet build
```

Expected output:

```
Build succeeded.
    0 Warning(s)
    0 Error(s)
```

The compiled library is in `bin/Debug/net8.0/MlirBindings.dll`.

## Using the Bindings

Let's rewrite the Chapter 02 hello-world example using the new bindings. Create a new console project:

```bash
cd $HOME/mlir-fsharp-tutorial
dotnet new console -lang F# -o HelloMlirWithBindings
cd HelloMlirWithBindings
dotnet add reference ../MlirBindings/MlirBindings.fsproj
```

Replace the contents of `Program.fs`:

```fsharp
open System
open MlirBindings

[<EntryPoint>]
let main argv =
    // Create context with dialects
    let ctx = MlirHelpers.createContextWithDialects()
    printfn "Created MLIR context with dialects loaded"

    // Create empty module
    let loc = MlirNative.mlirLocationUnknownGet(ctx)
    let mlirModule = MlirNative.mlirModuleCreateEmpty(loc)
    printfn "Created empty module"

    // Print the module
    printfn "\nGenerated MLIR IR:"
    printfn "%s" (MlirHelpers.moduleToString mlirModule)

    // Cleanup
    MlirNative.mlirModuleDestroy(mlirModule)
    MlirNative.mlirContextDestroy(ctx)
    printfn "\nCleaned up"

    0
```

Run it:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet run
```

Expected output:

```
Created MLIR context with dialects loaded
Created empty module

Generated MLIR IR:
module {
}

Cleaned up
```

Much cleaner than Chapter 02! The bindings module handles all the marshalling and boilerplate.

## What We've Learned

In this chapter, you:

1. **Organized MLIR bindings** into a reusable F# library module with logical sections.
2. **Defined comprehensive handle types** for all MLIR entities (context, module, operation, type, region, block, value, attribute).
3. **Implemented safe string marshalling** with `MlirStringRef` and helper utilities.
4. **Declared P/Invoke bindings** for the complete MLIR C API surface area needed for compilation.
5. **Created helper utilities** to reduce boilerplate (printing, context creation).
6. **Understood cross-platform considerations** for library loading.
7. **Built and used the bindings library** in a separate project.

You now have a complete, production-ready binding layer for MLIR. This `MlirBindings` module will serve as the foundation for all future chapters as we build the FunLang compiler.

## Next Chapter

In the next chapter, we'll start building the FunLang compiler backend. We'll define the data structures for representing the typed FunLang AST in F#, and begin writing the code generation logic that translates FunLang expressions into MLIR operations using the bindings we've built.

Continue to **Chapter 04: FunLang AST to MLIR** (to be written).

## Further Reading

- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) — Official C API guide
- [P/Invoke Best Practices](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices) — Microsoft's guidelines for safe and performant interop
- [Memory Management in P/Invoke](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/tutorial-custom-marshaller) — Understanding managed/unmanaged memory boundaries
