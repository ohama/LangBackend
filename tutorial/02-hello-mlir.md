# Chapter 02: Hello MLIR from F#

## Introduction

In Chapter 00, you built MLIR from source and installed the .NET SDK. In Chapter 01, you learned the fundamental concepts of MLIR: dialects, operations, regions, blocks, and SSA form. Now it's time to write code.

This chapter is your first "it works!" moment. You'll write an F# script that calls the MLIR C API using P/Invoke, creates an MLIR context and module, builds a simple function with arithmetic operations, and prints the resulting IR to the console. By the end, you'll have a working proof of concept that F# can interoperate with MLIR.

The code in this chapter is intentionally ad-hoc and exploratory. We'll define the P/Invoke bindings inline and focus on getting something running. In Chapter 03, we'll organize these bindings into a proper reusable module.

## What We're Building

Our first MLIR program will be a function that returns a constant integer. In MLIR textual form, it looks like this:

```mlir
module {
  func.func @return_forty_two() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
}
```

This is the simplest possible MLIR program:
- One function named `@return_forty_two`
- Zero parameters
- Returns an `i32` (32-bit integer)
- Body creates a constant `42` and returns it

We'll build this programmatically from F# using MLIR's C API.

## Understanding P/Invoke

P/Invoke (Platform Invoke) is .NET's Foreign Function Interface (FFI) mechanism. It allows managed code (F#, C#, etc.) to call unmanaged native functions from shared libraries (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows).

### The DllImport Attribute

To call a native function, you use the `[<DllImport>]` attribute to declare the function signature. Here's the pattern:

```fsharp
[<DllImport("library-name", CallingConvention = CallingConvention.Cdecl)>]
extern ReturnType functionName(ParamType1 param1, ParamType2 param2)
```

Let's break this down:

- **`[<DllImport("library-name")>]`**: Specifies which shared library contains the function. For MLIR, this is `"MLIR-C"` (without file extension — .NET adds `.so`, `.dylib`, or `.dll` automatically based on the platform).

- **`CallingConvention = CallingConvention.Cdecl`**: Specifies how arguments are passed and the stack is managed. The MLIR C API uses the C calling convention (`Cdecl`), which is standard for C libraries.

- **`extern`**: Marks this as an external function defined in native code.

- **Return type and parameters**: Must match the C function signature exactly. MLIR uses opaque struct handles (pointers to internal data structures), which we represent in F# as `nativeint`.

### MLIR Handle Types

The MLIR C API uses opaque struct types for all IR entities:

```c
// MLIR-C API (C header)
typedef struct MlirContext { void *ptr; } MlirContext;
typedef struct MlirModule { void *ptr; } MlirModule;
typedef struct MlirOperation { void *ptr; } MlirOperation;
// ... and many more
```

Each struct is a wrapper around a pointer. From F#'s perspective, we don't care about the internal structure — we just need to pass these handles between MLIR functions. We'll represent them as F# structs with a single `nativeint` field:

```fsharp
[<Struct>]
type MlirContext =
    val Handle: nativeint
    new(handle) = { Handle = handle }
```

This matches the C memory layout (a single pointer) and is safe to pass across the P/Invoke boundary.

## Creating an F# Script

Let's start writing code. Create a new file called `HelloMlir.fsx` in your working directory:

```bash
cd $HOME
mkdir -p mlir-fsharp-tutorial
cd mlir-fsharp-tutorial
touch HelloMlir.fsx
```

Open `HelloMlir.fsx` in your text editor and start with the necessary imports:

```fsharp
open System
open System.Runtime.InteropServices
```

- `System`: Core .NET types
- `System.Runtime.InteropServices`: Contains `DllImport`, `CallingConvention`, and marshalling attributes

## Defining Handle Types

First, define the MLIR handle types we'll need. For this simple example, we need:

- `MlirContext`: The root MLIR context (manages memory, dialects, etc.)
- `MlirModule`: A module (top-level container for functions)
- `MlirLocation`: Source location information (required for creating operations)
- `MlirType`: Type system (we'll use `i32`)
- `MlirBlock`: A basic block
- `MlirRegion`: A region containing blocks
- `MlirOperation`: An operation (the result of creating a function or arithmetic op)
- `MlirValue`: An SSA value (the result of an operation)

Add these type definitions to your script:

```fsharp
[<Struct>]
type MlirContext =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirModule =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirLocation =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirType =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirBlock =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirRegion =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirOperation =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirValue =
    val Handle: nativeint
    new(handle) = { Handle = handle }
```

Each handle is a thin wrapper around a native pointer. The `[<Struct>]` attribute ensures these are stack-allocated value types (not heap-allocated classes), which is more efficient for small wrappers.

## String Marshalling: MlirStringRef

MLIR's C API uses a custom string structure called `MlirStringRef` for passing strings without ownership concerns. It's defined in C as:

```c
typedef struct MlirStringRef {
    const char *data;
    size_t length;
} MlirStringRef;
```

We need to match this layout in F#:

```fsharp
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirStringRef =
    val Data: nativeint  // const char*
    val Length: nativeint  // size_t

    new(data, length) = { Data = data; Length = length }

    static member FromString(s: string) =
        let bytes = System.Text.Encoding.UTF8.GetBytes(s)
        let ptr = Marshal.AllocHGlobal(bytes.Length)
        Marshal.Copy(bytes, 0, ptr, bytes.Length)
        MlirStringRef(ptr, nativeint bytes.Length)

    member this.Free() =
        if this.Data <> nativeint 0 then
            Marshal.FreeHGlobal(this.Data)
```

Breaking this down:

- **`[<StructLayout(LayoutKind.Sequential)>]`**: Ensures fields are laid out in memory in the order declared (matching the C struct).

- **`FromString(s: string)`**: Helper to convert an F# string to `MlirStringRef`. It allocates unmanaged memory, copies the UTF-8 bytes, and returns a `MlirStringRef` pointing to that memory.

- **`Free()`**: Releases the unmanaged memory. You must call this after passing the string to MLIR, or you'll leak memory.

## Declaring P/Invoke Functions

Now for the P/Invoke declarations. We'll declare only the functions needed for this example. Add this to your script:

```fsharp
module MlirNative =
    // Context management
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirContext mlirContextCreate()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirContextDestroy(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__func__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirDialectHandle mlirGetDialectHandle__arith__()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirDialectHandleRegisterDialect(MlirDialectHandle handle, MlirContext ctx)

    // Module management
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirModuleGetOperation(MlirModule m)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirModuleDestroy(MlirModule m)

    // Location
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationUnknownGet(MlirContext ctx)

    // Types
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirIntegerTypeGet(MlirContext ctx, uint32 bitwidth)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunctionTypeGet(MlirContext ctx, nativeint numInputs, MlirType& inputs, nativeint numResults, MlirType& results)

    // Operation building
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirOperationCreate(MlirOperationState& state)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirOperationGetRegion(MlirOperation op, nativeint pos)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirRegionAppendOwnedBlock(MlirRegion region, MlirBlock block)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirBlock mlirBlockCreate(nativeint numArgs, MlirType& argTypes, MlirLocation& argLocs)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirBlockInsertOwnedOperation(MlirBlock block, nativeint pos, MlirOperation op)

    // Printing
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationPrint(MlirOperation op, MlirStringCallback callback, nativeint userData)
```

We also need a few more handle types that appeared in the function signatures:

```fsharp
[<Struct>]
type MlirDialectHandle =
    val Handle: nativeint
    new(handle) = { Handle = handle }

[<Struct>]
type MlirOperationState =
    val Name: MlirStringRef
    val Location: MlirLocation
    val NumResults: nativeint
    val Results: nativeint  // Pointer to MlirType array
    val NumOperands: nativeint
    val Operands: nativeint  // Pointer to MlirValue array
    val NumRegions: nativeint
    val Regions: nativeint  // Pointer to MlirRegion array
    val NumSuccessors: nativeint
    val Successors: nativeint  // Pointer to MlirBlock array
    val NumAttributes: nativeint
    val Attributes: nativeint  // Pointer to MlirNamedAttribute array
    val EnableResultTypeInference: bool
```

And the callback delegate for printing:

```fsharp
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirStringCallback = delegate of MlirStringRef * nativeint -> unit
```

This delegate allows MLIR to call back into F# code when printing IR. MLIR will invoke the callback with each chunk of the printed output.

## Building the MLIR Module

Now let's write the logic to create our MLIR module. Add this function to your script:

```fsharp
let buildHelloMlir() =
    // Step 1: Create MLIR context
    let ctx = MlirNative.mlirContextCreate()
    printfn "Created MLIR context"

    // Step 2: Load required dialects (func and arith)
    let funcDialect = MlirNative.mlirGetDialectHandle__func__()
    MlirNative.mlirDialectHandleRegisterDialect(funcDialect, ctx)
    let arithDialect = MlirNative.mlirGetDialectHandle__arith__()
    MlirNative.mlirDialectHandleRegisterDialect(arithDialect, ctx)
    printfn "Registered func and arith dialects"

    // Step 3: Create an empty module
    let loc = MlirNative.mlirLocationUnknownGet(ctx)
    let mlirModule = MlirNative.mlirModuleCreateEmpty(loc)
    printfn "Created empty module"

    // Step 4: Create the function type () -> i32
    let i32Type = MlirNative.mlirIntegerTypeGet(ctx, 32u)
    let mutable resultType = i32Type
    let funcType = MlirNative.mlirFunctionTypeGet(ctx, nativeint 0, &i32Type, nativeint 1, &resultType)
    printfn "Created function type () -> i32"

    // Step 5: Create func.func operation
    let funcName = MlirStringRef.FromString("func.func")
    let mutable funcState =
        { MlirOperationState.Name = funcName
          Location = loc
          NumResults = nativeint 0
          Results = nativeint 0
          NumOperands = nativeint 0
          Operands = nativeint 0
          NumRegions = nativeint 1  // Function body is a region
          Regions = nativeint 0
          NumSuccessors = nativeint 0
          Successors = nativeint 0
          NumAttributes = nativeint 0
          Attributes = nativeint 0
          EnableResultTypeInference = false }

    let funcOp = MlirNative.mlirOperationCreate(&funcState)
    funcName.Free()
    printfn "Created func.func operation"

    // Step 6: Create a block for the function body
    let funcRegion = MlirNative.mlirOperationGetRegion(funcOp, nativeint 0)
    let block = MlirNative.mlirBlockCreate(nativeint 0, &i32Type, &loc)
    MlirNative.mlirRegionAppendOwnedBlock(funcRegion, block)
    printfn "Created function body block"

    // Step 7: Create arith.constant 42 : i32
    let constantName = MlirStringRef.FromString("arith.constant")
    let mutable constantState =
        { MlirOperationState.Name = constantName
          Location = loc
          NumResults = nativeint 1
          Results = Marshal.AllocHGlobal(sizeof<nativeint>)
          NumOperands = nativeint 0
          Operands = nativeint 0
          NumRegions = nativeint 0
          Regions = nativeint 0
          NumSuccessors = nativeint 0
          Successors = nativeint 0
          NumAttributes = nativeint 0
          Attributes = nativeint 0
          EnableResultTypeInference = false }
    Marshal.StructureToPtr(i32Type, constantState.Results, false)

    let constantOp = MlirNative.mlirOperationCreate(&constantState)
    constantName.Free()
    Marshal.FreeHGlobal(constantState.Results)
    printfn "Created arith.constant operation"

    // Step 8: Create return operation
    let returnName = MlirStringRef.FromString("func.return")
    let mutable returnState =
        { MlirOperationState.Name = returnName
          Location = loc
          NumResults = nativeint 0
          Results = nativeint 0
          NumOperands = nativeint 1
          Operands = nativeint 0  // Should point to constant's result
          NumRegions = nativeint 0
          Regions = nativeint 0
          NumSuccessors = nativeint 0
          Successors = nativeint 0
          NumAttributes = nativeint 0
          Attributes = nativeint 0
          EnableResultTypeInference = false }

    let returnOp = MlirNative.mlirOperationCreate(&returnState)
    returnName.Free()
    printfn "Created func.return operation"

    // Step 9: Insert operations into the block
    MlirNative.mlirBlockInsertOwnedOperation(block, nativeint 0, constantOp)
    MlirNative.mlirBlockInsertOwnedOperation(block, nativeint 1, returnOp)
    printfn "Inserted operations into block"

    // Step 10: Get module operation and print
    let moduleOp = MlirNative.mlirModuleGetOperation(mlirModule)
    printfn "\n--- Generated MLIR IR ---"

    let mutable output = ""
    let callback = MlirStringCallback(fun strRef _ ->
        let length = int strRef.Length
        let bytes = Array.zeroCreate<byte> length
        Marshal.Copy(strRef.Data, bytes, 0, length)
        let text = System.Text.Encoding.UTF8.GetString(bytes)
        output <- output + text
    )

    MlirNative.mlirOperationPrint(moduleOp, callback, nativeint 0)
    printfn "%s" output
    printfn "--- End of IR ---\n"

    // Cleanup
    MlirNative.mlirModuleDestroy(mlirModule)
    MlirNative.mlirContextDestroy(ctx)
    printfn "Cleaned up MLIR context and module"
```

This function does a lot, so let's walk through it step by step.

## Step-by-Step Breakdown

### Step 1: Create MLIR Context

```fsharp
let ctx = MlirNative.mlirContextCreate()
```

The MLIR context is the root object that manages all MLIR state: registered dialects, type uniquing, memory management, etc. You must create a context before doing anything else.

### Step 2: Load Dialects

```fsharp
let funcDialect = MlirNative.mlirGetDialectHandle__func__()
MlirNative.mlirDialectHandleRegisterDialect(funcDialect, ctx)
let arithDialect = MlirNative.mlirGetDialectHandle__arith__()
MlirNative.mlirDialectHandleRegisterDialect(arithDialect, ctx)
```

MLIR dialects are loaded on-demand. We need the `func` dialect (for function definitions) and the `arith` dialect (for constants and arithmetic operations). Each dialect has a getter function (`mlirGetDialectHandle__<dialect>__`), and we register it with the context.

### Step 3: Create an Empty Module

```fsharp
let loc = MlirNative.mlirLocationUnknownGet(ctx)
let mlirModule = MlirNative.mlirModuleCreateEmpty(loc)
```

Every MLIR operation requires a source location. For generated code, we use an "unknown" location. Then we create an empty module.

### Step 4: Create Function Type

```fsharp
let i32Type = MlirNative.mlirIntegerTypeGet(ctx, 32u)
let mutable resultType = i32Type
let funcType = MlirNative.mlirFunctionTypeGet(ctx, nativeint 0, &i32Type, nativeint 1, &resultType)
```

We define the function signature: no inputs (`nativeint 0`), one output (`i32`). The `mlirFunctionTypeGet` function takes pointers to type arrays, so we use `&` to pass by reference.

### Step 5-6: Create Function Operation and Body Block

Creating operations in MLIR requires building an `MlirOperationState` and calling `mlirOperationCreate`. This is the general pattern for all operation creation:

1. Create `MlirOperationState` with operation name, location, operands, results, regions, etc.
2. Call `mlirOperationCreate(&state)`
3. Free any allocated memory (like the operation name string)

For the function, we also create a region (function body) and a block inside it.

### Step 7-8: Create Operations Inside the Function

We create two operations:

1. **`arith.constant 42 : i32`**: The constant operation. It has one result (the value 42).
2. **`func.return %result`**: The return operation. It has one operand (the constant's result).

Each operation follows the same pattern: create `MlirOperationState`, call `mlirOperationCreate`, clean up.

### Step 9: Insert Operations into Block

```fsharp
MlirNative.mlirBlockInsertOwnedOperation(block, nativeint 0, constantOp)
MlirNative.mlirBlockInsertOwnedOperation(block, nativeint 1, returnOp)
```

Operations must be inserted into a block in execution order. The constant comes first (position 0), then the return (position 1).

### Step 10: Print the IR

```fsharp
let callback = MlirStringCallback(fun strRef _ ->
    // Convert MlirStringRef to F# string
    // Accumulate in output variable
)
MlirNative.mlirOperationPrint(moduleOp, callback, nativeint 0)
```

MLIR's print functions use callbacks. The callback is invoked multiple times with chunks of the output. We accumulate these chunks into a single string and print it.

### Cleanup

```fsharp
MlirNative.mlirModuleDestroy(mlirModule)
MlirNative.mlirContextDestroy(ctx)
```

Always destroy the module and context to avoid memory leaks.

## Running the Script

Add this at the end of your `HelloMlir.fsx` file:

```fsharp
[<EntryPoint>]
let main argv =
    buildHelloMlir()
    0
```

Now run the script with F# Interactive:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet fsi HelloMlir.fsx
```

**Expected output:**

```
Created MLIR context
Registered func and arith dialects
Created empty module
Created function type () -> i32
Created func.func operation
Created function body block
Created arith.constant operation
Created func.return operation
Inserted operations into block

--- Generated MLIR IR ---
module {
  func.func @return_forty_two() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
}
--- End of IR ---

Cleaned up MLIR context and module
```

If you see this output, congratulations! You've successfully called MLIR from F# and generated IR programmatically.

## Troubleshooting

### DllNotFoundException: Unable to load shared library 'MLIR-C'

**Cause:** .NET runtime cannot find the MLIR-C shared library.

**Solution:** Ensure `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) includes `$HOME/mlir-install/lib`:

```bash
export LD_LIBRARY_PATH=$HOME/mlir-install/lib:$LD_LIBRARY_PATH
dotnet fsi HelloMlir.fsx
```

Or run with the environment variable inline:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet fsi HelloMlir.fsx
```

### AccessViolationException or Segmentation Fault

**Cause:** Incorrect P/Invoke signature (wrong parameter types, missing `&` for byref parameters, etc.).

**Solution:** Verify your `DllImport` declarations match the MLIR-C API header files exactly. Check the [MLIR-C API documentation](https://mlir.llvm.org/docs/CAPI/) and the header files in `$HOME/mlir-install/include/mlir-c/`.

### Empty or Malformed IR Output

**Cause:** Operations not properly inserted into blocks, or regions not properly attached to operations.

**Solution:** Verify the order of operations: create operation → get region → create block → insert operations into block.

## What We've Learned

In this chapter, you:

1. **Defined MLIR handle types** as F# structs wrapping native pointers.
2. **Used `[<DllImport>]`** to declare external MLIR-C API functions.
3. **Marshalled strings** using `MlirStringRef` and manual memory management.
4. **Created an MLIR context and module** from scratch.
5. **Built operations programmatically** using `MlirOperationState`.
6. **Printed MLIR IR** using callbacks.
7. **Managed memory** by destroying contexts and modules when done.

You now have proof that F# can interoperate with MLIR. But this code is messy — we're defining types and P/Invoke functions inline in a script. In a real compiler, we need these bindings organized into a reusable module.

## Next Chapter

Continue to [Chapter 03: P/Invoke Bindings](03-pinvoke-bindings.md) to learn how to organize these bindings into a proper F# module with clean APIs and comprehensive coverage of the MLIR-C API.

## Further Reading

- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) — Official guide to the MLIR C API design and usage patterns.
- [.NET P/Invoke Documentation](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke) — Comprehensive guide to Platform Invoke in .NET.
- [Marshalling in .NET](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/type-marshalling) — How .NET converts between managed and unmanaged types.
