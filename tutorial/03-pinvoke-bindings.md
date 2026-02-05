# Chapter 03: P/Invoke 바인딩

## 소개

Chapter 02에서는 MLIR IR을 생성하는 첫 번째 F# 프로그램을 작성했습니다. 핸들 타입을 정의하고, `DllImport` 선언을 작성하며, MLIR C API를 성공적으로 호출하여 간단한 함수를 만들었습니다. 하지만 그 코드는 탐색적이고 임시방편적이었습니다 -- 모든 바인딩이 스크립트 내에 인라인으로 정의되어 있었습니다.

실제 컴파일러에는 체계적이고 재사용 가능한 바인딩이 필요합니다. 이 장에서는 Chapter 02에서 배운 모든 것을 가져와 적절한 F# 모듈인 `MlirBindings.fs`로 체계화합니다. 이 모듈은 이후 모든 장의 기반이 됩니다. 이 장에서 배울 내용은 다음과 같습니다:

- 기능 영역별(context, module, type, operation 등)로 MLIR C API 바인딩을 구성하는 방법
- 문자열 마샬링을 올바르고 안전하게 처리하는 방법
- IR 출력을 위한 콜백 처리 방법
- 크로스 플랫폼 고려 사항 (Linux, macOS, Windows)

이 장을 마치면 MLIR C API에 대한 완전하고 프로덕션에 사용할 수 있는 바인딩 레이어를 갖추게 됩니다.

## 설계 철학

바인딩 레이어는 다음 원칙을 따릅니다:

1. **얇은 래퍼:** C API 위에 최소한의 추상화만 적용합니다. 각 F# 함수는 C 함수에 직접 대응됩니다.
2. **타입 안전성:** MLIR 핸들에 F# struct 타입을 사용하여 컴파일 시점에 타입 오류를 잡습니다.
3. **메모리 안전성:** 안전한 문자열 마샬링과 정리를 위한 유틸리티를 제공하되, destroy 함수를 호출해야 하는 필요성을 숨기지 않습니다.
4. **완전성:** 컴파일러에 필요한 모든 MLIR C API 함수를 다룹니다 (context, module, type, operation, region, block, location, attribute, value).
5. **문서화:** 모든 함수에 목적과 MLIR C API 대응 관계를 설명하는 주석이 있습니다.

## 프로젝트 구조

코드를 작성하기 전에 적절한 F# 프로젝트를 설정하겠습니다. Chapter 02에서는 스크립트(`.fsx`)를 사용했지만, 이제 라이브러리 프로젝트를 만들겠습니다:

```bash
cd $HOME/mlir-fsharp-tutorial
dotnet new classlib -lang F# -o MlirBindings
cd MlirBindings
```

이렇게 하면 다음과 같은 구조의 새 F# 라이브러리 프로젝트가 생성됩니다:

```
MlirBindings/
├── MlirBindings.fsproj
└── Library.fs
```

기본 `Library.fs`를 삭제합니다:

```bash
rm Library.fs
```

`MlirBindings.fs`를 처음부터 새로 만들겠습니다.

## 모듈 구성

바인딩 모듈은 다음과 같은 논리적 섹션으로 구성됩니다:

1. **핸들 타입:** MLIR 불투명 타입을 나타내는 F# struct
2. **문자열 마샬링:** `MlirStringRef`와 헬퍼 함수
3. **콜백 델리게이트:** MLIR 콜백을 위한 함수 포인터 타입
4. **Context 관리:** Context 생성, 소멸, dialect 로딩
5. **Module 관리:** Module 생성, 연산, 출력
6. **Location:** 소스 위치 유틸리티
7. **타입 시스템:** 정수 타입, 함수 타입, LLVM 타입
8. **Operation 빌딩:** Operation state 생성 및 조립
9. **Region과 Block:** Region 및 Block 생성과 관리
10. **Value와 Attribute:** SSA value 및 attribute 처리

단계별로 구축해 보겠습니다.

## 핸들 타입

`MlirBindings` 디렉토리에 새 파일 `MlirBindings.fs`를 생성합니다:

```bash
touch MlirBindings.fs
```

프로젝트 파일 `MlirBindings.fsproj`를 편집하여 파일을 추가합니다. 내용을 다음으로 교체합니다:

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

이제 `MlirBindings.fs`를 열고 namespace와 import부터 시작합니다:

```fsharp
namespace MlirBindings

open System
open System.Runtime.InteropServices
```

필요한 모든 핸들 타입을 정의합니다. 이것들은 MLIR 내부 구조체에 대한 불투명 포인터입니다:

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

각 핸들 타입에는 목적을 설명하는 문서 주석이 포함되어 있습니다. `[<Struct>]` 어트리뷰트는 이들이 스택에 할당되는 값 타입임을 보장합니다.

## 문자열 마샬링

MLIR은 소유권 의미 없이 문자열을 전달하기 위해 `MlirStringRef`를 사용합니다. 헬퍼 유틸리티와 함께 정의합니다:

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

`WithString` 헬퍼는 특히 유용합니다 -- 할당과 정리를 자동으로 처리합니다:

```fsharp
// 이렇게 하는 대신:
let strRef = MlirStringRef.FromString("func.func")
let op = createOp strRef
strRef.Free()

// 다음과 같이 작성할 수 있습니다:
MlirStringRef.WithString "func.func" (fun strRef ->
    createOp strRef
)
```

## 콜백 델리게이트

MLIR은 출력과 문자열 처리를 위해 콜백을 사용합니다. 델리게이트 타입을 정의합니다:

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

`MlirOperationState` struct는 operation을 빌드하는 데 사용됩니다. 배열에 대한 포인터를 포함하기 때문에 복잡합니다:

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

참고: `mlirOperationCreate`에 전달하기 전에 수정해야 하므로 모든 필드가 mutable입니다.

## P/Invoke 선언

이제 핵심 부분입니다: MLIR C API에 대한 P/Invoke 선언입니다. 모듈로 구성합니다:

```fsharp
module MlirNative =

    //==========================================================================
    // Context 관리
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
    // Module 관리
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
    // 타입 시스템
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
    // Attribute 시스템
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
    // Operation 빌딩
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
    // Region 관리
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
    // Block 관리
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

이것은 컴파일러 구축에 필요한 모든 MLIR C API 함수를 다루는 포괄적인 바인딩 레이어입니다. 각 함수에는 목적을 설명하는 문서가 포함되어 있습니다.

## 크로스 플랫폼 라이브러리 로딩

중요한 세부 사항이 하나 있습니다: 라이브러리 이름 `"MLIR-C"`는 .NET이 자동으로 올바른 확장자를 추가하기 때문에 플랫폼 간에 동작합니다:

- **Linux:** `libMLIR-C.so`
- **macOS:** `libMLIR-C.dylib`
- **Windows:** `MLIR-C.dll`

그러나 .NET은 런타임에 라이브러리를 어디서 찾을 수 있는지 알아야 합니다. 이 내용은 Chapter 00에서 다루었습니다 (`LD_LIBRARY_PATH` 또는 `DYLD_LIBRARY_PATH` 설정). 프로덕션 애플리케이션의 경우 여러 가지 옵션이 있습니다:

### 옵션 1: 환경 변수 (개발 시)

실행 전에 라이브러리 경로를 설정합니다:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet run
```

### 옵션 2: NativeLibrary.SetDllImportResolver (런타임)

.NET의 `NativeLibrary` API를 사용하여 커스텀 검색 경로를 지정합니다:

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

MLIR 함수를 호출하기 전에 `LibraryLoader.initialize()`를 호출합니다.

### 옵션 3: rpath (Linux/macOS 바이너리)

컴파일된 바이너리의 경우, rpath를 사용하여 실행 파일에 라이브러리 검색 경로를 내장합니다. 이 방법은 이 튜토리얼의 범위를 벗어나지만, 배포 애플리케이션의 표준 솔루션입니다.

## 헬퍼 유틸리티

자주 사용되는 패턴을 위한 고수준 헬퍼 함수를 추가합니다:

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

이 유틸리티들은 일반적인 작업을 래핑하여 사용자 코드에서 보일러플레이트를 줄여 줍니다.

## 전체 MlirBindings.fs 목록

다음은 모든 섹션이 통합된 완전한 `MlirBindings.fs` 파일입니다:

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

이것이 완전하고 프로덕션에 사용할 수 있는 MLIR 바인딩 레이어입니다.

## 라이브러리 빌드

라이브러리 프로젝트를 빌드합니다:

```bash
cd $HOME/mlir-fsharp-tutorial/MlirBindings
dotnet build
```

예상 출력:

```
Build succeeded.
    0 Warning(s)
    0 Error(s)
```

컴파일된 라이브러리는 `bin/Debug/net8.0/MlirBindings.dll`에 위치합니다.

## 바인딩 사용하기

새 바인딩을 사용하여 Chapter 02의 hello-world 예제를 다시 작성해 보겠습니다. 새 콘솔 프로젝트를 생성합니다:

```bash
cd $HOME/mlir-fsharp-tutorial
dotnet new console -lang F# -o HelloMlirWithBindings
cd HelloMlirWithBindings
dotnet add reference ../MlirBindings/MlirBindings.fsproj
```

`Program.fs`의 내용을 다음으로 교체합니다:

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

실행합니다:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet run
```

예상 출력:

```
Created MLIR context with dialects loaded
Created empty module

Generated MLIR IR:
module {
}

Cleaned up
```

Chapter 02보다 훨씬 깔끔합니다! 바인딩 모듈이 모든 마샬링과 보일러플레이트를 처리합니다.

## 이 장에서 배운 내용

이 장에서는 다음을 수행했습니다:

1. **MLIR 바인딩을 구성하여** 논리적 섹션으로 나뉜 재사용 가능한 F# 라이브러리 모듈을 만들었습니다.
2. **포괄적인 핸들 타입을 정의하여** 모든 MLIR 엔티티(context, module, operation, type, region, block, value, attribute)를 다루었습니다.
3. **안전한 문자열 마샬링을 구현하여** `MlirStringRef`와 헬퍼 유틸리티를 만들었습니다.
4. **P/Invoke 바인딩을 선언하여** 컴파일에 필요한 MLIR C API의 전체 표면적을 다루었습니다.
5. **헬퍼 유틸리티를 생성하여** 보일러플레이트를 줄였습니다 (출력, context 생성).
6. **크로스 플랫폼 고려 사항을** 이해하여 라이브러리 로딩을 다루었습니다.
7. **바인딩 라이브러리를 빌드하고 사용하여** 별도의 프로젝트에서 활용했습니다.

이제 MLIR에 대한 완전하고 프로덕션에 사용할 수 있는 바인딩 레이어를 갖추었습니다. 이 `MlirBindings` 모듈은 FunLang 컴파일러를 구축하는 이후 모든 장의 기반이 됩니다.

## 다음 장

다음 장에서는 FunLang 컴파일러 백엔드 구축을 시작합니다. 타입이 지정된 FunLang AST를 F#에서 표현하기 위한 데이터 구조를 정의하고, 여기서 만든 바인딩을 사용하여 FunLang 표현식을 MLIR operation으로 변환하는 코드 생성 로직을 작성하기 시작합니다.

**Chapter 04: FunLang AST에서 MLIR로** (작성 예정)로 이어집니다.

## 참고 자료

- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) -- 공식 C API 가이드
- [P/Invoke Best Practices](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices) -- 안전하고 고성능의 interop을 위한 Microsoft의 가이드라인
- [Memory Management in P/Invoke](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/tutorial-custom-marshaller) -- 관리/비관리 메모리 경계 이해
