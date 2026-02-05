# 챕터 02: F#에서 Hello MLIR

## 소개

챕터 00에서는 MLIR을 소스에서 빌드하고 .NET SDK를 설치했다. 챕터 01에서는 MLIR의 핵심 개념인 dialect, operation, region, block, SSA 형태에 대해 배웠다. 이제 코드를 작성할 차례다.

이 챕터는 처음으로 "동작한다!"를 경험하는 순간이다. F# 스크립트를 작성하여 P/Invoke를 통해 MLIR C API를 호출하고, MLIR context와 module을 생성하며, 산술 연산이 포함된 간단한 함수를 구성한 뒤, 결과 IR을 콘솔에 출력할 것이다. 이 챕터를 마치면 F#이 MLIR과 상호운용될 수 있다는 것을 증명하는 동작하는 프로토타입을 갖게 된다.

이 챕터의 코드는 의도적으로 즉흥적이고 탐색적이다. P/Invoke 바인딩을 인라인으로 정의하고 우선 동작하는 것에 집중한다. 챕터 03에서 이 바인딩들을 적절한 재사용 가능한 모듈로 구성할 것이다.

## 만들어 볼 것

첫 번째 MLIR 프로그램은 상수 정수를 반환하는 함수다. MLIR 텍스트 형식으로는 다음과 같다:

```mlir
module {
  func.func @return_forty_two() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
}
```

이것은 가장 간단한 MLIR 프로그램이다:
- `@return_forty_two`라는 이름의 함수 하나
- 매개변수 없음
- `i32` (32비트 정수) 반환
- 본문에서 상수 `42`를 생성하고 반환

이것을 MLIR의 C API를 사용하여 F#에서 프로그래밍 방식으로 구성할 것이다.

## P/Invoke 이해하기

P/Invoke (Platform Invoke)는 .NET의 외부 함수 인터페이스(FFI) 메커니즘이다. 관리 코드(F#, C# 등)에서 공유 라이브러리(Linux의 `.so`, macOS의 `.dylib`, Windows의 `.dll`)에 있는 비관리 네이티브 함수를 호출할 수 있게 해준다.

### DllImport 속성

네이티브 함수를 호출하려면 `[<DllImport>]` 속성을 사용하여 함수 시그니처를 선언한다. 패턴은 다음과 같다:

```fsharp
[<DllImport("library-name", CallingConvention = CallingConvention.Cdecl)>]
extern ReturnType functionName(ParamType1 param1, ParamType2 param2)
```

하나씩 살펴본다:

- **`[<DllImport("library-name")>]`**: 함수가 포함된 공유 라이브러리를 지정한다. MLIR의 경우 `"MLIR-C"`이다(파일 확장자 없이 -- .NET이 플랫폼에 따라 자동으로 `.so`, `.dylib`, `.dll`을 추가한다).

- **`CallingConvention = CallingConvention.Cdecl`**: 인수 전달 및 스택 관리 방식을 지정한다. MLIR C API는 C 라이브러리의 표준인 C 호출 규약(`Cdecl`)을 사용한다.

- **`extern`**: 네이티브 코드에 정의된 외부 함수임을 표시한다.

- **반환 타입과 매개변수**: C 함수 시그니처와 정확히 일치해야 한다. MLIR은 불투명 구조체 핸들(내부 데이터 구조에 대한 포인터)을 사용하며, F#에서는 이를 `nativeint`로 표현한다.

### MLIR 핸들 타입

MLIR C API는 모든 IR 엔티티에 불투명 구조체 타입을 사용한다:

```c
// MLIR-C API (C header)
typedef struct MlirContext { void *ptr; } MlirContext;
typedef struct MlirModule { void *ptr; } MlirModule;
typedef struct MlirOperation { void *ptr; } MlirOperation;
// ... and many more
```

각 구조체는 포인터를 감싸는 래퍼다. F#의 관점에서는 내부 구조에 관심이 없고, MLIR 함수 간에 이 핸들들을 전달하기만 하면 된다. 단일 `nativeint` 필드를 가진 F# 구조체로 표현한다:

```fsharp
[<Struct>]
type MlirContext =
    val Handle: nativeint
    new(handle) = { Handle = handle }
```

이는 C 메모리 레이아웃(단일 포인터)과 일치하며, P/Invoke 경계를 넘어 안전하게 전달할 수 있다.

## F# 스크립트 생성

코드를 작성해 본다. 작업 디렉터리에 `HelloMlir.fsx`라는 새 파일을 생성한다:

```bash
cd $HOME
mkdir -p mlir-fsharp-tutorial
cd mlir-fsharp-tutorial
touch HelloMlir.fsx
```

텍스트 편집기에서 `HelloMlir.fsx`를 열고 필요한 import부터 시작한다:

```fsharp
open System
open System.Runtime.InteropServices
```

- `System`: .NET 핵심 타입
- `System.Runtime.InteropServices`: `DllImport`, `CallingConvention`, 마샬링 속성 포함

## 핸들 타입 정의

먼저 필요한 MLIR 핸들 타입을 정의한다. 이 간단한 예제에서는 다음이 필요하다:

- `MlirContext`: MLIR 루트 context (메모리, dialect 등을 관리)
- `MlirModule`: module (함수의 최상위 컨테이너)
- `MlirLocation`: 소스 위치 정보 (operation 생성에 필요)
- `MlirType`: 타입 시스템 (`i32` 사용 예정)
- `MlirBlock`: 기본 블록
- `MlirRegion`: 블록을 포함하는 region
- `MlirOperation`: operation (함수나 산술 연산 생성 결과)
- `MlirValue`: SSA 값 (operation의 결과)

스크립트에 다음 타입 정의를 추가한다:

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

각 핸들은 네이티브 포인터를 감싸는 얇은 래퍼다. `[<Struct>]` 속성은 이들이 힙에 할당되는 클래스가 아닌 스택에 할당되는 값 타입임을 보장하며, 작은 래퍼에 대해 더 효율적이다.

## 문자열 마샬링: MlirStringRef

MLIR의 C API는 소유권 문제 없이 문자열을 전달하기 위해 `MlirStringRef`라는 사용자 정의 문자열 구조체를 사용한다. C에서는 다음과 같이 정의되어 있다:

```c
typedef struct MlirStringRef {
    const char *data;
    size_t length;
} MlirStringRef;
```

이 레이아웃을 F#에서 맞춰야 한다:

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

세부 사항을 살펴본다:

- **`[<StructLayout(LayoutKind.Sequential)>]`**: 필드가 선언된 순서대로 메모리에 배치되도록 보장한다 (C 구조체와 일치).

- **`FromString(s: string)`**: F# 문자열을 `MlirStringRef`로 변환하는 헬퍼다. 비관리 메모리를 할당하고, UTF-8 바이트를 복사한 후, 해당 메모리를 가리키는 `MlirStringRef`를 반환한다.

- **`Free()`**: 비관리 메모리를 해제한다. 문자열을 MLIR에 전달한 후 반드시 호출해야 하며, 그렇지 않으면 메모리 누수가 발생한다.

## P/Invoke 함수 선언

이제 P/Invoke 선언을 작성한다. 이 예제에 필요한 함수만 선언한다. 스크립트에 다음을 추가한다:

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

함수 시그니처에 등장한 추가 핸들 타입도 필요하다:

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

그리고 출력을 위한 콜백 delegate도 필요하다:

```fsharp
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type MlirStringCallback = delegate of MlirStringRef * nativeint -> unit
```

이 delegate는 IR 출력 시 MLIR이 F# 코드를 콜백할 수 있게 해준다. MLIR은 출력의 각 청크마다 콜백을 호출한다.

## MLIR Module 구성하기

이제 MLIR module을 생성하는 로직을 작성한다. 스크립트에 다음 함수를 추가한다:

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

이 함수에는 많은 내용이 있으므로 단계별로 살펴본다.

## 단계별 분석

### 1단계: MLIR Context 생성

```fsharp
let ctx = MlirNative.mlirContextCreate()
```

MLIR context는 등록된 dialect, 타입 고유화, 메모리 관리 등 모든 MLIR 상태를 관리하는 루트 객체다. 다른 작업을 하기 전에 반드시 context를 생성해야 한다.

### 2단계: Dialect 로드

```fsharp
let funcDialect = MlirNative.mlirGetDialectHandle__func__()
MlirNative.mlirDialectHandleRegisterDialect(funcDialect, ctx)
let arithDialect = MlirNative.mlirGetDialectHandle__arith__()
MlirNative.mlirDialectHandleRegisterDialect(arithDialect, ctx)
```

MLIR dialect은 요청 시 로드된다. 함수 정의를 위한 `func` dialect과 상수 및 산술 연산을 위한 `arith` dialect이 필요하다. 각 dialect에는 getter 함수(`mlirGetDialectHandle__<dialect>__`)가 있으며, 이를 context에 등록한다.

### 3단계: 빈 Module 생성

```fsharp
let loc = MlirNative.mlirLocationUnknownGet(ctx)
let mlirModule = MlirNative.mlirModuleCreateEmpty(loc)
```

모든 MLIR operation에는 소스 위치가 필요하다. 생성된 코드의 경우 "unknown" 위치를 사용한다. 그런 다음 빈 module을 생성한다.

### 4단계: 함수 타입 생성

```fsharp
let i32Type = MlirNative.mlirIntegerTypeGet(ctx, 32u)
let mutable resultType = i32Type
let funcType = MlirNative.mlirFunctionTypeGet(ctx, nativeint 0, &i32Type, nativeint 1, &resultType)
```

함수 시그니처를 정의한다: 입력 없음(`nativeint 0`), 출력 하나(`i32`). `mlirFunctionTypeGet` 함수는 타입 배열에 대한 포인터를 받으므로 `&`를 사용하여 참조로 전달한다.

### 5-6단계: 함수 Operation 및 본문 Block 생성

MLIR에서 operation을 생성하려면 `MlirOperationState`를 구성하고 `mlirOperationCreate`를 호출해야 한다. 이것이 모든 operation 생성의 일반적인 패턴이다:

1. operation 이름, 위치, 피연산자, 결과, region 등을 포함하는 `MlirOperationState` 생성
2. `mlirOperationCreate(&state)` 호출
3. 할당된 메모리(operation 이름 문자열 등) 해제

함수의 경우 region(함수 본문)과 그 안의 block도 생성한다.

### 7-8단계: 함수 내부 Operation 생성

두 개의 operation을 생성한다:

1. **`arith.constant 42 : i32`**: 상수 operation이다. 하나의 결과(값 42)를 가진다.
2. **`func.return %result`**: 반환 operation이다. 하나의 피연산자(상수의 결과)를 가진다.

각 operation은 동일한 패턴을 따른다: `MlirOperationState` 생성, `mlirOperationCreate` 호출, 정리.

### 9단계: Operation을 Block에 삽입

```fsharp
MlirNative.mlirBlockInsertOwnedOperation(block, nativeint 0, constantOp)
MlirNative.mlirBlockInsertOwnedOperation(block, nativeint 1, returnOp)
```

Operation은 실행 순서대로 block에 삽입해야 한다. 상수가 먼저(위치 0), 그다음 반환(위치 1)이다.

### 10단계: IR 출력

```fsharp
let callback = MlirStringCallback(fun strRef _ ->
    // MlirStringRef를 F# 문자열로 변환
    // output 변수에 누적
)
MlirNative.mlirOperationPrint(moduleOp, callback, nativeint 0)
```

MLIR의 출력 함수는 콜백을 사용한다. 콜백은 출력의 청크마다 여러 번 호출된다. 이 청크들을 하나의 문자열로 누적하여 출력한다.

### 정리

```fsharp
MlirNative.mlirModuleDestroy(mlirModule)
MlirNative.mlirContextDestroy(ctx)
```

메모리 누수를 방지하기 위해 항상 module과 context를 파괴해야 한다.

## 스크립트 실행

`HelloMlir.fsx` 파일 끝에 다음을 추가한다:

```fsharp
[<EntryPoint>]
let main argv =
    buildHelloMlir()
    0
```

이제 F# Interactive로 스크립트를 실행한다:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet fsi HelloMlir.fsx
```

**예상 출력:**

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

이 출력이 보인다면 성공이다! F#에서 MLIR을 호출하고 프로그래밍 방식으로 IR을 생성하는 데 성공한 것이다.

## 문제 해결

### DllNotFoundException: Unable to load shared library 'MLIR-C'

**원인:** .NET 런타임이 MLIR-C 공유 라이브러리를 찾을 수 없다.

**해결 방법:** `LD_LIBRARY_PATH` (Linux) 또는 `DYLD_LIBRARY_PATH` (macOS)에 `$HOME/mlir-install/lib`이 포함되어 있는지 확인한다:

```bash
export LD_LIBRARY_PATH=$HOME/mlir-install/lib:$LD_LIBRARY_PATH
dotnet fsi HelloMlir.fsx
```

또는 환경 변수를 인라인으로 지정하여 실행한다:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet fsi HelloMlir.fsx
```

### AccessViolationException 또는 Segmentation Fault

**원인:** 잘못된 P/Invoke 시그니처 (잘못된 매개변수 타입, byref 매개변수에 `&` 누락 등).

**해결 방법:** `DllImport` 선언이 MLIR-C API 헤더 파일과 정확히 일치하는지 확인한다. [MLIR-C API 문서](https://mlir.llvm.org/docs/CAPI/)와 `$HOME/mlir-install/include/mlir-c/`의 헤더 파일을 참고한다.

### 비어있거나 잘못된 형식의 IR 출력

**원인:** Operation이 block에 제대로 삽입되지 않았거나, region이 operation에 제대로 연결되지 않았다.

**해결 방법:** 연산 순서를 확인한다: operation 생성 -> region 가져오기 -> block 생성 -> block에 operation 삽입.

## 배운 내용

이 챕터에서 다음을 배웠다:

1. **MLIR 핸들 타입 정의** - 네이티브 포인터를 감싸는 F# 구조체로 정의했다.
2. **`[<DllImport>]` 사용** - 외부 MLIR-C API 함수를 선언했다.
3. **문자열 마샬링** - `MlirStringRef`와 수동 메모리 관리를 사용했다.
4. **MLIR context와 module 생성** - 처음부터 생성했다.
5. **프로그래밍 방식으로 operation 구성** - `MlirOperationState`를 사용했다.
6. **MLIR IR 출력** - 콜백을 사용했다.
7. **메모리 관리** - 완료 후 context와 module을 파괴했다.

이제 F#이 MLIR과 상호운용될 수 있다는 것이 증명되었다. 하지만 이 코드는 정돈되지 않았다 -- 타입과 P/Invoke 함수를 스크립트에 인라인으로 정의하고 있다. 실제 컴파일러에서는 이 바인딩들이 재사용 가능한 모듈로 구성되어야 한다.

## 다음 챕터

[챕터 03: P/Invoke 바인딩](03-pinvoke-bindings.md)으로 이어서 이 바인딩들을 깔끔한 API와 MLIR-C API의 포괄적인 커버리지를 갖춘 적절한 F# 모듈로 구성하는 방법을 배운다.

## 추가 참고 자료

- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) -- MLIR C API 설계 및 사용 패턴에 대한 공식 가이드.
- [.NET P/Invoke Documentation](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke) -- .NET에서의 Platform Invoke 종합 가이드.
- [Marshalling in .NET](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/type-marshalling) -- .NET이 관리 타입과 비관리 타입 간에 변환하는 방법.
