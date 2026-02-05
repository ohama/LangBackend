# Chapter 04: F# 래퍼 레이어

## 소개

Chapter 03에서는 MLIR C API에 대한 완전한 P/Invoke 바인딩 모듈인 `MlirBindings.fs`를 구축했다. 이제 Context를 생성하고, Module을 만들며, Operation을 구성하는 등 MLIR C API의 모든 기능을 F#에서 호출할 수 있다.

하지만 Chapter 02와 03의 코드를 살펴보면 몇 가지 문제점이 드러난다:

**문제 1: 리소스 누수 위험**

```fsharp
let ctx = MlirNative.mlirContextCreate()
let loc = MlirNative.mlirLocationUnknownGet(ctx)
let mlirMod = MlirNative.mlirModuleCreateEmpty(loc)

// ... IR 구축 ...

// 정리를 잊어버리면 메모리 누수 발생
MlirNative.mlirModuleDestroy(mlirMod)
MlirNative.mlirContextDestroy(ctx)
```

수동으로 `Destroy` 함수를 호출해야 한다. 예외가 발생하거나 조기 반환이 있으면 리소스가 누수된다.

**문제 2: 장황함**

```fsharp
let state = MlirNative.mlirOperationStateGet(
    MlirStringRef.FromString("arith.constant"),
    location)
MlirNative.mlirOperationStateAddResults(&state, 1, &intType)
// ... 더 많은 state 조작 ...
let op = MlirNative.mlirOperationCreate(&state)
```

Operation 하나를 만드는데 5-10줄의 코드가 필요하다. 반복적이고 오류가 발생하기 쉽다.

**문제 3: 타입 안전성 부족**

```fsharp
let ctx = MlirNative.mlirContextCreate()
MlirNative.mlirContextDestroy(ctx)
// ctx는 이제 무효하지만, 타입 시스템이 이를 막지 못한다
let loc = MlirNative.mlirLocationUnknownGet(ctx) // 버그!
```

핸들을 해제한 후에도 여전히 사용할 수 있다. C API는 이를 막지 못한다.

이 장에서는 이러한 문제들을 해결하는 **래퍼 레이어**를 구축한다. 이 레이어는 원시 P/Invoke 바인딩을 관용적인 F# API로 감싸서 다음을 제공한다:

- **자동 리소스 관리**: `IDisposable`과 `use` 키워드
- **간결한 API**: `OpBuilder.CreateConstant(42)` 같은 유창한 빌더
- **생명주기 안전성**: 부모 객체가 자식보다 먼저 파괴되는 것을 방지

이 장을 마치면 튜토리얼의 나머지 부분에서 사용할 깔끔하고 안전한 MLIR API를 갖게 된다.

## 소유권 문제

MLIR은 엄격한 소유권 계층 구조를 갖는다:

```
Context (root)
  └─ Module
       └─ Operation
            └─ Region
                 └─ Block
                      └─ Operation
```

각 객체는 부모에 속한다:
- **Module**은 **Context**가 소유한다
- **Operation**은 **Block**이 소유한다
- **Block**은 **Region**이 소유한다
- **Region**은 **Operation**이 소유한다

C++에서는 이 소유권이 자동으로 관리된다 (RAII와 unique_ptr). 부모가 파괴되면 자식도 자동으로 파괴된다.

P/Invoke에서는 이 소유권을 수동으로 관리해야 한다. 문제는 부모를 먼저 파괴하면 자식 핸들이 무효가 된다는 것이다:

```fsharp
// 버그가 있는 코드
let ctx = MlirNative.mlirContextCreate()
let loc = MlirNative.mlirLocationUnknownGet(ctx)
let mlirMod = MlirNative.mlirModuleCreateEmpty(loc)

// Context를 먼저 파괴
MlirNative.mlirContextDestroy(ctx)

// Module 핸들이 이제 무효 - 위험한 포인터!
MlirNative.mlirModuleGetOperation(mlirMod) // 세그멘테이션 폴트
```

F#의 가비지 컬렉터는 MLIR의 소유권 규칙을 알지 못한다. 따라서 우리가 강제해야 한다.

**해결책:** F# 래퍼는 부모 객체에 대한 참조를 저장한다. 자식이 살아있는 한 부모는 가비지 컬렉트되지 않는다.

```fsharp
type Module(context: Context, location: Location) =
    let handle = MlirNative.mlirModuleCreateEmpty(location.Handle)
    let contextRef = context  // 부모 참조 유지 - Context가 먼저 GC되는 것을 방지

    member _.Handle = handle

    interface IDisposable with
        member _.Dispose() =
            MlirNative.mlirModuleDestroy(handle)
```

## Context 래퍼

MLIR의 최상위 객체인 Context부터 시작한다. 새 파일 `MlirWrapper.fs`를 만든다:

```fsharp
namespace MlirWrapper

open System
open MlirBindings

/// MLIR Context를 나타낸다. 모든 MLIR 객체의 소유자이며 메모리 관리를 담당한다.
/// Context는 dialect와 type을 등록하고 IR 구성을 위한 전역 환경을 제공한다.
type Context() =
    let mutable handle = MlirNative.mlirContextCreate()
    let mutable disposed = false

    /// 기본 MLIR context 핸들
    member _.Handle = handle

    /// 이 context에 dialect를 로드한다.
    /// dialect: 로드할 dialect의 이름 (예: "arith", "func", "llvm")
    member _.LoadDialect(dialect: string) =
        if disposed then
            raise (ObjectDisposedException("Context"))

        MlirStringRef.WithString dialect (fun nameRef ->
            MlirNative.mlirContextGetOrLoadDialect(handle, nameRef)
            |> ignore)

    interface IDisposable with
        member this.Dispose() =
            this.Dispose(true)
            GC.SuppressFinalize(this)

    member private _.Dispose(disposing: bool) =
        if not disposed then
            if disposing then
                // 관리 리소스 정리 (이 경우 없음)
                ()

            // 비관리 리소스 정리
            MlirNative.mlirContextDestroy(handle)
            handle <- Unchecked.defaultof<_>
            disposed <- true
```

> **설계 결정:** `disposed` 플래그는 이중 해제를 방지한다. 동일한 Context에서 `Dispose()`를 두 번 호출하는 것은 안전하다 (두 번째 호출은 아무 작업도 하지 않는다).

**사용 예:**

```fsharp
let example () =
    use ctx = new Context()          // Context 생성
    ctx.LoadDialect("arith")         // Arithmetic dialect 로드
    ctx.LoadDialect("func")          // Function dialect 로드

    // ctx 사용...
    printfn "Context created: %A" ctx.Handle

    // 스코프가 끝나면 자동으로 Dispose 호출됨 - mlirContextDestroy 호출
```

F#의 `use` 키워드는 C#의 `using`과 동일하다. 스코프가 끝나면 자동으로 `Dispose()`를 호출한다. 예외가 발생해도 정리가 보장된다.

## Location 래퍼

Location은 MLIR의 가벼운 값 타입이다. 리소스를 소유하지 않으므로 `IDisposable`이 필요하지 않다:

```fsharp
/// MLIR IR에서 소스 위치를 나타낸다. 컴파일 오류 보고에 사용된다.
type Location =
    | Unknown of Context
    | FileLineCol of Context * filename: string * line: int * col: int

    /// 기본 MLIR location 핸들을 반환한다
    member this.Handle =
        match this with
        | Unknown ctx ->
            MlirNative.mlirLocationUnknownGet(ctx.Handle)

        | FileLineCol (ctx, filename, line, col) ->
            MlirStringRef.WithString filename (fun filenameRef ->
                MlirNative.mlirLocationFileLineColGet(
                    ctx.Handle,
                    filenameRef,
                    uint32 line,
                    uint32 col))
```

> **설계 결정:** 모든 MLIR 타입이 `IDisposable`을 필요로 하는 것은 아니다. Location, Type, Attribute는 값 타입이며 Context가 소유한다. 명시적 정리가 필요 없다.

**사용 예:**

```fsharp
use ctx = new Context()

let loc1 = Location.Unknown(ctx)
let loc2 = Location.FileLineCol(ctx, "example.fun", 10, 5)

printfn "Unknown location: %A" loc1.Handle
printfn "File location: %A" loc2.Handle
```

## Module 래퍼

Module은 MLIR IR의 최상위 컨테이너다. 여러 함수와 전역 선언을 포함한다:

```fsharp
/// MLIR Module - 최상위 IR 컨테이너. 함수와 전역 선언을 포함한다.
type Module(context: Context, location: Location) =
    let handle = MlirNative.mlirModuleCreateEmpty(location.Handle)
    let contextRef = context  // Context 참조 유지 - 조기 GC 방지
    let mutable disposed = false

    /// 기본 MLIR module 핸들
    member _.Handle = handle

    /// 이 module이 속한 context
    member _.Context = contextRef

    /// 이 module의 body block을 반환한다 (최상위 operation들을 포함)
    member _.Body =
        let op = MlirNative.mlirModuleGetOperation(handle)
        let region = MlirNative.mlirOperationGetRegion(op, 0n)
        MlirNative.mlirRegionGetFirstBlock(region)

    /// MLIR IR을 검증한다. 모든 operation이 올바른 형식인지 확인한다.
    member _.Verify() =
        let op = MlirNative.mlirModuleGetOperation(handle)
        MlirNative.mlirOperationVerify(op)

    /// MLIR IR을 문자열로 출력한다
    member _.Print() =
        let op = MlirNative.mlirModuleGetOperation(handle)
        MlirHelpers.operationToString(op)

    interface IDisposable with
        member this.Dispose() =
            this.Dispose(true)
            GC.SuppressFinalize(this)

    member private _.Dispose(disposing: bool) =
        if not disposed then
            if disposing then
                ()

            MlirNative.mlirModuleDestroy(handle)
            disposed <- true
```

> **설계 결정:** `contextRef` 필드는 Module이 존재하는 한 Context가 가비지 컬렉트되지 않도록 보장한다. 이는 소유권 안전성의 핵심이다.

**사용 예:**

```fsharp
use ctx = new Context()
ctx.LoadDialect("func")

let loc = Location.Unknown(ctx)
use mlirMod = new Module(ctx, loc)

// IR 구축...

if mlirMod.Verify() then
    printfn "Module IR:\n%s" (mlirMod.Print())
else
    failwith "IR verification failed"
```

## OpBuilder: IR 구축을 위한 유창한 API

Operation을 만드는 것은 MLIR에서 가장 복잡한 작업이다. 원시 C API는 다음과 같다:

```fsharp
// 원시 P/Invoke - 15줄
let mutable state = MlirNative.mlirOperationStateGet(
    MlirStringRef.FromString("arith.constant"), location)

let mutable intType = MlirNative.mlirIntegerTypeGet(ctx, 32u)
MlirNative.mlirOperationStateAddResults(&state, 1, &intType)

let value = 42
let mutable attr = MlirNative.mlirIntegerAttrGet(intType, int64 value)
let mutable attrName = MlirStringRef.FromString("value")
MlirNative.mlirOperationStateAddAttributes(&state, 1, &attrName, &attr)

let op = MlirNative.mlirOperationCreate(&state)
```

이것을 한 줄로 줄이고 싶다:

```fsharp
let op = builder.CreateConstant(42, intType, location)
```

`OpBuilder` 클래스가 이를 가능하게 한다:

```fsharp
/// MLIR operation을 구축하기 위한 유창한 빌더 API.
/// 원시 operation state 조작을 숨기고 일반적인 operation에 대한 고수준 메서드를 제공한다.
type OpBuilder(context: Context) =
    let contextRef = context

    /// i32 타입을 반환한다
    member _.I32Type() =
        MlirNative.mlirIntegerTypeGet(contextRef.Handle, 32u)

    /// i64 타입을 반환한다
    member _.I64Type() =
        MlirNative.mlirIntegerTypeGet(contextRef.Handle, 64u)

    /// 함수 타입을 생성한다 (inputs -> results)
    member _.FunctionType(inputs: MlirType[], results: MlirType[]) =
        let mutable inputsArray = inputs
        let mutable resultsArray = results
        MlirNative.mlirFunctionTypeGet(
            contextRef.Handle,
            unativeint inputs.Length,
            &&inputsArray.[0],
            unativeint results.Length,
            &&resultsArray.[0])

    /// 정수 상수 operation을 생성한다: arith.constant
    member _.CreateConstant(value: int, typ: MlirType, location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("arith.constant"),
            location.Handle)

        // 결과 타입 추가
        let mutable resultType = typ
        MlirNative.mlirOperationStateAddResults(&state, 1n, &&resultType)

        // value attribute 추가
        let mutable attr = MlirNative.mlirIntegerAttrGet(typ, int64 value)
        let mutable attrName = MlirStringRef.FromString("value")
        MlirNative.mlirOperationStateAddAttributes(&state, 1n, &&attrName, &&attr)

        MlirNative.mlirOperationCreate(&state)

    /// 함수 operation을 생성한다: func.func
    member _.CreateFunction(name: string, funcType: MlirType, location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("func.func"),
            location.Handle)

        // sym_name attribute 추가 (함수 이름)
        MlirStringRef.WithString name (fun nameRef ->
            let mutable attr = MlirNative.mlirStringAttrGet(contextRef.Handle, nameRef)
            let mutable attrName = MlirStringRef.FromString("sym_name")
            MlirNative.mlirOperationStateAddAttributes(&state, 1n, &&attrName, &&attr))

        // function_type attribute 추가
        let mutable funcTypeAttr = MlirNative.mlirTypeAttrGet(funcType)
        let mutable funcTypeAttrName = MlirStringRef.FromString("function_type")
        MlirNative.mlirOperationStateAddAttributes(&state, 1n, &&funcTypeAttrName, &&funcTypeAttr)

        // body region 추가
        let mutable numRegions = 1n
        MlirNative.mlirOperationStateAddOwnedRegions(&state, numRegions)

        MlirNative.mlirOperationCreate(&state)

    /// return operation을 생성한다: func.return
    member _.CreateReturn(values: MlirValue[], location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("func.return"),
            location.Handle)

        // operand 추가
        if values.Length > 0 then
            let mutable operands = values
            MlirNative.mlirOperationStateAddOperands(&state, unativeint values.Length, &&operands.[0])

        MlirNative.mlirOperationCreate(&state)

    /// Block에서 operation의 결과 value를 가져온다
    member _.GetResult(op: MlirOperation, index: int) =
        MlirNative.mlirOperationGetResult(op, unativeint index)
```

> **설계 결정:** `OpBuilder`는 MLIR의 복잡성 대부분을 숨긴다. 일반적인 operation (constant, function, return)에 대해 고수준 메서드를 제공한다. 드물게 사용되는 operation은 직접 원시 API를 사용할 수 있다.

## 타입 헬퍼

타입 생성을 더 편리하게 만드는 모듈:

```fsharp
/// MLIR 타입 생성을 위한 헬퍼 함수들
module MLIRType =
    /// i32 타입을 반환한다
    let i32 (ctx: Context) =
        MlirNative.mlirIntegerTypeGet(ctx.Handle, 32u)

    /// i64 타입을 반환한다
    let i64 (ctx: Context) =
        MlirNative.mlirIntegerTypeGet(ctx.Handle, 64u)

    /// 함수 타입을 생성한다
    let func (ctx: Context) (inputs: MlirType[]) (results: MlirType[]) =
        let mutable inputsArray = inputs
        let mutable resultsArray = results
        MlirNative.mlirFunctionTypeGet(
            ctx.Handle,
            unativeint inputs.Length,
            (if inputs.Length > 0 then &&inputsArray.[0] else nativeint 0),
            unativeint results.Length,
            (if results.Length > 0 then &&resultsArray.[0] else nativeint 0))
```

## 모두 함께 사용하기

이제 래퍼를 사용하여 Chapter 02의 "hello-mlir" 예제를 다시 작성해 본다. 비교를 위해 두 버전을 나란히 보자:

**원시 P/Invoke 버전 (Chapter 02):**

```fsharp
// 35+ 줄, 수동 정리, 장황함
let ctx = MlirNative.mlirContextCreate()

MlirStringRef.WithString "arith" (fun dialectName ->
    MlirNative.mlirContextGetOrLoadDialect(ctx, dialectName) |> ignore)

MlirStringRef.WithString "func" (fun dialectName ->
    MlirNative.mlirContextGetOrLoadDialect(ctx, dialectName) |> ignore)

let loc = MlirNative.mlirLocationUnknownGet(ctx)
let mlirMod = MlirNative.mlirModuleCreateEmpty(loc)

// ... 더 많은 장황한 코드 ...

MlirNative.mlirModuleDestroy(mlirMod)
MlirNative.mlirContextDestroy(ctx)
```

**래퍼 버전 (Chapter 04):**

```fsharp
// 20줄, 자동 정리, 간결함
open MlirWrapper

let buildHelloMlir () =
    use ctx = new Context()
    ctx.LoadDialect("arith")
    ctx.LoadDialect("func")

    let loc = Location.Unknown(ctx)
    use mlirMod = new Module(ctx, loc)

    let builder = OpBuilder(ctx)
    let i32Type = builder.I32Type()

    // 함수 타입 생성: () -> i32
    let funcType = builder.FunctionType([||], [| i32Type |])

    // 함수 operation 생성
    let funcOp = builder.CreateFunction("return_forty_two", funcType, loc)

    // 함수 body의 첫 번째 region과 block 가져오기
    let bodyRegion = MlirNative.mlirOperationGetRegion(funcOp, 0n)
    let entryBlock = MlirNative.mlirRegionGetFirstBlock(bodyRegion)

    // entry block이 비어있는지 확인하고, 비어있으면 새로 생성
    let block =
        if MlirNative.mlirBlockIsNull(entryBlock) then
            let newBlock = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
            MlirNative.mlirRegionAppendOwnedBlock(bodyRegion, newBlock)
            newBlock
        else
            entryBlock

    // 상수 operation 생성: %c42 = arith.constant 42 : i32
    let constOp = builder.CreateConstant(42, i32Type, loc)
    MlirNative.mlirBlockAppendOwnedOperation(block, constOp)

    // 상수의 결과 value 가져오기
    let constValue = builder.GetResult(constOp, 0)

    // return operation 생성: return %c42 : i32
    let returnOp = builder.CreateReturn([| constValue |], loc)
    MlirNative.mlirBlockAppendOwnedOperation(block, returnOp)

    // 함수를 module에 추가
    MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, funcOp)

    // 검증 및 출력
    if mlirMod.Verify() then
        printfn "Generated MLIR:\n%s" (mlirMod.Print())
    else
        failwith "Module verification failed"

    // use가 자동으로 정리 처리
```

**개선 사항:**

1. **자동 정리**: `use` 키워드가 `Dispose()`를 자동으로 호출한다
2. **간결성**: `builder.CreateConstant(42, i32Type, loc)` vs. 15줄의 state 조작
3. **타입 안전성**: Context 참조가 Module이 살아있는 동안 유지됨을 보장
4. **가독성**: 의도가 명확하고 보일러플레이트가 적음

## 완전한 래퍼 모듈 리스팅

다음은 완전한 `MlirWrapper.fs` 파일이다:

```fsharp
namespace MlirWrapper

open System
open MlirBindings

/// MLIR Context - 모든 MLIR 객체의 소유자
type Context() =
    let mutable handle = MlirNative.mlirContextCreate()
    let mutable disposed = false

    member _.Handle = handle

    member _.LoadDialect(dialect: string) =
        if disposed then
            raise (ObjectDisposedException("Context"))

        MlirStringRef.WithString dialect (fun nameRef ->
            MlirNative.mlirContextGetOrLoadDialect(handle, nameRef)
            |> ignore)

    interface IDisposable with
        member this.Dispose() =
            this.Dispose(true)
            GC.SuppressFinalize(this)

    member private _.Dispose(disposing: bool) =
        if not disposed then
            if disposing then
                ()
            MlirNative.mlirContextDestroy(handle)
            handle <- Unchecked.defaultof<_>
            disposed <- true

/// MLIR Location - 소스 위치 정보
type Location =
    | Unknown of Context
    | FileLineCol of Context * filename: string * line: int * col: int

    member this.Handle =
        match this with
        | Unknown ctx ->
            MlirNative.mlirLocationUnknownGet(ctx.Handle)
        | FileLineCol (ctx, filename, line, col) ->
            MlirStringRef.WithString filename (fun filenameRef ->
                MlirNative.mlirLocationFileLineColGet(
                    ctx.Handle, filenameRef, uint32 line, uint32 col))

/// MLIR Module - 최상위 IR 컨테이너
type Module(context: Context, location: Location) =
    let handle = MlirNative.mlirModuleCreateEmpty(location.Handle)
    let contextRef = context
    let mutable disposed = false

    member _.Handle = handle
    member _.Context = contextRef

    member _.Body =
        let op = MlirNative.mlirModuleGetOperation(handle)
        let region = MlirNative.mlirOperationGetRegion(op, 0n)
        MlirNative.mlirRegionGetFirstBlock(region)

    member _.Verify() =
        let op = MlirNative.mlirModuleGetOperation(handle)
        MlirNative.mlirOperationVerify(op)

    member _.Print() =
        let op = MlirNative.mlirModuleGetOperation(handle)
        MlirHelpers.operationToString(op)

    interface IDisposable with
        member this.Dispose() =
            this.Dispose(true)
            GC.SuppressFinalize(this)

    member private _.Dispose(disposing: bool) =
        if not disposed then
            if disposing then
                ()
            MlirNative.mlirModuleDestroy(handle)
            disposed <- true

/// Operation 빌더 - 유창한 IR 구축 API
type OpBuilder(context: Context) =
    let contextRef = context

    member _.I32Type() =
        MlirNative.mlirIntegerTypeGet(contextRef.Handle, 32u)

    member _.I64Type() =
        MlirNative.mlirIntegerTypeGet(contextRef.Handle, 64u)

    member _.FunctionType(inputs: MlirType[], results: MlirType[]) =
        let mutable inputsArray = inputs
        let mutable resultsArray = results
        MlirNative.mlirFunctionTypeGet(
            contextRef.Handle,
            unativeint inputs.Length,
            (if inputs.Length > 0 then &&inputsArray.[0] else nativeint 0),
            unativeint results.Length,
            (if results.Length > 0 then &&resultsArray.[0] else nativeint 0))

    member _.CreateConstant(value: int, typ: MlirType, location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("arith.constant"), location.Handle)

        let mutable resultType = typ
        MlirNative.mlirOperationStateAddResults(&state, 1n, &&resultType)

        let mutable attr = MlirNative.mlirIntegerAttrGet(typ, int64 value)
        let mutable attrName = MlirStringRef.FromString("value")
        MlirNative.mlirOperationStateAddAttributes(&state, 1n, &&attrName, &&attr)

        MlirNative.mlirOperationCreate(&state)

    member _.CreateFunction(name: string, funcType: MlirType, location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("func.func"), location.Handle)

        MlirStringRef.WithString name (fun nameRef ->
            let mutable attr = MlirNative.mlirStringAttrGet(contextRef.Handle, nameRef)
            let mutable attrName = MlirStringRef.FromString("sym_name")
            MlirNative.mlirOperationStateAddAttributes(&state, 1n, &&attrName, &&attr))

        let mutable funcTypeAttr = MlirNative.mlirTypeAttrGet(funcType)
        let mutable funcTypeAttrName = MlirStringRef.FromString("function_type")
        MlirNative.mlirOperationStateAddAttributes(&state, 1n, &&funcTypeAttrName, &&funcTypeAttr)

        let mutable numRegions = 1n
        MlirNative.mlirOperationStateAddOwnedRegions(&state, numRegions)

        MlirNative.mlirOperationCreate(&state)

    member _.CreateReturn(values: MlirValue[], location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("func.return"), location.Handle)

        if values.Length > 0 then
            let mutable operands = values
            MlirNative.mlirOperationStateAddOperands(&state, unativeint values.Length, &&operands.[0])

        MlirNative.mlirOperationCreate(&state)

    member _.GetResult(op: MlirOperation, index: int) =
        MlirNative.mlirOperationGetResult(op, unativeint index)

/// 타입 생성 헬퍼
module MLIRType =
    let i32 (ctx: Context) =
        MlirNative.mlirIntegerTypeGet(ctx.Handle, 32u)

    let i64 (ctx: Context) =
        MlirNative.mlirIntegerTypeGet(ctx.Handle, 64u)

    let func (ctx: Context) (inputs: MlirType[]) (results: MlirType[]) =
        let mutable inputsArray = inputs
        let mutable resultsArray = results
        MlirNative.mlirFunctionTypeGet(
            ctx.Handle,
            unativeint inputs.Length,
            (if inputs.Length > 0 then &&inputsArray.[0] else nativeint 0),
            unativeint results.Length,
            (if results.Length > 0 then &&resultsArray.[0] else nativeint 0))
```

## 배운 것

이 장에서 다음을 배웠다:

1. **소유권 관리**: MLIR의 계층적 소유권과 F#에서 부모 참조로 이를 강제하는 방법
2. **IDisposable 패턴**: 자동 리소스 정리를 위한 `use` 키워드
3. **빌더 패턴**: 복잡한 API를 간단한 메서드 호출로 감싸는 `OpBuilder`
4. **타입 안전성**: 장황함 없이 컴파일 시점 타입 검사를 제공하는 F# 래퍼

## 다음 단계

Chapter 05에서는 이 래퍼 레이어를 사용하여 **완전한 컴파일러**를 구축한다. 정수 리터럴을 갖는 간단한 FunLang 프로그램을 파싱하고, MLIR IR로 변환하며, LLVM dialect로 낮추고, 네이티브 바이너리로 컴파일하여 실행할 것이다.

이것이 Phase 1의 정점이다 -- 실제 코드를 실행하는 것이다!
