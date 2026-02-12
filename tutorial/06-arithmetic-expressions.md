# Chapter 06: 산술 표현식 - 연산자와 비교

## 소개

Chapter 05에서 정수 리터럴 하나만 컴파일하는 최소한의 컴파일러를 구축했다. `42`를 입력으로 받아 네이티브 바이너리로 출력하는 전체 파이프라인이 작동한다. 하지만 실제 프로그램을 작성하려면 산술 연산자가 필요하다.

이 장에서는 다음을 추가한다:

- **이진 연산자**: `+`, `-`, `*`, `/` (정수 산술)
- **비교 연산자**: `<`, `>`, `<=`, `>=`, `=`, `<>` (i1 boolean 반환)
- **단항 연산자**: `-` (부정)
- **출력 기능**: `print` 함수로 결과를 stdout에 출력

이 장을 마치면 `10 + 3 * 4`와 같은 표현식을 컴파일하고, 비교를 수행하고, 결과를 화면에 출력하는 완전한 계산기 컴파일러를 갖게 된다.

> **중요:** 산술 연산은 MLIR의 `arith` dialect를 사용한다 (Chapter 01의 primer에서 배웠다). 이 dialect는 SSA 형태의 연산을 제공하며 LLVM dialect로 깔끔하게 낮춰진다.

## 확장된 AST 정의

Chapter 05의 AST는 `IntLiteral` 하나만 가졌다. 이제 표현식을 확장한다:

```fsharp
namespace FunLangCompiler

/// 이진 연산자
type Operator =
    | Add       // +
    | Subtract  // -
    | Multiply  // *
    | Divide    // /

/// 비교 연산자
type CompareOp =
    | LessThan       // <
    | GreaterThan    // >
    | LessEqual      // <=
    | GreaterEqual   // >=
    | Equal          // =
    | NotEqual       // <>

/// 단항 연산자
type UnaryOp =
    | Negate  // -

/// FunLang 표현식 AST
type Expr =
    | IntLiteral of int
    | BinaryOp of Operator * Expr * Expr       // 예: Add(IntLiteral 10, IntLiteral 20)
    | UnaryOp of UnaryOp * Expr                // 예: Negate(IntLiteral 42)
    | Comparison of CompareOp * Expr * Expr    // 예: LessThan(IntLiteral 5, IntLiteral 10)

/// 최상위 프로그램
type Program =
    { expr: Expr }
```

**설계 결정:**

- **Operator와 CompareOp 분리**: 산술 연산은 i32를 반환하지만, 비교는 i1 (boolean)을 반환한다. 타입 시스템이 다르므로 별도의 타입으로 구분한다.
- **UnaryOp은 확장 가능**: 지금은 Negate만 있지만 나중에 논리 부정 (`not`) 등을 추가할 수 있다.

**AST 예시:**

```fsharp
// Source: 10 + 3 * 4
BinaryOp(Add,
  IntLiteral 10,
  BinaryOp(Multiply,
    IntLiteral 3,
    IntLiteral 4))

// Source: -(5 + 10)
UnaryOp(Negate,
  BinaryOp(Add,
    IntLiteral 5,
    IntLiteral 10))

// Source: 5 < 10
Comparison(LessThan,
  IntLiteral 5,
  IntLiteral 10)
```

> **파서 노트:** 실제 파서는 연산자 우선순위를 처리해야 한다 (`*`가 `+`보다 높음). 이 장에서는 코드 생성에 집중하므로 파서 구현은 생략한다. LangTutorial의 파서를 재사용하거나 간단한 재귀 하강 파서를 작성하면 된다.

## arith Dialect 연산 생성

Chapter 03-04에서 구축한 `OpBuilder.CreateOperation` 패턴을 사용하여 arith dialect 연산을 생성한다. 개별 P/Invoke 대신 generic operation builder를 사용하는 것이 더 유연하고 유지보수가 쉽다.

**CodeGen.fs에서 연산 생성 헬퍼:**

```fsharp
/// Create operation, append to block, return result value
let private emitOp (ctx: CompileContext) name resultTypes operands attrs regions =
    let op = ctx.Builder.CreateOperation(name, ctx.Location, resultTypes, operands, attrs, regions)
    ctx.Builder.AppendOperationToBlock(ctx.Block, op)
    op
```

**산술 연산 생성 예시:**

```fsharp
// arith.addi: 정수 덧셈
| Add(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let i32Type = builder.I32Type()
    let op = emitOp ctx "arith.addi" [| i32Type |] [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)

// arith.subi: 정수 뺄셈
| Subtract(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let op = emitOp ctx "arith.subi" [| i32Type |] [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)

// arith.muli: 정수 곱셈
| Multiply(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let op = emitOp ctx "arith.muli" [| i32Type |] [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)

// arith.divsi: 부호 있는 정수 나눗셈
| Divide(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let op = emitOp ctx "arith.divsi" [| i32Type |] [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)
```

**비교 연산 - arith.cmpi:**

비교 연산은 predicate 속성이 필요하다. **중요:** predicate는 반드시 **i64 타입**의 IntegerAttr로 전달해야 한다:

```fsharp
// arith.cmpi predicate 값:
//   0 = eq (equal)
//   1 = ne (not equal)
//   2 = slt (signed less than)
//   3 = sle (signed less or equal)
//   4 = sgt (signed greater than)
//   5 = sge (signed greater or equal)

| Equal(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let i64Type = builder.I64Type()  // 주의: i64 타입!
    let predicateAttr = builder.IntegerAttr(0L, i64Type)  // eq = 0
    let i1Type = builder.I1Type()  // 결과는 i1 (boolean)
    let op = emitOp ctx "arith.cmpi" [| i1Type |]
                [| leftVal; rightVal |]
                [| builder.NamedAttr("predicate", predicateAttr) |]
                [||]
    builder.GetResult(op, 0)
```

> **핵심 발견:** MLIR의 ArithOps.td 정의에 따르면 predicate 속성은 i64 타입이어야 한다. i32를 사용하면 "attribute 'predicate' expected integer type of width 64" 에러가 발생한다.

**연산자 매핑 표:**

| FunLang Operator | MLIR Operation | 타입 시그니처 |
|-----------------|----------------|-------------|
| `+` | `arith.addi` | `(i32, i32) -> i32` |
| `-` | `arith.subi` | `(i32, i32) -> i32` |
| `*` | `arith.muli` | `(i32, i32) -> i32` |
| `/` | `arith.divsi` | `(i32, i32) -> i32` (부호 있는 나눗셈) |
| `<` | `arith.cmpi slt` | `(i32, i32) -> i1` |
| `>` | `arith.cmpi sgt` | `(i32, i32) -> i1` |
| `<=` | `arith.cmpi sle` | `(i32, i32) -> i1` |
| `>=` | `arith.cmpi sge` | `(i32, i32) -> i1` |
| `=` | `arith.cmpi eq` | `(i32, i32) -> i1` |
| `<>` | `arith.cmpi ne` | `(i32, i32) -> i1` |

> **C API 노트:** MLIR C API는 `mlir-c/Dialect/Arith.h`에서 arith dialect 연산을 노출한다. 실제 함수 이름은 위와 다를 수 있다 (예: `mlirArithAddiOpCreate` vs `mlirArithAddiCreate`). MLIR 설치의 헤더 파일을 확인하여 정확한 시그니처를 사용한다.

**arith.cmpi predicate 값:**

```fsharp
/// arith.cmpi predicate enum
module ArithCmpIPredicate =
    let eq = 0    // equal
    let ne = 1    // not equal
    let slt = 2   // signed less than
    let sle = 3   // signed less or equal
    let sgt = 4   // signed greater than
    let sge = 5   // signed greater or equal
    let ult = 6   // unsigned less than (나중에 사용)
    let ule = 7   // unsigned less or equal
    let ugt = 8   // unsigned greater than
    let uge = 9   // unsigned greater or equal
```

## Boolean 리터럴과 논리 연산자

비교 연산 외에도 boolean 리터럴 (`true`, `false`)과 논리 연산자 (`&&`, `||`)를 지원해야 한다.

### Boolean 리터럴 컴파일

Boolean 값은 i1 타입 (1-bit integer)으로 표현된다:

```fsharp
| Bool(b, _) ->
    let i1Type = builder.I1Type()
    let value = if b then 1L else 0L
    let valueAttr = builder.IntegerAttr(value, i1Type)
    let op = emitOp ctx "arith.constant" [| i1Type |] [||]
                [| builder.NamedAttr("value", valueAttr) |] [||]
    builder.GetResult(op, 0)
```

**생성된 MLIR IR:**

```mlir
%true = arith.constant true    // 또는 arith.constant 1 : i1
%false = arith.constant false  // 또는 arith.constant 0 : i1
```

### 논리 AND/OR 연산자

논리 연산자는 `arith.andi`와 `arith.ori`를 사용한다:

```fsharp
| And(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let i1Type = builder.I1Type()
    let op = emitOp ctx "arith.andi" [| i1Type |] [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)

| Or(left, right, _) ->
    let leftVal = compileExpr ctx left
    let rightVal = compileExpr ctx right
    let i1Type = builder.I1Type()
    let op = emitOp ctx "arith.ori" [| i1Type |] [| leftVal; rightVal |] [||] [||]
    builder.GetResult(op, 0)
```

> **주의:** 이 구현은 **비단락 평가 (non-short-circuit evaluation)**이다. 양쪽 피연산자가 항상 평가된다. 진정한 단락 평가를 위해서는 `scf.if`를 사용해야 한다 (Chapter 08 참조).

**생성된 MLIR IR:**

```mlir
// true && false
%a = arith.constant true
%b = arith.constant false
%result = arith.andi %a, %b : i1  // 결과: false

// true || false
%result = arith.ori %a, %b : i1   // 결과: true
```

## 코드 생성 패턴

실제 구현에서는 개별 P/Invoke 대신 generic `CreateOperation` 패턴을 사용한다. 이것이 더 유지보수하기 쉽고 확장성이 좋다.

**설계 결정:**

- **Generic 패턴**: `CreateOperation(name, resultTypes, operands, attrs, regions)` 형식으로 모든 연산을 생성할 수 있다
- **emitOp 헬퍼**: CompileContext를 받아 operation 생성, block에 추가, operation 반환을 하나로 묶는다
- **부정 구현**: `-expr`은 `0 - expr`로 변환한다. 별도의 arith.negate 연산이 없으므로 이것이 표준 방법이다
- **타입 일관성**: 모든 정수는 i32, 모든 boolean은 i1로 컴파일한다

## 공통 에러 (1부)

### 에러 1: 잘못된 정수 타입 사용 (i64 vs i32)

**증상:**
```
MLIR verification failed:
  Type mismatch: expected i32, got i64
```

**원인:**
MLIR은 타입이 엄격하다. 상수를 i64로 생성했지만 함수 시그니처는 i32를 요구하는 경우.

**해결:**
```fsharp
// WRONG: i64 타입 사용
let i64Type = builder.Context.GetIntegerType(64)
let attr = builder.Context.GetIntegerAttr(i64Type, 42L)

// CORRECT: i32 타입 사용
let i32Type = builder.Context.GetIntegerType(32)
let attr = builder.Context.GetIntegerAttr(i32Type, 42L)
```

**규칙:** 모든 FunLang 정수는 i32로 컴파일한다. 타입을 일관되게 유지한다.

### 에러 2: 연산자 우선순위를 파서에서 처리하지 않음

**증상:**
```
Source: 10 + 3 * 4
Expected: 22
Actual: 52  (잘못된 결과)
```

**원인:**
파서가 우선순위를 무시하고 왼쪽에서 오른쪽으로 파싱하여 `(10 + 3) * 4 = 52`가 됨.

**해결:**
파서에서 연산자 우선순위를 구현한다:
- 곱셈/나눗셈 (`*`, `/`)이 덧셈/뺄셈 (`+`, `-`)보다 우선순위가 높다.
- 비교 연산자는 산술 연산보다 우선순위가 낮다.

**재귀 하강 파서 예시:**
```fsharp
// Precedence climbing algorithm
// additive := multiplicative (('+' | '-') multiplicative)*
// multiplicative := primary (('*' | '/') primary)*
// primary := number | '(' additive ')'
```

> **파서 구현은 이 장의 범위를 벗어난다.** LangTutorial의 기존 파서를 사용하거나 FParsec 같은 파서 라이브러리를 사용한다.

## 산술 표현식을 위한 코드 생성

이제 Chapter 05의 `compileExpr`을 확장하여 모든 산술 표현식을 처리한다.

**CodeGen.fs** 수정:

```fsharp
namespace FunLangCompiler

open System
open MlirWrapper
open MlirBindings

module CodeGen =

    /// 표현식을 MLIR value로 컴파일 (재귀적)
    let rec compileExpr
        (builder: OpBuilder)
        (block: MlirBlock)
        (location: Location)
        (expr: Expr)
        : MlirValue =

        match expr with
        | IntLiteral value ->
            // arith.constant operation 생성
            let i32Type = builder.I32Type()
            let attr = builder.Context.GetIntegerAttr(i32Type, int64 value)
            let constOp = builder.CreateConstant(attr, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, constOp)
            builder.GetResult(constOp, 0)

        | BinaryOp(op, lhs, rhs) ->
            // 왼쪽 피연산자 컴파일 (재귀)
            let lhsVal = compileExpr builder block location lhs

            // 오른쪽 피연산자 컴파일 (재귀)
            let rhsVal = compileExpr builder block location rhs

            // 이진 연산 생성
            let binOp = builder.CreateArithBinaryOp(op, lhsVal, rhsVal, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, binOp)
            builder.GetResult(binOp, 0)

        | UnaryOp(Negate, expr) ->
            // 피연산자 컴파일
            let val = compileExpr builder block location expr

            // 부정 연산 생성 (0 - val)
            let negOp = builder.CreateArithNegate(val, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, negOp)
            builder.GetResult(negOp, 0)

        | Comparison(compareOp, lhs, rhs) ->
            // 피연산자 컴파일
            let lhsVal = compileExpr builder block location lhs
            let rhsVal = compileExpr builder block location rhs

            // 비교 연산 생성 (i1 반환)
            let cmpOp = builder.CreateArithCompare(compareOp, lhsVal, rhsVal, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, cmpOp)
            builder.GetResult(cmpOp, 0)

    /// 프로그램을 MLIR module로 컴파일
    let translateToMlir (program: Program) : Module =
        let ctx = new Context()
        ctx.LoadDialect("arith")
        ctx.LoadDialect("func")

        let loc = Location.Unknown(ctx)
        let mlirMod = new Module(ctx, loc)

        let builder = OpBuilder(ctx)
        let i32Type = builder.I32Type()

        // main 함수 생성: () -> i32
        let funcType = builder.FunctionType([||], [| i32Type |])
        let funcOp = builder.CreateFunction("main", funcType, loc)

        // 함수 body에 entry block 생성
        let bodyRegion = MlirNative.mlirOperationGetRegion(funcOp, 0n)
        let entryBlock = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
        MlirNative.mlirRegionAppendOwnedBlock(bodyRegion, entryBlock)

        // 표현식 컴파일 (재귀적으로 모든 연산 처리)
        let resultValue = compileExpr builder entryBlock loc program.expr

        // return operation 생성
        let returnOp = builder.CreateReturn([| resultValue |], loc)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, returnOp)

        // 함수를 module에 추가
        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, funcOp)

        mlirMod

    /// MLIR module을 검증
    let verify (mlirMod: Module) =
        if not (mlirMod.Verify()) then
            eprintfn "MLIR verification failed:"
            eprintfn "%s" (mlirMod.Print())
            failwith "MLIR IR is invalid"
```

**SSA 형태 유지:**

재귀 호출이 SSA 형태를 자연스럽게 유지한다는 것을 주목한다:
- 각 `compileExpr` 호출은 새로운 SSA value를 반환한다.
- 중복 계산이 없다 (각 표현식은 정확히 한 번만 평가된다).
- 지배 관계가 자동으로 유지된다 (하위 표현식이 먼저 평가된다).

**예시: 복잡한 표현식 컴파일**

```fsharp
// Source: 10 + 3 * 4
let ast = BinaryOp(Add,
            IntLiteral 10,
            BinaryOp(Multiply,
              IntLiteral 3,
              IntLiteral 4))

let mlirMod = CodeGen.translateToMlir { expr = ast }
printfn "%s" (mlirMod.Print())
```

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %c10 = arith.constant 10 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %0 = arith.muli %c3, %c4 : i32     // 3 * 4 = 12
    %1 = arith.addi %c10, %0 : i32     // 10 + 12 = 22
    func.return %1 : i32
  }
}
```

**동작 분석:**
1. 상수 10, 3, 4가 생성된다 (`arith.constant`)
2. 먼저 곱셈 계산: `%0 = 3 * 4` (하위 표현식이 먼저)
3. 그 다음 덧셈: `%1 = 10 + %0`
4. 결과 반환: `return %1`

> **중요:** 연산 순서는 AST 구조가 결정한다. 파서가 올바른 우선순위로 AST를 구축하면 코드 생성이 자동으로 올바른 평가 순서를 생성한다.

**비교 예시:**

```fsharp
// Source: 5 < 10
let ast = Comparison(LessThan, IntLiteral 5, IntLiteral 10)

let mlirMod = CodeGen.translateToMlir { expr = ast }
printfn "%s" (mlirMod.Print())
```

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %0 = arith.cmpi slt, %c5, %c10 : i32  // returns i1
    // 문제: %0은 i1이지만 함수는 i32를 반환해야 함!
    func.return %0 : i32  // TYPE ERROR!
  }
}
```

**타입 불일치 문제:** 비교는 i1 (boolean)을 반환하지만 main 함수는 i32를 기대한다. 이를 해결하려면 boolean을 정수로 확장해야 한다:

```fsharp
// compileExpr 수정 (Comparison 케이스)
| Comparison(compareOp, lhs, rhs) ->
    let lhsVal = compileExpr builder block location lhs
    let rhsVal = compileExpr builder block location rhs

    // 비교 연산 (i1 반환)
    let cmpOp = builder.CreateArithCompare(compareOp, lhsVal, rhsVal, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, cmpOp)
    let cmpVal = builder.GetResult(cmpOp, 0)

    // i1 -> i32 확장 (zero extend)
    let i32Type = builder.I32Type()
    let extOp = builder.CreateArithExtUI(cmpVal, i32Type, location)  // unsigned extend
    MlirNative.mlirBlockAppendOwnedOperation(block, extOp)
    builder.GetResult(extOp, 0)
```

**생성된 MLIR IR (수정 후):**

```mlir
module {
  func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %0 = arith.cmpi slt, %c5, %c10 : i32   // returns i1
    %1 = arith.extui %0 : i1 to i32        // i1 -> i32 (0 or 1)
    func.return %1 : i32
  }
}
```

이제 비교 결과가 정수로 반환된다 (true = 1, false = 0).

**단항 부정 예시:**

```fsharp
// Source: -(10 + 5)
let ast = UnaryOp(Negate, BinaryOp(Add, IntLiteral 10, IntLiteral 5))

let mlirMod = CodeGen.translateToMlir { expr = ast }
printfn "%s" (mlirMod.Print())
```

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %c10 = arith.constant 10 : i32
    %c5 = arith.constant 5 : i32
    %0 = arith.addi %c10, %c5 : i32     // 10 + 5 = 15
    %c0 = arith.constant 0 : i32
    %1 = arith.subi %c0, %0 : i32       // 0 - 15 = -15
    func.return %1 : i32
  }
}
```

## 출력 기능 추가: printf로 결과 출력

지금까지 프로그램은 결과를 종료 코드로만 반환했다. 이제 `printf`를 사용하여 stdout에 출력하는 기능을 추가한다.

### llvm.call 연산을 위한 P/Invoke 바인딩

LLVM dialect는 외부 함수를 호출하는 `llvm.call` 연산을 제공한다.

**MlirBindings.fs**에 추가:

```fsharp
    // ===== LLVM dialect operations =====

    /// llvm.call: 외부 함수 호출
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirLLVMCallCreate(
        MlirContext context,
        MlirLocation location,
        MlirValue callee,
        MlirValue[] args,
        int numArgs)

    /// llvm.mlir.global: 전역 문자열 상수
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirLLVMGlobalCreate(
        MlirContext context,
        MlirLocation location,
        MlirType type,
        MlirAttribute initializer,
        MlirStringRef name)

    /// llvm.mlir.addressof: 전역 변수의 주소 가져오기
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirLLVMAddressOfCreate(
        MlirContext context,
        MlirLocation location,
        MlirStringRef globalName)
```

> **C API 경고:** 실제 MLIR C API는 LLVM dialect에 대한 직접 지원이 제한적일 수 있다. 필요한 경우 Chapter 05의 Appendix 패턴 (C++ wrapper)을 사용한다.

### printf 함수 선언 생성

printf를 호출하려면 먼저 함수 선언과 전역 포맷 문자열이 필요하다.

**CodeGen.fs**에 헬퍼 함수 추가:

```fsharp
    /// printf 함수 선언 생성 (module 레벨)
    let createPrintfDeclaration (builder: OpBuilder) (mlirMod: Module) (location: Location) =
        // printf 시그니처: (i8*, ...) -> i32
        let i8Type = builder.Context.GetIntegerType(8)
        let i8PtrType = builder.Context.GetPointerType(i8Type)
        let i32Type = builder.I32Type()

        // func.func @printf(%fmt: !llvm.ptr<i8>, ...) -> i32 attributes { sym_visibility = "private" }
        let printfType = builder.FunctionType([| i8PtrType |], [| i32Type |])
        let printfOp = builder.CreateFunction("printf", printfType, location)

        // 가변 인자 속성 추가 (실제 구현에서는 속성 API 필요)
        // 여기서는 단순화를 위해 생략

        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, printfOp)

    /// 전역 포맷 문자열 생성: "%d\n\0"
    let createFormatString (builder: OpBuilder) (mlirMod: Module) (location: Location) : string =
        let formatStrName = ".str.fmt"
        let formatStrValue = "%d\n\0"

        // LLVM global 생성
        let i8Type = builder.Context.GetIntegerType(8)
        let arrayType = builder.Context.GetArrayType(i8Type, formatStrValue.Length)
        let strAttr = builder.Context.GetStringAttr(formatStrValue)

        let globalOp = builder.CreateLLVMGlobal(arrayType, strAttr, formatStrName, location)
        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, globalOp)

        formatStrName

    /// print_int 헬퍼 함수 생성: 정수를 출력
    let createPrintIntHelper
        (builder: OpBuilder)
        (mlirMod: Module)
        (location: Location)
        (formatStrName: string)
        =

        // func.func @print_int(%arg: i32) -> i32
        let i32Type = builder.I32Type()
        let funcType = builder.FunctionType([| i32Type |], [| i32Type |])
        let funcOp = builder.CreateFunction("print_int", funcType, location)

        // 함수 body
        let bodyRegion = MlirNative.mlirOperationGetRegion(funcOp, 0n)
        let entryBlock = MlirNative.mlirBlockCreate(1n, &i32Type, nativeint 0)  // 1 argument
        MlirNative.mlirRegionAppendOwnedBlock(bodyRegion, entryBlock)

        // 인자 가져오기
        let arg = MlirNative.mlirBlockGetArgument(entryBlock, 0n)

        // 포맷 문자열 주소 가져오기
        let formatStrOp = builder.CreateLLVMAddressOf(formatStrName, location)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, formatStrOp)
        let formatStrPtr = builder.GetResult(formatStrOp, 0)

        // printf 호출
        let printfCallOp = builder.CreateLLVMCall("printf", [| formatStrPtr; arg |], location)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, printfCallOp)

        // 인자를 그대로 반환 (print는 부수 효과)
        let returnOp = builder.CreateReturn([| arg |], location)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, returnOp)

        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, funcOp)
```

### main 함수에서 print_int 호출

이제 main 함수를 수정하여 결과를 출력하도록 한다:

```fsharp
    /// 프로그램을 MLIR module로 컴파일 (print 지원)
    let translateToMlirWithPrint (program: Program) : Module =
        let ctx = new Context()
        ctx.LoadDialect("arith")
        ctx.LoadDialect("func")
        ctx.LoadDialect("llvm")

        let loc = Location.Unknown(ctx)
        let mlirMod = new Module(ctx, loc)

        let builder = OpBuilder(ctx)
        let i32Type = builder.I32Type()

        // printf 선언과 print_int 헬퍼 생성
        createPrintfDeclaration builder mlirMod loc
        let formatStrName = createFormatString builder mlirMod loc
        createPrintIntHelper builder mlirMod loc formatStrName

        // main 함수 생성
        let funcType = builder.FunctionType([||], [| i32Type |])
        let funcOp = builder.CreateFunction("main", funcType, loc)

        let bodyRegion = MlirNative.mlirOperationGetRegion(funcOp, 0n)
        let entryBlock = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
        MlirNative.mlirRegionAppendOwnedBlock(bodyRegion, entryBlock)

        // 표현식 컴파일
        let resultValue = compileExpr builder entryBlock loc program.expr

        // print_int 호출
        let printOp = builder.CreateFunctionCall("print_int", [| resultValue |], loc)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, printOp)
        let printedVal = builder.GetResult(printOp, 0)

        // 결과 반환
        let returnOp = builder.CreateReturn([| printedVal |], loc)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, returnOp)

        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, funcOp)

        mlirMod
```

**생성된 MLIR IR (전체):**

```mlir
module {
  // printf 선언
  func.func private @printf(!llvm.ptr<i8>, ...) -> i32

  // 포맷 문자열
  llvm.mlir.global private constant @.str.fmt("%d\n\00")

  // print_int 헬퍼
  func.func @print_int(%arg0: i32) -> i32 {
    %fmt = llvm.mlir.addressof @.str.fmt : !llvm.ptr<array<4 x i8>>
    %fmt_ptr = llvm.bitcast %fmt : !llvm.ptr<array<4 x i8>> to !llvm.ptr<i8>
    %result = llvm.call @printf(%fmt_ptr, %arg0) : (!llvm.ptr<i8>, i32) -> i32
    func.return %arg0 : i32
  }

  // main 함수
  func.func @main() -> i32 {
    %c10 = arith.constant 10 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %0 = arith.muli %c3, %c4 : i32
    %1 = arith.addi %c10, %0 : i32
    %2 = func.call @print_int(%1) : (i32) -> i32
    func.return %2 : i32
  }
}
```

**실행 결과:**

```bash
$ ./program
22
$ echo $?
22
```

결과가 stdout에 출력되고 종료 코드로도 반환된다!

## 완전한 컴파일러 드라이버

Chapter 05의 컴파일러 드라이버를 업데이트하여 새로운 기능을 지원한다:

**Compiler.fs 업데이트:**

```fsharp
namespace FunLangCompiler

open System
open System.IO

module Compiler =

    /// 소스 파일을 네이티브 실행 파일로 컴파일
    let compile (sourceFile: string) (outputFile: string) (withPrint: bool) =
        printfn "=== FunLang Compiler ==="
        printfn "Source: %s" sourceFile
        printfn "Output: %s" outputFile
        printfn ""

        // 1단계: 파싱
        printfn "[1/7] Parsing..."
        let source = File.ReadAllText(sourceFile)
        let program = Parser.parse source  // 실제 파서 사용 (LangTutorial 재사용)
        printfn "  AST: %A" program

        // 2단계: MLIR로 변환
        printfn "[2/7] Translating to MLIR..."
        let mlirMod =
            if withPrint then
                CodeGen.translateToMlirWithPrint program
            else
                CodeGen.translateToMlir program
        printfn "  MLIR (high-level):"
        printfn "%s" (mlirMod.Print())

        // 3단계: 검증
        printfn "[3/7] Verifying MLIR..."
        CodeGen.verify mlirMod
        printfn "  ✓ Verification passed"

        // 4-7단계: Lowering, LLVM IR, object file, linking (Chapter 05와 동일)
        Lowering.lowerToLLVMDialect mlirMod
        let llvmIR = Lowering.translateToLLVMIR mlirMod
        let objectFile = outputFile + ".o"
        NativeCodeGen.emitObjectFile llvmIR objectFile
        NativeCodeGen.linkExecutable objectFile outputFile

        mlirMod.Dispose()

        printfn ""
        printfn "=== Compilation successful ==="
        printfn "Run: ./%s" outputFile

[<EntryPoint>]
let main args =
    if args.Length < 2 then
        eprintfn "Usage: compiler <source.fun> <output> [--print]"
        exit 1

    let sourceFile = args.[0]
    let outputFile = args.[1]
    let withPrint = args.Length > 2 && args.[2] = "--print"

    Compiler.compile sourceFile outputFile withPrint
    0
```

**사용 예시:**

```bash
# 결과를 출력하지 않음 (종료 코드만)
$ dotnet run test.fun program

# 결과를 출력함 (stdout + 종료 코드)
$ dotnet run test.fun program --print
$ ./program
22
```

## 공통 에러 (2부)

### 에러 3: 비교가 i1을 반환하지만 i32가 필요한 곳에서 사용

**증상:**
```
MLIR verification failed:
  Type mismatch in func.return: expected i32, got i1
```

**원인:**
비교 연산은 i1 (boolean)을 반환하지만 main 함수는 i32를 반환해야 한다.

**해결:**
i1을 i32로 확장한다:

```fsharp
// arith.extui 사용 (zero extend)
let extOp = builder.CreateArithExtUI(cmpVal, i32Type, location)
```

**또는** main 함수가 i1을 반환하도록 변경 (덜 일반적):

```fsharp
// main 함수 시그니처를 i1으로 변경 (비권장)
let funcType = builder.FunctionType([||], [| builder.Context.GetIntegerType(1) |])
```

**권장 방법:** 항상 i32로 확장한다. Unix 종료 코드는 8비트 정수이므로 boolean을 정수로 표현하는 것이 자연스럽다.

### 에러 4: 0으로 나누기 (런타임 vs 컴파일 타임)

**증상:**
```bash
$ ./program
Floating point exception (core dumped)
```

**원인:**
`10 / 0`과 같은 표현식이 런타임에 0으로 나누기를 시도한다.

**컴파일 타임 해결:**
AST를 분석하여 상수 0으로 나누기를 감지한다:

```fsharp
| BinaryOp(Divide, lhs, IntLiteral 0) ->
    failwith "Compile error: division by zero"
```

**런타임 해결 (더 일반적):**
동적 검사 코드를 삽입한다:

```fsharp
| BinaryOp(Divide, lhs, rhs) ->
    let lhsVal = compileExpr builder block location lhs
    let rhsVal = compileExpr builder block location rhs

    // rhsVal == 0 검사
    let zero = builder.CreateConstant(0, builder.I32Type(), location)
    let isZero = builder.CreateArithCmpi(ArithCmpIPredicate.eq, rhsVal, zero, location)

    // if (rhsVal == 0) abort() else lhs / rhs
    // scf.if 사용 (Chapter 08에서 다룸)
    // 지금은 단순화를 위해 생략
```

**실용적 접근:** 대부분의 컴파일러는 0으로 나누기를 런타임 에러로 남긴다. 프로그램이 SIGFPE로 종료되는 것이 예상 동작이다.

### 에러 5: printf 포맷 문자열에 null terminator 누락

**증상:**
```bash
$ ./program
22ݠ�(garbage characters)
```

**원인:**
C 문자열은 null terminator (`\0`)가 필요하다. `"%d\n"` 대신 `"%d\n\0"`를 사용해야 한다.

**해결:**
```fsharp
// WRONG: null terminator 없음
let formatStrValue = "%d\n"

// CORRECT: null terminator 포함
let formatStrValue = "%d\n\0"
```

**MLIR IR:**
```mlir
// CORRECT
llvm.mlir.global private constant @.str.fmt("%d\0A\00") : !llvm.array<4 x i8>
```

### 에러 6: arith 연산 후 LLVM dialect로 낮추기 잊음

**증상:**
```
Translation error: Unhandled operation 'arith.addi'
```

**원인:**
arith dialect를 LLVM IR로 변환하려면 먼저 LLVM dialect로 낮춰야 한다.

**해결:**
Lowering 단계에서 `convert-arith-to-llvm` pass를 실행한다:

```fsharp
// Pass manager에 추가
MlirStringRef.WithString "convert-arith-to-llvm" (fun passName ->
    let pass = MlirNative.mlirCreateConversionPass(passName)
    MlirNative.mlirPassManagerAddOwnedPass(pm, pass))
```

**Pass 순서:**
1. `convert-func-to-llvm`
2. `convert-arith-to-llvm`
3. `reconcile-unrealized-casts`
4. 그 다음 `mlirTranslateModuleToLLVMIR`

## 구현 시 주의사항 (Common Pitfalls)

실제 구현에서 발견된 중요한 주의사항들:

### 1. arith.cmpi predicate는 i64 타입이어야 한다

```fsharp
// WRONG: i32 타입 predicate
let predicateAttr = builder.IntegerAttr(0L, builder.I32Type())

// CORRECT: i64 타입 predicate
let predicateAttr = builder.IntegerAttr(0L, builder.I64Type())
```

MLIR ArithOps.td 정의에서 predicate는 64비트 정수 속성으로 정의되어 있다. i32를 사용하면 검증 에러가 발생한다.

### 2. 비교 연산 결과는 i1, 정수 연산 결과는 i32

```fsharp
// 비교 연산: i1 결과
let op = emitOp ctx "arith.cmpi" [| builder.I1Type() |] ...

// 산술 연산: i32 결과
let op = emitOp ctx "arith.addi" [| builder.I32Type() |] ...
```

### 3. Boolean 리터럴은 i1 타입의 arith.constant

```fsharp
// Boolean true/false
let i1Type = builder.I1Type()
let value = if b then 1L else 0L
let valueAttr = builder.IntegerAttr(value, i1Type)  // i1 타입으로 생성
```

### 4. 비단락 평가에 주의

`arith.andi`와 `arith.ori`는 양쪽 피연산자를 모두 평가한다. 부수 효과가 있는 표현식에서 문제가 될 수 있다. 진정한 단락 평가가 필요하면 `scf.if`를 사용한다.

## 장 요약

이 장에서 다음을 성취했다:

1. **확장된 AST**: 이진 연산자, 비교, 단항 부정, boolean 리터럴, 논리 연산자를 지원하는 표현식 타입
2. **Generic 연산 생성**: `CreateOperation` 패턴으로 arith dialect 연산 생성
3. **비교 연산**: arith.cmpi와 i64 predicate 속성
4. **Boolean 지원**: i1 타입, arith.andi/ori 논리 연산자
5. **재귀 코드 생성**: SSA 형태를 유지하며 복잡한 표현식 컴파일
6. **출력 기능**: printf를 통한 결과 출력
7. **완전한 예제**: MLIR IR 출력을 보여주는 실행 가능한 코드

**독자가 할 수 있는 것:**
- `10 + 3 * 4` 컴파일 → 네이티브 바이너리 → 실행 → 결과: 22 ✓
- `5 < 10` 컴파일 → boolean 반환 (1 = true) ✓
- `-42` 컴파일 → 부정 연산 ✓
- `print(10 + 20)` 컴파일 → stdout 출력: 30 ✓

**다음 장 미리보기:**

Chapter 07에서는 **let 바인딩**을 추가한다:

```fsharp
let x = 10 in
let y = 20 in
x + y
```

이것은 다음을 도입한다:
- 변수 이름과 SSA value 간의 환경 (symbol table)
- 중첩된 스코프 (nested scopes)
- 변수 섀도잉 (shadowing) vs 뮤테이션 (mutation)

**Phase 2는 계속된다!**

---

**이제 독자는 산술 표현식을 컴파일하고 결과를 출력할 수 있다!**
