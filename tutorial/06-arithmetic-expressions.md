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

## arith Dialect를 위한 P/Invoke 바인딩

Chapter 03에서 기본 MLIR C API 바인딩을 구축했다. 이제 `arith` dialect 연산을 위한 바인딩을 추가한다.

**MlirBindings.fs**에 다음을 추가한다:

```fsharp
namespace FunLangCompiler

open System
open System.Runtime.InteropServices

module MlirBindings =

    // ... (기존 바인딩 코드)

    // ===== arith dialect operations =====

    /// arith.addi: 정수 덧셈 (SSA)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirArithAddiCreate(
        MlirContext context,
        MlirLocation location,
        MlirValue lhs,
        MlirValue rhs)

    /// arith.subi: 정수 뺄셈 (SSA)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirArithSubiCreate(
        MlirContext context,
        MlirLocation location,
        MlirValue lhs,
        MlirValue rhs)

    /// arith.muli: 정수 곱셈 (SSA)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirArithMuliCreate(
        MlirContext context,
        MlirLocation location,
        MlirValue lhs,
        MlirValue rhs)

    /// arith.divsi: 부호 있는 정수 나눗셈 (SSA)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirArithDivSICreate(
        MlirContext context,
        MlirLocation location,
        MlirValue lhs,
        MlirValue rhs)

    /// arith.cmpi: 정수 비교 (predicate 지정)
    /// predicate 값:
    ///   0 = eq (equal)
    ///   1 = ne (not equal)
    ///   2 = slt (signed less than)
    ///   3 = sle (signed less or equal)
    ///   4 = sgt (signed greater than)
    ///   5 = sge (signed greater or equal)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirArithCmpiCreate(
        MlirContext context,
        MlirLocation location,
        int predicate,
        MlirValue lhs,
        MlirValue rhs)

    /// arith.constant: 상수 정수/boolean
    /// (이미 Chapter 03에서 mlirArithConstantCreate가 있다고 가정.
    /// 없다면 여기에 추가)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirArithConstantCreate(
        MlirContext context,
        MlirLocation location,
        MlirAttribute value)
```

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

## F# 래퍼 확장

이제 Chapter 04의 `OpBuilder` 래퍼를 확장하여 산술 연산을 쉽게 생성하도록 한다.

**MlirWrapper.fs**에 `OpBuilder` 타입 확장:

```fsharp
namespace FunLangCompiler

open MlirBindings

/// OpBuilder 확장: 산술 연산 생성
type OpBuilder with

    /// arith.addi 생성 (정수 덧셈)
    member this.CreateArithAddi(lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        let loc = location.Handle
        MlirNative.mlirArithAddiCreate(this.Context.Handle, loc, lhs, rhs)

    /// arith.subi 생성 (정수 뺄셈)
    member this.CreateArithSubi(lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        let loc = location.Handle
        MlirNative.mlirArithSubiCreate(this.Context.Handle, loc, lhs, rhs)

    /// arith.muli 생성 (정수 곱셈)
    member this.CreateArithMuli(lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        let loc = location.Handle
        MlirNative.mlirArithMuliCreate(this.Context.Handle, loc, lhs, rhs)

    /// arith.divsi 생성 (부호 있는 정수 나눗셈)
    member this.CreateArithDivSI(lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        let loc = location.Handle
        MlirNative.mlirArithDivSICreate(this.Context.Handle, loc, lhs, rhs)

    /// arith.cmpi 생성 (정수 비교)
    member this.CreateArithCmpi(predicate: int, lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        let loc = location.Handle
        MlirNative.mlirArithCmpiCreate(this.Context.Handle, loc, predicate, lhs, rhs)

    /// 이진 연산 헬퍼: Operator -> arith operation
    member this.CreateArithBinaryOp(op: Operator, lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        match op with
        | Add -> this.CreateArithAddi(lhs, rhs, location)
        | Subtract -> this.CreateArithSubi(lhs, rhs, location)
        | Multiply -> this.CreateArithMuli(lhs, rhs, location)
        | Divide -> this.CreateArithDivSI(lhs, rhs, location)

    /// 비교 헬퍼: CompareOp -> arith.cmpi with predicate
    member this.CreateArithCompare(compareOp: CompareOp, lhs: MlirValue, rhs: MlirValue, location: Location) : MlirOperation =
        let predicate =
            match compareOp with
            | LessThan -> ArithCmpIPredicate.slt
            | GreaterThan -> ArithCmpIPredicate.sgt
            | LessEqual -> ArithCmpIPredicate.sle
            | GreaterEqual -> ArithCmpIPredicate.sge
            | Equal -> ArithCmpIPredicate.eq
            | NotEqual -> ArithCmpIPredicate.ne
        this.CreateArithCmpi(predicate, lhs, rhs, location)

    /// 단항 부정 헬퍼: -expr = 0 - expr
    member this.CreateArithNegate(value: MlirValue, location: Location) : MlirOperation =
        // 상수 0 생성
        let i32Type = this.I32Type()
        let zeroAttr = this.Context.GetIntegerAttr(i32Type, 0L)
        let zeroOp = this.CreateConstant(zeroAttr, location)
        let zeroVal = this.GetResult(zeroOp, 0)

        // 0 - value
        this.CreateArithSubi(zeroVal, value, location)
```

**설계 결정:**

- **헬퍼 메서드**: `CreateArithBinaryOp`은 Operator를 받아 적절한 arith 연산으로 매핑한다. 코드 생성 로직이 단순해진다.
- **부정 구현**: `-expr`은 `0 - expr`로 변환한다. 별도의 arith.negate 연산이 없으므로 이것이 표준 방법이다.
- **타입 안전성**: CompareOp -> predicate 매핑은 타입 세이프하다. 잘못된 predicate 값을 생성할 수 없다.

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

