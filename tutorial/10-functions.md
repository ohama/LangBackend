# Chapter 10: 함수와 func 다이얼렉트

## 소개

지금까지 FunLang 컴파일러는 **표현식(expression)**만 처리했다. Chapter 06부터 09까지 산술, 비교, let 바인딩, if 표현식을 컴파일하는 방법을 배웠다. 모든 것이 하나의 표현식이었고, 그 결과가 프로그램의 최종 값이었다.

```fsharp
// 지금까지의 FunLang - 단일 표현식
let x = 10 in
let y = 20 in
if x > y then x else y
```

이것은 단순한 스크립트에서는 작동하지만, 실제 프로그램은 **재사용 가능한 코드 단위**가 필요하다. 계산을 이름에 바인딩하고, 여러 곳에서 호출할 수 있어야 한다. 바로 **함수(function)**다.

이 장에서는 **최상위 명명된 함수(top-level named functions)**를 추가한다:

```fsharp
// 함수 정의
let add x y = x + y

// 함수 호출
add 10 20   // 결과: 30
```

**중요한 범위 구분:** 이 장은 **Phase 3의 첫 단계**로, 간단한 함수만 다룬다:
- 최상위 함수 정의 (module-level functions)
- 함수 파라미터 (고정된 개수)
- 함수 호출 (call-by-value)
- 함수 반환값

**제외되는 것 (Phase 4에서 다룸):**
- **클로저(Closures)**: 외부 변수를 캡처하는 함수
- **고차 함수(Higher-order functions)**: 함수를 인자로 받거나 반환하는 함수
- **익명 함수(Lambda expressions)**: `fun x -> x + 1`

왜 Phase 3과 Phase 4로 나누는가?
- Phase 3: 함수의 **정적 측면** (함수 정의, 호출, 재귀)
- Phase 4: 함수의 **동적 측면** (클로저, 환경 캡처, 함수 값)

Phase 3 함수는 C나 Java의 static method와 유사하다: 이름으로 호출하고, 외부 상태를 캡처하지 않는다. Phase 4에서 환경 캡처를 추가하면 진정한 함수형 언어가 된다.

**학습 목표:**
- MLIR func 다이얼렉트의 연산들 (`func.func`, `func.call`, `func.return`)
- 함수 파라미터를 block arguments로 표현하는 방법
- 함수 호출과 반환 값 처리
- LLVM 호출 규약(calling convention)의 기초
- 재귀 함수의 작동 원리 (Chapter 11 preview)

이 장을 마치면:
- 다중 함수 정의를 포함한 FunLang 프로그램을 컴파일할 수 있다
- 함수가 MLIR IR로, 그리고 네이티브 코드로 변환되는 과정을 이해한다
- 함수 파라미터가 SSA value로 처리되는 원리를 안다
- 모듈 레벨 심볼 테이블이 어떻게 재귀를 가능하게 하는지 안다

> **Preview:** Chapter 11에서는 재귀와 상호 재귀를 다룬다. Chapter 10은 함수의 기초를 확립한다.

## MLIR func 다이얼렉트

MLIR은 함수를 표현하기 위한 전용 다이얼렉트를 제공한다: **func 다이얼렉트**.

### func 다이얼렉트 개요

**func 다이얼렉트**는 함수 정의와 호출을 표현하는 고수준 추상화다. C, C++, Rust 같은 언어의 함수와 동일한 개념이다.

**핵심 연산:**

| 연산 | 목적 | 예시 |
|-----|------|------|
| `func.func` | 함수 정의 | `func.func @add(%arg0: i32, %arg1: i32) -> i32` |
| `func.call` | 함수 호출 | `%result = func.call @add(%x, %y) : (i32, i32) -> i32` |
| `func.return` | 함수에서 값 반환 | `func.return %result : i32` |

**func 다이얼렉트의 위치 (다이얼렉트 스택):**

```
High-level:  func 다이얼렉트 (함수 추상화)
             scf 다이얼렉트 (제어 흐름)
             arith 다이얼렉트 (산술)
             ↓ (lowering passes)
Middle:      LLVM 다이얼렉트 (LLVM IR 추상화)
             ↓ (mlir-translate)
Low-level:   LLVM IR (define, call, ret)
             ↓ (llc)
Native:      Machine code (x86-64, ARM, etc.)
```

func 다이얼렉트는 **고수준 추상화**다. 플랫폼 독립적으로 함수를 정의하고, 나중에 LLVM 다이얼렉트로 내려가면서 호출 규약, 레지스터 할당, 스택 프레임 관리가 추가된다.

### func.func 연산: 함수 정의

`func.func` 연산은 함수를 정의한다. C의 function definition, Java의 method declaration과 동일한 개념이다.

**Syntax:**

```mlir
func.func @function_name(%arg0: type0, %arg1: type1, ...) -> return_type {
  // function body
  func.return %result : return_type
}
```

**구성 요소:**

1. **Symbol name (`@function_name`)**: 함수의 이름. `@` 기호는 모듈 레벨 심볼을 나타낸다.
2. **Parameters (`%arg0`, `%arg1`)**: 함수의 파라미터. Block arguments로 표현된다.
3. **Function type (`(type0, type1) -> return_type`)**: 파라미터 타입과 반환 타입.
4. **Function body**: 함수 본체. Region (영역) 내부에 블록을 포함한다.
5. **Terminator (`func.return`)**: 함수 종료. 반환 값을 지정한다.

**예시 1: 단순한 함수 (두 정수 더하기)**

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %result = arith.addi %arg0, %arg1 : i32
  func.return %result : i32
}
```

**해석:**
- 함수 이름: `@add`
- 파라미터: `%arg0` (i32), `%arg1` (i32)
- 반환 타입: i32
- 본체: `%arg0 + %arg1` 계산
- 반환: `%result` 값 반환

이것은 C의 `int add(int arg0, int arg1) { return arg0 + arg1; }`와 동일하다.

**예시 2: 파라미터 없는 함수**

```mlir
func.func @get_constant() -> i32 {
  %c42 = arith.constant 42 : i32
  func.return %c42 : i32
}
```

파라미터가 없으면 괄호 내부가 비어있다: `()`.

**예시 3: 다중 연산을 포함하는 함수**

```mlir
func.func @compute(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %doubled = arith.muli %x, %c2 : i32
  %c10 = arith.constant 10 : i32
  %result = arith.addi %doubled, %c10 : i32
  func.return %result : i32
}
```

**해석:**
- `x * 2 + 10` 계산
- 중간 계산 (`%doubled`) 저장
- 최종 결과 반환

### func.call 연산: 함수 호출

`func.call` 연산은 함수를 호출한다. 함수 이름을 심볼 참조로 지정하고, 인자를 전달하고, 결과를 받는다.

**Syntax:**

```mlir
%result = func.call @function_name(%arg0, %arg1, ...) : (type0, type1, ...) -> return_type
```

**구성 요소:**

1. **Callee (`@function_name`)**: 호출할 함수의 심볼 참조.
2. **Arguments (`%arg0`, `%arg1`)**: 함수에 전달할 인자 (SSA values).
3. **Function type annotation**: 함수의 시그니처 (파라미터 타입과 반환 타입).
4. **Result (`%result`)**: 함수 호출의 결과 (SSA value).

**예시 1: add 함수 호출**

```mlir
func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32
  %sum = func.call @add(%c10, %c20) : (i32, i32) -> i32
  func.return %sum : i32
}

func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %result = arith.addi %arg0, %arg1 : i32
  func.return %result : i32
}
```

**실행 흐름:**
1. `@main` 함수 시작
2. `%c10 = 10`, `%c20 = 20` 생성
3. `@add` 함수 호출 (인자: 10, 20)
4. `@add` 내부: `%arg0 = 10`, `%arg1 = 20`
5. `%result = 10 + 20 = 30` 계산
6. `@add` 반환: 30
7. `@main`에서 `%sum = 30` 저장
8. `@main` 반환: 30

**예시 2: 중첩 호출 (함수 결과를 다른 함수의 인자로 사용)**

```mlir
func.func @main() -> i32 {
  %c5 = arith.constant 5 : i32
  %doubled = func.call @double(%c5) : (i32) -> i32
  %result = func.call @double(%doubled) : (i32) -> i32
  func.return %result : i32
}

func.func @double(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %result = arith.muli %x, %c2 : i32
  func.return %result : i32
}
```

**실행:**
- `double(5)` → 10
- `double(10)` → 20
- 최종 결과: 20

### func.return 연산: 함수 종료

`func.return` 연산은 함수를 종료하고 값을 반환한다. C의 `return` 문과 동일하다.

**Syntax:**

```mlir
func.return %value : type
```

**중요한 규칙:**

1. **모든 함수는 func.return으로 끝나야 한다**: `func.return`은 terminator operation이다. 함수 본체의 마지막 연산이어야 한다.

2. **반환 타입 일치**: 반환 값의 타입은 함수 시그니처의 반환 타입과 일치해야 한다.

   ```mlir
   // 올바름
   func.func @example() -> i32 {
     %c42 = arith.constant 42 : i32
     func.return %c42 : i32  // i32 반환 (시그니처와 일치)
   }

   // 오류: 타입 불일치
   func.func @wrong() -> i32 {
     %c1 = arith.constant 1 : i1  // i1 타입
     func.return %c1 : i1  // 오류! i32를 반환해야 함
   }
   ```

3. **Multiple returns (여러 반환 지점)**: 함수는 여러 반환 지점을 가질 수 있다 (조건부).

   ```mlir
   func.func @abs(%x: i32) -> i32 {
     %c0 = arith.constant 0 : i32
     %is_negative = arith.cmpi slt, %x, %c0 : i32
     %result = scf.if %is_negative -> (i32) {
       %neg = arith.subi %c0, %x : i32
       scf.yield %neg : i32
     } else {
       scf.yield %x : i32
     }
     func.return %result : i32
   }
   ```

### 함수 가시성 (Visibility)

함수는 **가시성(visibility)** 속성을 가질 수 있다:

| 가시성 | 의미 | 사용 예 |
|-------|------|---------|
| `public` (기본값) | 모듈 외부에서 접근 가능 | `func.func @main() -> i32` |
| `private` | 모듈 내부에서만 접근 가능 | `func.func private @helper() -> i32` |
| `nested` | 부모 함수 내부에서만 접근 가능 (Phase 4에서 다룸) | |

**예시: private 함수 (헬퍼 함수)**

```mlir
// Public 함수 - 외부에서 호출 가능
func.func @main() -> i32 {
  %result = func.call @helper() : () -> i32
  func.return %result : i32
}

// Private 함수 - main에서만 호출 가능
func.func private @helper() -> i32 {
  %c42 = arith.constant 42 : i32
  func.return %c42 : i32
}
```

Phase 3에서는 모든 함수가 `public`이다 (기본값). 가시성을 명시할 필요가 없다.

### 함수와 심볼 테이블

MLIR 모듈은 **심볼 테이블(symbol table)**을 유지한다. 모든 `func.func` 연산은 모듈 레벨 심볼로 등록된다.

**핵심 특성:**

1. **Flat namespace (평면 네임스페이스)**: 모든 함수가 동일한 네임스페이스에 있다. 함수 정의 순서는 중요하지 않다.

2. **Forward references (전방 참조)**: 함수를 정의하기 전에 호출할 수 있다.

   ```mlir
   // foo는 아직 정의되지 않았지만 호출 가능
   func.func @main() -> i32 {
     %result = func.call @foo() : () -> i32
     func.return %result : i32
   }

   // 나중에 정의됨
   func.func @foo() -> i32 {
     %c42 = arith.constant 42 : i32
     func.return %c42 : i32
   }
   ```

3. **재귀 가능**: 함수가 자기 자신을 호출할 수 있다 (심볼이 모듈에 등록되므로).

   ```mlir
   func.func @factorial(%n: i32) -> i32 {
     %c1 = arith.constant 1 : i32
     %is_one = arith.cmpi sle, %n, %c1 : i32
     %result = scf.if %is_one -> (i32) {
       scf.yield %c1 : i32
     } else {
       %n_minus_1 = arith.subi %n, %c1 : i32
       %rec = func.call @factorial(%n_minus_1) : (i32) -> i32  // 재귀 호출
       %product = arith.muli %n, %rec : i32
       scf.yield %product : i32
     }
     func.return %result : i32
   }
   ```

4. **상호 재귀 가능**: 두 함수가 서로를 호출할 수 있다.

   ```mlir
   func.func @is_even(%n: i32) -> i1 {
     // ... calls @is_odd ...
   }

   func.func @is_odd(%n: i32) -> i1 {
     // ... calls @is_even ...
   }
   ```

심볼 테이블 덕분에 함수 정의 순서나 전방 선언을 걱정할 필요가 없다. 모든 함수가 모듈 로드 시 등록된다.

### Phase 2와의 비교: 함수 vs 표현식

Phase 2에서는 모든 것이 단일 표현식이었다:

```mlir
// Phase 2 스타일 - 단일 main 함수
func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32
  %sum = arith.addi %c10, %c20 : i32
  func.return %sum : i32
}
```

Phase 3에서는 재사용 가능한 함수를 정의한다:

```mlir
// Phase 3 스타일 - 여러 함수
func.func @add(%a: i32, %b: i32) -> i32 {
  %result = arith.addi %a, %b : i32
  func.return %result : i32
}

func.func @main() -> i32 {
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32
  %sum = func.call @add(%c10, %c20) : (i32, i32) -> i32
  func.return %sum : i32
}
```

**차이점:**

| 측면 | Phase 2 (표현식) | Phase 3 (함수) |
|-----|------------------|----------------|
| 코드 조직 | 단일 main 함수 | 여러 함수 정의 |
| 재사용 | 불가능 (중복 코드) | 가능 (함수 호출) |
| 추상화 | 없음 | 함수 이름으로 추상화 |
| 모듈성 | 낮음 | 높음 (함수 단위) |
| 컴파일 결과 | 단일 함수 | 여러 함수 심볼 |

함수는 코드를 **모듈화**하고 **재사용 가능**하게 만든다. Phase 2의 표현식 컴파일러를 함수 본체 내부에서 재사용한다!

## AST 확장: FunDef와 App

FunLang에 함수를 추가하려면 AST를 확장해야 한다. 두 가지 새로운 노드가 필요하다:
1. **FunDef**: 함수 정의 (`let f x y = ...`)
2. **App**: 함수 적용 (호출) (`f 10 20`)

### FunDef: 함수 정의

**FunDef**는 최상위 함수 정의를 표현한다.

**F# AST 정의:**

```fsharp
type Expr =
    | Int of int
    | Bool of bool
    | Var of string
    | BinOp of Expr * Operator * Expr
    | UnaryOp of UnaryOperator * Expr
    | Compare of Expr * CompareOp * Expr
    | Let of string * Expr * Expr
    | If of Expr * Expr * Expr
    | App of string * Expr list              // NEW: 함수 호출
    // ... Lambda는 Phase 4에서 추가 ...

type FunDef = {                               // NEW: 함수 정의
    name: string                              // 함수 이름
    parameters: string list                   // 파라미터 이름 리스트
    body: Expr                                // 함수 본체 (표현식)
}

type Program = {                              // NEW: 프로그램 구조
    functions: FunDef list                    // 함수 정의 리스트
    main: Expr                                // Main 표현식
}
```

**예시: `let add x y = x + y`**

```fsharp
let addFunction = {
    name = "add"
    parameters = ["x"; "y"]
    body = BinOp(Var "x", Add, Var "y")
}
```

**구성 요소:**
- `name`: 함수 이름 (`"add"`)
- `parameters`: 파라미터 이름 리스트 (`["x"; "y"]`)
- `body`: 함수 본체 (`x + y` 표현식)

**예시: `let square x = x * x`**

```fsharp
let squareFunction = {
    name = "square"
    parameters = ["x"]
    body = BinOp(Var "x", Mul, Var "x")
}
```

**예시: 파라미터가 없는 함수 `let getConstant = 42`**

```fsharp
let constantFunction = {
    name = "getConstant"
    parameters = []                           // 빈 리스트
    body = Int 42
}
```

### App: 함수 적용 (호출)

**App**는 함수 호출을 표현한다. 함수 이름과 인자 리스트를 포함한다.

**F# AST 정의:**

```fsharp
type Expr =
    | ...
    | App of string * Expr list               // 함수 이름, 인자 리스트
```

**예시: `add 10 20`**

```fsharp
let callExpr = App("add", [Int 10; Int 20])
```

**구성 요소:**
- 함수 이름: `"add"`
- 인자 리스트: `[Int 10; Int 20]`

**예시: `square 5`**

```fsharp
let squareCall = App("square", [Int 5])
```

**예시: 중첩 호출 `add (square 3) (square 4)`**

```fsharp
let nestedCall =
    App("add", [
        App("square", [Int 3]);
        App("square", [Int 4])
    ])
```

**해석:**
- `square 3` → 9
- `square 4` → 16
- `add 9 16` → 25

### Program: 프로그램 구조

지금까지는 FunLang 프로그램이 단일 표현식이었다. 이제 **여러 함수 정의 + main 표현식**으로 구성된다.

**F# 정의:**

```fsharp
type Program = {
    functions: FunDef list                    // 함수 정의 리스트
    main: Expr                                // Main 표현식
}
```

**예시 프로그램:**

```fsharp
// FunLang 소스:
// let add x y = x + y
// let square x = x * x
// square (add 3 4)

let program = {
    functions = [
        { name = "add"
          parameters = ["x"; "y"]
          body = BinOp(Var "x", Add, Var "y") };
        { name = "square"
          parameters = ["x"]
          body = BinOp(Var "x", Mul, Var "x") }
    ]
    main = App("square", [App("add", [Int 3; Int 4])])
}
```

**실행:**
1. `add 3 4` → 7
2. `square 7` → 49
3. 최종 결과: 49

**프로그램 구조 시각화:**

```
Program
├── functions
│   ├── FunDef("add", ["x", "y"], x + y)
│   └── FunDef("square", ["x"], x * x)
└── main
    └── App("square", [App("add", [3, 4])])
```

### Lambda는 어디에?

함수형 언어의 핵심 기능인 **lambda (익명 함수)**는 어디에 있는가?

**Phase 3 범위: 최상위 명명된 함수만**
- `let f x = ...` (함수 정의)
- `f 10` (함수 호출)

**Phase 4에서 추가: Lambda와 클로저**
- `fun x -> x + 1` (익명 함수)
- `let makeAdder n = fun x -> x + n` (클로저, 외부 변수 캡처)
- 함수를 값으로 전달 (고차 함수)

Phase 3 함수는 **정적**이다:
- 컴파일 타임에 모든 함수가 알려진다
- 함수 이름은 고정된 심볼이다
- 외부 환경을 캡처하지 않는다

Phase 4 클로저는 **동적**이다:
- 런타임에 클로저가 생성된다
- 클로저는 값처럼 전달된다
- 외부 환경을 캡처하고 유지한다

Phase 3은 함수의 **기초**를 다진다. Phase 4는 그 위에 클로저를 추가한다.

## P/Invoke 바인딩: func 다이얼렉트

MLIR의 func 다이얼렉트 연산을 사용하려면 C API 바인딩이 필요하다. 이미 Phase 1에서 기본 바인딩을 작성했으므로, func 관련 함수를 추가한다.

### Function Type API

MLIR에서 함수는 **function type**을 가진다. Function type은 파라미터 타입과 반환 타입을 표현한다.

**Function type 생성:**

```c
// C API
MlirType mlirFunctionTypeGet(
    MlirContext ctx,
    intptr_t numInputs,
    MlirType const *inputs,
    intptr_t numResults,
    MlirType const *results
);
```

**파라미터:**
- `ctx`: MLIR context
- `numInputs`: 파라미터 개수
- `inputs`: 파라미터 타입 배열
- `numResults`: 반환 값 개수 (보통 0 또는 1)
- `results`: 반환 타입 배열

**예시: `(i32, i32) -> i32` 타입**

```c
MlirType i32Type = mlirIntegerTypeGet(ctx, 32);
MlirType paramTypes[] = { i32Type, i32Type };  // 두 개의 i32 파라미터
MlirType resultTypes[] = { i32Type };          // 하나의 i32 반환값

MlirType funcType = mlirFunctionTypeGet(
    ctx,
    2, paramTypes,   // 2개 파라미터
    1, resultTypes   // 1개 반환값
);
```

**Function type 쿼리:**

```c
// 파라미터 개수 가져오기
intptr_t mlirFunctionTypeGetNumInputs(MlirType type);

// 반환 값 개수 가져오기
intptr_t mlirFunctionTypeGetNumResults(MlirType type);

// N번째 파라미터 타입 가져오기
MlirType mlirFunctionTypeGetInput(MlirType type, intptr_t pos);

// N번째 반환 타입 가져오기
MlirType mlirFunctionTypeGetResult(MlirType type, intptr_t pos);
```

**F# P/Invoke 바인딩:**

```fsharp
// MlirBindings.fs에 추가

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunctionTypeGet(
    MlirContext ctx,
    nativeint numInputs,
    [<MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1s)>] MlirType[] inputs,
    nativeint numResults,
    [<MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 3s)>] MlirType[] results
)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirFunctionTypeGetNumInputs(MlirType funcType)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirFunctionTypeGetNumResults(MlirType funcType)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunctionTypeGetInput(MlirType funcType, nativeint pos)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunctionTypeGetResult(MlirType funcType, nativeint pos)
```

**사용 예시:**

```fsharp
// (i32, i32) -> i32 타입 생성
let i32Type = mlirIntegerTypeGet(ctx, 32u)
let paramTypes = [| i32Type; i32Type |]
let resultTypes = [| i32Type |]

let funcType = mlirFunctionTypeGet(
    ctx,
    2n, paramTypes,
    1n, resultTypes
)

// 타입 쿼리
let numParams = mlirFunctionTypeGetNumInputs(funcType)  // 2
let param0Type = mlirFunctionTypeGetInput(funcType, 0n)  // i32
```

### Symbol Reference Attribute

함수 호출 시 **symbol reference**가 필요하다. 심볼 참조는 `@function_name` 형태로, attribute로 표현된다.

**C API:**

```c
// Flat symbol reference (단일 심볼)
MlirAttribute mlirFlatSymbolRefAttrGet(
    MlirContext ctx,
    MlirStringRef symbol
);
```

**F# P/Invoke 바인딩:**

```fsharp
[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirAttribute mlirFlatSymbolRefAttrGet(
    MlirContext ctx,
    MlirStringRef symbol
)
```

**사용 예시:**

```fsharp
// @add 심볼 참조 생성
let addSymbol = MlirStringRef.FromString("add")
let addSymbolAttr = mlirFlatSymbolRefAttrGet(ctx, addSymbol)
```

### Generic Operation Creation for func.func

MLIR C API는 `func.func` 전용 생성 함수를 제공하지 않는다. 대신 **generic operation creation**을 사용한다.

**func.func 연산 생성 단계:**

1. **Operation state 초기화**

   ```c
   MlirOperationState state = mlirOperationStateGet(
       mlirStringRefCreateFromCString("func.func"),
       location
   );
   ```

2. **Attributes 추가 (sym_name, function_type)**

   ```c
   // sym_name: 함수 이름
   MlirAttribute nameAttr = mlirStringAttrGet(ctx, nameStringRef);
   MlirNamedAttribute symNameAttr = {
       mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
       nameAttr
   };

   // function_type: 함수 타입
   MlirAttribute typeAttr = mlirTypeAttrGet(functionType);
   MlirNamedAttribute funcTypeAttr = {
       mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")),
       typeAttr
   };

   MlirNamedAttribute attrs[] = { symNameAttr, funcTypeAttr };
   mlirOperationStateAddAttributes(&state, 2, attrs);
   ```

3. **Region 추가 (함수 본체)**

   ```c
   MlirRegion bodyRegion = mlirRegionCreate();
   MlirBlock entryBlock = mlirBlockCreate(numParams, paramTypes, NULL);
   mlirRegionAppendOwnedBlock(bodyRegion, entryBlock);
   mlirOperationStateAddOwnedRegions(&state, 1, &bodyRegion);
   ```

4. **Operation 생성**

   ```c
   MlirOperation funcOp = mlirOperationCreate(&state);
   ```

**F# 헬퍼 함수 (OpBuilder에 추가 예정):**

```fsharp
// OpBuilder.fs에 추가할 메서드 (다음 섹션에서 구현)
member this.CreateFuncOp(name: string, paramTypes: MlirType[], resultType: MlirType) =
    // ... implementation ...
```

### Generic Operation Creation for func.call

**func.call 연산 생성 단계:**

1. **Operation state 초기화**

   ```c
   MlirOperationState state = mlirOperationStateGet(
       mlirStringRefCreateFromCString("func.call"),
       location
   );
   ```

2. **Callee attribute 추가**

   ```c
   MlirAttribute calleeAttr = mlirFlatSymbolRefAttrGet(ctx, calleeSymbol);
   MlirNamedAttribute attr = {
       mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("callee")),
       calleeAttr
   };
   mlirOperationStateAddAttributes(&state, 1, &attr);
   ```

3. **Operands 추가 (인자)**

   ```c
   mlirOperationStateAddOperands(&state, numArgs, argValues);
   ```

4. **Result types 추가**

   ```c
   mlirOperationStateAddResults(&state, 1, &resultType);
   ```

5. **Operation 생성**

   ```c
   MlirOperation callOp = mlirOperationCreate(&state);
   ```

**F# 헬퍼 함수:**

```fsharp
member this.CreateFuncCall(calleeName: string, args: MlirValue[], resultType: MlirType) =
    // ... implementation ...
```

### Complete MlirBindings.fs Additions

**전체 추가 코드 (MlirBindings.fs):**

```fsharp
// ============================================================
// Function Type API
// ============================================================

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunctionTypeGet(
    MlirContext ctx,
    nativeint numInputs,
    [<MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1s)>] MlirType[] inputs,
    nativeint numResults,
    [<MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 3s)>] MlirType[] results
)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirFunctionTypeGetNumInputs(MlirType funcType)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirFunctionTypeGetNumResults(MlirType funcType)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunctionTypeGetInput(MlirType funcType, nativeint pos)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunctionTypeGetResult(MlirType funcType, nativeint pos)

// ============================================================
// Symbol Reference Attribute
// ============================================================

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirAttribute mlirFlatSymbolRefAttrGet(
    MlirContext ctx,
    MlirStringRef symbol
)

// ============================================================
// Block Arguments (for function parameters)
// ============================================================

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirValue mlirBlockGetArgument(
    MlirBlock block,
    nativeint pos
)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirBlockGetNumArguments(MlirBlock block)
```

**설명:**
- `mlirFunctionTypeGet`: 함수 타입 생성
- `mlirFunctionTypeGetInput/GetResult`: 함수 타입 쿼리
- `mlirFlatSymbolRefAttrGet`: 심볼 참조 attribute 생성
- `mlirBlockGetArgument`: 블록의 N번째 argument 가져오기 (함수 파라미터)
- `mlirBlockGetNumArguments`: 블록의 argument 개수 (파라미터 개수)

이 바인딩으로 func 다이얼렉트의 모든 연산을 생성할 수 있다!

## OpBuilder 확장: func 연산 헬퍼

P/Invoke 바인딩은 저수준 API다. 사용하기 편리한 F# 헬퍼 메서드를 OpBuilder 클래스에 추가한다.

### CreateFuncOp: 함수 생성

**목적:** func.func 연산을 생성한다. 함수 이름, 파라미터 타입, 반환 타입을 받아 함수 operation을 반환한다.

**시그니처:**

```fsharp
member this.CreateFuncOp(
    name: string,
    paramTypes: MlirType[],
    resultType: MlirType
) : MlirOperation
```

**구현:**

```fsharp
member this.CreateFuncOp(name: string, paramTypes: MlirType[], resultType: MlirType) =
    let loc = this.UnknownLoc()

    // 1. Function type 생성
    let resultTypes = [| resultType |]
    let funcType = mlirFunctionTypeGet(
        this.Context,
        nativeint paramTypes.Length, paramTypes,
        1n, resultTypes
    )

    // 2. Operation state 초기화
    let opName = MlirStringRef.FromString("func.func")
    let mutable state = mlirOperationStateGet(opName, loc)

    // 3. sym_name attribute 추가
    let nameStr = MlirStringRef.FromString(name)
    let nameAttr = mlirStringAttrGet(this.Context, nameStr)
    let symNameId = mlirIdentifierGet(this.Context, MlirStringRef.FromString("sym_name"))
    let mutable symNameAttr = MlirNamedAttribute(symNameId, nameAttr)

    // 4. function_type attribute 추가
    let typeAttr = mlirTypeAttrGet(funcType)
    let funcTypeId = mlirIdentifierGet(this.Context, MlirStringRef.FromString("function_type"))
    let mutable funcTypeAttr = MlirNamedAttribute(funcTypeId, typeAttr)

    // 5. Attributes 추가
    let attrs = [| symNameAttr; funcTypeAttr |]
    mlirOperationStateAddAttributes(&state, 2n, attrs)

    // 6. Body region 생성 (entry block with parameters)
    let bodyRegion = mlirRegionCreate()
    let entryBlock = mlirBlockCreate(
        nativeint paramTypes.Length,
        paramTypes,
        Array.zeroCreate paramTypes.Length  // Location array (null array)
    )
    mlirRegionAppendOwnedBlock(bodyRegion, entryBlock)

    let regions = [| bodyRegion |]
    mlirOperationStateAddOwnedRegions(&state, 1n, regions)

    // 7. Operation 생성
    let funcOp = mlirOperationCreate(&state)
    funcOp
```

**사용 예시:**

```fsharp
let builder = new OpBuilder(ctx, module)

// func.func @add(%arg0: i32, %arg1: i32) -> i32
let funcOp = builder.CreateFuncOp(
    "add",
    [| i32Type; i32Type |],
    i32Type
)

// 이제 funcOp 내부에 body를 추가해야 한다
```

**핵심 포인트:**
- `paramTypes`는 블록 arguments의 타입이 된다
- Entry block이 자동으로 생성되고 region에 추가된다
- 반환된 `MlirOperation`은 아직 비어있는 함수 (body를 채워야 함)

### GetFunctionEntryBlock: entry block 가져오기

함수 본체를 작성하려면 entry block을 가져와야 한다.

**시그니처:**

```fsharp
member this.GetFunctionEntryBlock(funcOp: MlirOperation) : MlirBlock
```

**구현:**

```fsharp
member this.GetFunctionEntryBlock(funcOp: MlirOperation) =
    // func.func operation은 region을 하나 가진다
    let bodyRegion = mlirOperationGetRegion(funcOp, 0n)
    // Region의 첫 번째 block이 entry block
    mlirRegionGetFirstBlock(bodyRegion)
```

**사용 예시:**

```fsharp
let funcOp = builder.CreateFuncOp("add", [| i32Type; i32Type |], i32Type)
let entryBlock = builder.GetFunctionEntryBlock(funcOp)

// 이제 entryBlock에 연산을 추가할 수 있다
builder.SetInsertionPointToEnd(entryBlock)
```

### GetFunctionBlockArg: 파라미터 가져오기

함수 파라미터는 entry block의 **block arguments**로 표현된다. 파라미터를 사용하려면 block argument를 가져와야 한다.

**시그니처:**

```fsharp
member this.GetFunctionBlockArg(block: MlirBlock, index: int) : MlirValue
```

**구현:**

```fsharp
member this.GetFunctionBlockArg(block: MlirBlock, index: int) =
    mlirBlockGetArgument(block, nativeint index)
```

**사용 예시:**

```fsharp
let funcOp = builder.CreateFuncOp("add", [| i32Type; i32Type |], i32Type)
let entryBlock = builder.GetFunctionEntryBlock(funcOp)

// 파라미터 가져오기
let arg0 = builder.GetFunctionBlockArg(entryBlock, 0)  // %arg0
let arg1 = builder.GetFunctionBlockArg(entryBlock, 1)  // %arg1

// 파라미터를 사용하여 연산 수행
builder.SetInsertionPointToEnd(entryBlock)
let sum = builder.CreateArithBinaryOp(ArithOp.Addi, arg0, arg1, i32Type)
```

### CreateFuncCall: 함수 호출 생성

**시그니처:**

```fsharp
member this.CreateFuncCall(
    calleeName: string,
    args: MlirValue[],
    resultType: MlirType
) : MlirValue
```

**구현:**

```fsharp
member this.CreateFuncCall(calleeName: string, args: MlirValue[], resultType: MlirType) =
    let loc = this.UnknownLoc()

    // 1. Operation state 초기화
    let opName = MlirStringRef.FromString("func.call")
    let mutable state = mlirOperationStateGet(opName, loc)

    // 2. callee attribute 추가
    let calleeSymbol = MlirStringRef.FromString(calleeName)
    let calleeAttr = mlirFlatSymbolRefAttrGet(this.Context, calleeSymbol)
    let calleeId = mlirIdentifierGet(this.Context, MlirStringRef.FromString("callee"))
    let mutable calleeNamedAttr = MlirNamedAttribute(calleeId, calleeAttr)

    mlirOperationStateAddAttributes(&state, 1n, [| calleeNamedAttr |])

    // 3. Operands 추가
    mlirOperationStateAddOperands(&state, nativeint args.Length, args)

    // 4. Result type 추가
    mlirOperationStateAddResults(&state, 1n, [| resultType |])

    // 5. Operation 생성
    let callOp = mlirOperationCreate(&state)

    // 6. 현재 insertion point에 추가
    mlirBlockAppendOwnedOperation(this.currentBlock, callOp)

    // 7. Result value 반환
    mlirOperationGetResult(callOp, 0n)
```

**사용 예시:**

```fsharp
builder.SetInsertionPointToEnd(mainBlock)

// func.call @add(%c10, %c20) : (i32, i32) -> i32
let c10 = builder.CreateConstant(10, i32Type)
let c20 = builder.CreateConstant(20, i32Type)
let result = builder.CreateFuncCall("add", [| c10; c20 |], i32Type)
```

### CreateFuncReturn: 함수 반환

**시그니처:**

```fsharp
member this.CreateFuncReturn(value: MlirValue) : unit
```

**구현:**

```fsharp
member this.CreateFuncReturn(value: MlirValue) =
    let loc = this.UnknownLoc()

    // 1. Operation state 초기화
    let opName = MlirStringRef.FromString("func.return")
    let mutable state = mlirOperationStateGet(opName, loc)

    // 2. Operand 추가 (반환 값)
    mlirOperationStateAddOperands(&state, 1n, [| value |])

    // 3. Operation 생성
    let returnOp = mlirOperationCreate(&state)

    // 4. 현재 insertion point에 추가
    mlirBlockAppendOwnedOperation(this.currentBlock, returnOp)
```

**사용 예시:**

```fsharp
builder.SetInsertionPointToEnd(entryBlock)
let sum = builder.CreateArithBinaryOp(ArithOp.Addi, arg0, arg1, i32Type)
builder.CreateFuncReturn(sum)
```

### 완전한 함수 생성 예시

**전체 흐름 (add 함수 생성):**

```fsharp
let builder = new OpBuilder(ctx, module)
let i32Type = builder.I32Type()

// 1. 함수 operation 생성
let funcOp = builder.CreateFuncOp("add", [| i32Type; i32Type |], i32Type)

// 2. Entry block 가져오기
let entryBlock = builder.GetFunctionEntryBlock(funcOp)

// 3. 파라미터 가져오기
let arg0 = builder.GetFunctionBlockArg(entryBlock, 0)
let arg1 = builder.GetFunctionBlockArg(entryBlock, 1)

// 4. Insertion point 설정
builder.SetInsertionPointToEnd(entryBlock)

// 5. 함수 본체 작성
let sum = builder.CreateArithBinaryOp(ArithOp.Addi, arg0, arg1, i32Type)

// 6. 반환
builder.CreateFuncReturn(sum)

// 7. 모듈에 함수 추가
builder.AddOperationToModule(funcOp)
```

**생성된 MLIR IR:**

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
```

이 헬퍼 메서드들로 func 다이얼렉트 연산을 쉽게 생성할 수 있다!

## 함수 파라미터와 Block Arguments

함수 파라미터는 MLIR에서 **block arguments**로 표현된다. 이것은 MLIR의 핵심 설계 원칙이며, Chapter 08에서 배운 block arguments 개념의 확장이다.

### 파라미터는 변수가 아니다

전통적인 프로그래밍 언어에서 함수 파라미터는 "변수"처럼 보인다:

```c
// C 함수
int add(int x, int y) {
    return x + y;
}
```

하지만 MLIR에서 파라미터는 **block arguments**다:

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %result = arith.addi %arg0, %arg1 : i32
  func.return %result : i32
}
```

**차이점:**

| 관점 | 변수 (C/Java) | Block Arguments (MLIR) |
|-----|---------------|------------------------|
| 저장 위치 | 스택 메모리 (또는 레지스터) | SSA value (레지스터 직접 사용) |
| 초기화 | 함수 진입 시 스택에 복사 | 블록 진입 시 이미 존재 |
| 뮤테이션 | 가능 (재할당 가능) | 불가능 (SSA, 한 번만 정의) |
| 주소 | 주소 가져오기 가능 (`&x`) | 주소 없음 (값 자체) |

MLIR에서 파라미터는 **이미 존재하는 SSA value**다. 함수가 호출되면, 인자 값들이 entry block의 arguments로 전달된다.

### Block Arguments 복습 (Chapter 08 연결)

Chapter 08에서 `scf.if`의 block arguments를 배웠다:

```mlir
%result = scf.if %condition -> (i32) {
  %c10 = arith.constant 10 : i32
  scf.yield %c10 : i32
} else {
  %c20 = arith.constant 20 : i32
  scf.yield %c20 : i32
}
// %result는 block argument (scf.if의 결과)
```

함수 파라미터도 동일한 메커니즘이다:

```mlir
func.func @example(%arg0: i32) -> i32 {
  // %arg0는 entry block의 argument
  func.return %arg0 : i32
}
```

**공통점:**
- 둘 다 **block arguments**다
- 둘 다 SSA values다
- 둘 다 블록 진입 시 이미 정의되어 있다

**차이점:**
- `scf.if` block arguments: 분기의 결과 값 (yield로 전달)
- 함수 block arguments: 함수의 입력 값 (호출자가 전달)

### Entry Block과 파라미터

함수의 entry block은 함수 정의 시 자동으로 생성된다. 파라미터 개수만큼 block arguments를 가진다.

**예시: 파라미터가 3개인 함수**

```mlir
func.func @sum3(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  // Entry block은 3개의 arguments를 가진다:
  // - %arg0 (첫 번째 파라미터)
  // - %arg1 (두 번째 파라미터)
  // - %arg2 (세 번째 파라미터)

  %sum01 = arith.addi %arg0, %arg1 : i32
  %sum012 = arith.addi %sum01, %arg2 : i32
  func.return %sum012 : i32
}
```

**MLIR IR 구조 시각화:**

```
func.func @sum3(...) {
^entry(%arg0: i32, %arg1: i32, %arg2: i32):
    // %arg0, %arg1, %arg2는 block arguments
    %sum01 = arith.addi %arg0, %arg1
    %sum012 = arith.addi %sum01, %arg2
    func.return %sum012
}
```

Entry block의 arguments는 함수 시그니처의 파라미터와 1:1 대응된다.

### 파라미터와 환경 (Environment)

Chapter 07에서 let 바인딩을 위한 **환경(environment)**을 구현했다:

```fsharp
type Environment = Map<string, MlirValue>
```

함수 파라미터도 환경에 추가해야 한다. 하지만 let 바인딩과는 다른 방식으로 처리한다:

**Let 바인딩:**
- 표현식을 컴파일하여 SSA value 생성
- 환경에 추가
- 본체 표현식 컴파일

**함수 파라미터:**
- Block arguments로 이미 존재
- 환경에 추가 (이름 → block argument 매핑)
- 본체 표현식 컴파일

**코드 비교:**

```fsharp
// Let 바인딩 (Phase 2)
| Let(name, valueExpr, bodyExpr) ->
    let value = compileExpr builder env valueExpr  // 표현식 컴파일
    let newEnv = Map.add name value env            // 환경 확장
    compileExpr builder newEnv bodyExpr

// 함수 파라미터 (Phase 3)
let compileFuncDef builder (funcDef: FunDef) =
    // ...
    let entryBlock = builder.GetFunctionEntryBlock(funcOp)

    // 파라미터를 환경에 추가
    let initialEnv =
        funcDef.parameters
        |> List.mapi (fun i name ->
            let arg = builder.GetFunctionBlockArg(entryBlock, i)
            (name, arg)
        )
        |> Map.ofList

    // 본체 컴파일 (환경 전달)
    let bodyValue = compileExpr builder initialEnv funcDef.body
    builder.CreateFuncReturn(bodyValue)
```

**핵심 차이:**
- Let 바인딩: `compileExpr`로 value 생성
- 함수 파라미터: `GetFunctionBlockArg`로 기존 value 가져오기

### 예시: 함수 본체에서 파라미터 사용

**FunLang 소스:**

```fsharp
let double x = x + x
```

**AST:**

```fsharp
{
    name = "double"
    parameters = ["x"]
    body = BinOp(Var "x", Add, Var "x")
}
```

**컴파일 과정:**

1. **함수 operation 생성**

   ```fsharp
   let funcOp = builder.CreateFuncOp("double", [| i32Type |], i32Type)
   ```

2. **Entry block 가져오기**

   ```fsharp
   let entryBlock = builder.GetFunctionEntryBlock(funcOp)
   ```

3. **파라미터를 환경에 추가**

   ```fsharp
   let arg0 = builder.GetFunctionBlockArg(entryBlock, 0)  // %arg0
   let env = Map.ofList [("x", arg0)]
   ```

4. **본체 컴파일 (`x + x`)**

   ```fsharp
   builder.SetInsertionPointToEnd(entryBlock)

   // BinOp(Var "x", Add, Var "x")
   // Var "x" → 환경에서 조회 → %arg0
   let lhs = env.["x"]  // %arg0
   let rhs = env.["x"]  // %arg0
   let sum = builder.CreateArithBinaryOp(ArithOp.Addi, lhs, rhs, i32Type)
   ```

5. **반환**

   ```fsharp
   builder.CreateFuncReturn(sum)
   ```

**생성된 MLIR IR:**

```mlir
func.func @double(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  func.return %0 : i32
}
```

### Let 바인딩 vs 함수 파라미터 구분

함수 본체 내부에서 let 바인딩과 파라미터를 모두 사용할 수 있다:

**FunLang 소스:**

```fsharp
let compute x y =
    let doubled = x + x in
    doubled + y
```

**환경 변화 추적:**

```fsharp
// 1. 초기 환경 (파라미터만)
env = { "x" -> %arg0, "y" -> %arg1 }

// 2. Let 바인딩 처리
// let doubled = x + x
let doubledValue = arith.addi %arg0, %arg0
env = { "x" -> %arg0, "y" -> %arg1, "doubled" -> %0 }

// 3. 본체 표현식 (doubled + y)
let result = arith.addi %0, %arg1
```

**생성된 MLIR IR:**

```mlir
func.func @compute(%arg0: i32, %arg1: i32) -> i32 {
  %doubled = arith.addi %arg0, %arg0 : i32   // let doubled = x + x
  %result = arith.addi %doubled, %arg1 : i32 // doubled + y
  func.return %result : i32
}
```

**결론:** 파라미터와 let 바인딩 모두 **환경을 통해 관리**된다. 차이점은 value의 출처뿐이다 (block argument vs 컴파일된 표현식).

## 코드 생성: 함수 정의

이제 FunLang 함수 정의 (FunDef)를 MLIR func.func 연산으로 컴파일하는 `compileFuncDef` 함수를 작성한다.

### compileFuncDef 시그니처

```fsharp
let compileFuncDef (builder: OpBuilder) (funcDef: FunDef) : unit =
    // ...
```

**입력:**
- `builder`: OpBuilder (MLIR IR 생성 도구)
- `funcDef`: FunDef (FunLang 함수 정의)

**출력:**
- `unit` (모듈에 함수를 추가하는 부수 효과)

### 단계별 구현

**Step 1: 타입 준비**

파라미터 타입과 반환 타입을 준비한다. Phase 3에서는 모든 값이 i32다.

```fsharp
let i32Type = builder.I32Type()
let paramTypes = Array.create funcDef.parameters.Length i32Type
let resultType = i32Type
```

**Step 2: 함수 operation 생성**

```fsharp
let funcOp = builder.CreateFuncOp(funcDef.name, paramTypes, resultType)
```

**Step 3: Entry block 가져오기**

```fsharp
let entryBlock = builder.GetFunctionEntryBlock(funcOp)
```

**Step 4: 초기 환경 구축 (파라미터 → block arguments)**

```fsharp
let initialEnv =
    funcDef.parameters
    |> List.mapi (fun i paramName ->
        let arg = builder.GetFunctionBlockArg(entryBlock, i)
        (paramName, arg)
    )
    |> Map.ofList
```

**Step 5: Insertion point 설정**

```fsharp
builder.SetInsertionPointToEnd(entryBlock)
```

**Step 6: 본체 표현식 컴파일**

```fsharp
let bodyValue = compileExpr builder initialEnv funcDef.body
```

`compileExpr`는 Phase 2에서 작성한 함수다. 환경을 받아서 표현식을 컴파일한다.

**Step 7: func.return 삽입**

```fsharp
builder.CreateFuncReturn(bodyValue)
```

**Step 8: 모듈에 함수 추가**

```fsharp
builder.AddOperationToModule(funcOp)
```

### 완전한 compileFuncDef 구현

```fsharp
let compileFuncDef (builder: OpBuilder) (funcDef: FunDef) : unit =
    // 1. 타입 준비
    let i32Type = builder.I32Type()
    let paramTypes = Array.create funcDef.parameters.Length i32Type
    let resultType = i32Type

    // 2. 함수 operation 생성
    let funcOp = builder.CreateFuncOp(funcDef.name, paramTypes, resultType)

    // 3. Entry block 가져오기
    let entryBlock = builder.GetFunctionEntryBlock(funcOp)

    // 4. 초기 환경 구축 (파라미터 → block arguments)
    let initialEnv =
        funcDef.parameters
        |> List.mapi (fun i paramName ->
            let arg = builder.GetFunctionBlockArg(entryBlock, i)
            (paramName, arg)
        )
        |> Map.ofList

    // 5. Insertion point 설정
    builder.SetInsertionPointToEnd(entryBlock)

    // 6. 본체 표현식 컴파일
    let bodyValue = compileExpr builder initialEnv funcDef.body

    // 7. func.return 삽입
    builder.CreateFuncReturn(bodyValue)

    // 8. 모듈에 함수 추가
    builder.AddOperationToModule(funcOp)
```

### 예시: let double x = x + x

**FunDef:**

```fsharp
{
    name = "double"
    parameters = ["x"]
    body = BinOp(Var "x", Add, Var "x")
}
```

**compileFuncDef 실행 과정:**

1. `paramTypes = [| i32Type |]`, `resultType = i32Type`
2. `funcOp = CreateFuncOp("double", [| i32 |], i32)`
3. `entryBlock = GetFunctionEntryBlock(funcOp)`
4. `arg0 = GetFunctionBlockArg(entryBlock, 0)`, `env = { "x" -> %arg0 }`
5. `SetInsertionPointToEnd(entryBlock)`
6. `bodyValue = compileExpr builder env (BinOp(Var "x", Add, Var "x"))`
   - `Var "x"` → `env.["x"]` → `%arg0`
   - `arith.addi %arg0, %arg0`
7. `CreateFuncReturn(bodyValue)`
8. `AddOperationToModule(funcOp)`

**생성된 MLIR IR:**

```mlir
func.func @double(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  func.return %0 : i32
}
```

### 예시: let add x y = x + y

**FunDef:**

```fsharp
{
    name = "add"
    parameters = ["x"; "y"]
    body = BinOp(Var "x", Add, Var "y")
}
```

**생성된 MLIR IR:**

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
```

### 복잡한 예시: let compute x y = (x + x) + y

**FunDef:**

```fsharp
{
    name = "compute"
    parameters = ["x"; "y"]
    body = BinOp(
        BinOp(Var "x", Add, Var "x"),
        Add,
        Var "y"
    )
}
```

**생성된 MLIR IR:**

```mlir
func.func @compute(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32      // x + x
  %1 = arith.addi %0, %arg1 : i32         // (x + x) + y
  func.return %1 : i32
}
```

`compileExpr`가 재귀적으로 호출되어 중첩된 연산을 처리한다!

## 코드 생성: 함수 호출

함수를 정의했으니 이제 호출할 수 있어야 한다. 함수 호출은 `App` 노드로 표현되며, `compileExpr`에 새로운 case를 추가한다.

### App case 추가

**compileExpr 확장:**

```fsharp
let rec compileExpr (builder: OpBuilder) (env: Environment) (expr: Expr) : MlirValue =
    match expr with
    | Int n -> builder.CreateConstant(n, builder.I32Type())
    | Bool b -> builder.CreateConstant((if b then 1 else 0), builder.I1Type())
    | Var name ->
        match Map.tryFind name env with
        | Some value -> value
        | None -> failwithf "Unbound variable: %s" name
    | BinOp(lhs, op, rhs) ->
        let lhsValue = compileExpr builder env lhs
        let rhsValue = compileExpr builder env rhs
        builder.CreateArithBinaryOp(op, lhsValue, rhsValue, builder.I32Type())
    | Compare(lhs, op, rhs) ->
        let lhsValue = compileExpr builder env lhs
        let rhsValue = compileExpr builder env rhs
        builder.CreateArithCompare(op, lhsValue, rhsValue)
    | Let(name, valueExpr, bodyExpr) ->
        let value = compileExpr builder env valueExpr
        let newEnv = Map.add name value env
        compileExpr builder newEnv bodyExpr
    | If(condition, thenExpr, elseExpr) ->
        let condValue = compileExpr builder env condition
        compileIfExpr builder env condValue thenExpr elseExpr
    | App(calleeName, argExprs) ->                  // NEW: 함수 호출
        // Step 1: 인자 표현식들을 컴파일
        let argValues =
            argExprs
            |> List.map (compileExpr builder env)
            |> List.toArray

        // Step 2: 함수 호출 생성
        let resultType = builder.I32Type()
        builder.CreateFuncCall(calleeName, argValues, resultType)
```

### App case 설명

**Step 1: 인자 컴파일**

함수 호출 전에 모든 인자 표현식을 먼저 컴파일한다 (call-by-value 의미론).

```fsharp
let argValues =
    argExprs
    |> List.map (compileExpr builder env)
    |> List.toArray
```

**예시:**

```fsharp
// add (5 + 3) (10 * 2)
App("add", [
    BinOp(Int 5, Add, Int 3);
    BinOp(Int 10, Mul, Int 2)
])
```

인자 컴파일 결과:
- `5 + 3` → `%0 = arith.addi ... (8)`
- `10 * 2` → `%1 = arith.muli ... (20)`
- `argValues = [| %0; %1 |]`

**Step 2: 함수 호출 생성**

```fsharp
let resultType = builder.I32Type()
builder.CreateFuncCall(calleeName, argValues, resultType)
```

`CreateFuncCall`이 `func.call` 연산을 생성하고 결과 SSA value를 반환한다.

### 예시: double 5

**FunLang 표현식:**

```fsharp
App("double", [Int 5])
```

**컴파일 과정:**

1. 인자 컴파일: `Int 5` → `%c5 = arith.constant 5 : i32`
2. 함수 호출: `CreateFuncCall("double", [| %c5 |], i32Type)`

**생성된 MLIR IR:**

```mlir
%c5 = arith.constant 5 : i32
%0 = func.call @double(%c5) : (i32) -> i32
```

### 예시: add 10 20

**FunLang 표현식:**

```fsharp
App("add", [Int 10; Int 20])
```

**생성된 MLIR IR:**

```mlir
%c10 = arith.constant 10 : i32
%c20 = arith.constant 20 : i32
%0 = func.call @add(%c10, %c20) : (i32, i32) -> i32
```

### 중첩 호출 예시: double (add 3 4)

**FunLang 표현식:**

```fsharp
App("double", [
    App("add", [Int 3; Int 4])
])
```

**컴파일 과정:**

1. 외부 호출의 인자 컴파일: `App("add", [Int 3; Int 4])`
   - 내부 호출의 인자 컴파일: `Int 3` → `%c3`, `Int 4` → `%c4`
   - 내부 호출: `%inner = func.call @add(%c3, %c4)`
2. 외부 호출: `%result = func.call @double(%inner)`

**생성된 MLIR IR:**

```mlir
%c3 = arith.constant 3 : i32
%c4 = arith.constant 4 : i32
%inner = func.call @add(%c3, %c4) : (i32, i32) -> i32
%result = func.call @double(%inner) : (i32) -> i32
```

중첩 호출이 자연스럽게 처리된다!

## 코드 생성: Program 컴파일

이제 전체 프로그램을 컴파일하는 `compileProgram` 함수를 작성한다. Program은 여러 함수 정의와 main 표현식으로 구성된다.

### compileProgram 시그니처

```fsharp
let compileProgram (builder: OpBuilder) (program: Program) : unit =
    // ...
```

**입력:**
- `builder`: OpBuilder
- `program`: Program (함수 정의 리스트 + main 표현식)

**출력:**
- `unit` (모듈에 함수들과 main을 추가)

### 단계별 구현

**Step 1: 모든 함수 정의 컴파일**

```fsharp
// 함수 정의들을 모듈에 추가
program.functions
|> List.iter (compileFuncDef builder)
```

각 FunDef를 `compileFuncDef`로 컴파일하여 모듈에 추가한다.

**Step 2: Main 함수 생성**

Main 표현식을 `@funlang_main` 함수로 컴파일한다. 이 함수가 프로그램의 진입점이 된다.

```fsharp
// Main 함수 생성
let i32Type = builder.I32Type()
let mainFuncOp = builder.CreateFuncOp("funlang_main", [||], i32Type)
let mainBlock = builder.GetFunctionEntryBlock(mainFuncOp)
builder.SetInsertionPointToEnd(mainBlock)

// Main 표현식 컴파일 (빈 환경)
let resultValue = compileExpr builder Map.empty program.main

// Main 반환
builder.CreateFuncReturn(resultValue)
builder.AddOperationToModule(mainFuncOp)
```

**Step 3 (선택적): C main 함수 생성**

실행 가능한 바이너리를 만들려면 C의 `main` 함수가 필요하다. `runtime.c`에서 제공한다 (Chapter 09 참조).

```c
// runtime.c
int funlang_main();

int main() {
    return funlang_main();
}
```

### 완전한 compileProgram 구현

```fsharp
let compileProgram (builder: OpBuilder) (program: Program) : unit =
    // 1. 모든 함수 정의 컴파일
    program.functions
    |> List.iter (compileFuncDef builder)

    // 2. Main 함수 생성 (프로그램 진입점)
    let i32Type = builder.I32Type()
    let mainFuncOp = builder.CreateFuncOp("funlang_main", [||], i32Type)
    let mainBlock = builder.GetFunctionEntryBlock(mainFuncOp)
    builder.SetInsertionPointToEnd(mainBlock)

    // 3. Main 표현식 컴파일 (빈 환경 - 함수 파라미터 없음)
    let resultValue = compileExpr builder Map.empty program.main

    // 4. Main 반환
    builder.CreateFuncReturn(resultValue)
    builder.AddOperationToModule(mainFuncOp)
```

### 함수 정의 순서와 심볼 테이블

**중요한 특성:** 함수 정의 순서는 중요하지 않다!

MLIR 모듈의 심볼 테이블은 flat namespace다. 모든 `func.func` 연산이 모듈 로드 시 등록되므로, 정의 순서와 무관하게 호출할 수 있다.

**예시:**

```fsharp
// 함수 정의 순서
let program = {
    functions = [
        { name = "bar"; parameters = []; body = App("foo", []) };  // foo를 호출
        { name = "foo"; parameters = []; body = Int 42 }           // foo 정의
    ]
    main = App("bar", [])
}
```

`bar`가 `foo`를 호출하지만, `foo`는 나중에 정의된다. MLIR에서는 문제없다:

```mlir
func.func @bar() -> i32 {
  %0 = func.call @foo() : () -> i32  // 전방 참조
  func.return %0 : i32
}

func.func @foo() -> i32 {
  %c42 = arith.constant 42 : i32
  func.return %c42 : i32
}
```

### 모든 함수가 모든 함수를 볼 수 있다

Flat namespace 덕분에 상호 재귀도 가능하다 (Chapter 11에서 자세히 다룸).

```mlir
func.func @is_even(%n: i32) -> i1 {
  // ... calls @is_odd ...
}

func.func @is_odd(%n: i32) -> i1 {
  // ... calls @is_even ...
}
```

정의 순서와 무관하게 모든 함수가 서로를 참조할 수 있다.

## 완전한 예시: 여러 함수와 Main

이제 완전한 프로그램 예시를 보자.

### FunLang 소스

```fsharp
let square x = x * x
let sumSquares a b = square a + square b
sumSquares 3 4
```

**의미:**
- `square 3` → 9
- `square 4` → 16
- `9 + 16` → 25

### AST 표현

```fsharp
let program = {
    functions = [
        { name = "square"
          parameters = ["x"]
          body = BinOp(Var "x", Mul, Var "x") };

        { name = "sumSquares"
          parameters = ["a"; "b"]
          body = BinOp(
              App("square", [Var "a"]),
              Add,
              App("square", [Var "b"])
          ) }
    ]
    main = App("sumSquares", [Int 3; Int 4])
}
```

### 컴파일 과정

**1. square 함수 컴파일**

```fsharp
compileFuncDef builder { name = "square"; parameters = ["x"]; body = ... }
```

생성된 MLIR IR:

```mlir
func.func @square(%arg0: i32) -> i32 {
  %0 = arith.muli %arg0, %arg0 : i32
  func.return %0 : i32
}
```

**2. sumSquares 함수 컴파일**

```fsharp
compileFuncDef builder { name = "sumSquares"; parameters = ["a"; "b"]; body = ... }
```

본체 컴파일:
- `App("square", [Var "a"])` → `%0 = func.call @square(%arg0)`
- `App("square", [Var "b"])` → `%1 = func.call @square(%arg1)`
- `BinOp(..., Add, ...)` → `%2 = arith.addi %0, %1`

생성된 MLIR IR:

```mlir
func.func @sumSquares(%arg0: i32, %arg1: i32) -> i32 {
  %0 = func.call @square(%arg0) : (i32) -> i32
  %1 = func.call @square(%arg1) : (i32) -> i32
  %2 = arith.addi %0, %1 : i32
  func.return %2 : i32
}
```

**3. Main 함수 컴파일**

```fsharp
// main = App("sumSquares", [Int 3; Int 4])
```

생성된 MLIR IR:

```mlir
func.func @funlang_main() -> i32 {
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %0 = func.call @sumSquares(%c3, %c4) : (i32, i32) -> i32
  func.return %0 : i32
}
```

### 완전한 MLIR 모듈

```mlir
module {
  func.func @square(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    func.return %0 : i32
  }

  func.func @sumSquares(%arg0: i32, %arg1: i32) -> i32 {
    %0 = func.call @square(%arg0) : (i32) -> i32
    %1 = func.call @square(%arg1) : (i32) -> i32
    %2 = arith.addi %0, %1 : i32
    func.return %2 : i32
  }

  func.func @funlang_main() -> i32 {
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %0 = func.call @sumSquares(%c3, %c4) : (i32, i32) -> i32
    func.return %0 : i32
  }
}
```

### Lowering to LLVM Dialect

MLIR의 `--convert-func-to-llvm` 패스를 적용하면 LLVM 다이얼렉트로 변환된다:

```bash
mlir-opt --convert-func-to-llvm \
         --convert-arith-to-llvm \
         --convert-scf-to-cf \
         --convert-cf-to-llvm \
         input.mlir -o lowered.mlir
```

**Lowered MLIR (LLVM dialect):**

```mlir
module {
  llvm.func @square(%arg0: i32) -> i32 {
    %0 = llvm.mul %arg0, %arg0 : i32
    llvm.return %0 : i32
  }

  llvm.func @sumSquares(%arg0: i32, %arg1: i32) -> i32 {
    %0 = llvm.call @square(%arg0) : (i32) -> i32
    %1 = llvm.call @square(%arg1) : (i32) -> i32
    %2 = llvm.add %0, %1 : i32
    llvm.return %2 : i32
  }

  llvm.func @funlang_main() -> i32 {
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %0 = llvm.call @sumSquares(%c3, %c4) : (i32, i32) -> i32
    llvm.return %0 : i32
  }
}
```

`func.*` 연산이 `llvm.*` 연산으로 변환되었다!

### LLVM IR 변환

```bash
mlir-translate --mlir-to-llvmir lowered.mlir -o output.ll
```

**LLVM IR:**

```llvm
define i32 @square(i32 %0) {
  %2 = mul i32 %0, %0
  ret i32 %2
}

define i32 @sumSquares(i32 %0, i32 %1) {
  %3 = call i32 @square(i32 %0)
  %4 = call i32 @square(i32 %1)
  %5 = add i32 %3, %4
  ret i32 %5
}

define i32 @funlang_main() {
  %1 = call i32 @sumSquares(i32 3, i32 4)
  ret i32 %1
}
```

### 컴파일과 실행

```bash
# LLVM IR을 object file로 컴파일
llc -filetype=obj output.ll -o funlang.o

# runtime.c와 링크
cc runtime.c funlang.o -o program

# 실행
./program
echo $?  # 25 (3*3 + 4*4)
```

프로그램이 실행되어 `25`를 반환한다!

## 호출 규약 (Calling Convention)

함수 호출이 실제로 어떻게 동작하는지 이해하려면 **호출 규약(calling convention)**을 알아야 한다.

### 호출 규약이란?

**호출 규약**은 함수 호출 시 인자, 반환 값, 레지스터, 스택이 어떻게 관리되는지 정의하는 규칙이다.

**규약에 포함되는 내용:**

1. **인자 전달 방법**: 레지스터? 스택? 어떤 순서?
2. **반환 값 위치**: 어느 레지스터에 반환 값을 넣는가?
3. **레지스터 보존**: 어떤 레지스터는 호출 전후에 보존되어야 하는가?
4. **스택 프레임**: 스택을 어떻게 정리하는가?

### C 호출 규약 (x86-64 System V ABI)

MLIR/LLVM은 기본적으로 **C 호출 규약**을 사용한다. x86-64 Linux에서는 **System V ABI**다.

**인자 전달 (x86-64 System V ABI):**

| 인자 순서 | 정수/포인터 | 부동소수점 |
|---------|------------|-----------|
| 1번째   | RDI        | XMM0      |
| 2번째   | RSI        | XMM1      |
| 3번째   | RDX        | XMM2      |
| 4번째   | RCX        | XMM3      |
| 5번째   | R8         | XMM4      |
| 6번째   | R9         | XMM5      |
| 7번째 이상 | 스택      | 스택      |

**예시: `add(10, 20)` 호출**

```asm
mov edi, 10      ; 첫 번째 인자 (RDI의 하위 32비트)
mov esi, 20      ; 두 번째 인자 (RSI의 하위 32비트)
call add         ; 함수 호출
; 반환 값은 EAX (RAX의 하위 32비트)에 저장됨
```

**반환 값:**
- 정수/포인터: RAX (32비트 정수는 EAX)
- 부동소수점: XMM0

**예시: `add` 함수 반환**

```asm
add:
    mov eax, edi
    add eax, esi   ; eax = edi + esi
    ret            ; eax에 반환 값
```

### LLVM이 호출 규약을 처리한다

**핵심 통찰력:** 우리는 호출 규약을 직접 구현하지 않는다. LLVM이 자동으로 처리한다!

**MLIR func 다이얼렉트 코드:**

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
```

**LLVM이 생성하는 네이티브 코드 (x86-64):**

```asm
add:
    ; 프롤로그 (스택 프레임 설정) - 단순 함수는 생략 가능
    lea eax, [rdi + rsi]  ; eax = edi + esi (최적화됨)
    ret
```

LLVM이 자동으로:
1. 파라미터를 적절한 레지스터에 배치 (EDI, ESI)
2. 반환 값을 EAX에 배치
3. 최적화 적용 (lea 사용)
4. 에필로그 생략 (단순 함수)

### 플랫폼별 차이

호출 규약은 플랫폼마다 다르다:

| 플랫폼 | 호출 규약 | 인자 전달 |
|--------|----------|----------|
| Linux x86-64 | System V ABI | RDI, RSI, RDX, RCX, R8, R9, 스택 |
| Windows x86-64 | Microsoft x64 | RCX, RDX, R8, R9, 스택 |
| ARM64 | AAPCS64 | X0-X7, 스택 |
| x86-32 | cdecl | 스택 (오른쪽부터) |

**LLVM의 역할:** 동일한 LLVM IR을 각 플랫폼에 맞게 변환한다. 우리는 신경 쓸 필요 없다!

### 왜 C 호출 규약을 사용하는가?

**장점:**

1. **C 라이브러리와 상호 운용**: printf, malloc 같은 C 함수를 호출할 수 있다.
2. **시스템 콜 호환성**: OS 시스템 콜이 C 규약을 따른다.
3. **디버거 지원**: GDB 같은 디버거가 C 호출 규약을 이해한다.
4. **ABI 안정성**: 표준 ABI로 다른 언어와 링크 가능.

**단점 (Phase 3에서는 해당 없음):**

- Tail call optimization이 보장되지 않음 (Chapter 11에서 다룸)
- 클로저 전달이 비효율적일 수 있음 (Phase 4에서 다룸)

Phase 3에서는 C 호출 규약이 완벽하게 작동한다. 단순한 값 전달과 반환만 있기 때문이다.

### 스택 프레임 관리

함수 호출 시 **스택 프레임(stack frame)**이 생성된다.

**스택 프레임 구조 (x86-64):**

```
High address
┌─────────────────┐
│ 인자 7, 8, ...  │  (레지스터에 들어가지 않는 인자들)
├─────────────────┤
│ 반환 주소       │  (call 명령어가 push)
├─────────────────┤
│ 이전 RBP        │  (함수 프롤로그가 push)
├─────────────────┤  ← RBP (base pointer)
│ 지역 변수       │
├─────────────────┤
│ 임시 값         │
└─────────────────┘  ← RSP (stack pointer)
Low address
```

**함수 프롤로그 (진입 시):**

```asm
push rbp           ; 이전 프레임 포인터 저장
mov rbp, rsp       ; 새 프레임 포인터 설정
sub rsp, 32        ; 지역 변수를 위한 공간 할당
```

**함수 에필로그 (종료 시):**

```asm
mov rsp, rbp       ; 스택 포인터 복원
pop rbp            ; 이전 프레임 포인터 복원
ret                ; 반환
```

**LLVM의 역할:** 이 모든 것을 자동으로 생성한다. 우리는 `func.func`와 `func.return`만 작성하면 된다!

### Tail Call Optimization (미리보기)

**Tail call**은 함수의 마지막 연산이 다른 함수 호출인 경우다:

```fsharp
let factorial_tail n acc =
    if n <= 1 then acc
    else factorial_tail (n - 1) (n * acc)  // Tail call!
```

일반 호출과 tail call의 차이:

**일반 호출 (스택 프레임 누적):**

```
factorial_tail(5, 1)
  → factorial_tail(4, 5)
    → factorial_tail(3, 20)
      → factorial_tail(2, 60)
        → factorial_tail(1, 120)
          → 120
```

5개의 스택 프레임이 누적된다.

**Tail call optimization (스택 프레임 재사용):**

```
factorial_tail(5, 1)
→ factorial_tail(4, 5)  (같은 프레임 재사용)
→ factorial_tail(3, 20)
→ factorial_tail(2, 60)
→ factorial_tail(1, 120)
→ 120
```

1개의 스택 프레임만 사용한다!

**Chapter 11에서 자세히 다룬다.** Tail call optimization은 재귀 함수를 효율적으로 만드는 핵심 기술이다.

## 일반적인 오류

함수를 처음 구현할 때 흔히 겪는 오류들을 살펴본다.

### 오류 1: 함수를 찾을 수 없음

**증상:**

```
error: 'func.call' op symbol reference '@foo' not found in symbol table
```

**원인:**
- 함수 이름 오타
- 함수가 정의되지 않음
- 함수를 모듈에 추가하지 않음

**예시 (잘못된 코드):**

```fsharp
// 'add' 함수를 정의했지만 'addd'로 호출
let program = {
    functions = [ { name = "add"; parameters = ["x"; "y"]; body = ... } ]
    main = App("addd", [Int 10; Int 20])  // 오타!
}
```

**해결 방법:**

1. **함수 이름 확인**: 정의와 호출 시 이름이 일치하는가?
2. **함수 정의 확인**: `compileFuncDef`가 호출되었는가?
3. **모듈 추가 확인**: `AddOperationToModule`이 호출되었는가?

```fsharp
// 올바른 코드
let program = {
    functions = [ { name = "add"; parameters = ["x"; "y"]; body = ... } ]
    main = App("add", [Int 10; Int 20])  // 일치!
}
```

### 오류 2: 인자 개수 불일치

**증상:**

```
error: 'func.call' op incorrect number of operands: expected 2 but got 1
```

**원인:**
- 함수 호출 시 인자 개수가 정의와 다름

**예시 (잘못된 코드):**

```fsharp
// add는 2개의 파라미터를 받는다
let addDef = { name = "add"; parameters = ["x"; "y"]; body = ... }

// 하지만 1개만 전달
let call = App("add", [Int 10])  // 오류!
```

**해결 방법:**

함수 정의의 파라미터 개수와 호출 시 인자 개수를 일치시킨다.

```fsharp
// 올바른 코드
let call = App("add", [Int 10; Int 20])  // 2개 인자
```

**디버깅 팁:**

함수 시그니처를 확인하는 유틸리티를 추가한다:

```fsharp
let checkFunctionArity (funcDef: FunDef) (argCount: int) =
    if argCount <> funcDef.parameters.Length then
        failwithf "Function %s expects %d arguments but got %d"
            funcDef.name
            funcDef.parameters.Length
            argCount
```

### 오류 3: 타입 불일치

**증상:**

```
error: 'func.call' op operand type mismatch: expected 'i32' but got 'i1'
```

**원인:**
- 함수 파라미터 타입과 인자 타입이 다름
- Phase 3에서는 모든 값이 i32이므로 비교 결과(i1)를 함수에 전달할 때 발생

**예시 (잘못된 코드):**

```fsharp
// compute는 i32를 받는다
let computeDef = { name = "compute"; parameters = ["x"]; body = ... }

// 하지만 i1 (비교 결과)를 전달
let cond = Compare(Int 10, Gt, Int 5)  // i1 타입
let call = App("compute", [cond])      // 타입 불일치!
```

**해결 방법:**

Phase 3에서는 모든 함수 파라미터가 i32다. 비교 결과를 전달하려면 i1을 i32로 확장한다:

```fsharp
// 비교 결과를 i32로 확장
let cond = Compare(Int 10, Gt, Int 5)  // i1
let condExtended = If(cond, Int 1, Int 0)  // i32
let call = App("compute", [condExtended])
```

또는 컴파일러가 자동으로 확장하도록 구현:

```fsharp
let rec compileExpr builder env expr =
    match expr with
    | App(name, args) ->
        let argValues =
            args
            |> List.map (fun argExpr ->
                let value = compileExpr builder env argExpr
                // i1 타입이면 i32로 확장
                if mlirTypeEqual (mlirValueGetType value) (builder.I1Type()) then
                    builder.CreateArithExtension(value, builder.I32Type())
                else
                    value
            )
            |> List.toArray
        builder.CreateFuncCall(name, argValues, builder.I32Type())
```

### 오류 4: func.return 누락

**증상:**

```
error: 'func.func' op block must be terminated with a func.return operation
```

**원인:**
- 함수 본체가 종결자(terminator) 없이 끝남
- `func.return`을 추가하지 않음

**예시 (잘못된 코드):**

```fsharp
let compileFuncDef builder funcDef =
    let funcOp = builder.CreateFuncOp(...)
    let entryBlock = builder.GetFunctionEntryBlock(funcOp)
    builder.SetInsertionPointToEnd(entryBlock)

    // 본체 컴파일
    let bodyValue = compileExpr builder env funcDef.body

    // func.return 누락!
    builder.AddOperationToModule(funcOp)
```

**해결 방법:**

항상 `func.return`을 추가한다:

```fsharp
let compileFuncDef builder funcDef =
    // ...
    let bodyValue = compileExpr builder env funcDef.body
    builder.CreateFuncReturn(bodyValue)  // 추가!
    builder.AddOperationToModule(funcOp)
```

### 오류 5: 파라미터와 let 바인딩 혼동

**증상:**

```
error: use of value '%arg0' requires an operation that dominates it
```

**원인:**
- 파라미터를 일반 변수처럼 처리함
- 환경에 파라미터를 추가하지 않음

**예시 (잘못된 코드):**

```fsharp
let compileFuncDef builder funcDef =
    // ...
    let entryBlock = builder.GetFunctionEntryBlock(funcOp)
    builder.SetInsertionPointToEnd(entryBlock)

    // 파라미터를 환경에 추가하지 않음!
    let env = Map.empty
    let bodyValue = compileExpr builder env funcDef.body  // Var "x"를 찾지 못함
```

**해결 방법:**

파라미터를 환경에 추가한다:

```fsharp
let compileFuncDef builder funcDef =
    // ...
    let entryBlock = builder.GetFunctionEntryBlock(funcOp)

    // 파라미터를 환경에 추가
    let initialEnv =
        funcDef.parameters
        |> List.mapi (fun i name ->
            let arg = builder.GetFunctionBlockArg(entryBlock, i)
            (name, arg)
        )
        |> Map.ofList

    builder.SetInsertionPointToEnd(entryBlock)
    let bodyValue = compileExpr builder initialEnv funcDef.body
    builder.CreateFuncReturn(bodyValue)
```

**핵심 원칙:** 파라미터는 block arguments다. 환경에 추가하여 이름으로 참조할 수 있게 한다.

## 장 요약

이 장에서 FunLang에 **함수**를 추가했다.

**배운 내용:**

1. **MLIR func 다이얼렉트**
   - `func.func`: 함수 정의
   - `func.call`: 함수 호출
   - `func.return`: 함수 반환
   - 모듈 레벨 심볼 테이블

2. **AST 확장**
   - `FunDef`: 함수 정의 (이름, 파라미터, 본체)
   - `App`: 함수 호출 (함수 이름, 인자 리스트)
   - `Program`: 함수 정의 리스트 + main 표현식

3. **P/Invoke 바인딩**
   - Function type API (`mlirFunctionTypeGet`)
   - Symbol reference (`mlirFlatSymbolRefAttrGet`)
   - Block arguments (`mlirBlockGetArgument`)

4. **OpBuilder 확장**
   - `CreateFuncOp`: 함수 생성
   - `GetFunctionEntryBlock`: entry block 가져오기
   - `GetFunctionBlockArg`: 파라미터 가져오기
   - `CreateFuncCall`: 함수 호출
   - `CreateFuncReturn`: 함수 반환

5. **함수 파라미터와 Block Arguments**
   - 파라미터는 block arguments로 표현
   - Entry block의 arguments로 자동 생성
   - 환경에 추가하여 이름으로 참조

6. **코드 생성**
   - `compileFuncDef`: 함수 정의 컴파일
   - `compileExpr`의 `App` case: 함수 호출 컴파일
   - `compileProgram`: 전체 프로그램 컴파일

7. **호출 규약 (Calling Convention)**
   - C 호출 규약 (System V ABI)
   - 인자 전달: 레지스터 → 스택
   - 반환 값: RAX 레지스터
   - LLVM이 자동 처리

**독자가 할 수 있는 것:**

- 다중 함수 정의를 포함한 FunLang 프로그램 작성
- 함수 호출과 중첩 호출 컴파일
- 생성된 MLIR IR 확인
- 네이티브 바이너리로 컴파일 및 실행

**다음 단계 (Chapter 11):**

- **재귀(Recursion)**: 함수가 자기 자신을 호출
- **상호 재귀(Mutual Recursion)**: 두 함수가 서로를 호출
- **Tail Call Optimization**: 재귀를 효율적으로 만들기

함수는 코드 재사용과 모듈화의 핵심이다. Phase 3은 함수의 기초를 확립했다. 다음 장에서는 재귀로 함수의 표현력을 극대화한다!

