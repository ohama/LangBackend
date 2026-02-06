# Chapter 07: Let 바인딩과 SSA 형태

## 소개

프로그래밍에서 변수는 필수적이다. 값을 이름에 바인딩하고, 나중에 그 이름을 참조하여 값을 재사용한다. Chapter 06까지는 표현식을 직접 계산했지만, 실제 프로그램을 작성하려면 중간 결과를 저장하고 참조할 수 있어야 한다.

이 장에서는 **let 바인딩**을 추가한다:

```fsharp
let x = 10 in
let y = 20 in
x + y
```

함수형 언어의 let 바인딩은 명령형 언어의 변수 할당과 다르다:

- **명령형**: `x = 5; x = 10;` (뮤테이션 - 값이 변경됨)
- **함수형**: `let x = 5 in let x = 10 in x` (섀도잉 - 새로운 바인딩 생성, 뮤테이션 아님)

**핵심 통찰력:** let 바인딩은 **불변(immutable)**이다. 이것이 MLIR의 SSA (Static Single Assignment) 형태와 완벽하게 일치한다. 함수형 프로그래밍은 SSA를 자연스럽게 표현한다!

이 장을 마치면:
- let 바인딩을 컴파일하여 네이티브 바이너리로 만들 수 있다
- 중첩된 바인딩과 스코프를 이해한다
- SSA 형태가 무엇이고 왜 중요한지 안다
- 환경 전달(environment passing)로 변수를 관리하는 방법을 안다

> **중요:** 이 장은 SSA 개념을 소개한다. SSA는 현대 컴파일러의 핵심 기술이며, MLIR은 SSA를 기본으로 한다.

## SSA 형태 설명

### SSA란 무엇인가?

**SSA (Static Single Assignment)**는 중간 표현(IR)의 속성이다:

> **정의:** 각 변수는 프로그램에서 정확히 한 번만 할당된다.

예시:

```fsharp
// SSA가 아님 (명령형):
x = 5
x = 10  // x가 두 번 할당됨!

// SSA (함수형):
let x1 = 5 in    // x1은 한 번만 할당
let x2 = 10 in   // x2는 한 번만 할당
x2
```

MLIR IR에서 SSA value는 `%` 기호로 표시된다:

```mlir
%x = arith.constant 5 : i32      // %x 정의 (한 번만)
%y = arith.addi %x, %x : i32     // %x 사용 (여러 번 가능)
%z = arith.muli %y, %x : i32     // %x, %y 사용
```

각 SSA value (`%x`, `%y`, `%z`)는 정확히 한 번만 정의된다. 사용은 여러 번 가능하다.

### 왜 SSA가 중요한가?

SSA 형태는 컴파일러 최적화를 극적으로 단순화한다:

#### 1. 상수 전파 (Constant Propagation)

```mlir
// SSA 형태
%c5 = arith.constant 5 : i32
%result = arith.addi %c5, %c5 : i32

// 최적화: %c5가 상수임을 알고 있으므로
%c10 = arith.constant 10 : i32  // 컴파일 타임에 계산
```

SSA value는 한 번만 정의되므로, 정의를 추적하여 상수 값을 전파할 수 있다.

#### 2. 죽은 코드 제거 (Dead Code Elimination)

```mlir
// SSA 형태
%unused = arith.constant 42 : i32  // 정의되지만 사용되지 않음
%result = arith.constant 10 : i32
func.return %result : i32

// 최적화: %unused는 사용되지 않으므로 제거 가능
```

SSA value가 사용되지 않으면 정의도 불필요하다. 쉽게 감지하고 제거할 수 있다.

#### 3. 레지스터 할당 (Register Allocation)

```mlir
%x = arith.constant 5 : i32       // %x의 수명 시작
%y = arith.constant 10 : i32      // %y의 수명 시작
%z = arith.addi %x, %y : i32      // %x, %y 사용 (%x, %y 수명 끝)
func.return %z : i32               // %z 사용 (%z 수명 끝)
```

SSA value의 수명(lifetime)이 명확하다:
- 정의 지점에서 시작
- 마지막 사용 지점에서 끝

레지스터 할당기가 수명 분석을 쉽게 수행하여 레지스터를 효율적으로 재사용할 수 있다.

### Let 바인딩은 자연스럽게 SSA다

함수형 언어의 let 바인딩은 불변이므로, 변환 없이 SSA로 직접 매핑된다:

```fsharp
// FunLang 소스
let x = 5 in
x + x
```

**MLIR IR로 변환:**

```mlir
func.func @main() -> i32 {
  %x = arith.constant 5 : i32      // let x = 5
  %result = arith.addi %x, %x : i32  // x + x
  func.return %result : i32
}
```

`let x = 5`가 SSA value `%x`의 **단일 정의**가 된다. 추가 작업이 필요 없다!

### 명령형 언어와의 대비

명령형 언어는 변수 뮤테이션을 허용하므로 SSA 변환이 필요하다:

```c
// C 코드 (SSA 아님)
int x = 5;
int y = x + x;
x = 10;       // 뮤테이션!
int z = x + y;
```

**SSA로 변환 (컴파일러가 수행):**

```mlir
%x_0 = arith.constant 5 : i32       // x = 5
%y = arith.addi %x_0, %x_0 : i32    // y = x + x
%x_1 = arith.constant 10 : i32      // x = 10 (새로운 SSA value)
%z = arith.addi %x_1, %y : i32      // z = x + y
```

각 "할당"이 새로운 SSA value를 생성한다 (`%x_0`, `%x_1`). 이것이 SSA 변환(SSA conversion)이다.

**함수형 언어의 이점:** 뮤테이션이 없으므로 SSA 변환이 불필요하다. let 바인딩이 이미 SSA다!

### 섀도잉: 새로운 값, 뮤테이션 아님

함수형 언어에서 같은 이름을 다시 바인딩하면 어떻게 될까?

```fsharp
let x = 5 in
let x = 10 in
x
```

이것은 **섀도잉(shadowing)**이다:

```mlir
func.func @main() -> i32 {
  %x = arith.constant 5 : i32      // 첫 번째 x 바인딩
  %x_0 = arith.constant 10 : i32   // 두 번째 x 바인딩 (새로운 값)
  func.return %x_0 : i32            // 내부 x 사용
}
```

**핵심:** MLIR은 자동으로 고유한 이름을 생성한다 (`%x`, `%x_0`, `%x_1`, ...). 섀도잉은 새로운 SSA value를 만들 뿐, 기존 값을 변경하지 않는다.

외부 `%x`는 여전히 존재하지만 내부 스코프에서는 가려진다 (shadowed). 스코프가 끝나면 외부 `%x`가 다시 보인다.

### SSA의 제약

SSA에서 **제어 흐름(control flow)** 합류 지점에서는 어떻게 될까?

```fsharp
let x = if condition then 10 else 20 in
x + x
```

if 표현식이 두 가지 다른 값 (10 또는 20)을 생성할 수 있다. 어떤 SSA value를 `x`에 바인딩해야 할까?

**해답:** MLIR은 **block arguments**를 사용한다. Chapter 08 (제어 흐름)에서 자세히 다룰 것이다. 지금은 let 바인딩이 단순한 값 바인딩이며 조건부 바인딩이 없다는 점만 기억하자.

### SSA 요약

**SSA 형태:**
- 각 value는 정확히 한 번만 정의된다
- 사용은 여러 번 가능하다
- 컴파일러 최적화를 단순화한다
- MLIR은 SSA를 기본으로 한다

**Let 바인딩과 SSA:**
- 함수형 언어의 let 바인딩은 불변이다
- 불변 = 자연스러운 SSA 형태
- 섀도잉은 새로운 SSA value를 생성한다
- 뮤테이션이 없으므로 SSA 변환이 불필요하다

> **명심:** SSA는 이론이 아니라 실용이다. 모든 현대 컴파일러 (LLVM, GCC, MLIR)는 SSA를 사용한다. 함수형 언어는 SSA를 "무료로" 얻는다!

## 확장된 AST: Let과 Var

이제 AST에 let 바인딩과 변수 참조를 추가한다.

**Ast.fs** 수정:

```fsharp
namespace FunLangCompiler

/// 이진 연산자 (Chapter 06)
type Operator =
    | Add
    | Subtract
    | Multiply
    | Divide

/// 비교 연산자 (Chapter 06)
type CompareOp =
    | LessThan
    | GreaterThan
    | LessEqual
    | GreaterEqual
    | Equal
    | NotEqual

/// 단항 연산자 (Chapter 06)
type UnaryOp =
    | Negate

/// FunLang 표현식 AST
type Expr =
    | IntLiteral of int
    | BinaryOp of Operator * Expr * Expr
    | UnaryOp of UnaryOp * Expr
    | Comparison of CompareOp * Expr * Expr
    // NEW: let 바인딩과 변수 참조
    | Let of name: string * binding: Expr * body: Expr
    | Var of name: string

/// 최상위 프로그램
type Program =
    { expr: Expr }
```

**새로운 케이스 설명:**

### Let of name * binding * body

```fsharp
| Let of name: string * binding: Expr * body: Expr
```

**의미:** `let {name} = {binding} in {body}`

**필드:**
- `name`: 바인딩할 변수 이름 (예: "x")
- `binding`: 변수에 바인딩할 표현식 (예: `IntLiteral 10`)
- `body`: 바인딩이 유효한 스코프 (예: `BinaryOp(Add, Var "x", Var "x")`)

**예시:**

```fsharp
// FunLang: let x = 10 in x + x
Let("x",
  IntLiteral 10,
  BinaryOp(Add, Var "x", Var "x"))
```

**스코프:** `body` 표현식 내에서만 `name`이 유효하다. 스코프 밖에서는 변수가 존재하지 않는다.

### Var of name

```fsharp
| Var of name: string
```

**의미:** 변수 참조 - 이전에 바인딩된 변수의 값을 사용한다.

**필드:**
- `name`: 참조할 변수 이름 (예: "x")

**예시:**

```fsharp
// FunLang: x
Var "x"
```

**바인딩 필요:** `Var "x"`를 사용하려면 스코프에서 `x`가 바인딩되어 있어야 한다. 바인딩되지 않은 변수를 참조하면 컴파일 에러다.

### 중첩된 Let 바인딩

```fsharp
// FunLang:
// let x = 10 in
// let y = 20 in
// x + y

Let("x",
  IntLiteral 10,
  Let("y",
    IntLiteral 20,
    BinaryOp(Add, Var "x", Var "y")))
```

**스코프 중첩:**
- 외부 let (`x`)의 body는 내부 let (`y`)이다
- 내부 let의 body에서 `x`와 `y` 모두 보인다
- 내부 스코프는 외부 스코프를 "확장"한다

### 섀도잉 예시

```fsharp
// FunLang:
// let x = 5 in
// let x = x + 1 in
// x

Let("x",
  IntLiteral 5,
  Let("x",
    BinaryOp(Add, Var "x", IntLiteral 1),  // 외부 x 사용
    Var "x"))  // 내부 x 반환
```

**섀도잉 동작:**
- 두 번째 `Let("x", ...)`: 새로운 `x` 바인딩
- `BinaryOp(Add, Var "x", ...)`: 여기서 `Var "x"`는 **외부 x** (값 5)를 참조한다
- `body`의 `Var "x"`: 여기서 `Var "x"`는 **내부 x** (값 6)를 참조한다

**결과:** 6을 반환한다.

### AST 완전한 예시

```fsharp
// FunLang:
// let x = 10 in
// let y = 20 in
// let z = x + y in
// z * 2

Let("x",
  IntLiteral 10,
  Let("y",
    IntLiteral 20,
    Let("z",
      BinaryOp(Add, Var "x", Var "y"),
      BinaryOp(Multiply, Var "z", IntLiteral 2))))
```

**예상 결과:**
- x = 10
- y = 20
- z = x + y = 30
- z * 2 = 60

이 AST를 컴파일하면 60을 반환하는 네이티브 바이너리가 생성된다.

## 환경 개념 (Environment)

변수를 컴파일하려면 **환경(environment)**이 필요하다.

### 환경이란?

**정의:** 환경은 변수 이름을 SSA value에 매핑하는 자료구조다.

**타입 정의:**

```fsharp
/// 변수 이름 -> MLIR SSA value 매핑
type Env = Map<string, MlirValue>
```

F#의 `Map` 타입은 불변 딕셔너리다. 키-값 쌍을 저장하며, 함수형 방식으로 확장할 수 있다.

### 환경 연산

#### 1. 빈 환경 생성

```fsharp
let emptyEnv : Env = Map.empty
```

프로그램 시작 시 환경은 비어 있다. 아직 변수가 바인딩되지 않았다.

#### 2. 환경 확장 (바인딩 추가)

```fsharp
// x를 %c5 SSA value에 바인딩
let env = Map.empty
let env' = env.Add("x", someValue)
```

`env.Add(name, value)`는 새로운 환경을 반환한다. 기존 환경 `env`는 변경되지 않는다 (불변성).

#### 3. 변수 조회

```fsharp
// x의 SSA value 찾기
match env.TryFind("x") with
| Some(value) -> value  // x가 바인딩되어 있음
| None -> failwith "Unbound variable: x"  // x가 바인딩되지 않음
```

`TryFind`는 `Option` 타입을 반환한다:
- `Some(value)`: 변수가 환경에 존재
- `None`: 변수가 존재하지 않음 (컴파일 에러)

### 환경과 스코프

스코프는 환경을 통해 구현된다:

```fsharp
// let x = 10 in let y = 20 in x + y
// 각 let이 환경을 확장한다

let env0 = Map.empty             // 초기 환경 (비어 있음)

// let x = 10
let env1 = env0.Add("x", %c10)   // env1 = { x -> %c10 }

// let y = 20
let env2 = env1.Add("y", %c20)   // env2 = { x -> %c10, y -> %c20 }

// x + y (env2에서 x와 y 조회)
// x = %c10, y = %c20
```

**환경 스택 다이어그램:**

```
let x = 5 in       env = { x -> %c5 }
  let y = 10 in    env = { x -> %c5, y -> %c10 }
    x + y          lookup x, lookup y -> arith.addi %c5, %c10
```

각 let 바인딩이 환경에 새로운 항목을 추가한다. 내부 스코프의 환경은 외부 스코프의 모든 바인딩을 포함한다.

### 섀도잉과 환경

같은 이름을 다시 바인딩하면?

```fsharp
// let x = 5 in let x = 10 in x
let env0 = Map.empty
let env1 = env0.Add("x", %c5)   // env1 = { x -> %c5 }
let env2 = env1.Add("x", %c10)  // env2 = { x -> %c10 }

// env2에서 x 조회 -> %c10 (새로운 바인딩)
```

`Map.Add`는 기존 키가 있으면 값을 덮어쓴다. 하지만 `env1`은 변경되지 않는다 (불변):

```fsharp
// env1에서 x 조회 -> 여전히 %c5
// env2에서 x 조회 -> %c10
```

이것이 **스코프 기반 섀도잉**이다. 내부 스코프가 끝나면 외부 바인딩이 다시 보인다:

```fsharp
// let x = 5 in (let x = 10 in x) + x
//               ^^^^^^^^^^^^^   ^^^
//               내부 x = 10     외부 x = 5

let env0 = Map.empty
let env1 = env0.Add("x", %c5)

// 내부 스코프
let env2 = env1.Add("x", %c10)
// 내부 body에서 x 조회 -> %c10

// 외부 스코프로 돌아옴 (env1 사용)
// 외부 body에서 x 조회 -> %c5
```

**결과:** `10 + 5 = 15`

### 환경 전달 패턴

컴파일러는 환경을 함수 인자로 전달한다:

```fsharp
let rec compileExpr (builder: OpBuilder) (expr: Expr) (env: Env) : MlirValue =
  match expr with
  | IntLiteral n -> ...  // env 사용 안 함
  | Var name ->
      // env에서 변수 조회
      match env.TryFind(name) with
      | Some(value) -> value
      | None -> failwithf "Unbound variable: %s" name
  | Let(name, binding, body) ->
      // 1. binding 표현식 컴파일 (현재 env 사용)
      let bindVal = compileExpr builder binding env
      // 2. env 확장
      let env' = env.Add(name, bindVal)
      // 3. body 표현식 컴파일 (확장된 env' 사용)
      compileExpr builder body env'
  | BinaryOp(op, lhs, rhs) ->
      // 재귀 호출에 env 전달
      let lhsVal = compileExpr builder lhs env
      let rhsVal = compileExpr builder rhs env
      ...
```

**핵심 패턴:**
- `compileExpr`이 `env` 파라미터를 받는다
- 모든 재귀 호출에서 `env`를 전달한다
- `Let` 케이스에서 `env`를 확장하고 body에 전달한다
- `Var` 케이스에서 `env`를 조회한다

이것이 **환경 전달(environment passing)**이다. 함수형 프로그래밍에서 흔한 패턴이다.

### 환경 요약

**환경:**
- 변수 이름 -> SSA value 매핑
- F# `Map<string, MlirValue>` 타입
- 불변 자료구조

**연산:**
- `Map.empty`: 빈 환경
- `env.Add(name, value)`: 바인딩 추가 (새 환경 반환)
- `env.TryFind(name)`: 변수 조회 (Option 반환)

**스코프:**
- 각 let 바인딩이 환경을 확장한다
- 내부 스코프는 외부 바인딩을 모두 포함한다
- 섀도잉은 `Map.Add`로 구현된다

**환경 전달:**
- `compileExpr`에 `env` 파라미터 추가
- 재귀 호출에서 `env` 전달
- `Let` 케이스에서 `env` 확장

> **다음 섹션:** 환경을 사용하여 let 바인딩을 MLIR IR로 컴파일하는 코드를 작성한다!

## 환경을 사용한 코드 생성

이제 Chapter 06의 `compileExpr`을 확장하여 let 바인딩과 변수를 처리한다.

### compileExpr 시그니처 변경

먼저 환경 파라미터를 추가한다:

```fsharp
// 기존 (Chapter 06):
let rec compileExpr
    (builder: OpBuilder)
    (block: MlirBlock)
    (location: Location)
    (expr: Expr)
    : MlirValue = ...

// 새로운 (Chapter 07):
let rec compileExpr
    (builder: OpBuilder)
    (block: MlirBlock)
    (location: Location)
    (expr: Expr)
    (env: Env)        // 환경 추가!
    : MlirValue = ...
```

**환경 타입 정의:**

```fsharp
/// 변수 이름 -> MLIR SSA value 매핑
type Env = Map<string, MlirValue>
```

### Let 케이스 구현

```fsharp
| Let(name, binding, body) ->
    // 1. binding 표현식 컴파일 (현재 환경 사용)
    let bindVal = compileExpr builder block location binding env

    // 2. 환경 확장: name -> bindVal 매핑 추가
    let env' = env.Add(name, bindVal)

    // 3. body 표현식 컴파일 (확장된 환경 사용)
    compileExpr builder block location body env'
```

**동작 설명:**

1. `binding` 표현식을 먼저 컴파일한다. 이것이 변수에 바인딩될 값이다.
2. 현재 환경 `env`를 확장하여 `name`을 `bindVal`에 매핑한다. 새로운 환경 `env'`가 생성된다.
3. `body` 표현식을 컴파일할 때 확장된 환경 `env'`를 사용한다. body 내에서 `name`을 참조할 수 있다.

**핵심:** let 바인딩은 MLIR IR에 새로운 연산을 생성하지 않는다. 단지 환경을 확장하고 body를 컴파일할 뿐이다. SSA value는 `binding` 표현식에서 이미 생성되었다.

### Var 케이스 구현

```fsharp
| Var(name) ->
    // 환경에서 변수 조회
    match env.TryFind(name) with
    | Some(value) -> value  // 바인딩된 SSA value 반환
    | None -> failwithf "Unbound variable: %s" name  // 컴파일 에러
```

**동작 설명:**

- `env.TryFind(name)`으로 변수를 조회한다.
- 바인딩되어 있으면 (`Some(value)`) 해당 SSA value를 반환한다.
- 바인딩되지 않았으면 (`None`) 에러를 발생시킨다.

**중요:** 변수 참조는 MLIR IR에 새로운 연산을 생성하지 않는다. 단지 기존 SSA value를 반환할 뿐이다. 이것이 SSA의 핵심이다!

### 기존 케이스 업데이트

모든 기존 케이스에서 재귀 호출에 `env`를 전달해야 한다:

```fsharp
| IntLiteral value ->
    // 환경 사용 안 함 (리터럴은 변수를 참조하지 않음)
    let i32Type = builder.I32Type()
    let attr = builder.Context.GetIntegerAttr(i32Type, int64 value)
    let constOp = builder.CreateConstant(attr, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, constOp)
    builder.GetResult(constOp, 0)

| BinaryOp(op, lhs, rhs) ->
    // 재귀 호출에 env 전달
    let lhsVal = compileExpr builder block location lhs env
    let rhsVal = compileExpr builder block location rhs env
    let binOp = builder.CreateArithBinaryOp(op, lhsVal, rhsVal, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, binOp)
    builder.GetResult(binOp, 0)

| UnaryOp(Negate, expr) ->
    // 재귀 호출에 env 전달
    let val = compileExpr builder block location expr env
    let negOp = builder.CreateArithNegate(val, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, negOp)
    builder.GetResult(negOp, 0)

| Comparison(compareOp, lhs, rhs) ->
    // 재귀 호출에 env 전달
    let lhsVal = compileExpr builder block location lhs env
    let rhsVal = compileExpr builder block location rhs env
    let cmpOp = builder.CreateArithCompare(compareOp, lhsVal, rhsVal, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, cmpOp)
    let cmpVal = builder.GetResult(cmpOp, 0)
    // i1 -> i32 확장 (Chapter 06과 동일)
    let i32Type = builder.I32Type()
    let extOp = builder.CreateArithExtUI(cmpVal, i32Type, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, extOp)
    builder.GetResult(extOp, 0)
```

**패턴:** 모든 재귀 호출에서 현재 환경 `env`를 그대로 전달한다. Let 케이스만 환경을 확장한다.

### 완전한 CodeGen.fs 리스팅

**CodeGen.fs** (환경 지원 버전):

```fsharp
namespace FunLangCompiler

open System
open MlirWrapper
open MlirBindings

module CodeGen =

    /// 변수 이름 -> MLIR SSA value 매핑
    type Env = Map<string, MlirValue>

    /// 표현식을 MLIR value로 컴파일 (재귀적, 환경 전달)
    let rec compileExpr
        (builder: OpBuilder)
        (block: MlirBlock)
        (location: Location)
        (expr: Expr)
        (env: Env)
        : MlirValue =

        match expr with
        | IntLiteral value ->
            let i32Type = builder.I32Type()
            let attr = builder.Context.GetIntegerAttr(i32Type, int64 value)
            let constOp = builder.CreateConstant(attr, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, constOp)
            builder.GetResult(constOp, 0)

        | Var(name) ->
            match env.TryFind(name) with
            | Some(value) -> value
            | None -> failwithf "Unbound variable: %s" name

        | Let(name, binding, body) ->
            let bindVal = compileExpr builder block location binding env
            let env' = env.Add(name, bindVal)
            compileExpr builder block location body env'

        | BinaryOp(op, lhs, rhs) ->
            let lhsVal = compileExpr builder block location lhs env
            let rhsVal = compileExpr builder block location rhs env
            let binOp = builder.CreateArithBinaryOp(op, lhsVal, rhsVal, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, binOp)
            builder.GetResult(binOp, 0)

        | UnaryOp(Negate, expr) ->
            let val = compileExpr builder block location expr env
            let negOp = builder.CreateArithNegate(val, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, negOp)
            builder.GetResult(negOp, 0)

        | Comparison(compareOp, lhs, rhs) ->
            let lhsVal = compileExpr builder block location lhs env
            let rhsVal = compileExpr builder block location rhs env
            let cmpOp = builder.CreateArithCompare(compareOp, lhsVal, rhsVal, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, cmpOp)
            let cmpVal = builder.GetResult(cmpOp, 0)
            let i32Type = builder.I32Type()
            let extOp = builder.CreateArithExtUI(cmpVal, i32Type, location)
            MlirNative.mlirBlockAppendOwnedOperation(block, extOp)
            builder.GetResult(extOp, 0)

    /// 프로그램을 MLIR module로 컴파일
    let translateToMlir (program: Program) : Module =
        let ctx = new Context()
        ctx.LoadDialect("arith")
        ctx.LoadDialect("func")

        let loc = Location.Unknown(ctx)
        let mlirMod = new Module(ctx, loc)

        let builder = OpBuilder(ctx)
        let i32Type = builder.I32Type()

        // main 함수 생성
        let funcType = builder.FunctionType([||], [| i32Type |])
        let funcOp = builder.CreateFunction("main", funcType, loc)

        let bodyRegion = MlirNative.mlirOperationGetRegion(funcOp, 0n)
        let entryBlock = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
        MlirNative.mlirRegionAppendOwnedBlock(bodyRegion, entryBlock)

        // 빈 환경에서 시작
        let env = Map.empty

        // 표현식 컴파일 (환경 전달)
        let resultValue = compileExpr builder entryBlock loc program.expr env

        // return operation 생성
        let returnOp = builder.CreateReturn([| resultValue |], loc)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, returnOp)

        // 함수를 module에 추가
        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, funcOp)

        mlirMod

    /// MLIR module 검증
    let verify (mlirMod: Module) =
        if not (mlirMod.Verify()) then
            eprintfn "MLIR verification failed:"
            eprintfn "%s" (mlirMod.Print())
            failwith "MLIR IR is invalid"
```

**주목할 점:**
- `compileExpr`이 `env` 파라미터를 받는다
- `translateToMlir`에서 빈 환경 (`Map.empty`)으로 시작한다
- 모든 재귀 호출에서 `env`를 전달한다

## 중첩된 Let 바인딩

중첩된 let 바인딩이 어떻게 컴파일되는지 보자.

**FunLang 소스:**

```fsharp
let x = 10 in
let y = 20 in
let z = x + y in
z * 2
```

**AST:**

```fsharp
Let("x",
  IntLiteral 10,
  Let("y",
    IntLiteral 20,
    Let("z",
      BinaryOp(Add, Var "x", Var "y"),
      BinaryOp(Multiply, Var "z", IntLiteral 2))))
```

**컴파일 과정:**

1. **Let("x", IntLiteral 10, ...)**
   - `binding` 컴파일: `%c10 = arith.constant 10 : i32`
   - `env0 = {}`
   - `env1 = env0.Add("x", %c10) = { x -> %c10 }`
   - `body` 컴파일 (env1 사용)

2. **Let("y", IntLiteral 20, ...)** (env1에서)
   - `binding` 컴파일: `%c20 = arith.constant 20 : i32`
   - `env2 = env1.Add("y", %c20) = { x -> %c10, y -> %c20 }`
   - `body` 컴파일 (env2 사용)

3. **Let("z", BinaryOp(Add, Var "x", Var "y"), ...)** (env2에서)
   - `binding` 컴파일:
     - `Var "x"`: env2에서 조회 → %c10
     - `Var "y"`: env2에서 조회 → %c20
     - `%z = arith.addi %c10, %c20 : i32`
   - `env3 = env2.Add("z", %z) = { x -> %c10, y -> %c20, z -> %z }`
   - `body` 컴파일 (env3 사용)

4. **BinaryOp(Multiply, Var "z", IntLiteral 2)** (env3에서)
   - `Var "z"`: env3에서 조회 → %z
   - `IntLiteral 2`: `%c2 = arith.constant 2 : i32`
   - `%result = arith.muli %z, %c2 : i32`

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %c10 = arith.constant 10 : i32      // let x = 10
    %c20 = arith.constant 20 : i32      // let y = 20
    %z = arith.addi %c10, %c20 : i32    // let z = x + y
    %c2 = arith.constant 2 : i32
    %result = arith.muli %z, %c2 : i32  // z * 2
    func.return %result : i32
  }
}
```

**분석:**

- 각 let 바인딩이 SSA value를 생성한다
- 변수 참조는 기존 SSA value를 재사용한다
- 명시적인 저장/로드 연산이 없다 (모든 것이 레지스터에 있다)
- SSA value가 자유롭게 흐른다 (%c10과 %c20이 %z에서 사용됨)

**실행:**

```bash
$ ./program
$ echo $?
60
```

예상대로 60을 반환한다!

## 변수 섀도잉

섀도잉이 어떻게 작동하는지 보자.

**FunLang 소스:**

```fsharp
let x = 5 in
let x = x + 1 in
x
```

**AST:**

```fsharp
Let("x",
  IntLiteral 5,
  Let("x",
    BinaryOp(Add, Var "x", IntLiteral 1),
    Var "x"))
```

**컴파일 과정:**

1. **첫 번째 Let("x", IntLiteral 5, ...)**
   - `binding`: `%x = arith.constant 5 : i32`
   - `env1 = { x -> %x }`

2. **두 번째 Let("x", BinaryOp(Add, Var "x", IntLiteral 1), ...)** (env1에서)
   - `binding`:
     - `Var "x"`: env1에서 조회 → %x (값 5)
     - `IntLiteral 1`: `%c1 = arith.constant 1 : i32`
     - `%x_0 = arith.addi %x, %c1 : i32`
   - `env2 = env1.Add("x", %x_0) = { x -> %x_0 }`  ← 섀도잉!

3. **Var "x"** (env2에서)
   - env2에서 조회 → %x_0 (값 6)

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %x = arith.constant 5 : i32        // 외부 x
    %c1 = arith.constant 1 : i32
    %x_0 = arith.addi %x, %c1 : i32    // 내부 x = 외부 x + 1
    func.return %x_0 : i32              // 내부 x 반환
  }
}
```

**핵심 통찰력:**

- MLIR은 자동으로 고유한 이름을 생성한다 (`%x`, `%x_0`)
- 두 번째 `Let("x", ...)`에서 `binding` 표현식은 **외부 x**를 참조한다 (env1에서 컴파일)
- body 표현식은 **내부 x**를 참조한다 (env2에서 컴파일)
- 섀도잉은 새로운 SSA value를 생성하지, 기존 value를 변경하지 않는다

**실행:**

```bash
$ ./program
$ echo $?
6
```

예상대로 6을 반환한다!

## 완전한 예시와 드라이버

이제 완전한 컴파일러 드라이버를 작성하자.

**Main.fs 예시:**

```fsharp
namespace FunLangCompiler

open System

module Main =

    [<EntryPoint>]
    let main args =
        printfn "=== FunLang Compiler with Let Bindings ==="

        // 예시: let x = 10 in let y = 20 in x + y
        let ast =
            Let("x",
              IntLiteral 10,
              Let("y",
                IntLiteral 20,
                BinaryOp(Add, Var "x", Var "y")))

        let program = { expr = ast }

        printfn "AST: %A" ast
        printfn ""

        // MLIR로 컴파일
        printfn "Compiling to MLIR..."
        let mlirMod = CodeGen.translateToMlir program
        printfn "%s" (mlirMod.Print())

        // 검증
        printfn "Verifying..."
        CodeGen.verify mlirMod
        printfn "✓ Verification passed"

        // Lowering과 네이티브 코드 생성 (Chapter 05와 동일)
        Lowering.lowerToLLVMDialect mlirMod
        let llvmIR = Lowering.translateToLLVMIR mlirMod
        NativeCodeGen.emitObjectFile llvmIR "program.o"
        NativeCodeGen.linkExecutable "program.o" "program"

        mlirMod.Dispose()

        printfn ""
        printfn "=== Compilation successful ==="
        printfn "Run: ./program"
        printfn "Expected output (exit code): 30"

        0
```

**컴파일과 실행:**

```bash
$ dotnet run
=== FunLang Compiler with Let Bindings ===
AST: Let ("x", IntLiteral 10, Let ("y", IntLiteral 20, BinaryOp (Add, Var "x", Var "y")))

Compiling to MLIR...
module {
  func.func @main() -> i32 {
    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32
    %0 = arith.addi %c10, %c20 : i32
    func.return %0 : i32
  }
}
Verifying...
✓ Verification passed
[... lowering과 linking ...]

=== Compilation successful ===
Run: ./program
Expected output (exit code): 30

$ ./program
$ echo $?
30
```

완벽하다!

## 공통 에러

### 에러 1: 바인딩되지 않은 변수 참조

**증상:**

```
Exception: Unbound variable: y
```

**원인:**

변수가 스코프에 없는데 참조하려고 했다.

**예시:**

```fsharp
// WRONG: y가 바인딩되지 않음
let x = 10 in
y + x
```

**해결:**

변수를 사용하기 전에 let 바인딩으로 정의한다:

```fsharp
// CORRECT
let x = 10 in
let y = 20 in
y + x
```

### 에러 2: 스코프 밖에서 변수 사용

**증상:**

```
Exception: Unbound variable: x
```

**원인:**

변수가 스코프 밖에서 사용되었다.

**예시:**

```fsharp
// WRONG: 두 번째 x는 스코프 밖
(let x = 10 in x + x) + x
//                      ^ x는 여기서 바인딩되지 않음
```

let 바인딩의 스코프는 `body` 표현식까지만이다. 밖에서는 보이지 않는다.

**해결:**

필요한 스코프 전체를 감싸거나, 바인딩을 외부로 이동한다:

```fsharp
// CORRECT: x를 외부에서 바인딩
let x = 10 in
(x + x) + x
```

### 에러 3: 환경을 재귀 호출에 전달하지 않음

**증상:**

```
Compilation error: 'env' is not defined
```

**원인:**

`compileExpr` 재귀 호출에서 `env` 파라미터를 빠뜨렸다.

**예시:**

```fsharp
// WRONG: env 파라미터 누락
| BinaryOp(op, lhs, rhs) ->
    let lhsVal = compileExpr builder block location lhs  // env 없음!
    ...
```

**해결:**

모든 `compileExpr` 호출에 `env`를 전달한다:

```fsharp
// CORRECT
| BinaryOp(op, lhs, rhs) ->
    let lhsVal = compileExpr builder block location lhs env
    let rhsVal = compileExpr builder block location rhs env
    ...
```

**패턴:** 각 케이스를 추가할 때마다 재귀 호출에 `env`를 전달하는지 확인한다.

### 에러 4: Let 바인딩에서 환경 확장 잊음

**증상:**

변수가 body에서 보이지 않는다.

**원인:**

`Let` 케이스에서 `env.Add`를 호출했지만 확장된 환경을 body에 전달하지 않았다.

**예시:**

```fsharp
// WRONG: 확장된 환경을 사용하지 않음
| Let(name, binding, body) ->
    let bindVal = compileExpr builder block location binding env
    let env' = env.Add(name, bindVal)
    compileExpr builder block location body env  // env' 대신 env 사용!
```

**해결:**

확장된 환경 `env'`를 body에 전달한다:

```fsharp
// CORRECT
| Let(name, binding, body) ->
    let bindVal = compileExpr builder block location binding env
    let env' = env.Add(name, bindVal)
    compileExpr builder block location body env'  // env' 사용!
```

### 에러 5: 섀도잉을 뮤테이션으로 착각

**개념 오류:**

```fsharp
// 이것은 뮤테이션이 아니다!
let x = 5 in
let x = 10 in
x
```

**설명:**

이것은 변수 "x"를 덮어쓰는 것이 아니다. 새로운 바인딩을 만드는 것이다:
- 외부 `x`는 값 5를 가진 SSA value `%x`
- 내부 `x`는 값 10을 가진 SSA value `%x_0`
- 두 value 모두 존재한다 (외부 x는 변경되지 않음)

**MLIR IR 확인:**

```mlir
%x = arith.constant 5 : i32    // 외부 x (여전히 존재)
%x_0 = arith.constant 10 : i32  // 내부 x (새로운 value)
func.return %x_0 : i32
```

## 장 요약

이 장에서 다음을 성취했다:

1. **SSA 형태 이해**: 각 value는 한 번만 정의되며, 이것이 컴파일러 최적화를 단순화한다
2. **Let 바인딩 추가**: 함수형 언어의 불변 바인딩이 SSA와 자연스럽게 일치한다
3. **환경 구현**: `Map<string, MlirValue>`로 변수 스코프 관리
4. **환경 전달 패턴**: 재귀 함수에 환경을 전달하여 중첩 스코프 구현
5. **섀도잉 vs 뮤테이션**: 섀도잉은 새로운 SSA value를 생성하지, 기존 value를 변경하지 않는다
6. **완전한 예제**: 중첩된 let 바인딩이 올바른 MLIR IR로 컴파일된다

**독자가 할 수 있는 것:**

- `let x = 5 in x + x` 컴파일 → 네이티브 바이너리 → 결과: 10 ✓
- `let x = 10 in let y = 20 in x + y` 컴파일 → 결과: 30 ✓
- 섀도잉 이해: `let x = 5 in let x = 10 in x` → 결과: 10 ✓
- 환경 전달로 스코프 관리 ✓
- 스코프 에러 디버깅 (바인딩되지 않은 변수) ✓

**핵심 개념:**

- **SSA 형태**: 각 value는 한 번만 정의된다
- **Let 바인딩 = SSA value**: 불변 바인딩이 SSA를 자연스럽게 표현한다
- **환경 = 변수 스코프**: Map으로 변수 이름을 SSA value에 매핑한다
- **환경 전달 = 스코프 중첩**: 재귀 호출로 스코프를 확장한다
- **섀도잉 ≠ 뮤테이션**: 새로운 value 생성, 기존 value 변경 아님

**다음 장 미리보기:**

Chapter 08에서는 **제어 흐름 (if/else)**을 추가한다:

```fsharp
let x = if 5 < 10 then 42 else 0 in
x + x
```

이것은 다음을 도입한다:
- **scf.if** 연산: 구조화된 제어 흐름
- **Block arguments**: MLIR의 PHI 노드 대안
- **scf.yield**: 분기에서 값 반환
- **SSA at control flow merges**: 조건부 값을 어떻게 SSA로 표현하는가

**Phase 2는 계속된다!**

---

**이제 독자는 let 바인딩과 변수를 컴파일하고, SSA 형태를 이해한다!**
