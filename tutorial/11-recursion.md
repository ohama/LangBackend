# Chapter 11: 재귀 (Recursion)

## 소개

함수형 프로그래밍에서 **재귀(recursion)**는 단순한 기법이 아니라 **필수 도구**다. 명령형 언어가 loop을 쓰는 곳에 함수형 언어는 재귀를 쓴다.

```fsharp
// 명령형 스타일 (loop)
let sum_to n =
    let mutable result = 0
    for i in 1 to n do
        result <- result + i
    result

// 함수형 스타일 (recursion)
let rec sum_to n =
    if n <= 0 then 0
    else n + sum_to (n - 1)
```

**왜 재귀인가?**

순수 함수형 언어에는 mutable 변수가 없다. 값은 불변이고, 상태는 함수 파라미터를 통해 전달된다. Loop은 카운터 변수를 변경하는데, 이것은 mutation이다. 재귀는 mutation 없이 반복을 표현할 수 있다.

FunLang은 순수 함수형 언어다. Loop 구문이 없다. 모든 반복은 재귀로 표현된다.

**재귀의 본질: 자기 참조(Self-reference)**

재귀 함수는 **자기 자신을 호출**한다:

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
           // ↑ 자기 자신을 호출!
```

`factorial` 함수가 본체 내부에서 `factorial`을 호출한다. 이것이 가능하려면:
- 함수 이름이 본체에서 보여야 한다 (scope 문제)
- 무한 재귀를 방지할 기저 사례(base case)가 필요하다

**Chapter 11의 범위:**

이 장에서 다루는 것:
1. **재귀 함수 (Recursive functions)**: 자기 자신을 호출하는 함수 (factorial, fibonacci)
2. **상호 재귀 (Mutual recursion)**: 두 함수가 서로를 호출 (is_even, is_odd)
3. **스택 프레임 (Stack frames)**: 재귀 호출이 스택 메모리를 어떻게 사용하는가
4. **꼬리 호출 최적화 (Tail call optimization)**: 스택 오버플로우를 방지하는 기법

이 장을 마치면:
- factorial, fibonacci 같은 재귀 함수를 컴파일할 수 있다
- 상호 재귀가 모듈 레벨 심볼 테이블을 통해 작동하는 원리를 안다
- 스택 프레임이 어떻게 생성되고 소멸되는지 이해한다
- 꼬리 호출 최적화가 무엇이고 왜 중요한지 안다

> **Preview:** Phase 3 (Chapter 10-11)은 최상위 명명된 함수를 다룬다. Phase 4에서 클로저와 고차 함수를 추가할 것이다.

## 재귀가 MLIR에서 작동하는 원리

### 모듈 레벨 심볼 테이블

Chapter 10에서 배운 것: MLIR 모듈은 **flat symbol table**을 가진다. 모든 `func.func` 연산이 모듈 레벨 심볼로 등록된다.

```mlir
module {
  func.func @factorial(%n: i32) -> i32 {
    // ...
  }

  func.func @fibonacci(%n: i32) -> i32 {
    // ...
  }

  func.func @main() -> i32 {
    // ...
  }
}
```

**핵심:** 모든 함수가 서로에게 보인다. 정의 순서는 중요하지 않다.

- `@factorial`은 `@fibonacci`를 호출할 수 있다
- `@fibonacci`는 `@factorial`을 호출할 수 있다
- `@factorial`은 **자기 자신**을 호출할 수 있다!

**자기 참조 (Self-reference):**

```mlir
func.func @factorial(%n: i32) -> i32 {
  // ...
  %rec = func.call @factorial(%n_minus_1) : (i32) -> i32
  //                 ↑ 자기 자신을 호출!
  // ...
}
```

`@factorial` 함수가 내부에서 `func.call @factorial`을 실행한다. 이것은 **심볼 참조(symbol reference)**다:
- `@factorial`이라는 심볼이 모듈에 존재하는가? **예** (자기 자신)
- 타입이 `(i32) -> i32`가 맞는가? **예**
- 호출 가능한가? **예**

MLIR verifier는 심볼 존재를 확인하지만, "자기 자신 호출"을 금지하지 않는다. 재귀가 자연스럽게 작동한다.

### Interpreter vs Compiler의 차이

**Interpreter에서 재귀 (LangTutorial FunLang):**

```fsharp
// AST
LetRec("factorial",
       Lambda(["n"],
              If(BinOp(Var "n", Le, Num 1),
                 Num 1,
                 BinOp(Var "n",
                       Mul,
                       App(Var "factorial", [BinOp(Var "n", Sub, Num 1)])))))

// Interpreter evaluation
let rec eval env ast =
    match ast with
    | LetRec(name, Lambda(params, body), rest) ->
        // 1. 재귀 환경 생성: env에 함수 자신을 추가
        let rec_env = env.Add(name, RecursiveClosure(params, body, rec_env))
        // 2. 본체 평가
        eval rec_env body
```

Interpreter는 **환경(environment)**에 함수를 바인딩한다. `LetRec`은 "재귀 바인딩"을 만든다 - 함수 본체가 평가되기 전에 환경에 자기 자신이 포함된다.

**Compiler에서 재귀 (FunLang MLIR):**

```fsharp
// 컴파일
let compileFuncDef builder moduleDef (FunDef(name, params, body)) =
    // 1. 함수 생성 (func.func @name)
    let funcOp = builder.CreateFuncOp(name, paramTypes, returnType)

    // 2. 본체 컴파일
    let bodyValue = compileExpr builder env body

    // 3. 반환
    builder.CreateFuncReturn(bodyValue)

    // 4. 모듈에 추가
    moduleDef.AddFunction(funcOp)
```

Compiler는 **심볼 테이블**을 사용한다:
- 함수가 `func.func` 연산으로 모듈에 추가되면, 심볼 `@name`이 등록된다
- 본체를 컴파일할 때 `func.call @name`을 만나면, 심볼 테이블에서 `@name`을 찾는다
- 심볼이 존재하므로 (자기 자신) 호출이 성공한다

**차이점:**

| 측면 | Interpreter | Compiler |
|------|-------------|----------|
| 함수 저장 | 환경 (Map<string, Value>) | 모듈 심볼 테이블 |
| 재귀 메커니즘 | 재귀 클로저 (self-reference in closure) | 심볼 참조 (symbol reference) |
| 평가 시점 | 런타임 (함수 호출할 때마다 환경 검색) | 컴파일 타임 (심볼 확인) + 런타임 (call instruction) |
| Forward declaration | 불필요 (LetRec이 재귀 환경 생성) | 불필요 (모듈 레벨 심볼은 정의 순서 무관) |

**핵심:** Interpreter는 환경을 사용하고, Compiler는 심볼을 사용한다. 둘 다 재귀를 지원하지만, 메커니즘이 다르다.

### 컴파일 타임 심볼 확인

MLIR은 **static symbol resolution**을 수행한다:

```mlir
// 잘못된 IR - verifier가 거부
func.func @foo(%n: i32) -> i32 {
  %result = func.call @bar(%n) : (i32) -> i32
  //                     ↑ @bar가 모듈에 없음!
  func.return %result : i32
}
```

MLIR verifier (`mlirOperationVerify`)는 심볼 참조를 검증한다:
- `@bar` 심볼이 모듈에 존재하는가?
- 타입이 `(i32) -> i32`와 호환되는가?

검증 실패 시 에러:

```
error: 'func.call' op 'bar' does not reference a valid function
```

**재귀 함수는 자연스럽게 통과:**

```mlir
func.func @factorial(%n: i32) -> i32 {
  // ...
  %rec = func.call @factorial(%n_minus_1) : (i32) -> i32
  // ✓ @factorial은 모듈에 존재 (자기 자신)
  // ✓ 타입 (i32) -> i32 일치
  // ...
}
```

Verifier는 심볼 존재만 확인한다. "자기 자신 호출"을 특별히 처리하지 않는다.

### MLIR IR 예시: Factorial 자기 참조

```mlir
module {
  func.func @factorial(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi sle, %arg0, %c1 : i32
    %result = scf.if %cmp -> (i32) {
      scf.yield %c1 : i32
    } else {
      %n_minus_1 = arith.subi %arg0, %c1 : i32
      %rec = func.call @factorial(%n_minus_1) : (i32) -> i32
      //                ↑ 자기 자신 호출
      %product = arith.muli %arg0, %rec : i32
      scf.yield %product : i32
    }
    func.return %result : i32
  }
}
```

**실행 시퀀스 (factorial 5):**

1. `@factorial`이 5로 호출됨
2. 조건 확인: `5 <= 1`? → false
3. else 블록 실행:
   - `n_minus_1 = 5 - 1 = 4`
   - `rec = func.call @factorial(4)` ← **재귀 호출**
4. 이제 **새로운 스택 프레임**에서 `@factorial(4)` 실행
5. 조건 확인: `4 <= 1`? → false
6. else 블록 실행:
   - `n_minus_1 = 4 - 1 = 3`
   - `rec = func.call @factorial(3)` ← **재귀 호출**
7. ... (계속)

재귀 호출마다 새로운 스택 프레임이 생성된다. 각 프레임은 독립적인 `%arg0`, `%n_minus_1`, `%rec` 값을 가진다.

**핵심:** 심볼 참조 `@factorial`은 컴파일 타임에 확인되고, 런타임에 `call` instruction으로 실행된다. LLVM이 스택 프레임 관리를 처리한다.

## 재귀 함수: Factorial

### Factorial 정의

**수학적 정의:**

```
factorial(n) = n! = n × (n-1) × (n-2) × ... × 2 × 1

예시:
  5! = 5 × 4 × 3 × 2 × 1 = 120
  3! = 3 × 2 × 1 = 6
  1! = 1
  0! = 1 (정의에 의해)
```

**재귀적 정의:**

```
factorial(n) = {
  1                        if n <= 1  (base case)
  n × factorial(n - 1)     if n > 1   (recursive case)
}
```

기저 사례(base case): `n <= 1`일 때 `1` 반환. 재귀 종료 조건.
재귀 사례(recursive case): `n × factorial(n - 1)`. 자기 자신을 더 작은 입력으로 호출.

**FunLang 소스:**

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
```

### AST 표현

Chapter 10에서 정의한 AST:

```fsharp
type Expr =
    | Num of int
    | Var of string
    | BinOp of Expr * Operator * Expr
    | If of Expr * Expr * Expr
    | Let of string * Expr * Expr
    | App of string * Expr list    // 함수 호출

type FunDef =
    | FunDef of string * string list * Expr

type Program =
    | Program of FunDef list * Expr
```

**factorial의 AST:**

```fsharp
FunDef("factorial",
       ["n"],
       If(BinOp(Var "n", Le, Num 1),
          Num 1,
          BinOp(Var "n",
                Mul,
                App("factorial", [BinOp(Var "n", Sub, Num 1)]))))
```

**주목할 점:**
- `App("factorial", ...)`: 함수 호출. 자기 자신을 호출한다.
- 기존 AST로 충분하다. `LetRec` 같은 새로운 AST 노드가 필요 없다.
- `FunDef`는 이미 모듈 레벨 함수를 표현한다. 이름으로 자기 참조가 가능하다.

### 컴파일 전략

**Chapter 10의 compileFuncDef 재사용:**

```fsharp
let compileFuncDef (builder: OpBuilder) (moduleDef: ModuleOp) (FunDef(name, params, body)) =
    // 1. 함수 타입 생성
    let paramTypes = List.replicate params.Length builder.GetI32Type()
    let returnType = builder.GetI32Type()
    let funcType = builder.GetFunctionType(paramTypes, returnType)

    // 2. func.func 생성
    let funcOp = builder.CreateFuncOp(name, funcType)

    // 3. Entry block 생성 및 파라미터 가져오기
    let entryBlock = funcOp.GetEntryBlock()
    builder.SetInsertionPointToEnd(entryBlock)

    // 4. 환경 구축: 파라미터를 환경에 추가
    let env =
        params
        |> List.mapi (fun i paramName ->
            let argValue = entryBlock.GetArgument(i)
            (paramName, argValue))
        |> Map.ofList

    // 5. 본체 컴파일
    let bodyValue = compileExpr builder env body

    // 6. 반환
    builder.CreateFuncReturn(bodyValue)

    // 7. 모듈에 추가
    moduleDef.AddFunction(funcOp)
```

**재귀 호출 처리 (compileExpr의 App case):**

```fsharp
let rec compileExpr (builder: OpBuilder) (env: Map<string, MlirValue>) (expr: Expr) =
    match expr with
    | App(funcName, args) ->
        // 1. 인자 컴파일
        let argValues = args |> List.map (compileExpr builder env)

        // 2. 함수 호출
        builder.CreateFuncCall(funcName, argValues)
    // ... other cases
```

**핵심:**
- `App("factorial", [arg])`를 만나면 `CreateFuncCall("factorial", [argValue])`
- `CreateFuncCall`은 `func.call @factorial(%arg) : (i32) -> i32` 생성
- 심볼 `@factorial`이 모듈에 존재 (자기 자신)
- 재귀 호출 완료!

**재귀 함수 컴파일에 특별한 처리가 필요 없다.** 일반 함수 호출과 동일하게 처리된다.

### 완전한 MLIR IR 출력

```mlir
module {
  func.func @factorial(%arg0: i32) -> i32 {
    // if n <= 1
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi sle, %arg0, %c1 : i32

    // scf.if with two branches
    %result = scf.if %cmp -> (i32) {
      // then: return 1
      scf.yield %c1 : i32
    } else {
      // else: return n * factorial(n - 1)

      // n - 1
      %n_minus_1 = arith.subi %arg0, %c1 : i32

      // factorial(n - 1) - 재귀 호출!
      %rec = func.call @factorial(%n_minus_1) : (i32) -> i32

      // n * factorial(n - 1)
      %product = arith.muli %arg0, %rec : i32

      scf.yield %product : i32
    }

    func.return %result : i32
  }
}
```

**구조:**
1. **조건 평가:** `%cmp = arith.cmpi sle, %arg0, %c1` (n <= 1?)
2. **scf.if 분기:**
   - **then 블록:** `scf.yield %c1` (기저 사례: 1 반환)
   - **else 블록:**
     - `%n_minus_1 = arith.subi %arg0, %c1` (n - 1 계산)
     - `%rec = func.call @factorial(%n_minus_1)` (**재귀 호출**)
     - `%product = arith.muli %arg0, %rec` (n * 재귀 결과)
     - `scf.yield %product` (재귀 사례: n * factorial(n-1))
3. **반환:** `func.return %result`

### 단계별 실행 추적

**factorial 5 실행 과정:**

```
1. factorial(5) 호출
   ├─ 조건: 5 <= 1? → false
   ├─ else 블록 진입
   ├─ n_minus_1 = 5 - 1 = 4
   ├─ factorial(4) 호출 ← 재귀
   │  ├─ 조건: 4 <= 1? → false
   │  ├─ else 블록 진입
   │  ├─ n_minus_1 = 4 - 1 = 3
   │  ├─ factorial(3) 호출 ← 재귀
   │  │  ├─ 조건: 3 <= 1? → false
   │  │  ├─ else 블록 진입
   │  │  ├─ n_minus_1 = 3 - 1 = 2
   │  │  ├─ factorial(2) 호출 ← 재귀
   │  │  │  ├─ 조건: 2 <= 1? → false
   │  │  │  ├─ else 블록 진입
   │  │  │  ├─ n_minus_1 = 2 - 1 = 1
   │  │  │  ├─ factorial(1) 호출 ← 재귀
   │  │  │  │  ├─ 조건: 1 <= 1? → true
   │  │  │  │  └─ then 블록: return 1 ← 기저 사례!
   │  │  │  ├─ rec = 1
   │  │  │  ├─ product = 2 * 1 = 2
   │  │  │  └─ return 2
   │  │  ├─ rec = 2
   │  │  ├─ product = 3 * 2 = 6
   │  │  └─ return 6
   │  ├─ rec = 6
   │  ├─ product = 4 * 6 = 24
   │  └─ return 24
   ├─ rec = 24
   ├─ product = 5 * 24 = 120
   └─ return 120

최종 결과: 120
```

**호출 깊이 (Call depth):** 5

각 재귀 호출은 새로운 스택 프레임을 생성한다. `factorial(5)`는 5개의 스택 프레임을 사용한다.

### Lowered LLVM IR

MLIR IR을 LLVM IR로 변환하면 (`mlir-opt --convert-scf-to-cf --convert-func-to-llvm --convert-arith-to-llvm`):

```llvm
define i32 @factorial(i32 %0) {
entry:
  %1 = icmp sle i32 %0, 1
  br i1 %1, label %then, label %else

then:
  br label %merge

else:
  %2 = sub i32 %0, 1
  %3 = call i32 @factorial(i32 %2)  ; 재귀 호출 (call instruction)
  %4 = mul i32 %0, %3
  br label %merge

merge:
  %5 = phi i32 [ 1, %then ], [ %4, %else ]
  ret i32 %5
}
```

**주목할 점:**
- `call i32 @factorial(i32 %2)`: LLVM IR의 재귀 호출
- 각 호출은 스택 프레임을 생성한다 (LLVM runtime이 처리)
- PHI 노드 (`phi i32 [ 1, %then ], [ %4, %else ]`)는 scf.if의 lowering 결과

**Native 코드로 컴파일:**

```bash
mlir-translate --mlir-to-llvmir factorial.mlir > factorial.ll
llc -filetype=obj factorial.ll -o factorial.o
gcc -o factorial factorial.o runtime.o -lgc
./factorial
```

## 재귀 함수: Fibonacci

### Fibonacci 정의

**수학적 정의:**

```
fibonacci(n) = {
  n                                if n <= 1  (base case)
  fibonacci(n-1) + fibonacci(n-2)  if n > 1   (recursive case)
}

수열:
  fib(0) = 0
  fib(1) = 1
  fib(2) = fib(1) + fib(0) = 1 + 0 = 1
  fib(3) = fib(2) + fib(1) = 1 + 1 = 2
  fib(4) = fib(3) + fib(2) = 2 + 1 = 3
  fib(5) = fib(4) + fib(3) = 3 + 2 = 5
  fib(6) = fib(5) + fib(4) = 5 + 3 = 8
```

**FunLang 소스:**

```fsharp
let rec fib n =
    if n <= 1 then n
    else fib (n - 1) + fib (n - 2)
```

### Double Recursion 패턴

Factorial은 **단일 재귀(single recursion)**: 한 번만 자기 자신을 호출.
Fibonacci는 **이중 재귀(double recursion)**: 두 번 자기 자신을 호출.

```fsharp
fib (n - 1) + fib (n - 2)
//  ↑             ↑
// 첫 번째 호출   두 번째 호출
```

**함의:**
- 각 재귀 호출이 또 다른 두 개의 호출을 만든다
- 호출 트리가 **지수적으로 증가**한다

**fib(5)의 호출 트리:**

```
                    fib(5)
                   /      \
              fib(4)      fib(3)
             /     \      /     \
        fib(3)   fib(2) fib(2) fib(1)
        /   \    /   \  /   \
    fib(2) fib(1) fib(1) fib(0) fib(1) fib(0)
    /   \
fib(1) fib(0)
```

**호출 횟수:** `fib(5)`를 계산하기 위해 15번의 함수 호출이 발생한다.

**시간 복잡도:** O(2^n) - 지수 시간. `fib(30)` ≈ 20억 번 호출!

### 컴파일: 두 개의 func.call

```mlir
func.func @fib(%arg0: i32) -> i32 {
  // if n <= 1
  %c1 = arith.constant 1 : i32
  %cmp = arith.cmpi sle, %arg0, %c1 : i32

  %result = scf.if %cmp -> (i32) {
    // then: return n
    scf.yield %arg0 : i32
  } else {
    // else: return fib(n-1) + fib(n-2)

    // n - 1
    %n_minus_1 = arith.subi %arg0, %c1 : i32

    // fib(n - 1) - 첫 번째 재귀 호출
    %fib_n_1 = func.call @fib(%n_minus_1) : (i32) -> i32

    // n - 2
    %c2 = arith.constant 2 : i32
    %n_minus_2 = arith.subi %arg0, %c2 : i32

    // fib(n - 2) - 두 번째 재귀 호출
    %fib_n_2 = func.call @fib(%n_minus_2) : (i32) -> i32

    // fib(n-1) + fib(n-2)
    %sum = arith.addi %fib_n_1, %fib_n_2 : i32

    scf.yield %sum : i32
  }

  func.return %result : i32
}
```

**구조:**
- **else 블록에서 두 번의 func.call:**
  - `%fib_n_1 = func.call @fib(%n_minus_1)`
  - `%fib_n_2 = func.call @fib(%n_minus_2)`
- **각 호출은 독립적:** `%fib_n_1`이 완료된 후 `%fib_n_2` 실행
- **결과를 더함:** `%sum = arith.addi %fib_n_1, %fib_n_2`

**실행 순서 (eager evaluation):**
1. `%n_minus_1` 계산
2. `func.call @fib(%n_minus_1)` 실행 → 결과를 `%fib_n_1`에 저장
3. `%n_minus_2` 계산
4. `func.call @fib(%n_minus_2)` 실행 → 결과를 `%fib_n_2`에 저장
5. `%sum = %fib_n_1 + %fib_n_2` 계산

### 성능 문제

**지수 시간 복잡도:**

```
fib(10) ≈ 177 호출
fib(20) ≈ 21,891 호출
fib(30) ≈ 2,692,537 호출
fib(40) ≈ 331,160,281 호출 (3억 번!)
```

**왜 느린가?**

중복 계산이 많다. `fib(5)`를 계산할 때 `fib(3)`을 두 번 계산하고, `fib(2)`를 세 번 계산한다.

```
fib(5)
├─ fib(4)
│  ├─ fib(3) ← 첫 번째 fib(3)
│  └─ fib(2)
└─ fib(3) ← 두 번째 fib(3) (중복!)
   ├─ fib(2) ← 중복!
   └─ fib(1)
```

**최적화 방법 (Phase 3 범위 밖):**
- **Memoization:** 이미 계산한 값을 저장 (hashtable 사용)
- **Dynamic Programming:** Bottom-up 방식으로 계산
- **Tail recursion:** 꼬리 재귀로 변환 (accumulator 사용)

이 장에서는 **순진한 재귀 구현**만 다룬다. 최적화는 나중 단계에서 배운다.

**교훈:** 재귀는 우아하지만, 항상 효율적이지는 않다. 알고리즘 선택이 중요하다.

## 스택 프레임 관리

### 스택 프레임이란?

**스택 프레임(stack frame)** (또는 **activation record**)은 함수 호출에 필요한 정보를 저장하는 메모리 영역이다.

**스택 프레임에 포함되는 것:**

1. **반환 주소(return address)**: 함수가 끝나면 돌아갈 위치
2. **함수 파라미터**: 호출자가 전달한 인자
3. **지역 변수**: 함수 내부에서 선언된 변수
4. **저장된 레지스터**: 호출 전 레지스터 상태 (ABI가 요구)
5. **임시 값**: 중간 계산 결과 (SSA values)

**함수 호출 시 스택 프레임 생성:**

```
main()
  |
  ├─ factorial(5) 호출
  │    ├─ 스택 프레임 생성
  │    │    - return address: main의 다음 instruction
  │    │    - arg0 = 5
  │    │    - 지역 변수 공간
  │    ├─ factorial(4) 호출
  │    │    ├─ 새로운 스택 프레임 생성
  │    │    │    - return address: factorial(5)의 다음 instruction
  │    │    │    - arg0 = 4
  │    │    ├─ factorial(3) 호출
  │    │    │    └─ 또 다른 스택 프레임...
```

**스택 성장 방향:**

대부분의 플랫폼에서 스택은 **아래로 성장**한다 (높은 주소 → 낮은 주소):

```
높은 주소
   ↓
 [main의 스택 프레임]
 [factorial(5)의 스택 프레임]  ← SP (Stack Pointer) 이동
 [factorial(4)의 스택 프레임]  ← SP 이동
 [factorial(3)의 스택 프레임]  ← SP 이동
 [factorial(2)의 스택 프레임]
 [factorial(1)의 스택 프레임]  ← SP (현재 위치)
   ↓
낮은 주소
```

**Stack Pointer (SP)**: 스택의 현재 끝을 가리키는 레지스터. 함수 호출 시 SP가 아래로 이동.

### 재귀 호출과 스택 깊이

**재귀 호출마다 새로운 스택 프레임:**

```fsharp
factorial(5)
  ├─ 스택 프레임 1: arg0=5, return_addr=main
  ├─ factorial(4) 호출
  │  ├─ 스택 프레임 2: arg0=4, return_addr=factorial(5)
  │  ├─ factorial(3) 호출
  │  │  ├─ 스택 프레임 3: arg0=3, return_addr=factorial(4)
  │  │  ├─ factorial(2) 호출
  │  │  │  ├─ 스택 프레임 4: arg0=2, return_addr=factorial(3)
  │  │  │  ├─ factorial(1) 호출
  │  │  │  │  └─ 스택 프레임 5: arg0=1, return_addr=factorial(2)
  │  │  │  │     ├─ 기저 사례: return 1
  │  │  │  │     └─ 스택 프레임 5 소멸
  │  │  │  ├─ 반환값 1 받음, 2*1=2 계산, return 2
  │  │  │  └─ 스택 프레임 4 소멸
  │  │  ├─ 반환값 2 받음, 3*2=6 계산, return 6
  │  │  └─ 스택 프레임 3 소멸
  │  ├─ 반환값 6 받음, 4*6=24 계산, return 24
  │  └─ 스택 프레임 2 소멸
  ├─ 반환값 24 받음, 5*24=120 계산, return 120
  └─ 스택 프레임 1 소멸
```

**최대 스택 깊이:** `factorial(5)`는 5개의 스택 프레임이 동시에 존재한다 (기저 사례에 도달했을 때).

**일반화:** `factorial(n)`의 최대 스택 깊이는 `n`.

### 스택 크기 제한

**운영체제는 스택 크기를 제한한다:**

| 플랫폼 | 기본 스택 크기 |
|--------|---------------|
| Linux (x86-64) | 8 MB |
| macOS | 8 MB |
| Windows | 1 MB |

**왜 제한이 필요한가?**
- 무한 재귀를 방지
- 메모리 보호 (스택이 다른 메모리 영역을 침범하지 않도록)

**스택 오버플로우(Stack Overflow):**

재귀 깊이가 너무 크면 스택 크기 한계에 도달한다:

```fsharp
factorial(100000)
  ├─ 100,000개의 스택 프레임 필요
  ├─ 각 프레임이 ~64 bytes라고 가정
  ├─ 총 스택 사용: 100,000 * 64 = 6.4 MB
  └─ Linux에서는 OK (8MB 한계), Windows에서는 실패 (1MB 한계)
```

**스택 오버플로우 에러:**

```bash
./factorial
Segmentation fault (core dumped)
# 또는
Stack overflow error
```

**해결책:**
1. **재귀 깊이 제한:** 입력 크기를 제한
2. **꼬리 호출 최적화(Tail Call Optimization):** 스택 프레임 재사용
3. **반복(Iteration)으로 변환:** Loop 사용 (함수형 언어에서는 덜 선호)
4. **Trampoline 기법:** 재귀를 CPS(Continuation-Passing Style)로 변환

이 장 후반부에서 꼬리 호출 최적화를 다룬다.

### LLVM의 스택 프레임 관리

**LLVM은 스택 프레임을 자동으로 관리한다:**

1. **함수 프롤로그(prologue):**
   - 스택 포인터(SP) 감소 (스택 공간 할당)
   - 프레임 포인터(FP) 저장
   - 필요한 레지스터 저장 (callee-saved registers)

2. **함수 에필로그(epilogue):**
   - 저장된 레지스터 복원
   - 프레임 포인터 복원
   - 스택 포인터 증가 (스택 공간 해제)
   - 반환 (ret instruction)

**예시 (x86-64 어셈블리):**

```asm
factorial:
  ; Prologue
  push    rbp              ; 이전 프레임 포인터 저장
  mov     rbp, rsp         ; 새로운 프레임 포인터 설정
  sub     rsp, 16          ; 지역 변수를 위한 스택 공간 할당

  ; Function body
  ; ... (factorial 계산)

  ; Epilogue
  add     rsp, 16          ; 스택 공간 해제
  pop     rbp              ; 이전 프레임 포인터 복원
  ret                      ; 반환 주소로 점프
```

**FunLang 컴파일러는 스택 관리를 직접 하지 않는다:**
- MLIR `func` 다이얼렉트로 함수 정의
- LLVM이 lowering 과정에서 프롤로그/에필로그 생성
- 플랫폼별 calling convention 자동 적용 (System V ABI for Linux, Microsoft x64 for Windows)

**이점:**
- 플랫폼 독립적인 코드
- ABI 호환성 자동 보장
- 최적화 (tail call elimination, frame pointer omission)

### Visualization: factorial 5의 스택

**시간별 스택 상태:**

```
시간 T1: main에서 factorial(5) 호출
┌──────────────────────┐
│ factorial(5)         │ ← SP
│  - arg0 = 5          │
│  - ret_addr = main+X │
├──────────────────────┤
│ main                 │
└──────────────────────┘

시간 T2: factorial(5)에서 factorial(4) 호출
┌──────────────────────┐
│ factorial(4)         │ ← SP
│  - arg0 = 4          │
│  - ret_addr = f(5)+Y │
├──────────────────────┤
│ factorial(5)         │
│  - arg0 = 5          │
├──────────────────────┤
│ main                 │
└──────────────────────┘

시간 T3: factorial(1) 도달 (최대 깊이)
┌──────────────────────┐
│ factorial(1)         │ ← SP (최대 깊이)
│  - arg0 = 1          │
├──────────────────────┤
│ factorial(2)         │
│  - arg0 = 2          │
├──────────────────────┤
│ factorial(3)         │
│  - arg0 = 3          │
├──────────────────────┤
│ factorial(4)         │
│  - arg0 = 4          │
├──────────────────────┤
│ factorial(5)         │
│  - arg0 = 5          │
├──────────────────────┤
│ main                 │
└──────────────────────┘

시간 T4: factorial(1) 반환 후 (1 반환)
┌──────────────────────┐
│ factorial(2)         │ ← SP
│  - arg0 = 2          │
│  - rec = 1           │
├──────────────────────┤
│ factorial(3)         │
├──────────────────────┤
│ factorial(4)         │
├──────────────────────┤
│ factorial(5)         │
├──────────────────────┤
│ main                 │
└──────────────────────┘

...

시간 T_final: 모든 호출 반환 완료
┌──────────────────────┐
│ main                 │ ← SP
│  - result = 120      │
└──────────────────────┘
```

**핵심:**
- 재귀 호출마다 스택이 **성장**한다
- 기저 사례에 도달하면 스택이 **수축**하기 시작한다
- 각 반환은 이전 스택 프레임을 복원한다

### 스택 vs 힙

**Phase 2에서 배운 것:**
- **스택(Stack):** 함수 로컬 값, LIFO, 자동 해제
- **힙(Heap):** 탈출하는 값(closures, data structures), 수동/GC 해제

**Phase 3에서 함수는 스택만 사용:**
- 파라미터: 스택 또는 레지스터 (calling convention)
- 반환 값: 레지스터 (작은 값) 또는 스택 (큰 구조체)
- 지역 변수: SSA values (레지스터 또는 스택 스필링)

**Phase 4에서 클로저는 힙 사용:**
- 클로저 환경: 힙에 할당 (GC_malloc)
- 클로저 포인터: 스택에 저장

**연결:**
- Chapter 9 (Boehm GC)는 Phase 4를 위한 준비였다
- Phase 3 함수는 GC를 사용하지 않는다 (메모리 할당 없음)
- Phase 4 클로저에서 GC가 활성화된다

## 왜 스택 오버플로우가 발생하는가

### 깊은 재귀의 위험

**문제:**

```fsharp
factorial(100000)
```

이 호출은 100,000개의 스택 프레임을 생성한다. 각 프레임이 64 bytes라면:

```
100,000 frames × 64 bytes/frame = 6,400,000 bytes = 6.4 MB
```

Linux 기본 스택 크기가 8 MB이므로 **아슬아슬하게 성공**할 수 있다. Windows (1 MB)에서는 **확실히 실패**한다.

**실제 테스트:**

```bash
# factorial 100000 컴파일 및 실행
./factorial 100000
Segmentation fault
```

**왜 Segmentation fault?**

스택 포인터(SP)가 스택 크기 한계를 넘어서 **guard page**에 도달한다. Guard page는 스택 오버플로우 감지를 위한 특수 메모리 페이지로, 접근 시 segfault를 발생시킨다.

### 최적화 없는 재귀

**일반 재귀 (Non-tail recursion):**

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
         // ↑ 재귀 호출 후 곱셈이 남아있음
```

재귀 호출 후에 **추가 작업**(곱셈)이 있으므로:
- 재귀 호출이 반환될 때까지 **현재 스택 프레임을 유지**해야 한다
- 반환 값을 받아서 `n`과 곱해야 한다
- 따라서 스택 프레임을 재사용할 수 없다

**스택 프레임 누적:**

```
factorial(5) 스택 프레임 유지 (n=5 저장 필요)
  factorial(4) 스택 프레임 유지 (n=4 저장 필요)
    factorial(3) 스택 프레임 유지 (n=3 저장 필요)
      factorial(2) 스택 프레임 유지 (n=2 저장 필요)
        factorial(1) 스택 프레임 생성
          return 1
        return 2 (= 2 * 1)
      return 6 (= 3 * 2)
    return 24 (= 4 * 6)
  return 120 (= 5 * 24)
```

모든 프레임이 동시에 존재해야 한다.

**결론:** 일반 재귀는 스택 크기에 제한받는다.

### 예시: factorial 100000은 왜 실패하는가

```
스택 크기: 8 MB = 8,388,608 bytes
필요한 스택: 100,000 frames × 64 bytes = 6,400,000 bytes

6,400,000 < 8,388,608 → 이론적으로 가능
```

하지만 실제로는:
- **다른 함수 프레임:** main, runtime initialization
- **스택 정렬 (alignment):** 16-byte 정렬 요구사항
- **추가 오버헤드:** 레지스터 저장, guard page

실제 사용 가능한 스택이 줄어든다. 그래서 6.4 MB도 실패할 수 있다.

**안전한 한계:**

대부분의 시스템에서 **~5,000 - 10,000 깊이**가 안전하다. 그 이상은 스택 오버플로우 위험.

**교훈:** 깊은 재귀는 위험하다. 꼬리 호출 최적화가 필요하다.

## 상호 재귀 (Mutual Recursion)

### 상호 재귀란?

**상호 재귀(mutual recursion)**는 두 개 이상의 함수가 서로를 호출하는 패턴이다.

```fsharp
// 함수 A가 함수 B를 호출하고,
// 함수 B가 함수 A를 호출한다.

let rec is_even n =
    if n = 0 then true
    else is_odd (n - 1)

let rec is_odd n =
    if n = 0 then false
    else is_even (n - 1)
```

**차이점:**
- **단순 재귀:** 함수가 자기 자신을 호출 (`factorial` → `factorial`)
- **상호 재귀:** 함수 A가 함수 B를 호출, 함수 B가 함수 A를 호출 (`is_even` ⇄ `is_odd`)

**왜 필요한가?**

어떤 문제는 자연스럽게 상호 재귀로 표현된다:
- 짝수/홀수 판정
- 문법 파서 (expression → term → factor → expression)
- 상태 기계 (state A → state B → state A)

### 예시: is_even과 is_odd

**수학적 정의:**

```
is_even(n) = {
  true                 if n = 0
  is_odd(n - 1)        if n > 0
}

is_odd(n) = {
  false                if n = 0
  is_even(n - 1)       if n > 0
}
```

**직관:**
- 0은 짝수
- n이 짝수인지 확인하려면: n-1이 홀수인지 확인
- n이 홀수인지 확인하려면: n-1이 짝수인지 확인

**FunLang 소스:**

```fsharp
let rec is_even n =
    if n = 0 then true
    else is_odd (n - 1)

let rec is_odd n =
    if n = 0 then false
    else is_even (n - 1)
```

**실행 예시 (is_even 4):**

```
is_even(4)
  ├─ 4 = 0? → false
  ├─ is_odd(3) 호출
  │  ├─ 3 = 0? → false
  │  ├─ is_even(2) 호출
  │  │  ├─ 2 = 0? → false
  │  │  ├─ is_odd(1) 호출
  │  │  │  ├─ 1 = 0? → false
  │  │  │  ├─ is_even(0) 호출
  │  │  │  │  ├─ 0 = 0? → true
  │  │  │  │  └─ return true
  │  │  │  └─ return true (is_even(0) = true)
  │  │  └─ return true (is_odd(1) = true)
  │  └─ return true (is_even(2) = true)
  └─ return true (is_odd(3) = true)

최종 결과: true (4는 짝수)
```

**호출 시퀀스:** is_even → is_odd → is_even → is_odd → is_even

### 모듈 레벨 심볼 테이블의 역할

**핵심:** MLIR 모듈은 flat symbol namespace를 가진다. 모든 함수가 동시에 보인다.

```mlir
module {
  func.func @is_even(%n: i32) -> i1 { ... }
  func.func @is_odd(%n: i32) -> i1 { ... }
}
```

**중요한 점:**
- **정의 순서는 무관:** `is_even`이 먼저 정의되든, `is_odd`가 먼저 정의되든 상관없다.
- **Forward declaration 불필요:** C에서는 forward declaration이 필요하지만, MLIR에서는 필요 없다.
- **모든 함수가 서로에게 보임:** `is_even` 본체에서 `is_odd`를 참조할 수 있고, `is_odd` 본체에서 `is_even`을 참조할 수 있다.

**C와 비교:**

```c
// C에서는 forward declaration 필요
int is_odd(int n);  // forward declaration

int is_even(int n) {
    if (n == 0) return 1;
    else return is_odd(n - 1);
}

int is_odd(int n) {
    if (n == 0) return 0;
    else return is_even(n - 1);
}
```

**MLIR/FunLang에서는 불필요:**

```mlir
// 정의 순서 무관 - 둘 다 작동
module {
  func.func @is_even(%n: i32) -> i1 { ... func.call @is_odd ... }
  func.func @is_odd(%n: i32) -> i1 { ... func.call @is_even ... }
}
```

### 컴파일: 크로스 참조 처리

**상호 재귀 함수 컴파일:**

```fsharp
let compileProgram (builder: OpBuilder) (moduleDef: ModuleOp) (Program(funcs, mainExpr)) =
    // 1. 모든 함수 정의를 모듈에 추가
    funcs |> List.iter (compileFuncDef builder moduleDef)

    // 2. Main 표현식 컴파일
    let mainValue = compileExpr builder Map.empty mainExpr
    ...
```

**핵심 아이디어:**

1. **모든 함수를 먼저 컴파일:** 모듈에 `func.func` 연산 추가
2. **심볼 등록 자동:** MLIR이 각 함수를 심볼 테이블에 등록
3. **본체 컴파일 시 심볼 참조:** `func.call @is_odd` → 심볼 테이블에서 찾기

**두 가지 접근법:**

**접근법 1: 순차 컴파일 (FunLang 사용)**

```fsharp
// 함수를 하나씩 컴파일
funcs |> List.iter (fun funcDef ->
    compileFuncDef builder moduleDef funcDef
)
```

- `is_even` 컴파일 시 본체에서 `func.call @is_odd` 생성
- `@is_odd` 심볼이 아직 등록 안 됨 → **문제 없음!**
- MLIR verifier는 **모든 함수가 컴파일된 후** 실행됨
- Verifier가 실행될 때는 `@is_odd`도 이미 등록되어 있음

**접근법 2: 스텁 먼저 생성 (대안)**

```fsharp
// 1단계: 모든 함수 헤더만 생성 (body 없음)
funcs |> List.iter (fun (FunDef(name, params, _)) ->
    let funcOp = builder.CreateFuncStub(name, paramTypes, returnType)
    moduleDef.AddFunction(funcOp)
)

// 2단계: 모든 함수 본체 채우기
funcs |> List.iter (fun (FunDef(name, params, body)) ->
    let funcOp = moduleDef.GetFunction(name)
    compileFuncBody builder funcOp params body
)
```

- 더 명시적이지만 복잡함
- FunLang은 접근법 1 사용 (더 간단)

**왜 작동하는가?**

MLIR의 **lazy verification**:
- 함수를 컴파일하는 동안 심볼 참조는 검증하지 않음
- 모듈이 완성된 후 `mlirOperationVerify()`를 호출
- 그때 모든 심볼 참조 확인

### 완전한 MLIR IR 출력

```mlir
module {
  // is_even 함수
  func.func @is_even(%arg0: i32) -> i1 {
    %c0 = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %arg0, %c0 : i32

    %result = scf.if %is_zero -> (i1) {
      // then: return true
      %true = arith.constant 1 : i1
      scf.yield %true : i1
    } else {
      // else: return is_odd(n - 1)
      %c1 = arith.constant 1 : i32
      %n_minus_1 = arith.subi %arg0, %c1 : i32

      // is_odd 호출 (상호 재귀!)
      %odd_result = func.call @is_odd(%n_minus_1) : (i32) -> i1

      scf.yield %odd_result : i1
    }

    func.return %result : i1
  }

  // is_odd 함수
  func.func @is_odd(%arg0: i32) -> i1 {
    %c0 = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %arg0, %c0 : i32

    %result = scf.if %is_zero -> (i1) {
      // then: return false
      %false = arith.constant 0 : i1
      scf.yield %false : i1
    } else {
      // else: return is_even(n - 1)
      %c1 = arith.constant 1 : i32
      %n_minus_1 = arith.subi %arg0, %c1 : i32

      // is_even 호출 (상호 재귀!)
      %even_result = func.call @is_even(%n_minus_1) : (i32) -> i1

      scf.yield %even_result : i1
    }

    func.return %result : i1
  }

  // Main 함수
  func.func @funlang_main() -> i32 {
    %c4 = arith.constant 4 : i32
    %result_i1 = func.call @is_even(%c4) : (i32) -> i1

    // i1 → i32 확장 (main 반환용)
    %result_i32 = arith.extui %result_i1 : i1 to i32

    func.return %result_i32 : i32
  }
}
```

**주목할 점:**
- `@is_even`이 `func.call @is_odd` 사용
- `@is_odd`가 `func.call @is_even` 사용
- **순환 참조(cyclic call graph)** 형성
- MLIR verifier가 허용 (심볼이 모두 존재)

### 실행 추적

**is_even(4) 호출:**

```
is_even(4)
  ├─ 4 = 0? → false
  ├─ else 블록: is_odd(4 - 1) = is_odd(3)
  │  ├─ 3 = 0? → false
  │  ├─ else 블록: is_even(3 - 1) = is_even(2)
  │  │  ├─ 2 = 0? → false
  │  │  ├─ else 블록: is_odd(2 - 1) = is_odd(1)
  │  │  │  ├─ 1 = 0? → false
  │  │  │  ├─ else 블록: is_even(1 - 1) = is_even(0)
  │  │  │  │  ├─ 0 = 0? → true
  │  │  │  │  └─ then 블록: return true
  │  │  │  ├─ odd_result = true
  │  │  │  └─ return true
  │  │  ├─ even_result = true
  │  │  └─ return true
  │  ├─ odd_result = true
  │  └─ return true
  ├─ even_result = true
  └─ return true (i1), 확장하여 1 (i32) 반환
```

**호출 스택 깊이:** 5 (is_even → is_odd → is_even → is_odd → is_even)

**상호 재귀의 스택 프레임:**

```
┌──────────────────────┐
│ is_even(0)           │ ← 최대 깊이 (기저 사례)
├──────────────────────┤
│ is_odd(1)            │
├──────────────────────┤
│ is_even(2)           │
├──────────────────────┤
│ is_odd(3)            │
├──────────────────────┤
│ is_even(4)           │ ← 최초 호출
├──────────────────────┤
│ funlang_main         │
└──────────────────────┘
```

스택 프레임이 번갈아가며 생성된다: is_even → is_odd → is_even → ...

### Verifier의 심볼 검증

**MLIR verifier는 모듈 완성 후 실행:**

```fsharp
// 컴파일러 코드
let compileProgram moduleDef funcs mainExpr =
    // 1. 모든 함수 컴파일
    funcs |> List.iter (compileFuncDef builder moduleDef)

    // 2. Main 컴파일
    let mainFunc = compileMain builder mainExpr
    moduleDef.AddFunction(mainFunc)

    // 3. Verify (모든 함수 추가 후)
    if not (mlirOperationVerify(moduleDef.GetOperation())) then
        failwith "Module verification failed"
```

**Verification 과정:**

1. **심볼 수집:** 모듈의 모든 `func.func` 연산에서 심볼 추출 (`@is_even`, `@is_odd`)
2. **심볼 참조 확인:** 각 `func.call` 연산의 callee 확인
   - `func.call @is_odd` → `@is_odd` 심볼이 존재하는가? **예**
   - `func.call @is_even` → `@is_even` 심볼이 존재하는가? **예**
3. **타입 검증:** 호출 타입과 함수 타입 일치 확인
   - `@is_even: (i32) -> i1`
   - `func.call @is_even(%n_minus_1) : (i32) -> i1` → **일치**

**실패 케이스 (존재하지 않는 함수 호출):**

```mlir
func.func @foo(%n: i32) -> i1 {
  %result = func.call @nonexistent(%n) : (i32) -> i1
  //                    ↑ 모듈에 없음
  func.return %result : i1
}
```

Verifier 에러:

```
error: 'func.call' op 'nonexistent' does not reference a valid function
```

**상호 재귀는 통과:** 모든 심볼이 존재하므로 검증 성공.

### FunLang Interpreter와의 차이

**Interpreter에서 상호 재귀:**

```fsharp
// FunLang interpreter (LangTutorial)
let rec eval env ast =
    match ast with
    | LetRec(funcs, body) ->
        // 재귀 환경 생성: 모든 함수를 env에 추가
        let rec_env =
            funcs |> List.fold (fun e (name, func) ->
                e.Add(name, RecursiveClosure(func, rec_env))
            ) env
        eval rec_env body
```

**문제:**
- 환경이 재귀적으로 정의됨 (`rec_env`가 자기 자신을 참조)
- F#의 `let rec` 또는 명시적인 mutation 필요

**Compiler는 더 간단:**

- 모듈 심볼 테이블이 자연스럽게 flat namespace 제공
- 순환 참조를 허용
- Lazy verification으로 정의 순서 무관

## 꼬리 재귀와 꼬리 호출 최적화

### 꼬리 위치 (Tail Position)

**꼬리 위치(tail position)**는 함수에서 **마지막으로 실행되는 표현식의 위치**다.

```fsharp
let rec factorial n =
    if n <= 1 then
        1           // ← 꼬리 위치 (then 분기의 마지막)
    else
        n * factorial (n - 1)
        //  ↑ factorial 호출은 꼬리 위치가 아님!
        //    호출 후 곱셈이 남아있음
```

**꼬리 위치 판단:**
- then 분기의 `1`: **꼬리 위치** (분기의 마지막 값)
- `factorial (n - 1)` 호출: **꼬리 위치 아님** (호출 후 `n *` 곱셈이 실행됨)
- `n * factorial(...)` 전체: **꼬리 위치** (else 분기의 마지막 값)

**일반 규칙:**

함수 본체에서:
- `if` then/else 각 분기의 마지막 표현식: 꼬리 위치
- `let x = ... in <expr>`: `<expr>`이 꼬리 위치
- 함수의 최상위 표현식: 꼬리 위치

**꼬리 호출(tail call):** 꼬리 위치에 있는 함수 호출.

```fsharp
let rec countdown n =
    if n <= 0 then
        0           // ← 꼬리 위치, 값 (호출 아님)
    else
        countdown (n - 1)
        // ↑ 꼬리 위치에 있는 호출 → 꼬리 호출!
```

`countdown (n - 1)`은 else 분기의 마지막이고, 호출 후 추가 작업이 없다. **꼬리 호출**이다.

### 꼬리 호출 최적화 (Tail Call Optimization)

**꼬리 호출 최적화(TCO, Tail Call Optimization)**는 꼬리 호출을 **점프(jump)**로 변환하여 스택 프레임을 재사용하는 최적화다.

**일반 재귀 (TCO 없음):**

```
factorial(5)
  ├─ 스택 프레임 1 생성
  ├─ factorial(4) 호출
  │  ├─ 스택 프레임 2 생성
  │  ├─ factorial(3) 호출
  │  │  └─ ... (스택 누적)
  │  ├─ 반환 후 n * result 계산 ← 추가 작업
  │  └─ 스택 프레임 2 해제
  ├─ 반환 후 n * result 계산
  └─ 스택 프레임 1 해제
```

**꼬리 재귀 (TCO 사용):**

```
countdown(5)
  ├─ 스택 프레임 1 생성
  ├─ countdown(4) 호출 → 점프로 변환!
  │    (스택 프레임 1 재사용, n 값만 업데이트)
  ├─ countdown(3) 호출 → 점프
  ├─ countdown(2) 호출 → 점프
  ├─ countdown(1) 호출 → 점프
  ├─ countdown(0) 호출 → 점프
  └─ 기저 사례: return 0
     (스택 프레임 1 해제)
```

**핵심 차이:**
- **일반 재귀:** 각 호출마다 스택 프레임 생성. 깊이 N → N개 프레임.
- **꼬리 재귀 + TCO:** 단일 스택 프레임 재사용. 깊이 N → 1개 프레임.

**왜 가능한가?**

꼬리 호출은 "호출 후 돌아올 필요가 없다":
- 현재 함수는 호출 결과를 그대로 반환
- 현재 스택 프레임에 남은 작업이 없음
- 따라서 현재 프레임을 버리고, 새 프레임으로 점프할 수 있음

### 꼬리 재귀 변환: Factorial

**일반 재귀 factorial (non-tail):**

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
         // ↑ 호출 후 곱셈 → 꼬리 호출 아님
```

**꼬리 재귀 factorial:**

```fsharp
let rec factorial_tail n acc =
    if n <= 1 then acc
    else factorial_tail (n - 1) (n * acc)
         // ↑ 호출이 마지막 → 꼬리 호출!
```

**차이점:**

| 측면 | 일반 재귀 | 꼬리 재귀 |
|------|----------|----------|
| Accumulator | 없음 | `acc` 파라미터 |
| 곱셈 위치 | 호출 **후** (`n * result`) | 호출 **전** (`n * acc`) |
| 반환값 | 재귀 호출 결과를 변환 | 재귀 호출 결과 그대로 |
| 꼬리 호출 | 아님 | 맞음 |

**Accumulator 패턴:**

꼬리 재귀는 **accumulator**를 사용하여 중간 결과를 전달한다:

```
factorial_tail(5, 1)
  → factorial_tail(4, 5*1=5)
    → factorial_tail(3, 4*5=20)
      → factorial_tail(2, 3*20=60)
        → factorial_tail(1, 2*60=120)
          → return 120
```

**Wrapper 함수:**

사용자는 accumulator를 모르므로, wrapper 함수 제공:

```fsharp
let factorial n =
    factorial_tail n 1
```

### MLIR/LLVM에서 TCO

**LLVM의 꼬리 호출 최적화:**

LLVM은 특정 조건에서 꼬리 호출을 최적화할 수 있다:

1. **함수 속성 (function attribute):** `"tailcc"` calling convention
2. **최적화 플래그:** `-tailcallopt`
3. **타겟 지원:** 플랫폼이 TCO를 지원해야 함 (대부분의 x86-64, ARM은 지원)

**MLIR IR에서 꼬리 호출 표시:**

MLIR `func` 다이얼렉트는 TCO를 명시적으로 표시하는 속성이 없다. 대신:
- LLVM dialect로 낮춘 후 `tail` 속성 추가
- 또는 LLVM 최적화 패스에 의존

**Lowered LLVM IR (꼬리 호출 속성):**

```llvm
define i32 @factorial_tail(i32 %n, i32 %acc) {
entry:
  %cmp = icmp sle i32 %n, 1
  br i1 %cmp, label %base, label %rec

base:
  ret i32 %acc

rec:
  %n_minus_1 = sub i32 %n, 1
  %new_acc = mul i32 %n, %acc
  ; tail 키워드 → TCO 힌트
  %result = tail call i32 @factorial_tail(i32 %n_minus_1, i32 %new_acc)
  ret i32 %result
}
```

**`tail call`의 의미:**
- "이 호출은 꼬리 호출입니다"
- LLVM 최적화 패스가 이를 점프로 변환 가능
- `-tailcallopt` 플래그와 함께 사용

**FunLang Phase 3에서 TCO:**

Phase 3에서는 **TCO를 보장하지 않는다**:
- 교육 목적: 재귀의 기본 개념 먼저 이해
- LLVM이 자동으로 최적화할 **수** 있지만, 보장되지 않음
- Phase 7 (최적화)에서 명시적 TCO 지원 추가 예정

**현재 동작:**

```fsharp
// FunLang 소스
let rec factorial_tail n acc =
    if n <= 1 then acc
    else factorial_tail (n - 1) (n * acc)

// MLIR IR
func.func @factorial_tail(%arg0: i32, %arg1: i32) -> i32 {
  // ... if 조건
  %result = scf.if %cmp -> (i32) {
    scf.yield %arg1 : i32
  } else {
    %n_minus_1 = arith.subi %arg0, %c1 : i32
    %new_acc = arith.muli %arg0, %arg1 : i32
    %rec = func.call @factorial_tail(%n_minus_1, %new_acc) : (i32, i32) -> i32
    // ↑ 일반 func.call (tail 속성 없음)
    scf.yield %rec : i32
  }
  func.return %result : i32
}
```

**LLVM이 최적화할 수 있음 (보장 안 됨):**
- LLVM `-O2` 또는 `-O3` 최적화 레벨
- 일부 경우 자동으로 TCO 적용
- 하지만 C calling convention에서는 보장되지 않음

### TCO 활성화 방법 (Preview)

**Phase 7에서 다룰 내용 (Preview):**

1. **`tailcc` calling convention 사용:**

```llvm
define tailcc i32 @factorial_tail(i32 %n, i32 %acc) {
  ; tailcc = 꼬리 호출 최적화에 특화된 calling convention
  ...
  %result = tail call tailcc i32 @factorial_tail(i32 %n_minus_1, i32 %new_acc)
  ret i32 %result
}
```

2. **Compiler 플래그:**

```bash
llc -tailcallopt factorial.ll -o factorial.s
```

3. **함수 속성:**

MLIR에서 LLVM dialect로 낮출 때 함수 속성 추가:
- `llvm.func @factorial_tail ... attributes { tail = true }`

**현재 (Phase 3):**

- 꼬리 재귀 패턴을 이해
- accumulator 사용법 배우기
- LLVM의 자동 최적화에 의존
- Phase 7에서 명시적 제어 추가

## 코드 생성 업데이트

### compileFuncDef 재사용

**좋은 소식:** 재귀 함수를 위한 특별한 코드 생성이 **필요 없다**.

Chapter 10의 `compileFuncDef`를 그대로 사용:

```fsharp
let compileFuncDef (builder: OpBuilder) (moduleDef: ModuleOp) (FunDef(name, params, body)) =
    // 1. 함수 타입 생성
    let paramTypes = List.replicate params.Length builder.GetI32Type()
    let returnType = builder.GetI32Type()
    let funcType = builder.GetFunctionType(paramTypes, returnType)

    // 2. func.func 생성
    let funcOp = builder.CreateFuncOp(name, funcType)

    // 3. Entry block에서 파라미터 가져오기
    let entryBlock = funcOp.GetEntryBlock()
    builder.SetInsertionPointToEnd(entryBlock)

    let env =
        params
        |> List.mapi (fun i paramName ->
            let argValue = entryBlock.GetArgument(i)
            (paramName, argValue))
        |> Map.ofList

    // 4. 본체 컴파일
    let bodyValue = compileExpr builder env body

    // 5. 반환
    builder.CreateFuncReturn(bodyValue)

    // 6. 모듈에 추가
    moduleDef.AddFunction(funcOp)
```

**재귀 호출은 자동으로 처리:**

`compileExpr`의 `App` case:

```fsharp
| App(funcName, args) ->
    let argValues = args |> List.map (compileExpr builder env)
    builder.CreateFuncCall(funcName, argValues)
```

- `App("factorial", [Num 5])` → `func.call @factorial(%c5) : (i32) -> i32`
- `App("factorial", [BinOp(...)])` → `func.call @factorial(%n_minus_1) : (i32) -> i32`

**자기 참조가 자연스럽게 작동:**
- `@factorial` 심볼이 모듈에 이미 존재 (본체 컴파일 중이지만 함수 자체는 이미 추가됨)
- `CreateFuncCall`이 심볼 참조 생성
- Verifier가 나중에 확인

### 상호 재귀 처리

**상호 재귀도 특별한 처리 불필요:**

```fsharp
let compileProgram (builder: OpBuilder) (moduleDef: ModuleOp) (Program(funcs, mainExpr)) =
    // 모든 함수 컴파일
    funcs |> List.iter (compileFuncDef builder moduleDef)

    // Main 표현식 컴파일
    // ...
```

**순서:**

1. `is_even` 컴파일:
   - `func.func @is_even` 생성, 모듈에 추가
   - 본체에서 `func.call @is_odd` 생성 (아직 `@is_odd` 없음 - OK!)

2. `is_odd` 컴파일:
   - `func.func @is_odd` 생성, 모듈에 추가
   - 본체에서 `func.call @is_even` 생성 (`@is_even` 이미 존재)

3. Verification:
   - `@is_even`의 `func.call @is_odd` → `@is_odd` 존재 확인 ✓
   - `@is_odd`의 `func.call @is_even` → `@is_even` 존재 확인 ✓

**핵심:** MLIR의 lazy verification 덕분에 순서 무관.

### compileProgram 전체 구조

**다중 함수 + Main 표현식:**

```fsharp
let compileProgram (builder: OpBuilder) (moduleDef: ModuleOp) (Program(funcs, mainExpr)) =
    // 1. 모든 함수 정의 컴파일
    funcs |> List.iter (fun funcDef ->
        compileFuncDef builder moduleDef funcDef
    )

    // 2. Main 함수 생성
    let mainFuncType = builder.GetFunctionType([], builder.GetI32Type())
    let mainFunc = builder.CreateFuncOp("funlang_main", mainFuncType)

    let mainBlock = mainFunc.GetEntryBlock()
    builder.SetInsertionPointToEnd(mainBlock)

    // 3. Main 표현식 컴파일
    let mainValue = compileExpr builder Map.empty mainExpr

    // 4. Main 반환
    builder.CreateFuncReturn(mainValue)

    moduleDef.AddFunction(mainFunc)

    // 5. Verification
    if not (mlirOperationVerify(moduleDef.GetOperation())) then
        failwith "Module verification failed"

    moduleDef
```

**프로그램 구조:**

```fsharp
Program([
    FunDef("factorial", ["n"], <body>),
    FunDef("fibonacci", ["n"], <body>),
    FunDef("is_even", ["n"], <body>),
    FunDef("is_odd", ["n"], <body>)
], App("factorial", [Num 5]))
```

**생성된 MLIR:**

```mlir
module {
  func.func @factorial(%arg0: i32) -> i32 { ... }
  func.func @fibonacci(%arg0: i32) -> i32 { ... }
  func.func @is_even(%arg0: i32) -> i1 { ... }
  func.func @is_odd(%arg0: i32) -> i1 { ... }

  func.func @funlang_main() -> i32 {
    %c5 = arith.constant 5 : i32
    %result = func.call @factorial(%c5) : (i32) -> i32
    func.return %result : i32
  }
}
```

## 완전한 예시: 여러 재귀 함수

### 프로그램 소스

```fsharp
// 함수 정의들
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

let rec fibonacci n =
    if n <= 1 then n
    else fibonacci (n - 1) + fibonacci (n - 2)

let rec is_even n =
    if n = 0 then true
    else is_odd (n - 1)

let rec is_odd n =
    if n = 0 then false
    else is_even (n - 1)

// Main 표현식
let result_fact = factorial 5 in
let result_fib = fibonacci 6 in
let result_even = is_even 4 in
result_fact + result_fib + result_even
```

### AST 표현 (간략)

```fsharp
Program([
    FunDef("factorial", ["n"], <factorial_body>),
    FunDef("fibonacci", ["n"], <fibonacci_body>),
    FunDef("is_even", ["n"], <is_even_body>),
    FunDef("is_odd", ["n"], <is_odd_body>)
],
Let("result_fact", App("factorial", [Num 5]),
Let("result_fib", App("fibonacci", [Num 6]),
Let("result_even", App("is_even", [Num 4]),
BinOp(
    BinOp(Var "result_fact", Add, Var "result_fib"),
    Add,
    Var "result_even"
)))))
```

### 생성된 MLIR IR (전체)

```mlir
module {
  // factorial 함수
  func.func @factorial(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi sle, %arg0, %c1 : i32
    %result = scf.if %cmp -> (i32) {
      scf.yield %c1 : i32
    } else {
      %n_minus_1 = arith.subi %arg0, %c1 : i32
      %rec = func.call @factorial(%n_minus_1) : (i32) -> i32
      %product = arith.muli %arg0, %rec : i32
      scf.yield %product : i32
    }
    func.return %result : i32
  }

  // fibonacci 함수
  func.func @fibonacci(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi sle, %arg0, %c1 : i32
    %result = scf.if %cmp -> (i32) {
      scf.yield %arg0 : i32
    } else {
      %n_minus_1 = arith.subi %arg0, %c1 : i32
      %fib_n_1 = func.call @fibonacci(%n_minus_1) : (i32) -> i32
      %c2 = arith.constant 2 : i32
      %n_minus_2 = arith.subi %arg0, %c2 : i32
      %fib_n_2 = func.call @fibonacci(%n_minus_2) : (i32) -> i32
      %sum = arith.addi %fib_n_1, %fib_n_2 : i32
      scf.yield %sum : i32
    }
    func.return %result : i32
  }

  // is_even 함수
  func.func @is_even(%arg0: i32) -> i1 {
    %c0 = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %arg0, %c0 : i32
    %result = scf.if %is_zero -> (i1) {
      %true = arith.constant 1 : i1
      scf.yield %true : i1
    } else {
      %c1 = arith.constant 1 : i32
      %n_minus_1 = arith.subi %arg0, %c1 : i32
      %odd_result = func.call @is_odd(%n_minus_1) : (i32) -> i1
      scf.yield %odd_result : i1
    }
    func.return %result : i1
  }

  // is_odd 함수
  func.func @is_odd(%arg0: i32) -> i1 {
    %c0 = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %arg0, %c0 : i32
    %result = scf.if %is_zero -> (i1) {
      %false = arith.constant 0 : i1
      scf.yield %false : i1
    } else {
      %c1 = arith.constant 1 : i32
      %n_minus_1 = arith.subi %arg0, %c1 : i32
      %even_result = func.call @is_even(%n_minus_1) : (i32) -> i1
      scf.yield %even_result : i1
    }
    func.return %result : i1
  }

  // Main 함수
  func.func @funlang_main() -> i32 {
    // result_fact = factorial(5)
    %c5 = arith.constant 5 : i32
    %result_fact = func.call @factorial(%c5) : (i32) -> i32

    // result_fib = fibonacci(6)
    %c6 = arith.constant 6 : i32
    %result_fib = func.call @fibonacci(%c6) : (i32) -> i32

    // result_even = is_even(4)
    %c4 = arith.constant 4 : i32
    %result_even_i1 = func.call @is_even(%c4) : (i32) -> i1
    %result_even = arith.extui %result_even_i1 : i1 to i32

    // result_fact + result_fib + result_even
    %sum1 = arith.addi %result_fact, %result_fib : i32
    %sum2 = arith.addi %sum1, %result_even : i32

    func.return %sum2 : i32
  }
}
```

### 컴파일 및 실행

```bash
# 1. MLIR 파일 저장
echo "<위 MLIR IR>" > recursion_example.mlir

# 2. Lowering passes 적용
mlir-opt \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  recursion_example.mlir \
  -o lowered.mlir

# 3. LLVM IR로 변환
mlir-translate --mlir-to-llvmir lowered.mlir -o recursion.ll

# 4. Object file 생성
llc -filetype=obj recursion.ll -o recursion.o

# 5. Runtime과 링크
gcc -o recursion recursion.o runtime.o -lgc

# 6. 실행
./recursion
# 출력: 129
# (factorial(5)=120, fibonacci(6)=8, is_even(4)=1, 120+8+1=129)
```

### 실행 결과 분석

**계산 과정:**

1. `factorial(5)` = 120
2. `fibonacci(6)` = 8
3. `is_even(4)` = true = 1 (i32로 확장)
4. 120 + 8 + 1 = 129

**스택 사용:**

- `factorial(5)`: 최대 5개 스택 프레임
- `fibonacci(6)`: 최대 6개 스택 프레임 (하지만 호출 트리가 넓음)
- `is_even(4)`: 최대 5개 스택 프레임 (is_even/is_odd 번갈아가며)

**총 호출 횟수:**

- `factorial(5)`: 5번
- `fibonacci(6)`: 25번 (지수 복잡도!)
- `is_even(4)` + `is_odd`: 5번

## 성능 고려사항

### 재귀 vs 반복 성능

**재귀의 오버헤드:**

1. **함수 호출 비용:**
   - 스택 프레임 생성/소멸
   - 레지스터 저장/복원
   - 점프 instruction (call/ret)

2. **스택 메모리 사용:**
   - 깊이 N → N개 스택 프레임
   - 각 프레임 ~64-128 bytes
   - 캐시 미스 가능성

3. **분기 예측:**
   - 재귀 호출은 간접 분기
   - CPU 분기 예측기가 학습하기 어려움

**반복(Loop)의 이점:**

1. **함수 호출 없음:**
   - 단일 스택 프레임
   - 레지스터 할당 효율적

2. **명령어 수 감소:**
   - 직접 점프 (conditional branch)
   - 예측 가능한 패턴

3. **메모리 효율:**
   - 스택 사용 최소

**언제 재귀가 괜찮은가?**

1. **얕은 재귀 (shallow recursion):**
   - 깊이 < 100: 성능 차이 미미
   - 예: 균형 트리 탐색 (깊이 ~log N)

2. **꼬리 재귀 + TCO:**
   - 컴파일러가 loop으로 변환
   - 성능이 반복과 동일

3. **알고리즘이 본질적으로 재귀적:**
   - 트리 순회, 퀵소트, 병합정렬
   - 재귀로 작성하는 것이 자연스럽고 명확

**언제 재귀를 피해야 하는가?**

1. **깊은 재귀 (deep recursion):**
   - 깊이 > 10,000: 스택 오버플로우 위험
   - 예: naive fibonacci

2. **중복 계산:**
   - Fibonacci 같은 지수 복잡도
   - Memoization 또는 DP로 해결

3. **성능이 중요한 경우:**
   - 내부 루프 (hot path)
   - 반복으로 작성 또는 TCO 보장

### 스택 프레임 오버헤드

**스택 프레임 구조 (x86-64):**

```
┌──────────────────────┐
│ Return address       │ 8 bytes
├──────────────────────┤
│ Saved rbp (frame ptr)│ 8 bytes
├──────────────────────┤
│ Local variables      │ Variable
├──────────────────────┤
│ Saved registers      │ Variable (callee-saved)
├──────────────────────┤
│ Padding (alignment)  │ 0-15 bytes (16-byte align)
└──────────────────────┘
```

**최소 크기:** ~16 bytes (return address + rbp)
**일반적 크기:** 64-128 bytes (지역 변수, 레지스터 저장 포함)

**호출 비용:**

- `call` instruction: ~1-2 CPU cycles (분기 예측 성공 시)
- 스택 프레임 setup: ~5-10 instructions (push rbp, mov, sub)
- 스택 프레임 teardown: ~5-10 instructions (mov, pop, ret)
- **총:** ~20-30 instructions per call

**비교 (factorial 1000):**

- **재귀:** 1,000 함수 호출 × 30 instructions = 30,000 instructions
- **반복:** ~5 instructions per iteration × 1,000 = 5,000 instructions

**6배 차이!** 하지만 절대 시간은 여전히 작음 (~수 마이크로초).

### LLVM 최적화 기회

**LLVM이 재귀에 적용하는 최적화:**

1. **Tail Call Elimination (TCO):**
   - 꼬리 재귀 → loop 변환
   - 스택 사용 O(1)

2. **Inlining:**
   - 작은 재귀 함수를 호출 사이트에 인라인
   - 함수 호출 오버헤드 제거

3. **Constant Folding:**
   - 컴파일 타임에 계산 가능한 재귀 (예: `factorial(5)`) → 상수 120

4. **Loop Optimization:**
   - 재귀를 loop으로 변환 후 loop unrolling, vectorization 적용

**예시 (LLVM -O3):**

```fsharp
// 소스
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

let result = factorial 5
```

**LLVM -O3 최적화 후:**

```llvm
define i32 @funlang_main() {
  ret i32 120  ; 컴파일 타임에 계산됨!
}
```

**변수 입력 (factorial n, n이 런타임 값):**

LLVM은 재귀를 그대로 유지하지만, 레지스터 할당과 분기 예측을 최적화.

### Phase 7 최적화 Preview

**Phase 7에서 다룰 내용:**

1. **명시적 TCO 지원:**
   - `tailcc` calling convention
   - 꼬리 재귀 자동 감지 및 변환

2. **Inlining 제어:**
   - 작은 함수 자동 인라인
   - `inline` 힌트

3. **Memoization:**
   - 함수 결과 캐싱 (fibonacci 최적화)

4. **Loop 변환:**
   - 재귀 → 반복 자동 변환 (특정 패턴)

**현재 (Phase 3):**

- 재귀의 기본 개념과 제약 이해
- 성능 트레이드오프 인지
- LLVM의 기본 최적화에 의존

## 일반적인 오류

### Error 1: 무한 재귀 (기저 사례 누락)

**문제:**

```fsharp
let rec infinite_loop n =
    infinite_loop (n - 1)
    // 기저 사례가 없음!
```

**증상:**

```bash
./program
Segmentation fault (core dumped)
```

**원인:**

- 재귀 종료 조건이 없음
- 스택이 무한히 성장
- 스택 오버플로우

**해결:**

```fsharp
let rec countdown n =
    if n <= 0 then 0  // ← 기저 사례 추가
    else countdown (n - 1)
```

**디버깅 팁:**

- 모든 재귀 함수에 기저 사례가 있는지 확인
- "언제 재귀가 멈추는가?" 질문

### Error 2: 스택 오버플로우 (깊은 재귀)

**문제:**

```fsharp
let rec sum_to n =
    if n <= 0 then 0
    else n + sum_to (n - 1)

let result = sum_to 100000  // 깊이 100,000!
```

**증상:**

```bash
./program
Segmentation fault (core dumped)
```

**원인:**

- 재귀 깊이 > 스택 크기
- 100,000 프레임 × 64 bytes = 6.4 MB > 일부 플랫폼 한계

**해결:**

1. **꼬리 재귀로 변환:**

```fsharp
let rec sum_to_tail n acc =
    if n <= 0 then acc
    else sum_to_tail (n - 1) (n + acc)

let sum_to n = sum_to_tail n 0
```

2. **입력 크기 제한:**

```fsharp
if n > 10000 then
    failwith "Input too large"
else
    sum_to n
```

3. **반복으로 변환:**

```fsharp
// FunLang은 loop 없지만, LLVM이 TCO로 변환 가능
```

### Error 3: 심볼을 찾을 수 없음 (타이포)

**문제:**

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorail (n - 1)  // typo: factorail
```

**증상 (MLIR verification):**

```
error: 'func.call' op 'factorail' does not reference a valid function
```

**원인:**

- 함수 이름 오타
- 심볼 `@factorail`이 모듈에 없음

**해결:**

- 함수 이름 철자 확인
- IDE의 자동완성 사용

### Error 4: 인자 순서 오류 (상호 재귀)

**문제:**

```fsharp
let rec is_even n =
    if n = 0 then true
    else is_odd n  // ← (n - 1) 빠뜨림!

let rec is_odd n =
    if n = 0 then false
    else is_even n  // ← 똑같은 오류
```

**증상:**

```bash
./program
Segmentation fault (core dumped)
```

**원인:**

- 무한 재귀: is_even(4) → is_odd(4) → is_even(4) → ...
- 인자가 감소하지 않음

**해결:**

```fsharp
let rec is_even n =
    if n = 0 then true
    else is_odd (n - 1)  // ← (n - 1) 추가

let rec is_odd n =
    if n = 0 then false
    else is_even (n - 1)
```

### Error 5: 꼬리 위치가 아닌 곳에서 TCO 기대

**문제:**

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
    //       ↑ 꼬리 위치 아님 (곱셈 후 실행)

// TCO가 적용될 것으로 기대하지만, 실제로는 안 됨
```

**증상:**

- 깊은 재귀에서 스택 오버플로우
- TCO가 적용되지 않음

**원인:**

- 재귀 호출 후 추가 작업 (`n *`)
- 꼬리 호출이 아님

**해결:**

```fsharp
// accumulator 패턴으로 변환
let rec factorial_tail n acc =
    if n <= 1 then acc
    else factorial_tail (n - 1) (n * acc)
    //   ↑ 꼬리 위치! (호출이 마지막)
```

### 디버깅 팁

1. **Print 디버깅:**

```fsharp
let rec factorial n =
    // 디버깅: 함수 호출 출력
    print_int n;
    if n <= 1 then 1
    else n * factorial (n - 1)
```

2. **기저 사례 먼저 확인:**

재귀 함수를 작성할 때:
- 먼저 기저 사례 작성
- 그 다음 재귀 사례 작성

3. **작은 입력으로 테스트:**

```fsharp
// factorial 100000 전에 factorial 5 먼저 테스트
```

4. **스택 크기 늘리기 (임시 해결):**

```bash
# Linux에서 스택 크기 늘리기
ulimit -s 16384  # 16 MB
./program
```

5. **MLIR IR 검증:**

```bash
mlir-opt --verify-diagnostics program.mlir
```

## 요약 및 Phase 3 완료

### Chapter 11 요약

**배운 내용:**

1. **재귀의 기본:**
   - 자기 자신을 호출하는 함수
   - 기저 사례 + 재귀 사례
   - 예시: factorial, fibonacci

2. **MLIR에서 재귀:**
   - 모듈 레벨 심볼 테이블
   - 자기 참조 (`func.call @factorial` inside `@factorial`)
   - 심볼 확인은 컴파일 타임, 호출은 런타임

3. **상호 재귀:**
   - 두 함수가 서로 호출 (is_even, is_odd)
   - Forward declaration 불필요
   - Flat symbol namespace 덕분에 자연스럽게 작동

4. **스택 프레임:**
   - 각 재귀 호출마다 스택 프레임 생성
   - 깊이 N → N개 프레임
   - 스택 크기 제한 (8 MB Linux, 1 MB Windows)

5. **꼬리 호출 최적화:**
   - 꼬리 위치 = 함수의 마지막 표현식
   - 꼬리 호출 = 꼬리 위치의 함수 호출
   - TCO = 꼬리 호출을 점프로 변환, 스택 재사용
   - Accumulator 패턴으로 꼬리 재귀 변환

6. **성능:**
   - 재귀는 오버헤드 있음 (함수 호출, 스택)
   - 얕은 재귀는 괜찮음
   - 깊은 재귀는 TCO 필요
   - LLVM 최적화 활용

7. **일반적인 오류:**
   - 무한 재귀 (기저 사례 누락)
   - 스택 오버플로우 (깊은 재귀)
   - 타이포 (심볼 참조 실패)
   - 인자 오류 (상호 재귀)
   - 꼬리 위치 오해

### Phase 3 완료!

**Phase 3 목표:**

- [x] 최상위 명명된 함수 (Chapter 10)
- [x] 함수 파라미터와 호출 (Chapter 10)
- [x] 재귀 함수 (Chapter 11)
- [x] 상호 재귀 (Chapter 11)
- [x] 스택 프레임 관리 (Chapter 11)
- [x] 꼬리 호출 최적화 개념 (Chapter 11)

**Phase 3에서 구축한 것:**

1. **func 다이얼렉트 통합:**
   - `func.func`, `func.call`, `func.return` 연산
   - P/Invoke 바인딩 및 OpBuilder 메서드

2. **함수 컴파일 인프라:**
   - `compileFuncDef`: AST → func.func
   - `compileProgram`: 다중 함수 + main
   - 환경 관리 (파라미터를 block arguments로)

3. **재귀 지원:**
   - 자기 참조 (심볼 테이블)
   - 상호 재귀 (lazy verification)
   - 스택 기반 실행 모델

4. **Calling convention:**
   - C calling convention (System V ABI)
   - LLVM의 자동 프롤로그/에필로그 생성

**Phase 3에서 제외된 것 (Phase 4로 연기):**

- **클로저:** 환경을 캡처하는 함수
- **고차 함수:** 함수를 인자로 받거나 반환
- **익명 함수:** Lambda 표현식
- **힙 할당:** 클로저 환경 (GC_malloc 사용)

### Phase 4 Preview

**Phase 4: 클로저와 고차 함수**

**목표:**

1. **Lambda 표현식:**
   ```fsharp
   let add_n n = fun x -> x + n
   ```

2. **환경 캡처:**
   ```fsharp
   let make_counter () =
       let count = ref 0 in
       fun () -> (count := !count + 1; !count)
   ```

3. **고차 함수:**
   ```fsharp
   let map f list = ...
   let result = map (fun x -> x * 2) [1; 2; 3]
   ```

4. **클로저 변환:**
   - 자유 변수 분석
   - 환경을 힙에 할당 (GC_malloc)
   - 클로저 = (function pointer, environment pointer)

5. **Heap 사용:**
   - Chapter 9 (Boehm GC) 활성화
   - memref 다이얼렉트 (alloc, load, store)

**연결:**

- Phase 3: 스택 기반 함수 (파라미터만 사용)
- Phase 4: 힙 기반 클로저 (파라미터 + 캡처된 환경)

### 다음 단계

**완성된 컴파일러 능력:**

Phase 3 완료 후 FunLang 컴파일러는 다음을 지원한다:

- [x] 산술 및 비교 연산 (Chapter 06)
- [x] Let 바인딩과 변수 (Chapter 07)
- [x] If/then/else 제어 흐름 (Chapter 08)
- [x] 메모리 관리 (Boehm GC 통합, Chapter 09)
- [x] 함수 정의 및 호출 (Chapter 10)
- [x] 재귀 및 상호 재귀 (Chapter 11)

**아직 지원하지 않는 것:**

- [ ] 클로저 및 lambda
- [ ] 고차 함수
- [ ] 패턴 매칭
- [ ] 대수적 데이터 타입 (ADT)
- [ ] 리스트, 튜플 등 데이터 구조
- [ ] 타입 시스템 (현재 모두 i32)

**학습 경로:**

```
Phase 1 (Foundation): MLIR 기초, P/Invoke
  ↓
Phase 2 (Core Language): 표현식, 제어 흐름, 메모리
  ↓
Phase 3 (Functions): 함수, 재귀, 스택 ← 현재 위치
  ↓
Phase 4 (Closures): 클로저, 고차 함수, 힙
  ↓
Phase 5 (Data Structures): 리스트, 튜플, ADT
  ↓
Phase 6 (Type System): 타입 추론, 다형성
  ↓
Phase 7 (Optimization): 인라인, TCO, 최적화 패스
```

**축하합니다!** Phase 3를 완료했습니다. FunLang 컴파일러는 이제 재귀 함수를 포함한 완전한 프로그램을 네이티브 코드로 컴파일할 수 있습니다.

**다음 장 (Phase 4)에서:** 클로저와 환경 캡처를 추가하여 진정한 함수형 프로그래밍 기능을 구현할 것입니다.

