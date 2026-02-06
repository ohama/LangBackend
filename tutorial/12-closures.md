# Chapter 12: 클로저 (Closures)

## 소개

**클로저(closure)**는 함수형 프로그래밍의 핵심 기능이다. 클로저는 단순한 함수가 아니라, **함수 + 환경(environment)**의 조합이다.

```fsharp
// Phase 3 함수 - 외부 변수 사용 불가
let add x y = x + y

// Phase 4 클로저 - 외부 변수 캡처 가능
let make_adder n =
    fun x -> x + n   // n을 캡처!
```

`fun x -> x + n`은 **클로저**다:
- `x`는 **파라미터** (bound variable)
- `n`은 **캡처된 변수** (free variable, captured from environment)

클로저가 생성될 때, `n`의 값이 **환경(environment)**에 저장된다. 나중에 클로저가 호출되면, 저장된 `n` 값을 사용한다.

**왜 클로저가 중요한가?**

1. **고차 함수(Higher-order functions)의 기초**: 함수를 반환하거나 인자로 전달하려면 클로저가 필요하다
2. **상태 캡처**: 함수 생성 시점의 환경을 저장할 수 있다
3. **추상화**: 공통 패턴을 클로저로 추상화할 수 있다 (map, filter, fold)

**클로저 vs Phase 3 함수:**

| Phase 3 함수 | Phase 4 클로저 |
|-------------|--------------|
| 이름이 있다 (named) | 익명 가능 (anonymous) |
| 외부 변수 사용 불가 | 외부 변수 캡처 가능 |
| `func.func` 연산 | 함수 포인터 + 환경 |
| 정적 바인딩 | 환경 저장 필요 |

**Chapter 12의 범위:**

이 장에서 다루는 것:
1. **클로저 이론**: 자유 변수(free variables), 바운드 변수(bound variables)
2. **자유 변수 분석**: 어떤 변수를 캡처해야 하는지 계산
3. **클로저 변환(Closure conversion)**: 암묵적 캡처를 명시적으로 만들기
4. **환경 구조체**: 캡처된 변수를 저장하는 힙 객체
5. **클로저 생성 코드**: GC_malloc으로 환경 할당하기

이 장을 마치면:
- 클로저가 무엇이고 왜 필요한지 이해한다
- 자유 변수 분석 알고리즘을 구현할 수 있다
- 클로저를 (함수 포인터, 환경 포인터) 쌍으로 표현할 수 있다
- 환경을 힙에 할당하고 변수를 저장/로드할 수 있다
- GC_malloc을 사용해 환경을 생성할 수 있다

> **Preview:** Chapter 13에서는 고차 함수 (map, filter)를 추가한다. Chapter 12는 클로저의 기초를 확립한다.

## 클로저 이론

### Lexical Scoping vs Dynamic Scoping

클로저를 이해하려면 먼저 **스코핑(scoping)** 개념을 알아야 한다. 변수의 값이 어떻게 결정되는가?

**Lexical scoping (정적 스코핑):**

변수는 **코드 작성 시점의 위치**로 결정된다.

```fsharp
let x = 10 in
let f = fun y -> x + y in
let x = 20 in
f 5   // 결과: 15 (x = 10 사용)
```

`fun y -> x + y`에서 `x`는 **정의 시점의 `x`** (10)을 참조한다. 나중에 `x`를 20으로 재바인딩해도 영향 없다.

**Dynamic scoping (동적 스코핑):**

변수는 **호출 시점의 환경**에서 찾는다.

```fsharp
// 동적 스코핑 가상 예시 (FunLang은 지원 안 함)
let x = 10 in
let f = fun y -> x + y in
let x = 20 in
f 5   // 결과: 25 (x = 20 사용)
```

`f`를 호출할 때, `x`는 **호출 시점의 환경**에서 찾는다 (20).

**FunLang은 lexical scoping을 사용한다.** 대부분의 현대 언어가 그렇다 (F#, JavaScript, Python, Rust, etc.). Dynamic scoping은 혼란스럽고 디버깅이 어렵다.

**Lexical scoping의 의미:**

- 함수가 정의될 때, **그 시점의 환경을 기억해야 한다**
- 함수가 호출될 때, **저장된 환경을 사용해야 한다**
- 이것이 **클로저**다: function + environment

### Free Variables vs Bound Variables

변수는 두 가지로 분류된다:

**Bound variable (바운드 변수):**

함수의 파라미터이거나, let 바인딩으로 정의된 변수.

```fsharp
fun x -> x + 1
//  ↑   ↑
//  바인딩  사용
```

`x`는 **bound variable**이다. `fun x`가 `x`를 바인딩한다.

**Free variable (자유 변수):**

함수 내부에서 사용되지만, 그 함수에서 바인딩되지 않은 변수.

```fsharp
fun x -> x + y
//           ↑
//       자유 변수!
```

`y`는 **free variable**이다. `fun x`는 `y`를 바인딩하지 않는다. `y`는 **외부 환경에서 와야 한다**.

**예시 1: 자유 변수 없음**

```fsharp
fun x -> x + 1
```

- Bound: {x}
- Free: {} (empty)

**예시 2: 자유 변수 하나**

```fsharp
fun x -> x + y
```

- Bound: {x}
- Free: {y}

**예시 3: 중첩된 람다**

```fsharp
fun x -> fun y -> x + y + z
```

내부 람다 `fun y -> x + y + z`를 보면:
- Bound: {y}
- Free: {x, z}

외부 람다 `fun x -> ...`를 보면:
- Bound: {x}
- Free: {z}

**전체 표현식의 자유 변수: {z}**

**예시 4: Let 바인딩**

```fsharp
let a = 10 in
fun x -> a + x
```

`fun x -> a + x`:
- Bound: {x}
- Free: {a}

하지만 전체 표현식은 `let a = 10`이 `a`를 바인딩하므로:
- 전체 자유 변수: {} (empty)

### 환경 캡처 (Environment Capture)

자유 변수가 있으면, 그 값을 어디선가 가져와야 한다. 클로저는 **환경(environment)**을 저장해서 해결한다.

**환경이란?**

환경은 **변수 이름 → 값**의 매핑이다.

```fsharp
let x = 10 in
let y = 20 in
fun z -> x + y + z
```

`fun z -> x + y + z`가 생성될 때:
- 자유 변수: {x, y}
- 환경에서 찾기: x = 10, y = 20
- **환경 캡처**: {x: 10, y: 20}을 저장

클로저는 **(함수 포인터, 환경 포인터)** 쌍이 된다:

```
Closure {
    fn_ptr: @lambda_123,
    env: { x: 10, y: 20 }
}
```

나중에 클로저를 호출할 때:
1. 함수 포인터를 찾는다 (@lambda_123)
2. 환경을 함수에 전달한다 ({x: 10, y: 20})
3. 함수는 환경에서 x, y 값을 로드한다
4. 계산: 10 + 20 + z

**Value capture vs Reference capture:**

FunLang은 **value capture**를 사용한다. 변수의 **현재 값**을 복사해서 저장한다.

```fsharp
let x = 10 in
let f = fun y -> x + y in
let x = 20 in   // x 재바인딩
f 5   // 결과: 15 (캡처된 x = 10 사용)
```

클로저가 생성될 때 `x = 10`이 환경에 복사된다. 나중에 `x`가 재바인딩되어도 영향 없다.

(참조 캡처는 C++의 `[&x]` 같은 개념인데, FunLang은 순수 함수형이므로 지원 안 함)

### 클로저의 구조

클로저는 **두 개의 포인터**로 표현된다:

```c
// C 스타일 표현
struct Closure {
    void* fn_ptr;      // 함수 코드 포인터
    void* env_ptr;     // 환경 데이터 포인터
};
```

**1. 함수 포인터 (fn_ptr):**

실행할 코드의 주소. MLIR에서는 `@lambda_N` 심볼.

**2. 환경 포인터 (env_ptr):**

캡처된 변수들을 저장한 힙 객체. 구조체의 주소.

**시각적 다이어그램:**

```
클로저 생성:
  let x = 10 in
  let y = 20 in
  fun z -> x + y + z

메모리 레이아웃:
┌─────────────────────┐
│ Closure (스택/레지스터) │
├─────────────────────┤
│ fn_ptr: @lambda_0   │───┐
│ env_ptr: 0x1a3b5c8  │───┼───────┐
└─────────────────────┘   │       │
                          │       │
                          │       v
                          │  ┌──────────────┐
                          │  │ Environment  │
                          │  │ (힙 할당)     │
                          │  ├──────────────┤
                          │  │ x: 10        │
                          │  │ y: 20        │
                          │  └──────────────┘
                          │
                          v
                    @lambda_0 코드:
                      ; env를 파라미터로 받음
                      ; env[0]에서 x 로드
                      ; env[1]에서 y 로드
                      ; x + y + z 계산
```

**핵심:**
- 클로저는 작은 객체 (포인터 2개)
- 환경은 힙에 할당 (크기는 캡처된 변수 개수에 따라 다름)
- 함수는 환경을 첫 번째 파라미터로 받음

## 자유 변수 분석 (Free Variable Analysis)

클로저를 컴파일하려면, **어떤 변수를 캡처해야 하는지** 알아야 한다. 이것이 **자유 변수 분석(free variable analysis)**이다.

### 분석 알고리즘

자유 변수를 찾는 알고리즘은 **set-based traversal**이다:

1. AST를 재귀적으로 순회
2. 각 표현식에서 자유 변수 set을 계산
3. 바운드 변수는 자유 변수 set에서 제거

**정의:**

```
FV(expr) = 표현식 expr의 자유 변수 집합
BV(expr) = 표현식 expr에서 바인딩되는 변수 집합
```

**규칙:**

| Expression | Free Variables | Bound Variables |
|------------|----------------|-----------------|
| `Var(x)` | {x} | {} |
| `Num(n)` | {} | {} |
| `Add(e1, e2)` | FV(e1) ∪ FV(e2) | {} |
| `Let(x, e1, e2)` | FV(e1) ∪ (FV(e2) - {x}) | {x} |
| `Lambda(x, body)` | FV(body) - {x} | {x} |
| `App(f, arg)` | FV(f) ∪ FV(arg) | {} |
| `If(cond, t, f)` | FV(cond) ∪ FV(t) ∪ FV(f) | {} |

**핵심 규칙 설명:**

**1. Var(x):**

변수 사용은 자유 변수다 (아직 바인딩 확인 안 함).

```
FV(x) = {x}
```

**2. Lambda(x, body):**

람다가 `x`를 바인딩하므로, body의 자유 변수에서 `x`를 제거.

```
FV(fun x -> body) = FV(body) - {x}
```

**3. Let(x, e1, e2):**

`e1`의 자유 변수 + (`e2`의 자유 변수 - {x})

```
FV(let x = e1 in e2) = FV(e1) ∪ (FV(e2) - {x})
```

**4. 기타 연산:**

자식 표현식들의 자유 변수를 합집합.

```
FV(e1 + e2) = FV(e1) ∪ FV(e2)
```

### F# 구현

```fsharp
// AST 정의 (간략화)
type Expr =
    | Var of string
    | Num of int
    | Add of Expr * Expr
    | Sub of Expr * Expr
    | Let of string * Expr * Expr
    | Lambda of string * Expr
    | App of Expr * Expr
    | If of Expr * Expr * Expr

// 자유 변수 분석
let rec freeVars (expr: Expr) : Set<string> =
    match expr with
    | Var(x) ->
        // 변수 사용 = 자유 변수 후보
        Set.singleton x

    | Num(_) ->
        // 리터럴 = 자유 변수 없음
        Set.empty

    | Add(e1, e2)
    | Sub(e1, e2) ->
        // 이항 연산 = 양쪽의 자유 변수 합
        Set.union (freeVars e1) (freeVars e2)

    | Let(x, e1, e2) ->
        // let x = e1 in e2
        // e1의 자유 변수 + (e2의 자유 변수 - {x})
        let fv1 = freeVars e1
        let fv2 = freeVars e2
        Set.union fv1 (Set.remove x fv2)

    | Lambda(param, body) ->
        // fun param -> body
        // body의 자유 변수 - {param}
        let fvBody = freeVars body
        Set.remove param fvBody

    | App(func, arg) ->
        // f arg
        // f의 자유 변수 + arg의 자유 변수
        Set.union (freeVars func) (freeVars arg)

    | If(cond, thenExpr, elseExpr) ->
        // if cond then thenExpr else elseExpr
        // 세 부분의 자유 변수 합
        freeVars cond
        |> Set.union (freeVars thenExpr)
        |> Set.union (freeVars elseExpr)
```

### 예시 분석

**예시 1: 단순 람다**

```fsharp
fun x -> x + 1
```

분석:
```
FV(fun x -> x + 1)
= FV(x + 1) - {x}
= (FV(x) ∪ FV(1)) - {x}
= ({x} ∪ {}) - {x}
= {} (empty)
```

**결과: 자유 변수 없음**

**예시 2: 하나의 자유 변수**

```fsharp
fun x -> x + y
```

분석:
```
FV(fun x -> x + y)
= FV(x + y) - {x}
= (FV(x) ∪ FV(y)) - {x}
= ({x} ∪ {y}) - {x}
= {y}
```

**결과: 자유 변수 = {y}**

**예시 3: 중첩 람다**

```fsharp
fun x -> fun y -> x + y + z
```

분석:
```
내부: FV(fun y -> x + y + z)
    = FV(x + y + z) - {y}
    = ({x} ∪ {y} ∪ {z}) - {y}
    = {x, z}

외부: FV(fun x -> (fun y -> x + y + z))
    = FV(fun y -> ...) - {x}
    = {x, z} - {x}
    = {z}
```

**결과: 자유 변수 = {z}**

**예시 4: Let 바인딩**

```fsharp
let a = 10 in
let b = a + 5 in
fun x -> a + b + x
```

분석:
```
1. FV(fun x -> a + b + x) = {a, b}

2. FV(let b = a + 5 in (fun x -> ...))
   = FV(a + 5) ∪ (FV(fun x -> ...) - {b})
   = {a} ∪ ({a, b} - {b})
   = {a} ∪ {a}
   = {a}

3. FV(let a = 10 in (let b = ...))
   = FV(10) ∪ (FV(let b = ...) - {a})
   = {} ∪ ({a} - {a})
   = {} (empty)
```

**결과: 전체 표현식의 자유 변수 없음** (모든 변수가 바인딩됨)

하지만 `fun x -> a + b + x` 자체는 {a, b}를 캡처해야 한다.

### 스코프와 섀도잉 (Shadowing)

섀도잉은 같은 이름의 변수를 재바인딩하는 것이다.

```fsharp
let x = 10 in
let f = fun y -> x + y in
let x = 20 in
f 5
```

분석:
```
1. 내부 x: let x = 10 에서 바인딩
2. fun y -> x + y: x는 첫 번째 x (10) 참조
3. 외부 x: let x = 20 에서 재바인딩 (다른 x)
4. f 5: f는 x = 10을 캡처한 클로저
```

**중요:** 자유 변수 분석은 **lexical scope**을 따른다. 변수는 **가장 가까운 바인딩 지점**을 참조한다.

F# 구현에서 `Set.remove`가 이것을 처리한다:
- `Let(x, e1, e2)`에서 `Set.remove x fv2`
- `Lambda(x, body)`에서 `Set.remove x fvBody`

**예시: 중첩된 섀도잉**

```fsharp
let x = 1 in
let f = fun y ->
    let x = 2 in
    fun z -> x + y + z
in
f 10 100
```

내부 람다 `fun z -> x + y + z`:
- x: let x = 2 참조 (가장 가까운 바인딩)
- y: fun y 참조
- 자유 변수: {x (inner), y}

외부 람다 `fun y -> let x = 2 in ...`:
- 자유 변수: {x (outer)}

**핵심:** 각 바인딩 지점이 새로운 스코프를 생성한다.

## 클로저 변환 (Closure Conversion)

자유 변수를 분석했으면, 이제 **클로저 변환(closure conversion)**을 적용한다. 클로저 변환은 **암묵적 환경 캡처를 명시적으로 만드는 변환**이다.

### 변환 개념

**변환 전 (source code):**

```fsharp
let x = 10 in
fun y -> x + y
```

`x`가 암묵적으로 캡처된다.

**변환 후 (closure-converted code):**

```fsharp
// 의사 코드
let x = 10 in
let env = { x: x } in
let closure = { fn: lambda_0, env: env } in
closure

// lambda_0 정의:
fun lambda_0 (env, y) =
    let x = env.x in
    x + y
```

**변화:**
1. **환경 생성**: `env = { x: 10 }`
2. **클로저 생성**: `closure = { fn: lambda_0, env: env }`
3. **함수 수정**: 환경을 첫 번째 파라미터로 받음
4. **자유 변수 접근**: 환경에서 로드 (`env.x`)

### Before/After 예시

**예시 1: 단순 클로저**

**Before:**
```fsharp
let make_adder n =
    fun x -> x + n
```

**After:**
```mlir
// make_adder 함수
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // 1. 환경 할당 (1개 변수)
    %env_size = arith.constant 16 : i64  // 8 (fn ptr) + 8 (n)
    %env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // 2. 함수 포인터 저장 (env[0])
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr

    // 3. 캡처된 변수 저장 (env[1])
    %n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr

    // 4. 환경 포인터 반환 (클로저)
    func.return %env_ptr : !llvm.ptr
}

// lambda_adder 함수 (환경 파라미터 추가)
func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // 1. 환경에서 n 로드
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32

    // 2. x + n 계산
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}
```

**핵심 변환:**
- `fun x -> x + n` → `func.func @lambda_adder(%env, %x)`
- `n` 접근 → `llvm.load from env[1]`
- 클로저 생성 → `GC_malloc` + store fn_ptr + store n

**예시 2: 여러 변수 캡처**

**Before:**
```fsharp
let x = 10 in
let y = 20 in
let z = 30 in
fun a -> x + y + z + a
```

**After (환경 구조):**
```c
// 환경 레이아웃
struct env {
    void* fn_ptr;   // [0] 함수 포인터
    i32 x;          // [1] 캡처된 x
    i32 y;          // [2] 캡처된 y
    i32 z;          // [3] 캡처된 z
};

// 환경 크기 = 8 + 4 + 4 + 4 = 20 바이트
```

```mlir
func.func @lambda_xyz(%env: !llvm.ptr, %a: i32) -> i32 {
    // x 로드
    %x_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %x = llvm.load %x_slot : !llvm.ptr -> i32

    // y 로드
    %y_slot = llvm.getelementptr %env[2] : (!llvm.ptr) -> !llvm.ptr
    %y = llvm.load %y_slot : !llvm.ptr -> i32

    // z 로드
    %z_slot = llvm.getelementptr %env[3] : (!llvm.ptr) -> !llvm.ptr
    %z = llvm.load %z_slot : !llvm.ptr -> i32

    // x + y + z + a
    %t1 = arith.addi %x, %y : i32
    %t2 = arith.addi %t1, %z : i32
    %result = arith.addi %t2, %a : i32
    func.return %result : i32
}
```

### 환경 파라미터

클로저 변환 후, 모든 람다 함수는 **환경을 첫 번째 파라미터로 받는다**.

**일반 함수 (Phase 3):**
```mlir
func.func @add(%x: i32, %y: i32) -> i32 {
    %result = arith.addi %x, %y : i32
    func.return %result : i32
}
```

**클로저 함수 (Phase 4):**
```mlir
func.func @lambda_closure(%env: !llvm.ptr, %x: i32, %y: i32) -> i32 {
    // 환경에서 캡처된 변수 로드
    // ...
    func.return %result : i32
}
```

**차이:**
- 일반 함수: 파라미터만
- 클로저 함수: `%env: !llvm.ptr` + 파라미터

**환경 타입:**

환경은 **opaque pointer**로 표현된다: `!llvm.ptr`

LLVM은 포인터 타입이 단순화되어, 모든 포인터가 `!llvm.ptr`이다. 내부 구조는 `getelementptr`의 인덱스로 관리한다.

### Flat Environment vs Linked Environment

환경을 표현하는 방법은 두 가지다:

**1. Flat environment (FunLang 선택):**

모든 캡처된 변수를 하나의 배열에 저장.

```c
struct env {
    void* fn_ptr;
    int var1;
    int var2;
    int var3;
};
```

**장점:**
- O(1) 접근: `env[index]`로 직접 접근
- 메모리 효율: 하나의 할당
- 간단한 구현

**단점:**
- 중첩 클로저가 부모 환경을 통째로 복사해야 함

**2. Linked environment (일부 컴파일러):**

환경을 체인으로 연결.

```c
struct env {
    void* fn_ptr;
    struct env* parent;  // 부모 환경 포인터
    int var1;
};
```

**장점:**
- 중첩 클로저가 부모 환경을 공유할 수 있음

**단점:**
- O(depth) 접근: 체인을 따라 탐색
- 메모리 간접 참조 증가

**FunLang 선택: Flat environment**

이유:
1. **단순성**: 구현이 간단함
2. **성능**: O(1) 접근이 빠름
3. **교육 목적**: 개념을 명확히 이해할 수 있음

중첩 클로저는 드물고, 복사 오버헤드가 크지 않다.

### 클로저 변환 요약

클로저 변환은 다음 단계를 수행한다:

1. **자유 변수 분석**: `freeVars(lambda)` → `{x, y, z}`
2. **환경 크기 계산**: `size = 8 (fn ptr) + 4*n (captured vars)`
3. **환경 할당**: `GC_malloc(size)` → heap 객체
4. **함수 포인터 저장**: `env[0] = @lambda_N`
5. **변수 저장**: `env[1] = x`, `env[2] = y`, ...
6. **함수 정의 수정**: `lambda_N(env, params...)` 형태로 변환
7. **변수 접근 수정**: `x` → `load from env[1]`

결과: 암묵적 캡처가 명시적 환경 조작으로 변환된다.

