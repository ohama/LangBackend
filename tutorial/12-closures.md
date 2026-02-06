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

## AST 확장: Lambda 표현식

클로저를 컴파일하려면, AST에 **Lambda** 케이스를 추가해야 한다.

### Expr 타입 확장

```fsharp
// Phase 3 AST (Chapter 10-11)
type Expr =
    | Var of string
    | Num of int
    | Add of Expr * Expr
    | Sub of Expr * Expr
    | Mul of Expr * Expr
    | Div of Expr * Expr
    | Eq of Expr * Expr
    | Lt of Expr * Expr
    | Let of string * Expr * Expr
    | If of Expr * Expr * Expr
    | App of string * Expr list  // 함수 호출: f(arg1, arg2, ...)

// Phase 4 AST (Chapter 12+)
type Expr =
    | Var of string
    | Num of int
    | Add of Expr * Expr
    | Sub of Expr * Expr
    | Mul of Expr * Expr
    | Div of Expr * Expr
    | Eq of Expr * Expr
    | Lt of Expr * Expr
    | Let of string * Expr * Expr
    | If of Expr * Expr * Expr
    | Lambda of string * Expr        // NEW: 람다 표현식
    | App of Expr * Expr             // CHANGED: 일반 함수 적용
```

**변경사항:**

1. **Lambda 추가**: `Lambda(param, body)`
   - `param`: 파라미터 이름 (단일 파라미터, 다중 파라미터는 currying으로 표현)
   - `body`: 함수 본체

2. **App 변경**: `App(Expr, Expr)` (함수 표현식 + 인자 표현식)
   - Phase 3: `App(string, Expr list)` - 이름으로 함수 호출
   - Phase 4: `App(Expr, Expr)` - 표현식이 함수가 될 수 있음 (클로저 호출)

### Lambda 예시

**예시 1: 단순 람다**

```fsharp
fun x -> x + 1
```

AST:
```fsharp
Lambda("x", Add(Var "x", Num 1))
```

**예시 2: 클로저**

```fsharp
let y = 10 in
fun x -> x + y
```

AST:
```fsharp
Let("y", Num 10,
    Lambda("x", Add(Var "x", Var "y")))
```

**예시 3: 고차 함수**

```fsharp
fun f -> fun x -> f x
```

AST:
```fsharp
Lambda("f",
    Lambda("x",
        App(Var "f", Var "x")))
```

### Currying으로 다중 파라미터 표현

FunLang은 단일 파라미터 람다만 지원한다. 다중 파라미터는 **currying**으로 표현한다.

```fsharp
// 다중 파라미터 (syntax sugar)
fun x y -> x + y

// Currying (desugared)
fun x -> fun y -> x + y
```

AST:
```fsharp
Lambda("x",
    Lambda("y",
        Add(Var "x", Var "y")))
```

이것이 표준 함수형 언어 패턴이다 (Haskell, OCaml, F#).

### Parser 업데이트 (개념)

Lambda를 파싱하려면, `fun` 키워드를 추가해야 한다.

```fsharp
// LangTutorial의 parser.fsy에서
// (독자는 LangTutorial을 참고하여 자신의 parser를 업데이트)

Expr:
    | FUN ID ARROW Expr    { Lambda($2, $4) }
    | ...
```

**토큰:**
- `FUN`: "fun" 키워드
- `ID`: 식별자
- `ARROW`: "->" 화살표
- `Expr`: 본체 표현식

**결합 순서:**
- `fun x -> fun y -> x + y`: 오른쪽 결합
- `f x y`: 왼쪽 결합 (App는 왼쪽 결합)

## MLIR 환경 구조체

클로저의 핵심은 **환경(environment)** 구조체다. 환경은 캡처된 변수들을 저장하는 힙 객체다.

### 환경 레이아웃

환경은 **헤테로지니어스 배열(heterogeneous array)**이다:

```c
// C 스타일 표현
struct closure_env {
    void* fn_ptr;   // [0] 함수 포인터
    int var1;       // [1] 첫 번째 캡처된 변수
    int var2;       // [2] 두 번째 캡처된 변수
    // ...
};
```

**인덱스 규칙:**

| Index | Content | Type | Size |
|-------|---------|------|------|
| 0 | 함수 포인터 | `!llvm.ptr` | 8 bytes |
| 1 | 첫 번째 변수 | `i32` | 4 bytes |
| 2 | 두 번째 변수 | `i32` | 4 bytes |
| ... | ... | ... | ... |

**상수 정의:**

```fsharp
// F# 컴파일러에서
let ENV_FN_PTR = 0         // 함수 포인터 인덱스
let ENV_FIRST_VAR = 1      // 첫 번째 변수 인덱스
```

### LLVM Struct Type

MLIR에서 환경은 `!llvm.struct` 타입으로 표현할 수도 있지만, **opaque pointer** 방식이 더 간단하다.

**Opaque pointer 방식 (FunLang 선택):**

```mlir
// 환경은 !llvm.ptr로 표현
// 내부 구조는 getelementptr 인덱스로 관리

%env_ptr = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr
%slot = llvm.getelementptr %env_ptr[index] : (!llvm.ptr) -> !llvm.ptr
```

**장점:**
- 타입 시스템이 간단함
- 동적 크기 환경 가능
- getelementptr가 바이트 오프셋 자동 계산

**Struct type 방식 (대안):**

```mlir
// 환경 타입 정의
!env_type = !llvm.struct<(ptr, i32, i32)>

// 사용
%env = llvm.alloca : !llvm.ptr
%slot = llvm.getelementptr %env[0, 1] : (!llvm.ptr) -> !llvm.ptr
```

**단점:**
- 각 클로저마다 다른 타입 필요
- 타입 정의가 복잡함

**FunLang 선택:** Opaque pointer 방식

### getelementptr로 슬롯 접근

`llvm.getelementptr`는 포인터 산술 연산이다. 배열 인덱스를 받아서 해당 위치의 포인터를 계산한다.

**Syntax:**

```mlir
%slot_ptr = llvm.getelementptr %base_ptr[index] : (!llvm.ptr) -> !llvm.ptr
```

**예시:**

```mlir
// 환경 포인터: %env
// 인덱스 1번 슬롯 접근 (첫 번째 변수)

%slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
%value = llvm.load %slot : !llvm.ptr -> i32
```

**중요:** getelementptr는 **포인터만 계산**한다. 실제 로드는 `llvm.load`로 수행한다.

**메모리 레이아웃 예시:**

```
환경 메모리 (3개 변수 캡처):
Address     Content
0x1000      @lambda_N (fn ptr, 8 bytes)
0x1008      10 (var1, 4 bytes)
0x100C      20 (var2, 4 bytes)
0x1010      30 (var3, 4 bytes)

getelementptr %env[0]: 0x1000
getelementptr %env[1]: 0x1008
getelementptr %env[2]: 0x100C
getelementptr %env[3]: 0x1010
```

**바이트 정렬:** LLVM이 자동으로 적절한 정렬을 수행한다.

### Helper 함수: CreateClosureEnv

환경 생성을 간단하게 만드는 helper 함수:

```fsharp
// F# 컴파일러에서
let createClosureEnv (builder: OpBuilder) (fnAddr: MlirValue) (capturedVars: MlirValue list) : MlirValue =
    // 1. 환경 크기 계산
    let fnPtrSize = 8L  // 포인터 크기
    let varSize = 4L    // i32 크기
    let totalSize = fnPtrSize + (int64 capturedVars.Length) * varSize
    let sizeConst = builder.CreateI64Const(totalSize)

    // 2. GC_malloc 호출
    let envPtr = builder.CreateCall("GC_malloc", [sizeConst])

    // 3. 함수 포인터 저장
    let fnSlot = builder.CreateGEP(envPtr, 0)
    builder.CreateStore(fnAddr, fnSlot)

    // 4. 캡처된 변수들 저장
    capturedVars |> List.iteri (fun i var ->
        let slot = builder.CreateGEP(envPtr, i + 1)
        builder.CreateStore(var, slot)
    )

    // 5. 환경 포인터 반환
    envPtr
```

### Helper 함수: GetEnvSlot

환경에서 변수 로드를 간단하게:

```fsharp
let getEnvSlot (builder: OpBuilder) (envPtr: MlirValue) (index: int) : MlirValue =
    // getelementptr + load
    let slot = builder.CreateGEP(envPtr, index)
    builder.CreateLoad(slot, "i32")
```

사용 예시:

```fsharp
// 환경에서 첫 번째 변수 로드
let var1 = getEnvSlot builder envPtr ENV_FIRST_VAR
```

## 클로저 생성 코드 (Closure Creation)

클로저 생성은 **환경 할당 + 변수 저장 + 환경 포인터 반환**이다.

### compileLambda 함수

```fsharp
// Lambda 표현식 컴파일
let compileLambda (builder: OpBuilder) (env: Environment) (param: string) (body: Expr) : MlirValue =
    // 1. 자유 변수 분석
    let freeVarSet = freeVars (Lambda(param, body))
    let freeVarList = Set.toList freeVarSet

    // 2. 캡처된 변수들의 SSA 값 가져오기
    let capturedValues =
        freeVarList |> List.map (fun varName ->
            match env.TryFind(varName) with
            | Some(value) -> value
            | None -> failwithf "Undefined variable: %s" varName
        )

    // 3. 람다 함수 정의 생성
    let lambdaName = generateLambdaName()  // @lambda_0, @lambda_1, ...
    let lambdaFunc = createLambdaFunction builder lambdaName param body freeVarList env

    // 4. 함수 포인터 얻기
    let fnAddr = builder.CreateAddressOf(lambdaName)

    // 5. 환경 생성 및 변수 저장
    let envPtr = createClosureEnv builder fnAddr capturedValues

    // 6. 환경 포인터 반환 (이것이 클로저)
    envPtr
```

**핵심 단계:**

1. **자유 변수 분석**: `freeVars`로 캡처할 변수 찾기
2. **값 가져오기**: 환경에서 SSA 값 로드
3. **람다 함수 정의**: 별도 함수로 생성 (환경 파라미터 포함)
4. **함수 포인터**: `llvm.mlir.addressof`로 주소 얻기
5. **환경 할당**: `GC_malloc` + 변수 저장
6. **반환**: 환경 포인터 (클로저 값)

### createLambdaFunction

람다 함수를 별도 `func.func`로 정의:

```fsharp
let createLambdaFunction (builder: OpBuilder) (name: string) (param: string) (body: Expr) (freeVars: string list) (outerEnv: Environment) : unit =
    // 1. 함수 시그니처: (%env: !llvm.ptr, %param: i32) -> i32
    let paramTypes = [builder.GetPtrType(); builder.GetI32Type()]
    let returnType = builder.GetI32Type()

    // 2. 함수 생성
    let funcOp = builder.CreateFuncOp(name, paramTypes, returnType)
    let entryBlock = builder.GetEntryBlock(funcOp)
    builder.SetInsertionPoint(entryBlock)

    // 3. Block arguments 얻기
    let envArg = builder.GetBlockArg(entryBlock, 0)   // %env
    let paramArg = builder.GetBlockArg(entryBlock, 1)  // %param

    // 4. 환경 구축: 파라미터 + 캡처된 변수들
    let mutable lambdaEnv = Map.empty
    lambdaEnv <- lambdaEnv.Add(param, paramArg)

    // 캡처된 변수들을 환경에서 로드
    freeVars |> List.iteri (fun i varName ->
        let value = getEnvSlot builder envArg (ENV_FIRST_VAR + i)
        lambdaEnv <- lambdaEnv.Add(varName, value)
    )

    // 5. 본체 컴파일
    let bodyValue = compileExpr builder lambdaEnv body

    // 6. 반환
    builder.CreateFuncReturn(bodyValue)
```

**핵심:**
- 환경 파라미터 `%env: !llvm.ptr`가 첫 번째
- 실제 파라미터 `%param: i32`가 두 번째
- 캡처된 변수들을 환경에서 로드하여 lambda 환경에 추가
- 본체 컴파일은 일반 표현식과 동일

### 전체 예시: make_adder 컴파일

**Source code:**

```fsharp
let make_adder n =
    fun x -> x + n
```

**AST:**

```fsharp
Let("make_adder",
    Lambda("n",
        Lambda("x", Add(Var "x", Var "n"))),
    ...)
```

**Generated MLIR IR:**

```mlir
// make_adder 함수
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // 내부 람다: fun x -> x + n
    // 자유 변수: {n}

    // 1. 환경 크기 계산: 8 (fn ptr) + 4 (n) = 12 bytes
    %env_size = arith.constant 12 : i64

    // 2. GC_malloc 호출
    %env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // 3. 함수 포인터 저장
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr

    // 4. 캡처된 변수 n 저장
    %n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr

    // 5. 환경 포인터 반환 (클로저)
    func.return %env_ptr : !llvm.ptr
}

// lambda_adder 함수
func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // 1. 환경에서 n 로드
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32

    // 2. x + n 계산
    %result = arith.addi %x, %n : i32

    // 3. 반환
    func.return %result : i32
}
```

**코드 흐름:**

1. `make_adder` 호출: `make_adder(5)`
2. 환경 할당: `env = GC_malloc(12)`
3. 함수 포인터 저장: `env[0] = @lambda_adder`
4. `n` 저장: `env[1] = 5`
5. 클로저 반환: `env` (포인터)
6. 나중에 클로저 호출: `closure(10)`
7. `lambda_adder` 호출: `@lambda_adder(env, 10)`
8. `n` 로드: `env[1]` → `5`
9. 계산: `10 + 5` → `15`

## 클로저 본체 컴파일 (Closure Body)

클로저 본체는 **lifted function**으로 컴파일된다. Lifted function은 최상위 함수로 추출된 람다 함수다.

### Lifting 개념

**Before lifting (nested lambda):**

```fsharp
let make_adder n =
    fun x -> x + n
```

**After lifting (top-level functions):**

```fsharp
// Lifted lambda
let lambda_adder (env, x) =
    let n = env[1] in
    x + n

// make_adder는 클로저 생성기
let make_adder n =
    let env = allocate_env(@lambda_adder, n) in
    env
```

모든 람다 함수가 최상위로 **lift**된다. 중첩된 함수가 flat structure로 변환된다.

### 환경 파라미터 타입

Lifted function의 시그니처:

```mlir
func.func @lambda_N(%env: !llvm.ptr, %param1: i32, %param2: i32, ...) -> i32
```

**첫 번째 파라미터:**
- 이름: `%env`
- 타입: `!llvm.ptr` (opaque pointer)
- 목적: 캡처된 변수 접근

**나머지 파라미터:**
- 람다의 실제 파라미터들

### 환경에서 변수 로드

캡처된 변수를 사용하려면, 환경에서 로드해야 한다:

```mlir
// 첫 번째 캡처된 변수 로드
%var1_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
%var1 = llvm.load %var1_slot : !llvm.ptr -> i32

// 두 번째 캡처된 변수 로드
%var2_slot = llvm.getelementptr %env[2] : (!llvm.ptr) -> !llvm.ptr
%var2 = llvm.load %var2_slot : !llvm.ptr -> i32
```

**패턴:**
1. `getelementptr`로 슬롯 포인터 계산
2. `llvm.load`로 값 로드
3. SSA 값으로 사용

### 전체 예시: 중첩 클로저

**Source code:**

```fsharp
let x = 10 in
let y = 20 in
fun z -> x + y + z
```

**Lifted function:**

```mlir
func.func @lambda_xyz(%env: !llvm.ptr, %z: i32) -> i32 {
    // 1. x 로드 (env[1])
    %x_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %x = llvm.load %x_slot : !llvm.ptr -> i32

    // 2. y 로드 (env[2])
    %y_slot = llvm.getelementptr %env[2] : (!llvm.ptr) -> !llvm.ptr
    %y = llvm.load %y_slot : !llvm.ptr -> i32

    // 3. x + y 계산
    %t1 = arith.addi %x, %y : i32

    // 4. t1 + z 계산
    %result = arith.addi %t1, %z : i32

    // 5. 반환
    func.return %result : i32
}
```

**클로저 생성 부분:**

```mlir
func.func @main() -> i32 {
    // 1. x, y 정의
    %x = arith.constant 10 : i32
    %y = arith.constant 20 : i32

    // 2. 환경 크기: 8 (fn ptr) + 4 (x) + 4 (y) = 16 bytes
    %env_size = arith.constant 16 : i64

    // 3. 환경 할당
    %env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // 4. 함수 포인터 저장
    %fn = llvm.mlir.addressof @lambda_xyz : !llvm.ptr
    %fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn, %fn_slot : !llvm.ptr, !llvm.ptr

    // 5. x 저장
    %x_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %x, %x_slot : i32, !llvm.ptr

    // 6. y 저장
    %y_slot = llvm.getelementptr %env[2] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %y, %y_slot : i32, !llvm.ptr

    // 7. 클로저 호출 (나중에 Chapter 13에서)
    // %result = call_closure %env (%z)

    func.return %0 : i32
}
```

### 함수 명명 규칙

Lifted function의 이름은 자동 생성된다:

```fsharp
let mutable lambdaCounter = 0

let generateLambdaName() =
    let name = sprintf "lambda_%d" lambdaCounter
    lambdaCounter <- lambdaCounter + 1
    name
```

**예시:**
- 첫 번째 람다: `@lambda_0`
- 두 번째 람다: `@lambda_1`
- ...

**중요:** 이름은 unique해야 한다. 같은 이름의 함수가 여러 개 있으면 링커 오류가 발생한다.

## 공통 오류 (Common Errors)

클로저 컴파일에서 자주 발생하는 오류들:

### Error 1: 환경 인덱스 off-by-one

**증상:**

```mlir
// 잘못된 코드 - 함수 포인터를 변수로 로드
%var1_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
%var1 = llvm.load %var1_slot : !llvm.ptr -> i32  // ERROR: 타입 불일치!
```

**원인:**

환경 레이아웃을 잊음:
- `env[0]`: 함수 포인터 (`!llvm.ptr`)
- `env[1]`: 첫 번째 변수 (`i32`)
- `env[2]`: 두 번째 변수 (`i32`)

**해결:**

```mlir
// 올바른 코드 - 첫 번째 변수는 env[1]
%var1_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
%var1 = llvm.load %var1_slot : !llvm.ptr -> i32  // OK
```

**팁:** `ENV_FN_PTR = 0`, `ENV_FIRST_VAR = 1` 상수 사용하기.

### Error 2: 환경 파라미터 누락

**증상:**

```mlir
// 잘못된 코드 - 환경 파라미터 없음
func.func @lambda_adder(%x: i32) -> i32 {
    %n = ??? // n을 어디서 가져오나?
}
```

**원인:**

Lifted function에 환경 파라미터를 추가하지 않음.

**해결:**

```mlir
// 올바른 코드 - 환경 파라미터 추가
func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    // ...
}
```

**팁:** 모든 람다 함수는 첫 번째 파라미터로 `%env: !llvm.ptr`를 받는다.

### Error 3: 스택 vs 힙 할당

**증상:**

```mlir
// 잘못된 코드 - 스택 할당
%env = llvm.alloca 16, i8 : (i32, i8) -> !llvm.ptr
// ...
func.return %env : !llvm.ptr  // ERROR: 스택 메모리를 반환!
```

**원인:**

환경을 스택에 할당했는데, 함수 반환 후 사라진다.

**해결:**

```mlir
// 올바른 코드 - 힙 할당
%env_size = arith.constant 16 : i64
%env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
func.return %env : !llvm.ptr  // OK: 힙 메모리는 살아있음
```

**원칙:**
- **스택 할당 (`llvm.alloca`)**: 함수 로컬 변수 (현재 스택 프레임에서만 유효)
- **힙 할당 (`GC_malloc`)**: 탈출하는 값 (함수 반환 후에도 유효)

클로저는 항상 **힙 할당**해야 한다. 클로저가 생성된 함수가 반환된 후에도 사용되기 때문이다.

### Error 4: 타입 불일치

**증상:**

```mlir
// 잘못된 코드
%fn_addr = llvm.mlir.addressof @lambda_0 : !llvm.ptr
%fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %fn_addr, %fn_slot : i32, !llvm.ptr  // ERROR: 타입 불일치!
```

**원인:**

함수 포인터 타입을 `i32`로 잘못 지정.

**해결:**

```mlir
// 올바른 코드
%fn_addr = llvm.mlir.addressof @lambda_0 : !llvm.ptr
%fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr  // OK
```

**타입 체크:**
- 함수 포인터: `!llvm.ptr`
- i32 변수: `i32`
- `llvm.store` 시그니처: `llvm.store %value, %ptr : value_type, !llvm.ptr`

## 요약

**이 장에서 배운 것:**

1. **클로저 이론**
   - Lexical scoping: 정의 시점의 환경을 기억
   - Free variables: 람다에서 바인딩되지 않은 변수
   - Bound variables: 람다 파라미터로 바인딩된 변수
   - 환경 캡처: 자유 변수의 값을 저장

2. **자유 변수 분석**
   - Set-based traversal 알고리즘
   - `FV(Lambda(x, body)) = FV(body) - {x}`
   - F# 구현: `freeVars` 재귀 함수

3. **클로저 변환**
   - 암묵적 캡처 → 명시적 환경 조작
   - Flat environment: 모든 변수를 배열에 저장
   - Lifted functions: 람다를 최상위 함수로 추출

4. **환경 구조체**
   - 레이아웃: `[fn_ptr, var1, var2, ...]`
   - `env[0]`: 함수 포인터
   - `env[1+]`: 캡처된 변수들
   - `getelementptr`로 슬롯 접근

5. **클로저 생성 코드**
   - `GC_malloc`로 환경 힙 할당
   - 함수 포인터 저장 (`llvm.mlir.addressof`)
   - 캡처된 변수들 저장 (`llvm.store`)
   - 환경 포인터 반환 (클로저 값)

6. **클로저 본체 컴파일**
   - Lifted function: `@lambda_N(%env, %params...)`
   - 환경 파라미터를 첫 번째로 받음
   - `getelementptr` + `llvm.load`로 변수 접근

**다음 장 (Chapter 13):**

- **고차 함수 (Higher-order functions)**: 함수를 인자로 받거나 반환
- **클로저 호출**: 환경 포인터에서 함수 포인터 추출 + 간접 호출
- **Map/Filter/Fold**: 표준 고차 함수 구현
- **Function 타입**: 함수를 first-class value로 취급

클로저는 함수형 프로그래밍의 핵심이다. 이 장에서 확립한 환경 캡처 메커니즘이 고차 함수의 기초가 된다.

