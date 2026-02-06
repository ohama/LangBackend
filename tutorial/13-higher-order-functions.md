# Chapter 13: 고차 함수 (Higher-Order Functions)

## 소개

**고차 함수(higher-order function, HOF)**는 함수를 **일급 값(first-class value)**으로 다루는 함수다:

1. **함수를 인자로 받는 함수**: `apply f x = f x`
2. **함수를 반환하는 함수**: `makeAdder n = fun x -> x + n`

```fsharp
// 고차 함수 예시
let apply f x = f x           // 함수를 인자로 받는다
let twice f x = f (f x)       // 함수를 두 번 적용한다
let compose f g x = f (g x)   // 함수 합성

let inc x = x + 1
let result = twice inc 10     // 결과: 12
```

**왜 고차 함수가 중요한가?**

1. **추상화(Abstraction)**: 공통 패턴을 재사용 가능하게 만든다 (map, filter, fold)
2. **합성(Composition)**: 작은 함수를 조합해 복잡한 동작을 만든다
3. **지연 평가(Lazy evaluation)**: 계산을 함수로 감싸서 나중에 실행할 수 있다
4. **콜백 패턴**: 비동기 작업, 이벤트 처리에 필수

**고차 함수 vs 일반 함수:**

| 일반 함수 (Phase 3) | 고차 함수 (Phase 4) |
|-------------------|-------------------|
| 데이터를 인자로 받는다 | 함수를 인자로 받는다 |
| 데이터를 반환한다 | 함수를 반환할 수 있다 |
| 직접 호출 (`func.call @symbol`) | 간접 호출 (function pointer) |
| 타입: `int -> int` | 타입: `(int -> int) -> int` |

**Chapter 13의 목표:**

이 장을 마치면 다음을 컴파일할 수 있다:

```fsharp
// 함수를 인자로 받기
let apply f x = f x
let result1 = apply inc 42   // 43

// 함수를 반환하기
let makeAdder n = fun x -> x + n
let add5 = makeAdder 5
let result2 = add5 10        // 15

// 함수 합성
let compose f g x = f (g x)
let inc x = x + 1
let double x = x * 2
let incThenDouble = compose double inc
let result3 = incThenDouble 5   // 12
```

**Chapter 13의 범위:**

1. **함수를 일급 값으로 다루기**: 클로저가 함수의 런타임 표현이다
2. **간접 호출(Indirect call) 패턴**: `llvm.call`로 함수 포인터를 호출한다
3. **Apply 함수**: 가장 단순한 고차 함수
4. **Compose 함수**: 여러 함수 인자를 다루기
5. **함수를 반환하기**: makeAdder 패턴, upward funarg problem
6. **커링(Currying)**: 다중 인자 함수를 중첩 클로저로 표현
7. **메모리 관리**: GC가 클로저 생명주기를 처리한다
8. **Complete 예시**: map 함수 (개념적, Phase 6에서 완전 구현)

**Prerequisites:**

- **Chapter 12 (Closures)**: 클로저 표현, 환경 구조, 자유 변수 분석
- Phase 3 함수 (이름 있는 함수, func.call)
- Phase 2 메모리 관리 (GC_malloc, 힙 할당)

이 장은 **클로저 + 고차 함수 = 함수형 프로그래밍 핵심**을 완성한다.

## 함수를 일급 값으로 다루기

### First-Class Functions

**일급 값(first-class value)**이란:

1. 변수에 할당할 수 있다
2. 함수 인자로 전달할 수 있다
3. 함수 반환값으로 반환할 수 있다
4. 데이터 구조에 저장할 수 있다

FunLang에서 **함수는 일급 값이다**:

```fsharp
// 1. 변수에 할당
let f = fun x -> x + 1

// 2. 함수 인자로 전달
let apply g x = g x
let result = apply f 10   // 11

// 3. 함수 반환값으로 반환
let makeAdder n = fun x -> x + n
let add5 = makeAdder 5

// 4. 데이터 구조에 저장 (Phase 6에서 리스트 구현 후)
// let funcs = [inc; double; square]
```

**일급 함수의 런타임 표현:**

Chapter 12에서 배운 **클로저**가 바로 함수의 런타임 표현이다:

```
Closure = (function_pointer, environment_pointer)
```

- `function_pointer`: 실행할 코드 (lifted function의 주소)
- `environment_pointer`: 캡처된 변수들 (힙에 할당된 환경)

**모든 함수가 클로저인가?**

논리적으로는 YES. 실제로는 최적화로 구분된다:

| 함수 종류 | 환경 | 표현 | 예시 |
|---------|-----|-----|-----|
| **Top-level named** | 비어있음 | 함수 포인터만 | `let add x y = x + y` |
| **Lambda (no capture)** | 비어있음 | 함수 포인터만 | `fun x -> x + 1` |
| **Lambda (capture)** | 변수 캡처 | 클로저 (ptr, env) | `fun x -> x + n` |

**Uniform representation:**

컴파일러 구현을 단순화하기 위해, **모든 함수를 클로저로 표현**할 수 있다:

- Top-level 함수: 환경이 null이거나 빈 환경
- 캡처 없는 람다: 환경이 빈 환경
- 캡처 있는 람다: 환경에 변수 저장

이 장에서는 **uniform representation**을 사용한다. 모든 함수는 `(fn_ptr, env_ptr)` 쌍이다.

### Named Functions vs Anonymous Lambdas

**Named function (Phase 3 스타일):**

```fsharp
let inc x = x + 1
```

컴파일 결과:

```mlir
func.func @inc(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %x, %c1 : i32
    func.return %result : i32
}
```

- MLIR 심볼 `@inc`로 직접 참조 가능
- `func.call @inc(%arg)` 직접 호출

**Anonymous lambda (Chapter 12 스타일):**

```fsharp
fun x -> x + 1
```

컴파일 결과:

```mlir
// Lifted function
func.func @lambda_0(%env: !llvm.ptr, %x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %x, %c1 : i32
    func.return %result : i32
}

// Closure 생성 (호출 지점에서)
%c0 = arith.constant 0 : i64
%env_size = arith.constant 8 : i64  // 환경 없음, fn_ptr만
%env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @lambda_0 : !llvm.ptr
llvm.store %fn_ptr, %env : !llvm.ptr, !llvm.ptr
// %env가 클로저다
```

**Named function을 클로저로 wrapping:**

Named function도 고차 함수에 전달하려면 클로저로 감싸야 한다:

```fsharp
let inc x = x + 1        // Named function
let apply f x = f x      // HOF
let result = apply inc 42   // inc를 클로저로 wrap
```

컴파일:

```mlir
// Named function (그대로)
func.func @inc(%x: i32) -> i32 { ... }

// inc를 클로저로 wrap
%env_size = arith.constant 8 : i64
%env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @inc : !llvm.ptr
llvm.store %fn_ptr, %env : !llvm.ptr, !llvm.ptr
%closure_inc = %env : !llvm.ptr

// apply에 전달
%result = func.call @apply(%closure_inc, %c42) : (!llvm.ptr, i32) -> i32
```

**요약:**

- **Named function**: MLIR 심볼로 정의, 직접 호출 가능
- **Anonymous lambda**: 항상 클로저로 표현
- **Named function을 HOF에 전달**: 클로저로 wrapping 필요
- **Uniform representation**: 모두 `!llvm.ptr` (클로저 포인터)로 표현

## 클로저 호출: 간접 호출 패턴

### Direct Call vs Indirect Call

**Direct call (Phase 3):**

```mlir
%result = func.call @inc(%x) : (i32) -> i32
```

- 호출 대상이 **컴파일 타임에 결정**됨 (`@inc` 심볼)
- 최적화 가능 (인라이닝, 특수화)

**Indirect call (Phase 4):**

```mlir
%fn_ptr = /* 클로저에서 추출 */
%result = llvm.call %fn_ptr(%closure, %x) : !llvm.ptr, (i32) -> i32
```

- 호출 대상이 **런타임에 결정**됨 (함수 포인터)
- 최적화 어려움 (가상 함수처럼 동작)

**왜 간접 호출이 필요한가?**

고차 함수는 **어떤 함수가 전달될지 컴파일 타임에 모른다**:

```fsharp
let apply f x = f x   // f는 런타임에 결정된다

apply inc 10      // f = inc
apply double 10   // f = double
```

컴파일러는 `f`가 무엇인지 모르므로, **간접 호출**을 생성해야 한다.

### 간접 호출 패턴 (Indirect Call Pattern)

클로저를 호출하는 3단계:

**1. 함수 포인터 추출:**

환경의 slot 0에서 함수 포인터를 로드한다:

```mlir
// %closure: !llvm.ptr (클로저 포인터)
%c0 = arith.constant 0 : i64
%fn_ptr_addr = llvm.getelementptr %closure[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
%fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
```

**2. 인자 준비:**

- **첫 번째 인자**: 클로저 자신 (환경 포인터)
- **나머지 인자**: 원래 함수 파라미터

```mlir
%env = %closure : !llvm.ptr   // 클로저 = 환경
%arg1 = %x : i32               // 실제 인자
```

**3. 간접 호출:**

함수 포인터를 통해 호출한다:

```mlir
%result = llvm.call %fn_ptr(%env, %arg1) : !llvm.ptr, (i32) -> i32
```

**완전한 예시:**

```mlir
// 클로저 %closure를 호출: closure(42)
%c0 = arith.constant 0 : i64
%c42 = arith.constant 42 : i32

// Step 1: 함수 포인터 추출
%fn_ptr_addr = llvm.getelementptr %closure[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
%fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr

// Step 2: 인자 준비
%env = %closure : !llvm.ptr

// Step 3: 간접 호출
%result = llvm.call %fn_ptr(%env, %c42) : !llvm.ptr, (i32) -> i32
```

### F# Helper: CallClosure

반복되는 패턴을 헬퍼 함수로 추출한다:

```fsharp
type OpBuilder with
    /// 클로저를 간접 호출한다
    /// closure: !llvm.ptr (클로저 포인터)
    /// args: 함수 인자들 (환경 제외)
    /// Returns: 함수 호출 결과
    member this.CallClosure(closure: MlirValue, args: MlirValue list, resultType: MlirType) : MlirValue =
        // 1. 함수 포인터 추출
        let c0 = this.ConstantInt(0L, 64)
        let fnPtrAddr = this.CreateGEP(closure, [c0])
        let fnPtr = this.CreateLoad(this.PtrType(), fnPtrAddr)

        // 2. 인자 리스트 구성 (환경 + 원래 인자)
        let allArgs = closure :: args

        // 3. 간접 호출
        this.CreateLLVMCall(fnPtr, allArgs, resultType)
```

**사용 예시:**

```fsharp
// compileExpr에서 클로저 호출
| App(funcExpr, argExpr) ->
    let funcVal = compileExpr builder env funcExpr
    let argVal = compileExpr builder env argExpr

    // funcVal은 클로저 (!llvm.ptr)
    // argVal은 인자 (i32)
    builder.CallClosure(funcVal, [argVal], builder.IntType(32))
```

**간접 호출의 비용:**

- **성능**: 직접 호출보다 느리다 (포인터 로드, 인라이닝 불가)
- **유연성**: 런타임에 함수 선택 가능 (고차 함수의 핵심)

**최적화 가능성:**

- **인라이닝**: 클로저가 상수라면 특수화 가능
- **Devirtualization**: 타입 분석으로 호출 대상 추론
- Phase 4는 최적화 없이 단순 구현만 한다

## Apply 함수

### Apply의 의미

**Apply 함수**는 가장 단순한 고차 함수다:

```fsharp
let apply f x = f x
```

- 타입: `(a -> b) -> a -> b`
- 의미: 함수 `f`를 인자 `x`에 적용한다

**왜 apply가 유용한가?**

일견 쓸모없어 보인다 (`f x`와 `apply f x`는 같다). 하지만:

1. **HOF 테스트**: 가장 단순한 고차 함수로 컴파일러 검증
2. **파이프라인**: `x |> apply f` 스타일 (Phase 7 파이프 연산자)
3. **교육적**: 간접 호출 패턴을 명확히 보여줌

**Apply 예시:**

```fsharp
let inc x = x + 1
let double x = x * 2

let result1 = apply inc 42      // 43
let result2 = apply double 10   // 20
```

### Apply 컴파일: F# 구현

**AST 표현:**

```fsharp
// apply f x = f x
Let("apply",
    Lambda("f",
        Lambda("x",
            App(Var "f", Var "x"))),
    ...)
```

**컴파일 단계:**

1. **외부 람다**: `fun f -> ...` (f를 캡처)
2. **내부 람다**: `fun x -> f x` (f 사용)
3. **App**: `f x` (간접 호출)

**F# 컴파일 함수:**

```fsharp
let rec compileExpr (builder: OpBuilder) (env: Map<string, MlirValue>) (expr: Expr) : MlirValue =
    match expr with
    // ... (기존 케이스들)

    | App(funcExpr, argExpr) ->
        // funcExpr를 평가 -> 클로저
        let closureVal = compileExpr builder env funcExpr

        // argExpr를 평가 -> 인자
        let argVal = compileExpr builder env argExpr

        // 클로저 간접 호출
        builder.CallClosure(closureVal, [argVal], builder.IntType(32))
```

**Apply 전체 컴파일:**

```fsharp
// let apply f x = f x
let compileApply (builder: OpBuilder) : MlirValue =
    // Lifted inner function: fun(env, x) -> env[1](x)
    //   env[0] = fn_ptr (inner)
    //   env[1] = f (captured)
    let innerFunc = builder.CreateFunction("apply_inner",
        [builder.PtrType(); builder.IntType(32)],
        builder.IntType(32))

    // Inner function body
    builder.WithInsertionPoint(innerFunc, fun () ->
        let env = innerFunc.GetArgument(0)
        let x = innerFunc.GetArgument(1)

        // Load captured f from env[1]
        let c1 = builder.ConstantInt(1L, 64)
        let fAddr = builder.CreateGEP(env, [c1])
        let f = builder.CreateLoad(builder.PtrType(), fAddr)

        // Call f(x) indirectly
        let result = builder.CallClosure(f, [x], builder.IntType(32))
        builder.CreateReturn(result)
    )

    // Lifted outer function: fun(env_outer, f) -> closure(inner, [f])
    let outerFunc = builder.CreateFunction("apply_outer",
        [builder.PtrType(); builder.PtrType()],
        builder.PtrType())

    // Outer function body
    builder.WithInsertionPoint(outerFunc, fun () ->
        let envOuter = outerFunc.GetArgument(0)
        let f = outerFunc.GetArgument(1)

        // Allocate environment for inner closure
        let envSize = builder.ConstantInt(16L, 64)  // 2 slots
        let envInner = builder.CreateGCMalloc(envSize)

        // env[0] = fn_ptr(inner)
        let c0 = builder.ConstantInt(0L, 64)
        let fnPtrInner = builder.CreateAddressOf(innerFunc)
        let slot0 = builder.CreateGEP(envInner, [c0])
        builder.CreateStore(fnPtrInner, slot0)

        // env[1] = f (captured)
        let c1 = builder.ConstantInt(1L, 64)
        let slot1 = builder.CreateGEP(envInner, [c1])
        builder.CreateStore(f, slot1)

        // Return closure
        builder.CreateReturn(envInner)
    )

    // Return outer closure (no captures, empty env)
    let envOuter = builder.CreateEmptyClosure(outerFunc)
    envOuter
```

### Apply MLIR IR

**예상 MLIR 출력:**

```mlir
// Inner lifted function
func.func @apply_inner(%env: !llvm.ptr, %x: i32) -> i32 {
    // Load captured f from env[1]
    %c1 = arith.constant 1 : i64
    %f_addr = llvm.getelementptr %env[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %f = llvm.load %f_addr : !llvm.ptr -> !llvm.ptr

    // Extract f's function pointer
    %c0 = arith.constant 0 : i64
    %fn_ptr_addr = llvm.getelementptr %f[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr

    // Call f(x) - indirect call
    %result = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32
    func.return %result : i32
}

// Outer lifted function
func.func @apply_outer(%env_outer: !llvm.ptr, %f: !llvm.ptr) -> !llvm.ptr {
    // Allocate environment for inner closure (2 slots)
    %env_size = arith.constant 16 : i64
    %env_inner = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // env[0] = fn_ptr(inner)
    %c0 = arith.constant 0 : i64
    %fn_ptr_inner = llvm.mlir.addressof @apply_inner : !llvm.ptr
    %slot0 = llvm.getelementptr %env_inner[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_inner, %slot0 : !llvm.ptr, !llvm.ptr

    // env[1] = f (captured)
    %c1 = arith.constant 1 : i64
    %slot1 = llvm.getelementptr %env_inner[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %f, %slot1 : !llvm.ptr, !llvm.ptr

    // Return inner closure
    func.return %env_inner : !llvm.ptr
}
```

**사용 예시 MLIR:**

```mlir
// let inc x = x + 1
func.func @inc(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %x, %c1 : i32
    func.return %result : i32
}

// let result = apply inc 42
func.func @main() -> i32 {
    // Wrap inc as closure
    %c8 = arith.constant 8 : i64
    %env_inc = llvm.call @GC_malloc(%c8) : (i64) -> !llvm.ptr
    %fn_ptr_inc = llvm.mlir.addressof @inc : !llvm.ptr
    %c0 = arith.constant 0 : i64
    %slot0 = llvm.getelementptr %env_inc[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_inc, %slot0 : !llvm.ptr, !llvm.ptr
    %closure_inc = %env_inc : !llvm.ptr

    // Create apply closure
    %env_apply_outer = llvm.call @GC_malloc(%c8) : (i64) -> !llvm.ptr
    %fn_ptr_apply = llvm.mlir.addressof @apply_outer : !llvm.ptr
    %slot0_apply = llvm.getelementptr %env_apply_outer[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_apply, %slot0_apply : !llvm.ptr, !llvm.ptr
    %closure_apply = %env_apply_outer : !llvm.ptr

    // Call apply(inc)
    %fn_ptr_apply_outer = llvm.load %slot0_apply : !llvm.ptr -> !llvm.ptr
    %closure_partial = llvm.call %fn_ptr_apply_outer(%closure_apply, %closure_inc)
        : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr

    // Call (apply inc)(42)
    %c42 = arith.constant 42 : i32
    %fn_ptr_partial = llvm.getelementptr %closure_partial[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_partial : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr(%closure_partial, %c42) : (!llvm.ptr, i32) -> i32

    func.return %result : i32
}
```

**테스트:**

```bash
$ ./funlang apply_test.fun
43
```

## 여러 함수 인자 받기

### Compose 함수

**Compose**는 두 함수를 합성한다:

```fsharp
let compose f g x = f (g x)
```

- 타입: `(b -> c) -> (a -> b) -> a -> c`
- 의미: `g`를 먼저 적용하고, 그 결과에 `f`를 적용한다

**Compose 예시:**

```fsharp
let inc x = x + 1
let double x = x * 2

let incThenDouble = compose double inc
let result = incThenDouble 5   // double(inc(5)) = double(6) = 12
```

**왜 compose가 유용한가?**

1. **함수 조합**: 작은 함수를 연결해 복잡한 동작 만들기
2. **파이프라인**: `f << g << h` 스타일 (Phase 7)
3. **포인트-프리 스타일**: `let process = compose validate transform`

### Compose 컴파일

**AST:**

```fsharp
// compose f g x = f (g x)
Let("compose",
    Lambda("f",
        Lambda("g",
            Lambda("x",
                App(Var "f", App(Var "g", Var "x"))))),
    ...)
```

**중첩 람다:**

- 외부: `fun f -> ...` (f 캡처)
- 중간: `fun g -> ...` (f, g 캡처)
- 내부: `fun x -> f (g x)` (f, g 사용)

**Lifted functions:**

1. **Innermost**: `compose_inner(env, x)` - env에 f, g 저장
2. **Middle**: `compose_middle(env, g)` - env에 f 저장, g와 f로 inner closure 생성
3. **Outermost**: `compose_outer(env, f)` - f로 middle closure 생성

**MLIR IR (간략):**

```mlir
// Innermost: fun x -> f (g x)
func.func @compose_inner(%env: !llvm.ptr, %x: i32) -> i32 {
    // Load g from env[1]
    %c1 = arith.constant 1 : i64
    %g_addr = llvm.getelementptr %env[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %g = llvm.load %g_addr : !llvm.ptr -> !llvm.ptr

    // Call g(x)
    %gx = /* CallClosure(g, x) */ : i32

    // Load f from env[2]
    %c2 = arith.constant 2 : i64
    %f_addr = llvm.getelementptr %env[0, %c2] : (!llvm.ptr, i64) -> !llvm.ptr
    %f = llvm.load %f_addr : !llvm.ptr -> !llvm.ptr

    // Call f(g(x))
    %result = /* CallClosure(f, gx) */ : i32
    func.return %result : i32
}

// Middle: fun g -> <inner closure with f, g>
func.func @compose_middle(%env: !llvm.ptr, %g: !llvm.ptr) -> !llvm.ptr {
    // Load f from env[1]
    %c1 = arith.constant 1 : i64
    %f_addr = llvm.getelementptr %env[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %f = llvm.load %f_addr : !llvm.ptr -> !llvm.ptr

    // Allocate environment for inner (3 slots: fn_ptr, g, f)
    %c24 = arith.constant 24 : i64
    %env_inner = llvm.call @GC_malloc(%c24) : (i64) -> !llvm.ptr

    // env[0] = fn_ptr(inner)
    %c0 = arith.constant 0 : i64
    %fn_ptr_inner = llvm.mlir.addressof @compose_inner : !llvm.ptr
    %slot0 = llvm.getelementptr %env_inner[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_inner, %slot0 : !llvm.ptr, !llvm.ptr

    // env[1] = g
    %slot1 = llvm.getelementptr %env_inner[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %g, %slot1 : !llvm.ptr, !llvm.ptr

    // env[2] = f
    %c2 = arith.constant 2 : i64
    %slot2 = llvm.getelementptr %env_inner[0, %c2] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %f, %slot2 : !llvm.ptr, !llvm.ptr

    func.return %env_inner : !llvm.ptr
}

// Outermost: fun f -> <middle closure with f>
func.func @compose_outer(%env: !llvm.ptr, %f: !llvm.ptr) -> !llvm.ptr {
    // Allocate environment for middle (2 slots: fn_ptr, f)
    %c16 = arith.constant 16 : i64
    %env_middle = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr

    // env[0] = fn_ptr(middle)
    %c0 = arith.constant 0 : i64
    %fn_ptr_middle = llvm.mlir.addressof @compose_middle : !llvm.ptr
    %slot0 = llvm.getelementptr %env_middle[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_middle, %slot0 : !llvm.ptr, !llvm.ptr

    // env[1] = f
    %c1 = arith.constant 1 : i64
    %slot1 = llvm.getelementptr %env_middle[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %f, %slot1 : !llvm.ptr, !llvm.ptr

    func.return %env_middle : !llvm.ptr
}
```

### 여러 클로저 호출 체이닝

**Compose 사용:**

```fsharp
let inc x = x + 1
let double x = x * 2
let incThenDouble = compose double inc
let result = incThenDouble 5
```

**컴파일 과정:**

1. `compose double inc` → 중간 클로저 반환 (middle closure with f=double, g=inc)
2. `(compose double inc) 5` → 내부 클로저 호출 (inner with f, g, x=5)
3. 내부에서 `g(5)` → 6
4. 내부에서 `f(6)` → 12

**MLIR 호출 체인:**

```mlir
// 1. Wrap double as closure
%closure_double = /* ... */

// 2. Wrap inc as closure
%closure_inc = /* ... */

// 3. Create compose closure
%closure_compose = /* compose_outer의 empty closure */

// 4. Call compose(double)
%closure_partial1 = llvm.call %fn_ptr_compose(%closure_compose, %closure_double)
    : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr

// 5. Call (compose double)(inc)
%closure_partial2 = llvm.call %fn_ptr_partial1(%closure_partial1, %closure_inc)
    : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr

// 6. Call (compose double inc)(5)
%c5 = arith.constant 5 : i32
%result = llvm.call %fn_ptr_partial2(%closure_partial2, %c5)
    : (!llvm.ptr, i32) -> i32
```

**간접 호출의 연쇄:**

- `compose` 호출 → 클로저 반환
- `(compose double)` 호출 → 클로저 반환
- `(compose double inc)` 호출 → 클로저 반환
- `(compose double inc 5)` 호출 → 값 반환 (12)

모든 중간 단계가 **간접 호출**을 사용한다.

## 함수를 반환하기

### Upward Funarg Problem

**함수를 반환하는 함수**는 특별한 문제를 야기한다:

```fsharp
let makeAdder n =
    fun x -> x + n   // 이 클로저가 함수를 벗어나 반환된다
```

- `makeAdder`가 호출되면, 내부 람다 `fun x -> x + n`이 생성된다
- 이 람다는 `n`을 **캡처**한다
- 람다가 **makeAdder를 벗어나** 반환된다
- 반환된 후에도 `n`에 접근할 수 있어야 한다!

**Upward funarg problem:**

> 함수가 생성된 스코프를 벗어나 반환될 때, 캡처된 변수들이 어떻게 유지되는가?

**잘못된 해결책: 스택 할당**

```c
// 안 되는 C 코드
typedef int (*func_ptr)(int);

func_ptr makeAdder(int n) {
    int captured_n = n;   // 스택 변수
    return &inner_func;    // inner_func이 captured_n을 참조
}   // 여기서 captured_n이 소멸! Dangling pointer!
```

함수가 반환되면 스택 프레임이 소멸되므로, `captured_n`에 접근하면 **undefined behavior**다.

**올바른 해결책: 힙 할당**

환경을 **힙(heap)**에 할당하면, 함수가 반환되어도 환경이 유지된다:

```mlir
func.func @makeAdder(%n: i32) -> !llvm.ptr {
    // Allocate environment on heap (NOT stack!)
    %env_size = arith.constant 16 : i64
    %env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // env[0] = fn_ptr
    %fn_ptr = llvm.mlir.addressof @makeAdder_inner : !llvm.ptr
    // ... store fn_ptr ...

    // env[1] = n (captured)
    // ... store n ...

    func.return %env : !llvm.ptr   // 환경이 함수를 벗어나 반환
}
```

**GC의 역할:**

- 힙에 할당된 환경은 **GC가 관리**한다
- 클로저가 살아있는 동안 환경도 유지된다
- 클로저가 더 이상 사용되지 않으면 환경도 해제된다

**Chapter 12 설계의 정당성:**

Chapter 12에서 **모든 클로저를 힙에 할당**한 이유가 바로 이것이다. 클로저가 생성 스코프를 벗어날 수 있으므로, 항상 힙에 할당해야 안전하다.

### MakeAdder 구현

**MakeAdder 함수:**

```fsharp
let makeAdder n =
    fun x -> x + n
```

- 타입: `int -> (int -> int)`
- 의미: `n`을 받아서, "n을 더하는 함수"를 반환한다

**사용 예시:**

```fsharp
let add5 = makeAdder 5
let result1 = add5 10   // 15

let add10 = makeAdder 10
let result2 = add10 20  // 30
```

**AST:**

```fsharp
Let("makeAdder",
    Lambda("n",
        Lambda("x",
            Add(Var "x", Var "n"))),
    ...)
```

**Closure conversion:**

1. **내부 람다**: `fun x -> x + n` (n 캡처)
   - Lifted: `makeAdder_inner(env, x) = x + env[1]`
2. **외부 람다**: `fun n -> <inner closure>`
   - Lifted: `makeAdder_outer(env, n) = create_closure(makeAdder_inner, [n])`

**F# 컴파일 (간략):**

```fsharp
let compileMakeAdder (builder: OpBuilder) : unit =
    // Inner function: fun x -> x + n
    let innerFunc = builder.CreateFunction("makeAdder_inner",
        [builder.PtrType(); builder.IntType(32)],
        builder.IntType(32))

    builder.WithInsertionPoint(innerFunc, fun () ->
        let env = innerFunc.GetArgument(0)
        let x = innerFunc.GetArgument(1)

        // Load n from env[1]
        let c1 = builder.ConstantInt(1L, 64)
        let nAddr = builder.CreateGEP(env, [c1])
        let n = builder.CreateLoad(builder.IntType(32), nAddr)

        // Compute x + n
        let result = builder.CreateAdd(x, n)
        builder.CreateReturn(result)
    )

    // Outer function: fun n -> <inner closure>
    let outerFunc = builder.CreateFunction("makeAdder_outer",
        [builder.PtrType(); builder.IntType(32)],
        builder.PtrType())

    builder.WithInsertionPoint(outerFunc, fun () ->
        let envOuter = outerFunc.GetArgument(0)
        let n = outerFunc.GetArgument(1)

        // Allocate environment for inner closure (2 slots)
        let envSize = builder.ConstantInt(16L, 64)
        let envInner = builder.CreateGCMalloc(envSize)

        // env[0] = fn_ptr(inner)
        let c0 = builder.ConstantInt(0L, 64)
        let fnPtrInner = builder.CreateAddressOf(innerFunc)
        let slot0 = builder.CreateGEP(envInner, [c0])
        builder.CreateStore(fnPtrInner, slot0)

        // env[1] = n (captured)
        let c1 = builder.ConstantInt(1L, 64)
        let slot1 = builder.CreateGEP(envInner, [c1])
        builder.CreateStore(n, slot1)

        // Return inner closure (escapes function!)
        builder.CreateReturn(envInner)
    )
```

**완전한 MLIR IR:**

```mlir
// Inner function
func.func @makeAdder_inner(%env: !llvm.ptr, %x: i32) -> i32 {
    // Load n from env[1]
    %c1 = arith.constant 1 : i64
    %n_addr = llvm.getelementptr %env[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %n = llvm.load %n_addr : !llvm.ptr -> i32

    // x + n
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}

// Outer function
func.func @makeAdder_outer(%env_outer: !llvm.ptr, %n: i32) -> !llvm.ptr {
    // Allocate environment for inner closure
    %c16 = arith.constant 16 : i64
    %env_inner = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr

    // env[0] = fn_ptr(inner)
    %c0 = arith.constant 0 : i64
    %fn_ptr_inner = llvm.mlir.addressof @makeAdder_inner : !llvm.ptr
    %slot0 = llvm.getelementptr %env_inner[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_inner, %slot0 : !llvm.ptr, !llvm.ptr

    // env[1] = n
    %c1 = arith.constant 1 : i64
    %slot1 = llvm.getelementptr %env_inner[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %n, %slot1 : !llvm.ptr, !llvm.ptr

    // Return closure (environment escapes!)
    func.return %env_inner : !llvm.ptr
}
```

**테스트 코드:**

```mlir
func.func @main() -> i32 {
    // Create makeAdder closure (empty env)
    %c8 = arith.constant 8 : i64
    %env_makeAdder = llvm.call @GC_malloc(%c8) : (i64) -> !llvm.ptr
    %fn_ptr_makeAdder = llvm.mlir.addressof @makeAdder_outer : !llvm.ptr
    %c0 = arith.constant 0 : i64
    %slot0 = llvm.getelementptr %env_makeAdder[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr_makeAdder, %slot0 : !llvm.ptr, !llvm.ptr

    // Call makeAdder(5)
    %c5 = arith.constant 5 : i32
    %fn_ptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %add5 = llvm.call %fn_ptr(%env_makeAdder, %c5) : (!llvm.ptr, i32) -> !llvm.ptr
    // %add5 is a closure (inner function with n=5)

    // Call add5(10)
    %c10 = arith.constant 10 : i32
    %fn_ptr_inner = llvm.getelementptr %add5[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr_loaded = llvm.load %fn_ptr_inner : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr_loaded(%add5, %c10) : (!llvm.ptr, i32) -> i32
    // %result = 15

    func.return %result : i32
}
```

**실행:**

```bash
$ ./funlang makeAdder_test.fun
15
```

### 반환된 클로저의 생명주기

**환경은 언제 해제되는가?**

```fsharp
let add5 = makeAdder 5
// add5가 살아있는 동안, makeAdder의 환경도 유지된다
let result = add5 10   // OK
// add5가 스코프를 벗어나면, 환경도 GC에 의해 해제된다
```

**GC의 추적:**

- `add5` (클로저 포인터)가 살아있으면 → 환경 유지
- `add5`가 더 이상 참조되지 않으면 → GC가 환경 수거

**여러 클로저가 같은 환경을 공유하지 않는다:**

```fsharp
let add5 = makeAdder 5
let add10 = makeAdder 10
```

- `add5`와 `add10`은 **서로 다른 환경**을 가진다
- 각 `makeAdder` 호출이 새로운 환경을 힙에 할당한다

**메모리 누수 없음:**

GC가 자동으로 관리하므로, 프로그래머가 `free`를 호출할 필요 없다.

## 커링 패턴

### 다중 인자 함수를 클로저 체인으로 표현

**커링(Currying)**은 다중 인자 함수를 **중첩된 단일 인자 함수로 변환**하는 것이다:

```fsharp
// 다중 인자 함수 (Phase 3 스타일)
let add x y = x + y

// 커리된 함수 (Phase 4 스타일)
let add = fun x -> fun y -> x + y
```

- `add`의 타입: `int -> (int -> int)`
- `add`는 함수를 반환하는 함수다

**커링의 장점:**

1. **부분 적용(Partial application)**: `let add5 = add 5`
2. **합성 용이**: 커리된 함수는 파이프라인에 쉽게 통합됨
3. **일관된 타입 시스템**: 모든 함수가 단일 인자

**커링 예시:**

```fsharp
let add x y = x + y     // 실제로는 fun x -> fun y -> x + y

let add5 = add 5        // 부분 적용
let result = add5 10    // 15
```

### 커리된 함수의 컴파일

**AST:**

```fsharp
Let("add",
    Lambda("x",
        Lambda("y",
            Add(Var "x", Var "y"))),
    ...)
```

**Closure conversion:**

1. **내부 람다**: `fun y -> x + y` (x 캡처)
   - Lifted: `add_inner(env, y) = env[1] + y`
2. **외부 람다**: `fun x -> <inner closure>`
   - Lifted: `add_outer(env, x) = create_closure(add_inner, [x])`

**MLIR IR:**

```mlir
// Inner: fun y -> x + y
func.func @add_inner(%env: !llvm.ptr, %y: i32) -> i32 {
    // Load x from env[1]
    %c1 = arith.constant 1 : i64
    %x_addr = llvm.getelementptr %env[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %x = llvm.load %x_addr : !llvm.ptr -> i32

    // x + y
    %result = arith.addi %x, %y : i32
    func.return %result : i32
}

// Outer: fun x -> <inner closure>
func.func @add_outer(%env_outer: !llvm.ptr, %x: i32) -> !llvm.ptr {
    // Allocate environment for inner
    %c16 = arith.constant 16 : i64
    %env_inner = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr

    // env[0] = fn_ptr
    %c0 = arith.constant 0 : i64
    %fn_ptr = llvm.mlir.addressof @add_inner : !llvm.ptr
    %slot0 = llvm.getelementptr %env_inner[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr

    // env[1] = x
    %c1 = arith.constant 1 : i64
    %slot1 = llvm.getelementptr %env_inner[0, %c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %x, %slot1 : !llvm.ptr, !llvm.ptr

    func.return %env_inner : !llvm.ptr
}
```

**부분 적용:**

```fsharp
let add5 = add 5
```

```mlir
// Call add(5) -> returns closure with x=5
%c5 = arith.constant 5 : i32
%closure_add = /* ... */
%fn_ptr_add = /* load from closure_add[0] */
%add5 = llvm.call %fn_ptr_add(%closure_add, %c5) : (!llvm.ptr, i32) -> !llvm.ptr
// %add5 is inner closure with x=5 captured
```

**완전 적용:**

```fsharp
let result = add5 10
```

```mlir
// Call add5(10)
%c10 = arith.constant 10 : i32
%fn_ptr_inner = /* load from %add5[0] */
%result = llvm.call %fn_ptr_inner(%add5, %c10) : (!llvm.ptr, i32) -> i32
// %result = 15
```

**커링과 makeAdder의 유사성:**

- `makeAdder`는 명시적 함수 반환
- 커링은 암묵적 함수 반환 (다중 인자를 중첩 람다로 변환)
- 둘 다 **upward funarg problem** 해결 필요 (힙 할당)

### 3개 이상의 인자

```fsharp
let add3 x y z = x + y + z
// = fun x -> fun y -> fun z -> x + y + z
```

**중첩 구조:**

- 외부: `fun x -> ...` (비어있음)
- 중간: `fun y -> ...` (x 캡처)
- 내부: `fun z -> x + y + z` (x, y 캡처)

**MLIR에서 3단계 중첩:**

각 단계가 새로운 클로저를 생성하고, 이전 환경을 캡처한다. 복잡하지만 패턴은 동일하다.

## 메모리 관리와 클로저

### GC가 클로저 생명주기를 관리한다

**핵심 원칙:**

- 모든 클로저 환경은 **힙에 할당**된다 (`GC_malloc`)
- **GC가 자동으로 추적**하여, 사용되지 않으면 해제한다
- 프로그래머는 메모리 관리를 신경 쓸 필요 없다

**생명주기 예시:**

```fsharp
let createAdders () =
    let add5 = makeAdder 5
    let add10 = makeAdder 10
    add5    // add5만 반환, add10은 버려진다

let adder = createAdders()
let result = adder 20   // 25
```

**메모리 추적:**

1. `makeAdder 5` 호출 → 환경1 할당 (n=5)
2. `makeAdder 10` 호출 → 환경2 할당 (n=10)
3. `add5` 반환 → 환경1은 유지
4. `add10`은 스코프 벗어남 → 환경2는 GC 수거 대상
5. `adder` 사용 → 환경1 유지
6. `adder` 스코프 벗어남 → 환경1도 GC 수거

**Dangling pointer 없음:**

C/C++에서는 스택 포인터 반환이 위험하지만, GC 덕분에 FunLang은 안전하다:

```fsharp
let unsafeInC () =
    let local = 42
    fun () -> local   // C에서는 dangling pointer, FunLang에서는 OK
```

FunLang 컴파일러는 `local`을 환경에 캡처하고 힙에 할당하므로 안전하다.

### 순환 참조와 GC

**Cyclic closures (순환 클로저):**

```fsharp
let rec isEven n =
    if n = 0 then true
    else isOdd (n - 1)
and isOdd n =
    if n = 0 then false
    else isEven (n - 1)
```

- `isEven` 클로저가 `isOdd`를 캡처
- `isOdd` 클로저가 `isEven`을 캡처
- **순환 참조!**

**GC의 처리:**

Boehm GC는 **tracing GC**이므로, 순환 참조를 정확히 감지하고 해제한다:

- 루트(스택, 전역)에서 도달 가능한 객체만 유지
- 순환 참조가 루트에서 도달 불가능하면 → 수거

**Reference counting과의 차이:**

- **Reference counting**: 순환 참조를 해제하지 못함 (메모리 누수)
- **Tracing GC**: 순환 참조도 정확히 처리

Phase 2에서 Boehm GC를 선택한 이유가 이것이다.

### 클로저 생성 비용

**힙 할당 비용:**

- 클로저 생성 = `GC_malloc` 호출
- 스택 할당보다 느리지만, 안전성 보장

**최적화 가능성 (Phase 7):**

- **Escape analysis**: 클로저가 함수를 벗어나지 않으면 스택 할당 가능
- **Closure inlining**: 클로저가 즉시 호출되면 인라이닝 가능
- Phase 4는 최적화 없이 항상 힙 할당

**GC 오버헤드:**

- 주기적인 GC 실행 (pause time)
- 메모리 오버헤드 (fragmentation)
- 하지만 프로그래머 생산성은 크게 향상

## Complete 예시: Map 함수

### Map의 개념

**Map 함수**는 리스트의 각 원소에 함수를 적용한다:

```fsharp
// 개념적 정의 (Phase 6에서 완전 구현)
let rec map f list =
    match list with
    | [] -> []
    | head :: tail -> (f head) :: (map f tail)
```

- 타입: `(a -> b) -> list a -> list b`
- 의미: `f`를 각 원소에 적용해 새 리스트 생성

**Map 예시:**

```fsharp
let inc x = x + 1
let numbers = [1; 2; 3; 4]
let incremented = map inc numbers   // [2; 3; 4; 5]

let double x = x * 2
let doubled = map double numbers    // [2; 4; 6; 8]
```

### Phase 4의 Map (단순화 버전)

Phase 4에는 리스트가 없으므로, **개념적 설명**만 한다. 핵심은 **HOF 패턴**이다:

```fsharp
// 단순화: 두 원소 "리스트"만 처리
let map2 f x y =
    let fx = f x
    let fy = f y
    (fx, fy)   // Phase 6에서는 실제 리스트 반환
```

**컴파일:**

```mlir
func.func @map2(%f: !llvm.ptr, %x: i32, %y: i32) -> (i32, i32) {
    // Call f(x)
    %c0 = arith.constant 0 : i64
    %fn_ptr_addr = llvm.getelementptr %f[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %fx = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32

    // Call f(y)
    %fy = llvm.call %fn_ptr(%f, %y) : (!llvm.ptr, i32) -> i32

    // Return pair (fx, fy) - Phase 6에서는 리스트
    // ... (tuple 구현 생략)
}
```

**Map의 핵심:**

- 함수 `f`를 **인자로 받는다** (고차 함수)
- `f`를 **여러 번 호출**한다 (간접 호출)
- 각 호출마다 `f`의 환경을 전달한다

**Map + Closure:**

```fsharp
let addN n = fun x -> x + n
let add5 = addN 5
let result = map2 add5 10 20   // (15, 25)
```

- `add5`는 클로저 (n=5 캡처)
- `map2`가 `add5`를 받아서 두 번 호출
- 각 호출마다 캡처된 `n` 사용

이것이 **클로저 + 고차 함수 = 함수형 프로그래밍 핵심** 조합이다.

## 자주 하는 실수 (Common Errors)

### Error 1: 클로저를 첫 인자로 전달하지 않음

**문제:**

```mlir
// 잘못된 호출 - 환경 누락
%result = llvm.call %fn_ptr(%arg) : (i32) -> i32
```

Lifted function은 **첫 번째 파라미터로 환경을 받는다**:

```mlir
func.func @lifted(%env: !llvm.ptr, %arg: i32) -> i32
```

환경 없이 호출하면 **타입 미스매치** 또는 **segfault**:

```
ERROR: Call argument count mismatch (expected 2, got 1)
```

**해결:**

```mlir
// 올바른 호출 - 클로저(환경)를 첫 인자로
%result = llvm.call %fn_ptr(%closure, %arg) : (!llvm.ptr, i32) -> i32
```

**F# 헬퍼 사용:**

```fsharp
// 자동으로 클로저를 첫 인자로 전달
builder.CallClosure(closure, [arg], resultType)
```

### Error 2: 클로저 본체를 직접 호출

**문제:**

```mlir
// 잘못된 호출 - lifted function을 직접 호출
%result = func.call @lifted_func(%env, %arg) : (!llvm.ptr, i32) -> i32
```

Lifted function은 **내부 함수**이고, 직접 호출하면 **환경이 잘못 전달**될 수 있다.

**올바른 방법:**

1. 클로저에서 함수 포인터 추출
2. 간접 호출 (`llvm.call`)

```mlir
// 올바른 호출 - 클로저를 통해 간접 호출
%fn_ptr_addr = llvm.getelementptr %closure[0, 0] : (!llvm.ptr, i64) -> !llvm.ptr
%fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
%result = llvm.call %fn_ptr(%closure, %arg) : (!llvm.ptr, i32) -> i32
```

**예외:**

테스트 목적으로 직접 호출할 수는 있지만, 일반적인 패턴은 아니다.

### Error 3: 스택에 반환 클로저의 환경 할당

**문제:**

```mlir
func.func @makeAdder_wrong(%n: i32) -> !llvm.ptr {
    // 잘못됨 - 스택 할당!
    %c1 = arith.constant 1 : index
    %env = memref.alloca(%c1) : memref<?xi64>
    // ... store function pointer and n ...
    %ptr = memref.extract_aligned_pointer_as_index %env : memref<?xi64> -> !llvm.ptr
    func.return %ptr : !llvm.ptr
}   // 함수 종료 시 %env 소멸! Dangling pointer!
```

함수가 반환되면 **스택 프레임이 소멸**되므로, 환경 접근 시 undefined behavior.

**해결:**

```mlir
func.func @makeAdder_correct(%n: i32) -> !llvm.ptr {
    // 올바름 - 힙 할당
    %env_size = arith.constant 16 : i64
    %env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
    // ... store function pointer and n ...
    func.return %env : !llvm.ptr
}   // %env는 힙에 있으므로 안전
```

**원칙:**

- **반환되는 클로저**: 항상 `GC_malloc` 사용
- **로컬 클로저** (스코프 벗어나지 않음): 스택 가능 (Phase 7 최적화)

### Error 4: 간접 호출 시 타입 미스매치

**문제:**

```mlir
// Lifted function: (%env: !llvm.ptr, %x: i32) -> i32
func.func @lifted(%env: !llvm.ptr, %x: i32) -> i32

// 잘못된 호출 - 타입 불일치
%result = llvm.call %fn_ptr(%closure) : (!llvm.ptr) -> i32   // 인자 누락
```

LLVM IR에서 타입 불일치는 **검증 실패** 또는 **런타임 크래시**:

```
ERROR: Function signature mismatch in indirect call
```

**해결:**

간접 호출 시 **정확한 시그니처** 명시:

```mlir
// 올바른 호출 - 모든 인자와 정확한 타입
%result = llvm.call %fn_ptr(%closure, %x) : (!llvm.ptr, i32) -> i32
```

**타입 시그니처 유지:**

- Lifted function 정의와 간접 호출 타입이 **정확히 일치**해야 함
- 컴파일러가 자동으로 추론하도록 구현

### Error 5: 클로저 동일성 혼동

**문제:**

```fsharp
let f = fun x -> x + 1
let g = fun x -> x + 1
// f와 g는 같은가?
```

**답: 아니다!**

- `f`와 `g`는 **서로 다른 클로저**다
- 각각 **다른 환경 포인터**를 가진다 (빈 환경이더라도)
- **포인터 비교**: `f != g` (주소가 다름)

**의미적 동등성 vs 포인터 동등성:**

- **의미적 동등성**: 같은 동작 (extensional equality)
- **포인터 동등성**: 같은 객체 (intensional equality)

FunLang은 **포인터 동등성**만 지원한다 (대부분의 언어와 동일).

**예시:**

```mlir
// 두 클로저 생성
%closure1 = /* fun x -> x + 1 */
%closure2 = /* fun x -> x + 1 */

// 포인터 비교
%same = llvm.icmp "eq" %closure1, %closure2 : !llvm.ptr
// %same = false (주소가 다름)
```

**함수 메모이제이션:**

의미적 동등성이 필요하면 **명시적 비교 로직** 구현 필요 (Phase 7).

## Phase 4 완료 요약

### 무엇을 구현했는가

**Phase 4 - Closures & Higher-Order Functions:**

1. **Chapter 12 - Closures:**
   - 클로저 이론 (lexical scoping, free/bound variables)
   - 자유 변수 분석 알고리즘
   - 클로저 변환 (closure conversion)
   - 환경 구조체 (힙 할당)
   - GC_malloc으로 클로저 생성

2. **Chapter 13 - Higher-Order Functions:**
   - 함수를 일급 값으로 다루기
   - 간접 호출 패턴 (`llvm.call` with function pointer)
   - Apply 함수 (함수를 인자로 받기)
   - Compose 함수 (여러 함수 인자)
   - 함수를 반환하기 (makeAdder, upward funarg problem)
   - 커링 패턴 (다중 인자 → 중첩 람다)
   - 메모리 관리 (GC가 클로저 생명주기 관리)
   - 자주 하는 실수 5가지

**핵심 구현 항목:**

| 항목 | 설명 | MLIR 패턴 |
|-----|------|----------|
| **클로저 표현** | (fn_ptr, env) 쌍 | `!llvm.ptr` |
| **환경 할당** | 힙에 GC_malloc | `llvm.call @GC_malloc` |
| **간접 호출** | 함수 포인터 로드 후 호출 | `llvm.call %fn_ptr(...)` |
| **환경 접근** | GEP + load | `llvm.getelementptr + llvm.load` |
| **클로저 생성** | 환경 할당 + 변수 저장 | `GC_malloc + store` |
| **함수 반환** | 클로저 반환 (escaping) | `func.return %env` |

**타입 시스템:**

- 모든 함수/클로저: `!llvm.ptr` (opaque pointer)
- 함수 타입 (개념적): `a -> b` = `(!llvm.ptr, a) -> b` (lifted)

### Phase 4가 가능하게 한 것

**이제 컴파일할 수 있는 것:**

```fsharp
// 1. 클로저 생성
let makeAdder n = fun x -> x + n

// 2. 고차 함수
let apply f x = f x
let compose f g x = f (g x)

// 3. 부분 적용
let add5 = makeAdder 5
let result = add5 10   // 15

// 4. 함수 합성
let inc x = x + 1
let double x = x * 2
let incThenDouble = compose double inc
let result2 = incThenDouble 5   // 12

// 5. 콜백 패턴
let processWithCallback callback data =
    let result = compute data
    callback result

// 6. 커링
let add x y = x + y   // = fun x -> fun y -> x + y
let add5 = add 5
```

**함수형 프로그래밍의 핵심:**

- ✅ 클로저 (환경 캡처)
- ✅ 고차 함수 (함수 인자/반환)
- ✅ 부분 적용
- ✅ 함수 합성
- ⏸️ Map, filter, fold (Phase 6에서 리스트 추가 후)

### 다음 단계: Phase 5 - Custom MLIR Dialect

**Phase 5 목표:**

FunLang 전용 MLIR dialect 설계 및 구현:

1. **FunLang Dialect 정의:**
   - `funlang.closure` 연산 (클로저 생성 추상화)
   - `funlang.closure_call` 연산 (간접 호출 추상화)
   - `funlang.capture` 연산 (환경 저장 추상화)

2. **Lowering passes:**
   - FunLang dialect → Func/LLVM dialect
   - 고수준 의미론 → 저수준 MLIR

3. **이점:**
   - 컴파일러 코드 단순화 (고수준 연산 사용)
   - 최적화 pass 추가 용이 (dialect-specific 변환)
   - 타입 안전성 향상 (dialect 타입 시스템)

**Phase 4 vs Phase 5:**

| Phase 4 | Phase 5 |
|---------|---------|
| 저수준 LLVM dialect 직접 생성 | 고수준 FunLang dialect 생성 |
| GEP, load, store 수동 관리 | 추상화된 연산 사용 |
| 최적화 어려움 | Dialect 최적화 pass |

**Preview:**

```mlir
// Phase 4 (저수준)
%env = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @func : !llvm.ptr
%slot0 = llvm.getelementptr %env[0, 0] : (!llvm.ptr, i64) -> !llvm.ptr
llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
// ... (환경 저장)

// Phase 5 (고수준)
%closure = funlang.closure @func, %captured_vars : (i32) -> i32
```

Phase 5는 **컴파일러 품질**을 향상시킨다. Phase 4는 **기능 완성**이다.

---

**Congratulations!** Phase 4 완료. 이제 FunLang은 **완전한 함수형 언어**의 핵심 기능을 지원한다.
