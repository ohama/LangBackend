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

