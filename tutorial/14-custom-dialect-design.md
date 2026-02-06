# Chapter 14: 커스텀 Dialect 설계 (Custom Dialect Design)

## 소개

**커스텀 MLIR dialect**는 도메인별 언어(DSL)를 높은 수준에서 표현하는 방법이다. FunLang 컴파일러는 Phase 4까지 빌트인 dialect만 사용했다:

- `arith`: 산술 연산
- `func`: 함수 정의와 호출
- `scf`: 구조적 제어 흐름 (if/while)
- `llvm`: LLVM IR 타입과 operation

이 dialect들은 강력하고 범용적이지만, **FunLang의 고수준 의미론을 직접 표현하지 못한다**. Phase 4에서 클로저를 구현할 때 복잡성이 드러났다.

### Phase 4의 문제점: 저수준 LLVM Dialect의 복잡성

Chapter 12에서 클로저를 구현할 때, 단순한 개념을 복잡한 저수준 코드로 표현해야 했다:

```fsharp
// 개념적으로 단순한 코드
let make_adder n =
    fun x -> x + n
```

**생성된 MLIR (Phase 4 방식):**

```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // 1. 환경 크기 계산: 8 (fn ptr) + 4 (n) = 12 bytes
    %env_size = arith.constant 12 : i64

    // 2. GC_malloc 호출
    %env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // 3. 함수 포인터 저장 (env[0])
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr

    // 4. 캡처된 변수 n 저장 (env[1])
    %n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr

    // 5. 환경 포인터 반환 (클로저)
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

**문제점:**

1. **GEP (getelementptr) 패턴의 반복**
   - 환경 슬롯 접근할 때마다 `getelementptr + load` 패턴 필요
   - 인덱스 관리 오류 발생 가능 (env[0] vs env[1])
   - 코드 가독성 저하

2. **저수준 메모리 관리 노출**
   - `GC_malloc` 크기 계산 (8 + 4 bytes?)
   - 포인터 산술 명시적 작성
   - 타입 불일치 가능성 (i32 vs !llvm.ptr)

3. **도메인 의미론 상실**
   - "클로저"라는 개념이 안 보인다
   - "환경 포인터" = `!llvm.ptr` (opaque, 타입 안전성 없음)
   - 최적화 pass 작성 어려움 (어떤 포인터가 클로저인지?)

4. **컴파일러 코드 복잡성 폭발**
   - F# 컴파일러 코드가 저수준 세부사항 처리
   - 변수 1개 추가할 때마다 GEP 인덱스 계산
   - 에러 가능성 증가

**실제 컴파일러 코드 (Phase 4):**

```fsharp
// Lambda 컴파일 (Phase 4 버전)
let compileLambda (builder: OpBuilder) (param: string) (body: Expr) (capturedVars: (string * MlirValue) list) =
    // 1. 환경 크기 계산 (수동!)
    let fnPtrSize = 8L
    let varSize = 4L  // i32 가정
    let totalSize = fnPtrSize + (int64 capturedVars.Length) * varSize
    let sizeConst = builder.CreateI64Const(totalSize)

    // 2. GC_malloc 호출
    let envPtr = builder.CreateCall("GC_malloc", [sizeConst])

    // 3. 함수 포인터 저장 (getelementptr 0)
    let fnSlot = builder.CreateGEP(envPtr, 0)
    builder.CreateStore(fnAddr, fnSlot)

    // 4. 캡처된 변수들 저장 (getelementptr 1, 2, 3...)
    capturedVars |> List.iteri (fun i (name, value) ->
        let slot = builder.CreateGEP(envPtr, i + 1)
        builder.CreateStore(value, slot)
    )

    envPtr
```

크기 계산, GEP 인덱스 관리, 타입 추론 등 저수준 세부사항이 컴파일러 로직에 섞여있다.

### Custom Dialect의 이점

**커스텀 dialect**를 사용하면 **높은 수준에서 의미론을 표현**할 수 있다. 같은 코드를 FunLang dialect로 표현한다면:

```mlir
func.func @make_adder(%n: i32) -> !funlang.closure {
    // 클로저 생성 - 고수준 operation
    %closure = funlang.make_closure @lambda_adder(%n) : !funlang.closure
    func.return %closure : !funlang.closure
}

func.func @lambda_adder(%x: i32, %n: i32) -> i32 {
    // 캡처된 변수는 파라미터로 전달 (환경 명시적 관리 불필요)
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}
```

**변화:**

1. **도메인 의미론 보존**
   - `!funlang.closure` 타입: 클로저임을 명시
   - `funlang.make_closure`: 클로저 생성의 의도가 명확
   - GEP, malloc 등 구현 세부사항 숨김

2. **컴파일러 코드 단순화**

```fsharp
// Lambda 컴파일 (Phase 5 버전 - 커스텀 dialect 사용)
let compileLambda (builder: OpBuilder) (param: string) (body: Expr) (capturedVars: (string * MlirValue) list) =
    // 간단! dialect operation 호출
    let capturedValues = capturedVars |> List.map snd
    builder.CreateFunLangClosure(lambdaFuncName, capturedValues)
```

환경 크기, GEP 인덱스, 메모리 레이아웃 등이 dialect operation 구현 안으로 캡슐화된다.

3. **타입 안전성 향상**
   - `!llvm.ptr` (모든 포인터) → `!funlang.closure` (클로저 전용)
   - 타입 체커가 클로저 오용 방지 가능
   - 예: 정수 포인터를 클로저로 사용하려는 시도 방지

4. **최적화 기회 증가**
   - Dialect-specific optimization pass 작성 가능
   - 예: 환경에 변수 1개만 있을 때 inline 최적화
   - 예: 탈출하지 않는 클로저는 stack 할당

5. **디버깅 용이성**
   - 높은 수준 IR을 먼저 검증 가능
   - 에러 메시지가 도메인 용어 사용 ("closure type mismatch" vs "pointer type mismatch")

### Progressive Lowering: 왜 점진적으로 낮추는가?

**Progressive lowering (점진적 하강)**은 높은 수준 표현을 여러 단계로 낮추는 전략이다:

```
FunLang Dialect (highest level, domain-specific)
    ↓ (FunLangToStandard lowering pass)
Func + SCF + MemRef (mid-level, general purpose)
    ↓ (StandardToLLVM lowering pass)
LLVM Dialect (low-level, machine-oriented)
    ↓ (MLIR-to-LLVM translation)
LLVM IR → Machine Code
```

**Before/After 비교:**

| Phase 4 (Direct lowering) | Phase 5 (Progressive lowering) |
|--------------------------|--------------------------------|
| FunLang AST → LLVM Dialect | FunLang AST → FunLang Dialect |
| 단일 거대 변환 | → Func/SCF/MemRef Dialect |
| 의미론 상실 즉시 | → LLVM Dialect |
| 최적화 불가 | 각 단계에서 최적화 가능 |
| 디버깅 어려움 | 각 단계 독립 검증 가능 |

### Chapter 14의 목표

이 장에서 다루는 것:

1. **MLIR Dialect 아키텍처**: Operation, Type, Attribute의 역할
2. **Progressive Lowering 철학**: 왜 여러 단계로 낮추는가?
3. **TableGen ODS**: MLIR operation 정의 DSL
4. **C API Shim 패턴**: C++ dialect를 F#에 연결
5. **FunLang Dialect 설계**: 어떤 operation을 만들 것인가?

이 장을 마치면:
- 커스텀 dialect가 왜 필요한지 이해한다
- TableGen ODS 문법을 읽고 쓸 수 있다
- C API shim 패턴으로 F# interop 할 수 있다
- FunLang dialect의 operation과 type을 설계할 수 있다
- Progressive lowering 경로를 계획할 수 있다

> **Preview:** Chapter 15에서는 실제로 FunLang dialect를 구현하고 lowering pass를 작성한다. Chapter 14는 이론적 기초를 확립한다.

## MLIR Dialect 아키텍처

MLIR의 핵심 강점은 **extensibility (확장성)**다. 새 dialect를 정의해서 도메인별 개념을 표현할 수 있다.

### Dialect Hierarchy 개념

MLIR 프로그램은 **여러 dialect의 operation이 섞여있다**:

```mlir
func.func @example(%arg: i32) -> i32 {
    // arith dialect
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg, %c1 : i32

    // scf dialect
    %result = scf.if %cond -> i32 {
        scf.yield %sum : i32
    } else {
        scf.yield %arg : i32
    }

    // func dialect
    func.return %result : i32
}
```

각 operation은 `dialect.operation` 형식으로 네임스페이스를 가진다:
- `arith.constant`: arith dialect의 constant operation
- `scf.if`: scf dialect의 if operation
- `func.return`: func dialect의 return operation

**Dialect hierarchy (계층 구조):**

```
┌────────────────────────────────────────┐
│  FunLang Dialect (highest level)      │
│  - funlang.closure                     │
│  - funlang.apply                       │
│  - funlang.match (Phase 6)             │
└──────────────┬─────────────────────────┘
               │ (lowering pass)
               ↓
┌────────────────────────────────────────┐
│  Standard Dialects (mid-level)         │
│  - func.func, func.call                │
│  - scf.if, scf.while                   │
│  - memref.alloc, memref.load           │
└──────────────┬─────────────────────────┘
               │ (lowering pass)
               ↓
┌────────────────────────────────────────┐
│  LLVM Dialect (low-level)              │
│  - llvm.getelementptr                  │
│  - llvm.load, llvm.store               │
│  - llvm.call                           │
└──────────────┬─────────────────────────┘
               │ (translation)
               ↓
┌────────────────────────────────────────┐
│  LLVM IR                               │
└────────────────────────────────────────┘
```

**높은 수준일수록:**
- 도메인 개념 명확 (funlang.closure vs !llvm.ptr)
- 최적화 기회 많음 (의미론 활용 가능)
- 플랫폼 독립적

**낮은 수준일수록:**
- 기계 모델에 가까움 (레지스터, 메모리, 포인터)
- 구현 세부사항 노출
- 플랫폼 특화

### Operation, Type, Attribute의 역할

MLIR dialect는 세 가지 확장 포인트를 제공한다:

#### 1. Operation (연산)

**Operation**은 계산 단위다. FunLang dialect operation 예시:

```mlir
// funlang.make_closure operation
%closure = funlang.make_closure @lambda_func(%n, %m) : !funlang.closure

// funlang.apply operation
%result = funlang.apply %closure(%x) : (i32) -> i32
```

**Operation 구성 요소:**

- **Name**: `funlang.make_closure` (dialect.operation 형식)
- **Operands**: `@lambda_func`, `%n`, `%m` (입력 값)
- **Results**: `%closure` (출력 값)
- **Types**: `!funlang.closure`, `i32` (타입 정보)
- **Attributes**: `@lambda_func` (컴파일 타임 상수)
- **Regions**: 중첩 코드 블록 (예: scf.if의 then/else 블록)

**Operation의 역할:**
- 도메인별 계산 표현 (클로저 생성, 패턴 매칭 등)
- Verifier로 정적 검증 (타입 체크, 불변식)
- Lowering 대상 (다른 dialect operation으로 변환)

#### 2. Type (타입)

**Type**은 값의 종류를 표현한다. FunLang dialect type 예시:

```mlir
// funlang.closure 타입
%closure : !funlang.closure

// funlang.list 타입 (Phase 6)
%list : !funlang.list<i32>
```

**빌트인 타입 vs 커스텀 타입:**

| 빌트인 타입 | 커스텀 타입 |
|-----------|-----------|
| `i32`, `i64`, `f32` | `!funlang.closure` |
| `!llvm.ptr` | `!funlang.list<i32>` |
| `tensor<10xf32>` | `!funlang.record<{x:i32, y:i32}>` |
| 범용적 | 도메인 특화 |

**타입의 역할:**
- 값의 의미론 표현 (closure vs raw pointer)
- 타입 체커가 오용 방지
- 최적화 hint (closure는 함수 포인터 + 환경)

#### 3. Attribute (속성)

**Attribute**는 컴파일 타임 상수 값이다:

```mlir
// IntegerAttr
%c1 = arith.constant 1 : i32

// SymbolRefAttr (함수 이름)
%fn = func.call @my_function(%arg) : (i32) -> i32

// StringAttr
%str = llvm.mlir.global "hello"

// ArrayAttr
#array = [1, 2, 3, 4]
```

FunLang dialect에서 attribute 사용:

```mlir
// 클로저가 참조하는 함수 (SymbolRefAttr)
%closure = funlang.make_closure @lambda_func(%n) : !funlang.closure

// 패턴 매칭 케이스 (ArrayAttr)
%result = funlang.match %value {
    #funlang.pattern<constructor="Nil"> -> { ... }
    #funlang.pattern<constructor="Cons"> -> { ... }
}
```

**Attribute의 역할:**
- 컴파일 타임 정보 저장 (함수 이름, 상수 등)
- Serialization (MLIR IR을 파일에 저장)
- Lowering 힌트

### Region과 Block (Phase 1 복습)

Chapter 01에서 배운 개념 다시 보기:

**Region**: operation 내부의 코드 영역

```mlir
scf.if %cond -> i32 {
    // ↑ Region 1 (then block)
    %result = arith.addi %a, %b : i32
    scf.yield %result : i32
} else {
    // ↑ Region 2 (else block)
    %result = arith.subi %a, %b : i32
    scf.yield %result : i32
}
```

**Block**: region 내부의 명령어 시퀀스

```mlir
func.func @example(%arg: i32) -> i32 {
^entry:  // ↑ Block label
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg, %c1 : i32
    func.return %sum : i32
}
```

**FunLang dialect에서 region 사용 가능?**

가능하다. 예를 들어 `funlang.match` operation은 패턴별 region을 가질 수 있다:

```mlir
%result = funlang.match %list : !funlang.list<i32> -> i32 {
    // Nil case
    ^nil_case:
        %zero = arith.constant 0 : i32
        funlang.yield %zero : i32

    // Cons case
    ^cons_case(%head: i32, %tail: !funlang.list<i32>):
        %sum = funlang.apply %f(%head) : (i32) -> i32
        funlang.yield %sum : i32
}
```

각 케이스가 별도 block을 가진다. 이렇게 **structured control flow**를 dialect operation으로 표현할 수 있다.

### Symbol Table과 함수 참조

MLIR은 **symbol table**을 사용해 함수, 전역 변수 등을 참조한다.

**Symbol (심볼):**

```mlir
// 함수 정의 - symbol
func.func @my_function(%arg: i32) -> i32 {
    func.return %arg : i32
}

// 함수 참조 - SymbolRefAttr
%result = func.call @my_function(%x) : (i32) -> i32
```

`@my_function`은 **SymbolRefAttr**이다:
- 컴파일 타임에 해석됨
- 타입 체커가 함수 시그니처 검증
- Linker가 심볼 해석

**FunLang dialect에서 symbol 사용:**

```mlir
// 람다 함수 정의 (lifted)
func.func private @lambda_adder(%env: !funlang.env, %x: i32) -> i32 {
    // ...
}

// 클로저 생성 - 함수 심볼 참조
%closure = funlang.make_closure @lambda_adder(%n) : !funlang.closure
```

`@lambda_adder`가 심볼이다. 클로저는 이 심볼을 참조해서 함수 포인터를 얻는다.

**Symbol vs SSA Value:**

| Symbol | SSA Value |
|--------|-----------|
| `@func_name` | `%result` |
| 컴파일 타임 상수 | 런타임 값 |
| 전역 참조 가능 | 로컬 스코프만 |
| 함수, 전역 변수 | operation 결과 |

Phase 4에서 사용한 `llvm.mlir.addressof @lambda_func`도 심볼을 사용한다:

```mlir
// 함수 심볼 주소 얻기
%fn_addr = llvm.mlir.addressof @lambda_func : !llvm.ptr
```

### DialectRegistry와 의존성 선언

**DialectRegistry**는 context에 dialect를 등록하는 메커니즘이다.

**Phase 1-4 코드 (빌트인 dialect 등록):**

```fsharp
// MlirHelpers.fs
let createContextWithDialects() =
    let ctx = MlirContext.Create()

    // 빌트인 dialect 등록
    let arithHandle = Mlir.mlirGetDialectHandle__arith__()
    Mlir.mlirDialectHandleRegisterDialect(arithHandle, ctx.Handle)

    let funcHandle = Mlir.mlirGetDialectHandle__func__()
    Mlir.mlirDialectHandleRegisterDialect(funcHandle, ctx.Handle)

    // ... scf, llvm 등

    ctx
```

**Phase 5 코드 (커스텀 dialect 추가):**

```fsharp
// FunLang dialect 등록
let ctx = createContextWithDialects()

// C API shim 호출
FunLangDialect.RegisterDialect(ctx)
```

**의존성 선언:**

FunLang dialect는 다른 dialect를 사용할 수 있다:

```cpp
// FunLang dialect 정의 (C++)
class FunLangDialect : public Dialect {
public:
    FunLangDialect(MLIRContext *context) : ... {
        // 의존성 선언
        addDependentDialect<func::FuncDialect>();
        addDependentDialect<arith::ArithDialect>();
        addDependentDialect<LLVM::LLVMDialect>();
    }
};
```

이렇게 하면:
- FunLang operation이 func, arith operation을 사용 가능
- Lowering pass에서 func.call, arith.addi 생성 가능
- Context가 필요한 dialect 자동 로드

### FunLang Dialect 계층 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLIR Context                                 │
│  (모든 dialect의 컨테이너)                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ FunLang       │   │ BuiltIn       │   │ LLVM          │
│ Dialect       │   │ Dialect       │   │ Dialect       │
│               │   │ (func, scf,   │   │               │
│ - closure     │   │  arith)       │   │ - ptr         │
│ - apply       │   │               │   │ - call        │
│ - match       │   │ - func.func   │   │ - gep         │
└───────┬───────┘   │ - scf.if      │   │ - load/store  │
        │           │ - arith.addi  │   └───────────────┘
        │           └───────────────┘
        │
        │  (의존성)
        └──────────────────┐
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌───────────────┐     ┌───────────────┐
        │ Types         │     │ Operations    │
        │               │     │               │
        │ - closure     │     │ - make_closure│
        │ - list<T>     │     │ - apply       │
        │ - record<...> │     │ - match       │
        └───────────────┘     └───────────────┘
```

**Dialect 간 관계:**

1. **FunLang Dialect**: 최상위, 도메인 특화
   - 의존: func, scf, arith, llvm dialect
   - 제공: funlang.* operation/type

2. **BuiltIn Dialects**: 중간 수준, 범용
   - 의존: 최소 (arith는 독립적)
   - 제공: func.*, scf.*, arith.* operation

3. **LLVM Dialect**: 최하위, 기계 지향
   - 의존: 없음 (target-independent LLVM IR)
   - 제공: llvm.* operation

**Lowering 경로:**

```
funlang.make_closure
    ↓ (FunLangToFunc lowering)
func.func + memref.alloc + func.call
    ↓ (FuncToLLVM lowering)
llvm.call + llvm.getelementptr + llvm.store
    ↓ (MLIR-to-LLVM translation)
LLVM IR: call, getelementptr, store
```

## Progressive Lowering 철학

### Why Not Direct FunLang → LLVM Lowering?

컴파일러를 설계할 때 유혹이 있다: "FunLang AST를 바로 LLVM dialect로 낮추면 빠르지 않을까?"

**직접 lowering의 문제점:**

#### 1. 최적화 기회 상실

**예시: 클로저 inlining**

```fsharp
// FunLang 코드
let apply f x = f x

let result = apply (fun y -> y + 1) 42
```

**Direct lowering (FunLang → LLVM):**

```mlir
// 클로저 생성 (즉시 LLVM dialect)
%env = llvm.call @GC_malloc(...) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @lambda_0 : !llvm.ptr
%fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %fn_ptr, %fn_slot : !llvm.ptr, !llvm.ptr
// ... (환경 저장)

// 클로저 호출 (간접 호출)
%fn_ptr_loaded = llvm.load %fn_slot : !llvm.ptr -> !llvm.ptr
%result = llvm.call %fn_ptr_loaded(%env, %x) : (!llvm.ptr, i32) -> i32
```

문제: **LLVM 수준에서는 이것이 즉시 사용되는 클로저인지 알 수 없다**. 최적화 pass가 malloc, store, load, call 패턴을 분석해야 하는데, 이미 의미론이 상실됨.

**Progressive lowering (FunLang → Func → LLVM):**

```mlir
// Step 1: FunLang dialect (high-level)
%closure = funlang.make_closure @lambda_0() : !funlang.closure
%result = funlang.apply %closure(%x) : (i32) -> i32

// Optimization pass: closure inlining (FunLang dialect level)
// "이 클로저는 즉시 사용되고 탈출하지 않는다" → inline!
%result = func.call @lambda_0(%x) : (i32) -> i32

// Step 2: Lower to LLVM (이미 최적화됨)
%result = llvm.call @lambda_0(%x) : (i32) -> i32
```

**높은 수준에서 최적화하면:**
- 의미론이 명확 (closure + apply = inline candidate)
- 패턴 매칭 쉬움 (GEP + load 추적 불필요)
- 변환이 안전함 (타입 체커가 검증)

#### 2. 코드 복잡성 폭발

**Direct lowering 컴파일러 코드:**

```fsharp
// compileLambda: FunLang AST → LLVM dialect
let rec compileLambda (builder: OpBuilder) (lambda: Expr) =
    match lambda with
    | Lambda(param, body) ->
        // 1. 자유 변수 분석
        let freeVars = analyzeFreeVars lambda

        // 2. 환경 크기 계산 (수동!)
        let envSize = 8L + (int64 freeVars.Length) * 4L
        let sizeConst = builder.CreateI64Const(envSize)

        // 3. GC_malloc 호출
        let malloc = builder.CreateCall("GC_malloc", [sizeConst])

        // 4. 함수 포인터 저장 (GEP 0)
        let fnAddr = builder.CreateAddressOf(lambdaName)
        let fnSlot = builder.CreateGEP(malloc, 0)
        builder.CreateStore(fnAddr, fnSlot)

        // 5. 변수 저장 (GEP 1, 2, 3...)
        freeVars |> List.iteri (fun i var ->
            let value = compileExpr builder var
            let slot = builder.CreateGEP(malloc, i + 1)
            builder.CreateStore(value, slot)
        )

        // 6. 람다 함수 정의 (별도 함수)
        let lambdaFunc = builder.CreateFunction(lambdaName)
        // ... (환경 파라미터, body 컴파일, GEP + load for captures)

        malloc
```

**모든 세부사항이 한 함수에 섞여있다:**
- 메모리 레이아웃 계산
- GEP 인덱스 관리
- 타입 변환
- 함수 생성

**Progressive lowering 컴파일러 코드:**

```fsharp
// Step 1: FunLang AST → FunLang dialect
let rec compileLambda (builder: OpBuilder) (lambda: Expr) =
    match lambda with
    | Lambda(param, body) ->
        let freeVars = analyzeFreeVars lambda
        let capturedValues = freeVars |> List.map (compileExpr builder)

        // 간단! dialect operation 호출
        builder.CreateFunLangClosure(lambdaName, capturedValues)

// Step 2: FunLang dialect → Func dialect (별도 lowering pass)
// 이 pass에서 malloc, GEP, store 처리
class FunLangToFuncLowering : public RewritePattern {
    LogicalResult matchAndRewrite(MakeClosureOp op, ...) {
        // 여기서 환경 할당, 함수 포인터 저장 등 처리
        // 재사용 가능한 로직, 독립적 테스트 가능
    }
};
```

**코드가 계층화된다:**
- AST → Dialect: 의미론 변환 (단순)
- Dialect → Dialect: 구현 세부사항 (재사용 가능)
- Dialect → LLVM: 기계 코드 생성 (표준 패턴)

#### 3. 디버깅 어려움

**Direct lowering:**

```
FunLang AST → [Giant Black Box] → LLVM Dialect
```

에러가 발생하면:
- LLVM IR에서 segfault 발견
- 원인 추적 어려움 (GEP 인덱스? 타입? 메모리?)
- AST와 LLVM IR 사이 gap이 크다

**Progressive lowering:**

```
FunLang AST → FunLang Dialect → Func Dialect → LLVM Dialect
               ↑ verify       ↑ verify      ↑ verify
```

각 단계에서 검증 가능:
1. FunLang Dialect: 타입 체크 (`!funlang.closure` vs `i32`)
2. Func Dialect: 함수 시그니처, region 구조
3. LLVM Dialect: 포인터 연산, 메모리 안전성

**에러 메시지 비교:**

Direct lowering:
```
error: 'llvm.load' op requires result type '!llvm.ptr' but found 'i32'
  %value = llvm.load %slot : !llvm.ptr -> i32
```
"어디서 잘못됐지? GEP 인덱스? 타입 계산?"

Progressive lowering:
```
error: 'funlang.apply' op operand type mismatch
  expected: !funlang.closure
  found: i32
  %result = funlang.apply %x(%y) : (i32) -> i32
```
"아, 클로저가 아니라 정수를 apply하려고 했구나!"

### Progressive Lowering 단계 설계

FunLang 컴파일러의 lowering 경로:

```
┌─────────────────────────────────────────┐
│  FunLang AST (F# data structures)       │
│  - Lambda(param, body)                  │
│  - Apply(fn, arg)                       │
│  - Let(name, value, body)               │
└───────────────┬─────────────────────────┘
                │ (AST → Dialect)
                ↓
┌─────────────────────────────────────────┐
│  FunLang Dialect (MLIR IR)              │
│  - funlang.make_closure                 │
│  - funlang.apply                        │
│  - funlang.match                        │
│                                         │
│  Optimization:                          │
│  - Closure inlining                     │
│  - Dead closure elimination             │
│  - Escape analysis                      │
└───────────────┬─────────────────────────┘
                │ (FunLangToFunc lowering pass)
                ↓
┌─────────────────────────────────────────┐
│  Func + SCF + MemRef (MLIR IR)          │
│  - func.func, func.call                 │
│  - scf.if, scf.while                    │
│  - memref.alloc, memref.load/store      │
│                                         │
│  Optimization:                          │
│  - Inlining                             │
│  - Dead code elimination                │
│  - Loop optimization                    │
└───────────────┬─────────────────────────┘
                │ (FuncToLLVM lowering pass)
                ↓
┌─────────────────────────────────────────┐
│  LLVM Dialect (MLIR IR)                 │
│  - llvm.call                            │
│  - llvm.getelementptr                   │
│  - llvm.load, llvm.store                │
│                                         │
│  Optimization:                          │
│  - (LLVM's own optimization passes)     │
└───────────────┬─────────────────────────┘
                │ (MLIR → LLVM IR translation)
                ↓
┌─────────────────────────────────────────┐
│  LLVM IR                                │
│  - call, getelementptr, load, store     │
└───────────────┬─────────────────────────┘
                │ (LLVM backend)
                ↓
┌─────────────────────────────────────────┐
│  Machine Code (x86, ARM, etc.)          │
└─────────────────────────────────────────┘
```

### 각 단계의 역할

#### Stage 1: FunLang Dialect

**표현:** 도메인 의미론 (클로저, 패턴 매칭, 리스트)

**Example:**

```mlir
func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.make_closure @lambda_adder(%n) : !funlang.closure
    func.return %closure : !funlang.closure
}

func.func private @lambda_adder(%x: i32, %n: i32) -> i32 {
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}
```

**특징:**
- `!funlang.closure` 타입 사용
- 구현 세부사항 숨김 (malloc, GEP 없음)
- 최적화 가능 (클로저 inlining, escape analysis)

**최적화 예시:**

```mlir
// Before optimization
%closure = funlang.make_closure @lambda_inc() : !funlang.closure
%result = funlang.apply %closure(%x) : (i32) -> i32

// After closure inlining (FunLang dialect pass)
%result = func.call @lambda_inc(%x) : (i32) -> i32
```

#### Stage 2: Func + SCF + MemRef Dialect

**표현:** 범용 추상화 (함수, 제어 흐름, 메모리)

**Example (Stage 1 lowering 후):**

```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // 환경 할당 (memref.alloc)
    %c2 = arith.constant 2 : index
    %env = memref.alloc(%c2) : memref<?xi32>

    // 함수 포인터 저장 (conceptual, 실제는 다름)
    // ... (이 단계에서 여전히 추상적)

    // 캡처된 변수 저장
    %c1 = arith.constant 1 : index
    memref.store %n, %env[%c1] : memref<?xi32>

    // 포인터 반환
    %ptr = memref.cast %env : memref<?xi32> to !llvm.ptr
    func.return %ptr : !llvm.ptr
}
```

**특징:**
- 여전히 플랫폼 독립적
- 메모리 연산이 추상적 (memref vs raw pointer)
- 구조적 제어 흐름 (scf.if vs cf.br)

**최적화 예시:**

```mlir
// Inlining (func dialect level)
%result = func.call @small_function(%x) : (i32) -> i32

// After inlining
// (함수 본체 inline됨)
%result = arith.addi %x, %c1 : i32
```

#### Stage 3: LLVM Dialect

**표현:** 기계 모델 (포인터, 레지스터, 메모리)

**Example (Stage 2 lowering 후):**

```mlir
llvm.func @make_adder(%n: i32) -> !llvm.ptr {
    // GC_malloc 호출
    %c12 = llvm.mlir.constant(12 : i64) : i64
    %env = llvm.call @GC_malloc(%c12) : (i64) -> !llvm.ptr

    // 함수 포인터 저장
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr

    // 캡처된 변수 저장
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr

    llvm.return %env : !llvm.ptr
}
```

**특징:**
- 구현 세부사항 완전 노출 (GEP, malloc, store)
- LLVM IR과 1:1 대응
- 플랫폼 특화 최적화 가능 (LLVM backend)

### ConversionTarget과 Legal/Illegal Dialects

**Lowering pass**는 특정 dialect operation을 다른 dialect operation으로 변환한다. MLIR은 **ConversionTarget**으로 이를 제어한다.

**ConversionTarget 개념:**

"이 pass 이후 어떤 operation이 허용되는가?"

```cpp
// FunLangToFunc lowering pass
class FunLangToFuncLowering : public Pass {
    void runOnOperation() override {
        ConversionTarget target(getContext());

        // FunLang dialect operation은 불법 (lowering 대상)
        target.addIllegalDialect<FunLangDialect>();

        // Func, SCF, Arith dialect operation은 합법
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<arith::ArithDialect>();

        // Lowering 수행
        if (failed(applyPartialConversion(module, target, patterns)))
            signalPassFailure();
    }
};
```

**Legal vs Illegal:**

| Legal Operations | Illegal Operations |
|-----------------|-------------------|
| Pass 후 존재 가능 | Pass 후 제거되어야 함 |
| 변환 불필요 | 변환 패턴 필요 |
| 예: func.call | 예: funlang.make_closure |

**예시: FunLangToFunc lowering**

Before (FunLang dialect):
```mlir
%closure = funlang.make_closure @lambda_func(%n) : !funlang.closure
```

After (Func + MemRef dialect):
```mlir
%env = memref.alloc(...) : memref<?xi32>
memref.store %n, %env[%c1] : memref<?xi32>
%ptr = memref.cast %env : memref<?xi32> to !llvm.ptr
```

**ConversionTarget이 보장:**
- `funlang.make_closure`는 pass 후 존재하지 않음
- `memref.alloc`, `memref.store`는 합법

### RewritePatternSet 개념

**RewritePattern**은 operation 변환 규칙이다.

**구조:**

```cpp
struct MakeClosureOpLowering : public OpRewritePattern<MakeClosureOp> {
    using OpRewritePattern<MakeClosureOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MakeClosureOp op,
                                   PatternRewriter &rewriter) const override {
        // 1. Match: 이 operation을 변환할 수 있는가?
        // (OpRewritePattern이 자동으로 매칭)

        // 2. Rewrite: 어떻게 변환하는가?

        // 환경 할당
        Value envSize = rewriter.create<arith::ConstantOp>(...);
        Value env = rewriter.create<memref::AllocOp>(...);

        // 캡처된 변수 저장
        for (auto [idx, captured] : enumerate(op.getCapturedValues())) {
            Value index = rewriter.create<arith::ConstantIndexOp>(idx);
            rewriter.create<memref::StoreOp>(captured, env, index);
        }

        // 원래 operation 교체
        rewriter.replaceOp(op, env);
        return success();
    }
};
```

**RewritePatternSet 사용:**

```cpp
void FunLangToFuncPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());

    // 변환 패턴 등록
    patterns.add<MakeClosureOpLowering>(&getContext());
    patterns.add<ApplyOpLowering>(&getContext());
    patterns.add<MatchOpLowering>(&getContext());

    // Conversion target 설정
    ConversionTarget target(getContext());
    target.addIllegalDialect<FunLangDialect>();
    target.addLegalDialect<func::FuncDialect, memref::MemRefDialect, arith::ArithDialect>();

    // 변환 적용
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
        signalPassFailure();
}
```

**각 pattern이 처리:**
- `MakeClosureOpLowering`: `funlang.make_closure` → `memref.alloc` + stores
- `ApplyOpLowering`: `funlang.apply` → `func.call` (indirect)
- `MatchOpLowering`: `funlang.match` → `scf.if` cascade

### 실제 Lowering Pass 구조 미리보기

**FunLangToFunc.cpp 구조:**

```cpp
// 1. Pattern 정의들
namespace {

struct MakeClosureOpLowering : public OpRewritePattern<MakeClosureOp> {
    LogicalResult matchAndRewrite(...) const override {
        // funlang.make_closure → memref operations
    }
};

struct ApplyOpLowering : public OpRewritePattern<ApplyOp> {
    LogicalResult matchAndRewrite(...) const override {
        // funlang.apply → func.call (indirect)
    }
};

} // namespace

// 2. Pass 정의
struct FunLangToFuncPass : public PassWrapper<FunLangToFuncPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect, memref::MemRefDialect, arith::ArithDialect>();
    }

    void runOnOperation() override {
        // Pattern set 구성
        RewritePatternSet patterns(&getContext());
        patterns.add<MakeClosureOpLowering, ApplyOpLowering>(&getContext());

        // Target 설정
        ConversionTarget target(getContext());
        target.addIllegalDialect<FunLangDialect>();
        target.addLegalDialect<func::FuncDialect, memref::MemRefDialect, arith::ArithDialect>();

        // 변환 실행
        if (failed(applyPartialConversion(getOperation(), target, patterns)))
            signalPassFailure();
    }
};

// 3. Pass 등록
std::unique_ptr<Pass> createFunLangToFuncPass() {
    return std::make_unique<FunLangToFuncPass>();
}
```

**Pass 실행 순서 (Compiler.fs):**

```fsharp
// MLIR pass pipeline
let runLoweringPasses (module: MlirModule) =
    let pm = PassManager.Create(module.Context)

    // 1. FunLang dialect → Func/MemRef dialect
    pm.AddPass(FunLangPasses.CreateFunLangToFuncPass())

    // 2. SCF → CF (structured control flow → control flow)
    pm.AddPass(Passes.CreateSCFToCFPass())

    // 3. Func/MemRef/Arith → LLVM dialect
    pm.AddPass(Passes.CreateFuncToLLVMPass())
    pm.AddPass(Passes.CreateMemRefToLLVMPass())
    pm.AddPass(Passes.CreateArithToLLVMPass())

    pm.Run(module)
```

## 요약

**Chapter 14에서 배운 것:**

1. **Phase 4의 문제점**: 저수준 LLVM dialect 직접 사용 시 GEP 패턴 반복, 도메인 의미론 상실, 컴파일러 코드 복잡도 증가

2. **Custom Dialect의 이점**: 도메인 의미론 보존, 컴파일러 코드 단순화, 타입 안전성 향상, 최적화 기회 증가

3. **MLIR Dialect 아키텍처**: Operation (계산), Type (값 종류), Attribute (컴파일 타임 상수), Region/Block (중첩 코드), Symbol Table (전역 참조)

4. **Progressive Lowering 철학**:
   - 직접 lowering의 문제 (최적화 상실, 복잡도 폭발, 디버깅 어려움)
   - 단계적 lowering의 이점 (각 단계 최적화, 독립 검증, 명확한 책임)
   - FunLang → Func/MemRef → LLVM 경로

5. **ConversionTarget과 RewritePattern**: Legal/Illegal dialect 정의, 변환 규칙 작성, pass 구조

**다음 장 (Chapter 15) Preview:**

Chapter 15에서는:
- TableGen ODS로 FunLang operation 정의
- C API shim 작성 (F# interop)
- Lowering pass 구현 (FunLangToFunc)
- 전체 컴파일 파이프라인 구축

이론적 기초를 확립했으므로, 실제 구현으로 넘어갈 준비가 됐다.
