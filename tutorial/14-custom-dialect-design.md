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

## TableGen ODS (Operation Definition Specification) 기초

### TableGen이란?

**TableGen**은 LLVM 프로젝트의 **DSL (Domain-Specific Language)**이다. 코드 생성(code generation)을 위한 선언적 언어다.

**Why TableGen?**

MLIR operation을 C++로 직접 정의하면:

```cpp
// C++ 직접 정의 (verbose!)
class MakeClosureOp : public Op<MakeClosureOp, OpTrait::OneResult, OpTrait::ZeroRegions> {
public:
    static StringRef getOperationName() { return "funlang.make_closure"; }

    static void build(OpBuilder &builder, OperationState &state,
                      FlatSymbolRefAttr funcName, ValueRange capturedValues) {
        // 복잡한 builder 로직...
    }

    LogicalResult verify() {
        // 복잡한 verification 로직...
    }

    // parser, printer, folders, canonicalizers...
    // 100+ lines of boilerplate!
};
```

**문제점:**
- Boilerplate 코드 많음 (parser, printer, builder)
- 타입 안전성 수동 관리
- 일관성 유지 어려움 (operation마다 다른 스타일)

**TableGen 사용:**

```tablegen
// TableGen 정의 (concise!)
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  let summary = "Creates a closure value";
  let description = [{
    Creates a closure by capturing values into an environment.
  }];

  let arguments = (ins FlatSymbolRefAttr:$funcName,
                       Variadic<AnyType>:$capturedValues);
  let results = (outs FunLang_ClosureType:$result);

  let assemblyFormat = "$funcName `(` $capturedValues `)` attr-dict `:` type($result)";
}
```

**장점:**
- 선언적 (what, not how)
- 코드 자동 생성 (parser, printer, builder, verifier)
- 타입 안전성 자동 보장
- 일관된 스타일

**TableGen 빌드 프로세스:**

```
FunLangOps.td (TableGen source)
    ↓ (mlir-tblgen tool)
FunLangOps.h.inc (Generated C++ header)
FunLangOps.cpp.inc (Generated C++ implementation)
    ↓ (C++ compiler)
libMLIRFunLangDialect.so (Shared library)
```

### FunLang Dialect 정의

**FunLangDialect.td:**

```tablegen
// FunLang dialect 정의
def FunLang_Dialect : Dialect {
  // Dialect 이름 (operation prefix)
  let name = "funlang";

  // C++ namespace
  let cppNamespace = "::mlir::funlang";

  // 의존성 선언
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect",
    "LLVM::LLVMDialect"
  ];

  // Description (documentation)
  let description = [{
    The FunLang dialect represents high-level functional programming constructs
    for the FunLang compiler. It provides operations for closures, pattern matching,
    and other domain-specific features.
  }];

  // Extra class declarations (C++ 코드 삽입)
  let extraClassDeclaration = [{
    // Custom dialect methods (optional)
    void registerTypes();
    void registerOperations();
  }];
}
```

**각 필드 의미:**

1. **`name`**: Dialect 네임스페이스
   - Operation: `funlang.make_closure`
   - Type: `!funlang.closure`

2. **`cppNamespace`**: 생성되는 C++ 코드의 네임스페이스
   - `mlir::funlang::MakeClosureOp`
   - `mlir::funlang::ClosureType`

3. **`dependentDialects`**: 이 dialect가 사용하는 다른 dialect
   - FunLang operation이 `func.func`, `arith.addi` 등 사용 가능
   - Context에 자동 로드됨

4. **`description`**: Documentation (mlir-doc tool이 사용)

5. **`extraClassDeclaration`**: 추가 C++ 메서드 선언

### Operation 정의 구조

**Base class 정의:**

```tablegen
// FunLang operation base class
class FunLang_Op<string mnemonic, list<Trait> traits = []>
    : Op<FunLang_Dialect, mnemonic, traits>;
```

모든 FunLang operation이 이 base class를 상속한다.

**Operation 정의 예시: make_closure**

```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  // 한 줄 요약
  let summary = "Creates a closure value";

  // 상세 설명 (multi-line string)
  let description = [{
    The `funlang.make_closure` operation creates a closure by capturing
    values into an environment. The closure can later be invoked using
    `funlang.apply`.

    Example:
    ```mlir
    %closure = funlang.make_closure @my_lambda(%x, %y) : !funlang.closure
    ```
  }];

  // 입력 인자 (arguments)
  let arguments = (ins
    FlatSymbolRefAttr:$funcName,        // 함수 심볼 (@lambda_0)
    Variadic<AnyType>:$capturedValues   // 캡처된 값들 (%x, %y, ...)
  );

  // 출력 결과 (results)
  let results = (outs
    FunLang_ClosureType:$result         // 클로저 값
  );

  // Assembly format (parser/printer)
  let assemblyFormat = [{
    $funcName `(` $capturedValues `)` attr-dict `:` type($result)
  }];

  // Traits (operation 특성)
  // [Pure]: no side effects, result depends only on operands
}
```

**Arguments (ins):**

| Type | Name | Meaning |
|------|------|---------|
| `FlatSymbolRefAttr` | `funcName` | 함수 이름 attribute (`@lambda_0`) |
| `Variadic<AnyType>` | `capturedValues` | 가변 길이 값 목록 (captured variables) |

**Results (outs):**

| Type | Name | Meaning |
|------|------|---------|
| `FunLang_ClosureType` | `result` | 클로저 타입 값 |

**Assembly Format:**

- `$funcName`: `@lambda_func` 출력
- `` `(` ``: 리터럴 `(` 문자
- `$capturedValues`: 캡처된 값들 출력 (`%x, %y`)
- `` `)` ``: 리터럴 `)` 문자
- `attr-dict`: attribute dictionary (선택적)
- `` `:` ``: 리터럴 `:` 문자
- `type($result)`: 결과 타입 출력 (`!funlang.closure`)

생성되는 IR:
```mlir
%closure = funlang.make_closure @lambda_func(%x, %y) : !funlang.closure
```

### Operation Traits

**Trait**는 operation의 특성을 선언한다. MLIR이 최적화/검증에 사용한다.

**Pure trait:**

```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  // ...
}
```

`Pure` = **순수 함수** (no side effects, deterministic)
- 같은 입력 → 같은 출력
- 메모리 쓰기 없음, I/O 없음
- 최적화 가능: 중복 제거, 재배치

**MemoryEffects trait:**

```tablegen
def FunLang_AllocClosureOp : FunLang_Op<"alloc_closure",
    [MemoryEffects<[MemAlloc]>]> {
  // Memory allocation operation
}
```

`MemoryEffects<[MemAlloc]>` = 메모리 할당만 함 (읽기/쓰기 없음)

**다른 유용한 traits:**

| Trait | Meaning | Example |
|-------|---------|---------|
| `NoSideEffect` | 부작용 없음 (Pure와 비슷) | 산술 연산 |
| `Terminator` | Basic block 종료 operation | `func.return` |
| `IsolatedFromAbove` | 외부 값 참조 불가 | `func.func` |
| `SameOperandsAndResultType` | 입력과 출력 타입 동일 | `arith.addi` |

### hasVerifier 속성

Custom verification 로직이 필요하면:

```tablegen
def FunLang_ApplyOp : FunLang_Op<"apply"> {
  let arguments = (ins FunLang_ClosureType:$closure,
                       Variadic<AnyType>:$arguments);
  let results = (outs AnyType:$result);

  // Custom verifier 필요
  let hasVerifier = 1;
}
```

생성된 C++ 코드에 `verify()` 메서드 선언:

```cpp
// FunLangOps.h.inc에 생성됨
class ApplyOp : public ... {
public:
    LogicalResult verify();  // Custom implementation 필요
};
```

**Verifier 구현 (FunLangOps.cpp):**

```cpp
LogicalResult ApplyOp::verify() {
    // 클로저 타입 체크
    if (!getClosure().getType().isa<ClosureType>())
        return emitError("operand must be a closure type");

    // 인자 개수 체크 (optional, 런타임 체크 가능)
    // ...

    return success();
}
```

### Type 정의 (TypeDef)

**FunLang Closure 타입:**

```tablegen
def FunLang_ClosureType : TypeDef<FunLang_Dialect, "Closure"> {
  let mnemonic = "closure";

  let summary = "FunLang closure type";

  let description = [{
    Represents a closure value (function pointer + captured environment).
  }];

  // Parameters (타입 파라미터)
  // Closure는 파라미터 없음 (단순 타입)
  let parameters = (ins);

  // Assembly format
  let assemblyFormat = "";
}
```

**생성되는 C++ 코드:**

```cpp
// FunLangTypes.h.inc
class ClosureType : public Type::TypeBase<ClosureType, Type, TypeStorage> {
public:
    static constexpr StringLiteral getMnemonic() { return "closure"; }
    // ...
};
```

**사용 예:**

```mlir
// MLIR IR
%closure : !funlang.closure

// F# 코드
let closureType = FunLangType.GetClosure(ctx)
```

### FunLang 타입 설계

#### 1. ClosureType (클로저)

```tablegen
def FunLang_ClosureType : TypeDef<FunLang_Dialect, "Closure"> {
  let mnemonic = "closure";
  let summary = "Function closure (function pointer + environment)";
  let parameters = (ins);
  let assemblyFormat = "";
}
```

**용도:** 클로저 값 표현

```mlir
%closure = funlang.make_closure @lambda_func(%x) : !funlang.closure
```

#### 2. ListType (리스트, Phase 6 preview)

```tablegen
def FunLang_ListType : TypeDef<FunLang_Dialect, "List"> {
  let mnemonic = "list";
  let summary = "Immutable list of elements";

  // 파라미터: element type
  let parameters = (ins "Type":$elementType);

  // Assembly format: list<i32>
  let assemblyFormat = "`<` $elementType `>`";
}
```

**파라미터화된 타입:**
- `!funlang.list<i32>`: 정수 리스트
- `!funlang.list<!funlang.closure>`: 클로저 리스트

**생성된 C++ 코드:**

```cpp
class ListType : public Type::TypeBase<...> {
public:
    static ListType get(Type elementType);
    Type getElementType() const;
};
```

**사용 예:**

```mlir
// 빈 리스트
%nil = funlang.nil : !funlang.list<i32>

// Cons (head::tail)
%list = funlang.cons %head, %tail : (i32, !funlang.list<i32>) -> !funlang.list<i32>
```

#### 3. RecordType (레코드, Phase 7 preview)

```tablegen
def FunLang_RecordType : TypeDef<FunLang_Dialect, "Record"> {
  let mnemonic = "record";
  let summary = "Record with named fields";

  // 파라미터: field names + types
  let parameters = (ins
    ArrayRefParameter<"StringAttr">:$fieldNames,
    ArrayRefParameter<"Type">:$fieldTypes
  );

  let assemblyFormat = "`<` `{` $fieldNames `:` $fieldTypes `}` `>`";
}
```

**사용 예:**

```mlir
// {x: i32, y: i32}
%point : !funlang.record<{x: i32, y: i32}>
```

### FunLang Operations 정의 예시

#### funlang.make_closure

**TableGen 정의:**

```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  let summary = "Creates a closure value";

  let arguments = (ins
    FlatSymbolRefAttr:$funcName,
    Variadic<AnyType>:$capturedValues
  );

  let results = (outs FunLang_ClosureType:$result);

  let assemblyFormat = "$funcName `(` $capturedValues `)` attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$funcName,
                   "ValueRange":$capturedValues), [{
      build($_builder, $_state, ClosureType::get($_builder.getContext()),
            funcName, capturedValues);
    }]>
  ];
}
```

**생성된 C++ API:**

```cpp
// FunLangOps.h.inc
class MakeClosureOp : public Op<...> {
public:
    static MakeClosureOp create(OpBuilder &builder, Location loc,
                                FlatSymbolRefAttr funcName,
                                ValueRange capturedValues);

    FlatSymbolRefAttr getFuncName();
    OperandRange getCapturedValues();
    Value getResult();
};
```

#### funlang.apply

**TableGen 정의:**

```tablegen
def FunLang_ApplyOp : FunLang_Op<"apply"> {
  let summary = "Applies a closure to arguments";

  let arguments = (ins
    FunLang_ClosureType:$closure,
    Variadic<AnyType>:$arguments
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $closure `(` $arguments `)` attr-dict `:` functional-type($arguments, $result)
  }];

  let hasVerifier = 1;
}
```

**사용 예:**

```mlir
%result = funlang.apply %closure(%x, %y) : (i32, i32) -> i32
```

**Verifier (FunLangOps.cpp):**

```cpp
LogicalResult ApplyOp::verify() {
    if (!getClosure().getType().isa<ClosureType>())
        return emitError("first operand must be a closure");
    return success();
}
```

### 생성되는 C++ 코드 설명

**mlir-tblgen 실행:**

```bash
mlir-tblgen -gen-op-decls FunLangOps.td > FunLangOps.h.inc
mlir-tblgen -gen-op-defs FunLangOps.td > FunLangOps.cpp.inc
mlir-tblgen -gen-typedef-decls FunLangTypes.td > FunLangTypes.h.inc
mlir-tblgen -gen-typedef-defs FunLangTypes.td > FunLangTypes.cpp.inc
```

**FunLangOps.h.inc (생성된 헤더):**

```cpp
class MakeClosureOp : public Op<MakeClosureOp, OpTrait::ZeroRegions,
                                OpTrait::OneResult, OpTrait::Pure> {
public:
    static constexpr StringLiteral getOperationName() {
        return StringLiteral("funlang.make_closure");
    }

    // Accessors
    FlatSymbolRefAttr getFuncName();
    OperandRange getCapturedValues();
    Value getResult();

    // Builder
    static void build(OpBuilder &builder, OperationState &state, ...);

    // Parser/Printer (assemblyFormat에서 생성)
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
    void print(OpAsmPrinter &p);

    // Verifier (기본 타입 체크)
    LogicalResult verify();
};
```

**사용 (C++ dialect code):**

```cpp
// Operation 생성
auto closureOp = builder.create<MakeClosureOp>(
    loc,
    funcNameAttr,
    capturedValues
);

// Accessors 사용
FlatSymbolRefAttr funcName = closureOp.getFuncName();
Value result = closureOp.getResult();
```

**FunLangTypes.h.inc:**

```cpp
class ClosureType : public Type::TypeBase<ClosureType, Type, TypeStorage> {
public:
    static constexpr StringLiteral getMnemonic() { return "closure"; }

    static ClosureType get(MLIRContext *ctx) {
        return Base::get(ctx);
    }

    // Parser/Printer
    static ParseResult parse(AsmParser &parser);
    void print(AsmPrinter &printer) const;
};
```

**사용:**

```cpp
// 타입 생성
ClosureType closureType = ClosureType::get(ctx);

// 타입 체크
if (auto ct = value.getType().dyn_cast<ClosureType>()) {
    // This is a closure!
}
```

## C API Shim 패턴 (F# Interop)

### 문제: TableGen은 C++ 코드 생성, F#은 C API 필요

**상황:**

1. **TableGen → C++ 코드 생성**
   - `MakeClosureOp` 클래스 (C++)
   - `ClosureType::get()` 메서드 (C++)

2. **F#은 C API만 호출 가능**
   - P/Invoke는 `extern "C"` 함수만 지원
   - C++ 클래스 직접 호출 불가

**문제:**

```fsharp
// 이런 코드를 쓰고 싶지만...
let closure = MakeClosureOp.Create(builder, funcName, capturedValues)  // ERROR: C++ class!
```

### 해결책: extern "C" Wrapper Functions

**아키텍처:**

```
┌─────────────────────────────────────────┐
│ F# Code (Compiler.fs)                   │
│                                         │
│ let closure = FunLang.CreateClosure(...) │
└────────────────┬────────────────────────┘
                 │ P/Invoke
                 ▼
┌─────────────────────────────────────────┐
│ C API Shim (FunLangCAPI.h/.cpp)         │
│                                         │
│ extern "C" {                            │
│   MlirOperation mlirFunLangClosure...() │
│ }                                       │
└────────────────┬────────────────────────┘
                 │ Call C++ API
                 ▼
┌─────────────────────────────────────────┐
│ C++ Dialect (FunLangOps.h/.cpp)         │
│                                         │
│ class MakeClosureOp { ... }             │
│ (TableGen generated)                    │
└─────────────────────────────────────────┘
```

### FunLangCAPI.h 구조

**헤더 파일:**

```c
// FunLangCAPI.h - C API for FunLang Dialect
#ifndef FUNLANG_C_API_H
#define FUNLANG_C_API_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

/// Register FunLang dialect in the given context
MLIR_CAPI_EXPORTED void mlirContextRegisterFunLangDialect(MlirContext ctx);

/// Load FunLang dialect into the given context
MLIR_CAPI_EXPORTED MlirDialect mlirContextLoadFunLangDialect(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// Returns true if the given type is a FunLang closure type
MLIR_CAPI_EXPORTED bool mlirTypeIsAFunLangClosure(MlirType type);

/// Creates a FunLang closure type
MLIR_CAPI_EXPORTED MlirType mlirFunLangClosureTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

/// Creates a funlang.make_closure operation
MLIR_CAPI_EXPORTED MlirOperation mlirFunLangMakeClosureOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirAttribute funcName,       // FlatSymbolRefAttr
    intptr_t numCaptured,
    MlirValue *capturedValues     // Array of values
);

/// Creates a funlang.apply operation
MLIR_CAPI_EXPORTED MlirOperation mlirFunLangApplyOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue closure,
    intptr_t numArgs,
    MlirValue *arguments,
    MlirType resultType
);

#ifdef __cplusplus
}
#endif

#endif // FUNLANG_C_API_H
```

**핵심 패턴:**

1. **`extern "C"`**: C linkage (name mangling 없음)
2. **MLIR C API 타입 사용**: `MlirContext`, `MlirOperation`, `MlirValue`
3. **배열 전달**: `intptr_t num` + `MlirValue *array` 패턴

### FunLangCAPI.cpp 구현 패턴

**구현 파일:**

```cpp
// FunLangCAPI.cpp
#include "FunLangCAPI.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "FunLang/IR/FunLangDialect.h"
#include "FunLang/IR/FunLangOps.h"
#include "FunLang/IR/FunLangTypes.h"

using namespace mlir;
using namespace mlir::funlang;

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

void mlirContextRegisterFunLangDialect(MlirContext ctx) {
    // unwrap: C handle → C++ pointer
    MLIRContext *context = unwrap(ctx);

    // Register dialect
    DialectRegistry registry;
    registry.insert<FunLangDialect>();
    context->appendDialectRegistry(registry);
}

MlirDialect mlirContextLoadFunLangDialect(MlirContext ctx) {
    MLIRContext *context = unwrap(ctx);
    Dialect *dialect = context->loadDialect<FunLangDialect>();

    // wrap: C++ pointer → C handle
    return wrap(dialect);
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFunLangClosure(MlirType type) {
    return unwrap(type).isa<ClosureType>();
}

MlirType mlirFunLangClosureTypeGet(MlirContext ctx) {
    MLIRContext *context = unwrap(ctx);
    Type closureType = ClosureType::get(context);
    return wrap(closureType);
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

MlirOperation mlirFunLangMakeClosureOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirAttribute funcName,
    intptr_t numCaptured,
    MlirValue *capturedValues)
{
    // Unwrap C handles
    MLIRContext *context = unwrap(ctx);
    Location location = unwrap(loc);
    FlatSymbolRefAttr funcNameAttr = unwrap(funcName).cast<FlatSymbolRefAttr>();

    // Convert array to ValueRange
    SmallVector<Value, 4> captured;
    for (intptr_t i = 0; i < numCaptured; ++i) {
        captured.push_back(unwrap(capturedValues[i]));
    }

    // Create operation using OpBuilder
    OpBuilder builder(context);
    auto op = builder.create<MakeClosureOp>(location, funcNameAttr, captured);

    // Wrap and return
    return wrap(op.getOperation());
}

MlirOperation mlirFunLangApplyOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue closure,
    intptr_t numArgs,
    MlirValue *arguments,
    MlirType resultType)
{
    MLIRContext *context = unwrap(ctx);
    Location location = unwrap(loc);
    Value closureValue = unwrap(closure);
    Type resType = unwrap(resultType);

    SmallVector<Value, 4> args;
    for (intptr_t i = 0; i < numArgs; ++i) {
        args.push_back(unwrap(arguments[i]));
    }

    OpBuilder builder(context);
    auto op = builder.create<ApplyOp>(location, resType, closureValue, args);

    return wrap(op.getOperation());
}
```

#### wrap/unwrap 헬퍼 사용

**MLIR C API convention:**

- **`unwrap()`**: C handle → C++ pointer
- **`wrap()`**: C++ pointer → C handle

```cpp
// C handle types (opaque)
typedef struct MlirContext { void *ptr; } MlirContext;
typedef struct MlirType { void *ptr; } MlirType;
typedef struct MlirValue { void *ptr; } MlirValue;

// Unwrap/Wrap (MLIR/CAPI/Support.h)
inline MLIRContext *unwrap(MlirContext ctx) {
    return static_cast<MLIRContext *>(ctx.ptr);
}

inline MlirContext wrap(MLIRContext *ctx) {
    return MlirContext{static_cast<void *>(ctx)};
}
```

**사용 패턴:**

```cpp
// C API function signature (C handles)
MlirType mlirFunLangClosureTypeGet(MlirContext ctx);

// Implementation (unwrap → use C++ API → wrap)
MlirType mlirFunLangClosureTypeGet(MlirContext ctx) {
    MLIRContext *context = unwrap(ctx);           // C → C++
    Type closureType = ClosureType::get(context); // C++ API
    return wrap(closureType);                      // C++ → C
}
```

#### OpBuilder 활용

**OpBuilder**는 MLIR operation 생성 헬퍼다:

```cpp
OpBuilder builder(context);

// Operation 생성
auto op = builder.create<MakeClosureOp>(
    location,       // Location (source info)
    funcNameAttr,   // Symbol reference
    capturedValues  // Operands
);

// Block에 삽입
builder.setInsertionPointToEnd(block);
auto op2 = builder.create<ApplyOp>(...);
```

**C API shim에서:**

```cpp
MlirOperation mlirFunLangMakeClosureOpCreate(...) {
    OpBuilder builder(context);
    auto op = builder.create<MakeClosureOp>(...);
    return wrap(op.getOperation());  // Operation* → MlirOperation
}
```

#### 타입 생성 및 검증

**타입 생성:**

```cpp
MlirType mlirFunLangClosureTypeGet(MlirContext ctx) {
    MLIRContext *context = unwrap(ctx);
    Type closureType = ClosureType::get(context);
    return wrap(closureType);
}
```

**타입 검증:**

```cpp
bool mlirTypeIsAFunLangClosure(MlirType type) {
    Type t = unwrap(type);
    return t.isa<ClosureType>();  // C++ RTTI
}
```

F#에서 사용:

```fsharp
// 타입 생성
let closureType = FunLang.GetClosureType(ctx)

// 타입 체크
if FunLang.IsClosureType(value.Type) then
    printfn "This is a closure!"
```

### CMakeLists.txt 빌드 설정

**FunLang dialect CMake:**

```cmake
# CMakeLists.txt
add_mlir_dialect_library(MLIRFunLangDialect
  # TableGen sources
  FunLangDialect.cpp
  FunLangOps.cpp
  FunLangTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/FunLang

  DEPENDS
  MLIRFunLangOpsIncGen        # TableGen generated files
  MLIRFunLangTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
)

# C API shim
add_mlir_public_c_api_library(MLIRFunLangCAPI
  FunLangCAPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/FunLang-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRFunLangDialect
)
```

**빌드 출력:**

- `libMLIRFunLangDialect.so`: C++ dialect library
- `libMLIRFunLangCAPI.so`: C API shim library

F#은 `MLIRFunLangCAPI.so`를 로드한다.

### F# P/Invoke 바인딩 (Mlir.FunLang 모듈)

**FunLangBindings.fs:**

```fsharp
module Mlir.FunLang

open System
open System.Runtime.InteropServices

// P/Invoke declarations
[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirContextRegisterFunLangDialect(MlirContext ctx)

[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern MlirDialect mlirContextLoadFunLangDialect(MlirContext ctx)

[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern bool mlirTypeIsAFunLangClosure(MlirType ty)

[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType mlirFunLangClosureTypeGet(MlirContext ctx)

[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern MlirOperation mlirFunLangMakeClosureOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirAttribute funcName,
    nativeint numCaptured,
    MlirValue[] capturedValues
)

[<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
extern MlirOperation mlirFunLangApplyOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue closure,
    nativeint numArgs,
    MlirValue[] arguments,
    MlirType resultType
)

// High-level F# API
type FunLangDialect =
    static member Register(ctx: MlirContext) =
        mlirContextRegisterFunLangDialect(ctx)

    static member Load(ctx: MlirContext) : MlirDialect =
        mlirContextLoadFunLangDialect(ctx)

type FunLangType =
    static member GetClosure(ctx: MlirContext) : MlirType =
        mlirFunLangClosureTypeGet(ctx)

    static member IsClosure(ty: MlirType) : bool =
        mlirTypeIsAFunLangClosure(ty)

type FunLangOps =
    static member CreateMakeClosure(ctx: MlirContext, loc: MlirLocation,
                                     funcName: MlirAttribute,
                                     capturedValues: MlirValue[]) : MlirOperation =
        mlirFunLangMakeClosureOpCreate(ctx, loc, funcName, nativeint capturedValues.Length, capturedValues)

    static member CreateApply(ctx: MlirContext, loc: MlirLocation,
                               closure: MlirValue, arguments: MlirValue[],
                               resultType: MlirType) : MlirOperation =
        mlirFunLangApplyOpCreate(ctx, loc, closure, nativeint arguments.Length, arguments, resultType)
```

**사용 예 (Compiler.fs):**

```fsharp
// Dialect 등록
let ctx = MlirContext.Create()
FunLangDialect.Register(ctx)
FunLangDialect.Load(ctx)

// 클로저 타입 얻기
let closureType = FunLangType.GetClosure(ctx)

// make_closure operation 생성
let funcNameAttr = ... // SymbolRefAttr
let capturedValues = [| %x; %y |]
let makeClosureOp = FunLangOps.CreateMakeClosure(ctx, loc, funcNameAttr, capturedValues)

// apply operation 생성
let closureValue = ... // %closure
let arguments = [| %arg1; %arg2 |]
let resultType = ... // i32
let applyOp = FunLangOps.CreateApply(ctx, loc, closureValue, arguments, resultType)
```

### 전체 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────────┐
│                        F# Compiler                               │
│                                                                  │
│  let closure = FunLangOps.CreateMakeClosure(...)                │
│  let result = FunLangOps.CreateApply(...)                       │
└─────────────────────────┬────────────────────────────────────────┘
                          │ P/Invoke
                          │ (CallingConvention.Cdecl)
                          ↓
┌──────────────────────────────────────────────────────────────────┐
│              C API Shim (FunLangCAPI.h/.cpp)                     │
│                                                                  │
│  extern "C" {                                                    │
│    MlirOperation mlirFunLangMakeClosureOpCreate(...) {          │
│      MLIRContext *ctx = unwrap(ctxHandle);                      │
│      OpBuilder builder(ctx);                                     │
│      auto op = builder.create<MakeClosureOp>(...);              │
│      return wrap(op.getOperation());                             │
│    }                                                             │
│  }                                                               │
└─────────────────────────┬────────────────────────────────────────┘
                          │ Call C++ API
                          ↓
┌──────────────────────────────────────────────────────────────────┐
│         C++ Dialect (FunLangOps.h/.cpp, TableGen generated)      │
│                                                                  │
│  class MakeClosureOp : public Op<...> {                         │
│    // Generated by TableGen                                      │
│    static void build(OpBuilder &, OperationState &, ...);       │
│    LogicalResult verify();                                       │
│  };                                                              │
└─────────────────────────┬────────────────────────────────────────┘
                          │ Uses MLIR Core API
                          ↓
┌──────────────────────────────────────────────────────────────────┐
│                      MLIR Core (C++)                             │
│                                                                  │
│  - Operation, Type, Attribute classes                            │
│  - OpBuilder, PatternRewriter                                    │
│  - Dialect, DialectRegistry                                      │
└──────────────────────────────────────────────────────────────────┘
```

**데이터 흐름:**

1. **F# → C API**: P/Invoke로 C 함수 호출
   - `MlirContext`, `MlirValue` 등 opaque handle 전달
   - 배열은 `nativeint len` + `array` 패턴

2. **C API → C++**: unwrap으로 handle → pointer 변환
   - `unwrap(MlirContext)` → `MLIRContext*`
   - `OpBuilder.create<Op>(...)` 호출

3. **C++ → MLIR Core**: TableGen 생성 코드 사용
   - `MakeClosureOp::build()` 호출
   - Operation 생성, 타입 체크

4. **C++ → C API**: wrap으로 pointer → handle 변환
   - `wrap(Operation*)` → `MlirOperation`
   - F#에 반환

## FunLang Dialect Operations Preview

Phase 5-6에서 구현할 operations 목록:

### 1. funlang.make_closure

**의미:** 클로저 생성 (함수 포인터 + 캡처된 변수)

**시그니처:**

```tablegen
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$funcName,
                       Variadic<AnyType>:$capturedValues);
  let results = (outs FunLang_ClosureType:$result);
}
```

**사용 예:**

```mlir
%closure = funlang.make_closure @lambda_adder(%n, %m) : !funlang.closure
```

**Lowering (Phase 5):**

```mlir
// FunLang dialect
%closure = funlang.make_closure @lambda_adder(%n, %m) : !funlang.closure

// ↓ Lower to Func + MemRef

// 환경 할당
%c3 = arith.constant 3 : index
%env = memref.alloc(%c3) : memref<?xi32>

// 함수 포인터 저장 (slot 0)
// ... (conceptual)

// 변수 저장 (slot 1, 2)
%c1 = arith.constant 1 : index
memref.store %n, %env[%c1] : memref<?xi32>
%c2 = arith.constant 2 : index
memref.store %m, %env[%c2] : memref<?xi32>

// 포인터 반환
%closure_ptr = memref.cast %env : memref<?xi32> to !llvm.ptr
```

### 2. funlang.apply

**의미:** 클로저 호출 (간접 함수 호출)

**시그니처:**

```tablegen
def FunLang_ApplyOp : FunLang_Op<"apply"> {
  let arguments = (ins FunLang_ClosureType:$closure,
                       Variadic<AnyType>:$arguments);
  let results = (outs AnyType:$result);
}
```

**사용 예:**

```mlir
%result = funlang.apply %closure(%x, %y) : (i32, i32) -> i32
```

**Lowering:**

```mlir
// FunLang dialect
%result = funlang.apply %closure(%x, %y) : (i32, i32) -> i32

// ↓ Lower to Func + LLVM

// 환경에서 함수 포인터 로드
%fn_slot = llvm.getelementptr %closure[0] : (!llvm.ptr) -> !llvm.ptr
%fn_ptr = llvm.load %fn_slot : !llvm.ptr -> !llvm.ptr

// 간접 호출 (환경 + 인자들)
%result = llvm.call %fn_ptr(%closure, %x, %y) : (!llvm.ptr, i32, i32) -> i32
```

### 3. funlang.match (Phase 6)

**의미:** 패턴 매칭 (리스트, ADT)

**시그니처:**

```tablegen
def FunLang_MatchOp : FunLang_Op<"match", [RecursiveSideEffect]> {
  let arguments = (ins AnyType:$scrutinee);
  let results = (outs AnyType:$result);
  let regions = (region VariadicRegion<AnyRegion>:$cases);
}
```

**사용 예:**

```mlir
%result = funlang.match %list : !funlang.list<i32> -> i32 {
^nil_case:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32

^cons_case(%head: i32, %tail: !funlang.list<i32>):
    // ... (재귀 호출)
    funlang.yield %sum : i32
}
```

**Lowering:**

```mlir
// FunLang dialect
%result = funlang.match %list { ... }

// ↓ Lower to SCF (structured control flow)

// 리스트 태그 확인
%tag = llvm.load %list[0] : !llvm.ptr -> i32

// if (tag == NIL)
%is_nil = arith.cmpi eq, %tag, %c0 : i32
%result = scf.if %is_nil -> i32 {
    // Nil case
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
} else {
    // Cons case - head/tail 추출
    %head = llvm.load %list[1] : !llvm.ptr -> i32
    %tail = llvm.load %list[2] : !llvm.ptr -> !llvm.ptr
    // ... (body)
    scf.yield %sum : i32
}
```

### 4. funlang.nil / funlang.cons (Phase 6)

**리스트 생성:**

```tablegen
def FunLang_NilOp : FunLang_Op<"nil", [Pure]> {
  let arguments = (ins);
  let results = (outs FunLang_ListType:$result);
}

def FunLang_ConsOp : FunLang_Op<"cons", [Pure]> {
  let arguments = (ins AnyType:$head, FunLang_ListType:$tail);
  let results = (outs FunLang_ListType:$result);
}
```

**사용 예:**

```mlir
%nil = funlang.nil : !funlang.list<i32>
%list1 = funlang.cons %c1, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
%list2 = funlang.cons %c2, %list1 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
// list2 = [2, 1]
```

### Chapter 15에서 구현할 내용

**Phase 5 (Chapter 15-16):**

1. **TableGen 정의**
   - `FunLangDialect.td`
   - `FunLangOps.td` (make_closure, apply)
   - `FunLangTypes.td` (closure)

2. **C API Shim**
   - `FunLangCAPI.h`
   - `FunLangCAPI.cpp`

3. **F# Bindings**
   - `FunLangBindings.fs`

4. **Lowering Pass**
   - `FunLangToFunc.cpp` (make_closure → memref.alloc)
   - Pattern: `MakeClosureOpLowering`, `ApplyOpLowering`

5. **컴파일러 통합**
   - `Compiler.fs` 수정: FunLang dialect operations 생성
   - Pass pipeline: `FunLangToFunc → FuncToLLVM`

**Phase 6 (Chapter 17-18):**

- `funlang.match`, `funlang.nil`, `funlang.cons`
- `ListType` 구현
- Pattern matching lowering

## Common Pitfalls (흔한 실수들)

### Pitfall 1: 불완전한 타입 시스템 (AnyType 남용)

**문제:**

```tablegen
// 잘못된 설계 - 모든 것이 AnyType
def FunLang_MakeClosureOp : FunLang_Op<"make_closure"> {
  let arguments = (ins AnyType:$func, Variadic<AnyType>:$captured);
  let results = (outs AnyType:$result);  // ERROR: 타입 안전성 없음!
}
```

**왜 문제인가?**

- `AnyType`은 컴파일 타임 체크 불가
- 정수를 클로저로 사용 가능 (버그!)
- 최적화 pass가 타입 정보 활용 불가

**해결:**

```tablegen
// 올바른 설계 - 명확한 타입
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$funcName,  // 함수 심볼
                       Variadic<AnyType>:$captured);  // 캡처된 값 (다양한 타입)
  let results = (outs FunLang_ClosureType:$result);  // GOOD: 명확한 타입!
}
```

**원칙:**
- 도메인 타입 (closure, list)은 커스텀 타입 사용
- 범용 값 (캡처된 변수)은 `AnyType` 허용

### Pitfall 2: Missing Operation Traits (Pure, MemoryEffects)

**문제:**

```tablegen
// Trait 없는 operation
def FunLang_MakeClosureOp : FunLang_Op<"make_closure"> {
  // No traits specified!
}
```

**왜 문제인가?**

- MLIR이 side effect 가정 (보수적 최적화)
- CSE (Common Subexpression Elimination) 불가
- Dead code elimination 불가

**예시:**

```mlir
// 중복 클로저 생성
%closure1 = funlang.make_closure @lambda(%x) : !funlang.closure
%closure2 = funlang.make_closure @lambda(%x) : !funlang.closure
// Trait 없으면: 둘 다 유지 (side effect 가능성 가정)
// Pure trait 있으면: %closure2 = %closure1 (CSE 적용)
```

**해결:**

```tablegen
// 올바른 설계 - Trait 명시
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  // Pure = no side effects, deterministic
}

def FunLang_AllocEnvOp : FunLang_Op<"alloc_env", [MemoryEffects<[MemAlloc]>]> {
  // MemAlloc = allocates memory (but no read/write side effects)
}
```

**자주 사용하는 traits:**

| Trait | 의미 | 예시 |
|-------|------|------|
| `Pure` | 부작용 없음 | `arith.addi`, `funlang.make_closure` |
| `MemoryEffects<[MemRead]>` | 메모리 읽기만 | `memref.load` |
| `MemoryEffects<[MemWrite]>` | 메모리 쓰기만 | `memref.store` |
| `MemoryEffects<[MemAlloc]>` | 메모리 할당만 | `memref.alloc` |

### Pitfall 3: Symbol Table 미사용 (String 함수 참조)

**문제:**

```tablegen
// 잘못된 설계 - 함수 이름을 문자열로
def FunLang_MakeClosureOp : FunLang_Op<"make_closure"> {
  let arguments = (ins StrAttr:$funcName);  // ERROR: 타입 체크 불가!
}
```

**왜 문제인가?**

- 함수 존재 여부 체크 불가 (컴파일 타임)
- 함수 시그니처 검증 불가
- Linker가 심볼 해석 불가

**예시:**

```mlir
// 문자열 사용 - 에러 발견 안 됨!
%closure = funlang.make_closure "typo_func"  // 함수 없어도 pass!
```

**해결:**

```tablegen
// 올바른 설계 - SymbolRefAttr 사용
def FunLang_MakeClosureOp : FunLang_Op<"make_closure", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$funcName);  // GOOD: 심볼 참조
}
```

**사용:**

```mlir
// 심볼 참조 - 컴파일 타임 체크!
%closure = funlang.make_closure @lambda_func  // 함수 없으면 에러!

// 함수 정의 필요
func.func private @lambda_func(%env: !llvm.ptr, %x: i32) -> i32 {
  // ...
}
```

**SymbolRefAttr의 이점:**
- 컴파일 타임 심볼 해석
- 함수 시그니처 체크 가능
- IDE 지원 (jump to definition)

### Pitfall 4: C API 메모리 관리 혼동

**문제:**

```cpp
// 잘못된 C API - 포인터 반환
extern "C" {
    MlirValue* mlirFunLangGetCapturedValues(MlirOperation op) {
        auto makeClosureOp = cast<MakeClosureOp>(unwrap(op));
        auto captured = makeClosureOp.getCapturedValues();

        // ERROR: SmallVector 로컬 변수!
        SmallVector<MlirValue, 4> result;
        for (Value v : captured) {
            result.push_back(wrap(v));
        }

        // DANGER: 댕글링 포인터! (result는 스택)
        return result.data();
    }
}
```

**왜 문제인가?**

- C API는 ownership 명확히 해야 함
- 스택 메모리 반환 → use-after-free
- F#은 언제 메모리 해제할지 모름

**해결 1: 호출자가 버퍼 제공**

```cpp
extern "C" {
    intptr_t mlirFunLangGetCapturedValuesInto(MlirOperation op,
                                               MlirValue *buffer,
                                               intptr_t bufferSize) {
        auto makeClosureOp = cast<MakeClosureOp>(unwrap(op));
        auto captured = makeClosureOp.getCapturedValues();

        intptr_t numCaptured = captured.size();
        if (numCaptured > bufferSize)
            return -1;  // Buffer too small

        for (intptr_t i = 0; i < numCaptured; ++i) {
            buffer[i] = wrap(captured[i]);
        }

        return numCaptured;
    }
}
```

F#에서:

```fsharp
let buffer = Array.zeroCreate<MlirValue> 10
let count = mlirFunLangGetCapturedValuesInto(op, buffer, 10n)
let capturedValues = buffer.[0..int count - 1]
```

**해결 2: Iterator 패턴**

```cpp
extern "C" {
    void mlirFunLangMakeClosureForEachCaptured(MlirOperation op,
                                                 void (*callback)(MlirValue, void*),
                                                 void *userData) {
        auto makeClosureOp = cast<MakeClosureOp>(unwrap(op));
        for (Value v : makeClosureOp.getCapturedValues()) {
            callback(wrap(v), userData);
        }
    }
}
```

**원칙:**
- C API는 ownership 명확히 (caller owns? callee owns?)
- 배열 반환: caller-provided buffer 또는 callback
- 문서화: "caller must free" vs "MLIR owns"

## 요약

**Chapter 14에서 배운 것:**

1. **Progressive Lowering의 필요성**: Phase 4 직접 lowering의 문제 (복잡도, 최적화 상실, 디버깅 어려움)

2. **MLIR Dialect 아키텍처**: Operation (계산), Type (값), Attribute (상수), Region/Block (제어 흐름), Symbol Table (전역 참조)

3. **TableGen ODS 기초**:
   - Dialect 정의 (`FunLang_Dialect`)
   - Operation 정의 (arguments, results, traits, assemblyFormat)
   - Type 정의 (`ClosureType`, `ListType`)
   - 생성된 C++ 코드 (parser, printer, builder, verifier)

4. **C API Shim 패턴**:
   - 문제: TableGen → C++, F# → C API
   - 해결: `extern "C"` wrapper (FunLangCAPI.h/.cpp)
   - wrap/unwrap helpers (C ↔ C++ 변환)
   - OpBuilder 활용 (operation 생성)
   - F# P/Invoke bindings

5. **FunLang Operations 설계**:
   - `funlang.make_closure`: 클로저 생성
   - `funlang.apply`: 클로저 호출
   - `funlang.match`: 패턴 매칭 (Phase 6)
   - Lowering 전략 (FunLang → Func/MemRef → LLVM)

6. **Common Pitfalls**:
   - AnyType 남용 → 커스텀 타입 사용
   - Trait 누락 → Pure, MemoryEffects 명시
   - 문자열 함수 참조 → SymbolRefAttr 사용
   - C API 메모리 관리 → ownership 명확히

**다음 장 (Chapter 15) Preview:**

Chapter 15에서는:
- FunLang dialect 실제 구현 (C++ 코드 작성)
- TableGen 파일 작성 (FunLangOps.td, FunLangTypes.td)
- C API shim 구현 (FunLangCAPI.cpp)
- F# bindings 작성 (FunLangBindings.fs)
- Lowering pass 구현 (FunLangToFunc.cpp)
- 컴파일러 통합 (Compiler.fs 수정)
- 전체 빌드 시스템 (CMakeLists.txt)

이론적 기초를 확립했으므로, 실제 구현으로 넘어갈 준비가 됐다.
