# Chapter 15: 커스텀 Operations (Custom Operations)

## 소개

**Chapter 14**에서는 커스텀 MLIR dialect의 **이론**을 다뤘다:
- Progressive lowering 철학
- TableGen ODS 문법
- C API shim 패턴
- FunLang dialect 설계 방향

**Chapter 15**에서는 **실제 구현**을 진행한다. FunLang dialect의 핵심 operations를 정의하고 F#에서 사용할 수 있게 만든다.

### Chapter 15의 목표

1. **funlang.closure Operation**: Chapter 12의 12줄 클로저 생성 코드를 1줄로 압축
2. **funlang.apply Operation**: Chapter 13의 8줄 간접 호출 코드를 1줄로 압축
3. **funlang.match Operation (Preview)**: Phase 6 패턴 매칭을 위한 준비
4. **FunLang Custom Types**: `!funlang.closure`, `!funlang.list` 타입 정의
5. **Complete F# Integration**: C API shim부터 F# wrapper까지 전체 스택 구축

### Before vs After: 코드 압축의 위력

**Before (Phase 4 - Chapter 12):**

```mlir
// 클로저 생성: 12줄
func.func @make_adder(%n: i32) -> !llvm.ptr {
    %env_size = arith.constant 16 : i64
    %env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr
    %n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr
    func.return %env_ptr : !llvm.ptr
}

// 클로저 호출: 8줄
func.func @apply(%f: !llvm.ptr, %x: i32) -> i32 {
    %c0 = arith.constant 0 : i64
    %fn_ptr_addr = llvm.getelementptr %f[0, %c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32
    func.return %result : i32
}
```

**After (Phase 5 - Chapter 15):**

```mlir
// 클로저 생성: 1줄!
func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    func.return %closure : !funlang.closure
}

// 클로저 호출: 1줄!
func.func @apply(%f: !funlang.closure, %x: i32) -> i32 {
    %result = funlang.apply %f(%x) : (i32) -> i32
    func.return %result : i32
}
```

**개선 효과:**
- **코드 줄 수**: 20줄 → 4줄 (80% 감소!)
- **가독성**: GEP/store 패턴 제거, 의도 명확
- **타입 안전성**: `!llvm.ptr` → `!funlang.closure` (타입 시스템 활용)
- **최적화 가능성**: 클로저 인라이닝, escape analysis 등

### Chapter 14 복습

커스텀 dialect를 만드는 3가지 핵심 요소:

**1. TableGen ODS (Operation Definition Specification)**

- 선언적으로 operation 정의 (파서/프린터/빌더 자동 생성)
- `.td` 파일로 작성

**2. C++ Dialect 구현**

- TableGen이 생성한 클래스를 활용
- Verifier, lowering pass 구현

**3. C API Shim**

- `extern "C"` wrapper로 F# P/Invoke 연결
- `wrap`/`unwrap` 헬퍼로 C handle ↔ C++ pointer 변환

**이 장에서는 이 세 요소를 모두 구현한다.**

### 구현할 Operations

| Operation | Purpose | Phase |
|-----------|---------|-------|
| `funlang.closure` | 클로저 생성 (GC_malloc + store 추상화) | 5 |
| `funlang.apply` | 클로저 호출 (GEP + load + llvm.call 추상화) | 5 |
| `funlang.match` | 패턴 매칭 (region-based control flow) | 6 preview |

### 구현할 Types

| Type | Purpose | Phase |
|------|---------|-------|
| `!funlang.closure` | 클로저 값 (opaque type) | 5 |
| `!funlang.list<T>` | 불변 리스트 (parameterized type) | 6 preview |

### Chapter 15 성공 기준

이 장을 완료하면:
- [ ] `funlang.closure` operation을 TableGen으로 정의할 수 있다
- [ ] C API shim 함수를 작성해 F#에서 호출할 수 있다
- [ ] F# P/Invoke 바인딩을 작성할 수 있다
- [ ] Chapter 12-13의 compileExpr 코드를 리팩토링할 수 있다
- [ ] Phase 4 대비 코드 줄 수가 60% 이상 감소한다
- [ ] Region-based operation (funlang.match)의 구조를 이해한다

> **Preview:** Chapter 16에서는 FunLang dialect을 LLVM dialect으로 lowering하는 pass를 구현한다.

---

## Part 1: funlang.closure Operation

### Phase 4 패턴 분석: 무엇을 추상화하는가?

Chapter 12에서 클로저를 생성할 때, 12줄의 LLVM dialect 코드가 필요했다:

```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // Step 1: 환경 크기 계산
    // 함수 포인터 (8 bytes) + 캡처된 변수 (4 bytes * count)
    %env_size = arith.constant 16 : i64

    // Step 2: GC_malloc 호출로 환경 할당
    %env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // Step 3: 함수 포인터 저장 (env[0])
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr

    // Step 4: 캡처된 변수 n 저장 (env[1])
    %n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr

    // Step 5: 환경 포인터 반환 (클로저 값)
    func.return %env_ptr : !llvm.ptr
}
```

**패턴 분석:**

1. **환경 크기 계산**: 8 (fn ptr) + 4 * n (captured vars)
   - 컴파일 타임에 결정 가능
   - 하지만 컴파일러 코드에서 수동 계산 필요

2. **GC_malloc 호출**: 힙 할당
   - 모든 클로저에 공통
   - 크기만 다름

3. **함수 포인터 저장**: `env[0]` 슬롯에 `@lambda_N` 주소
   - 모든 클로저에 공통
   - 슬롯 인덱스는 항상 0

4. **변수 저장**: `env[1..n]` 슬롯에 캡처된 변수들
   - 변수 개수만 다름
   - GEP + store 패턴 반복

5. **타입**: `!llvm.ptr` (opaque)
   - 타입 안전성 없음
   - 클로저인지 일반 포인터인지 구별 불가

**문제점:**

- **반복 코드**: 모든 람다마다 동일한 패턴 12줄
- **인덱스 오류 가능성**: `env[0]` vs `env[1]` 수동 관리
- **타입 안전성 부족**: 모든 포인터가 `!llvm.ptr`
- **최적화 어려움**: 클로저인지 알 수 없음
- **가독성 저하**: 저수준 메모리 조작 노출

**해결책: funlang.closure Operation**

이 패턴을 단일 operation으로 추상화한다:

```mlir
// Before: 12 lines
%env_size = arith.constant 16 : i64
%env_ptr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
%fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
%fn_slot = llvm.getelementptr %env_ptr[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr
%n_slot = llvm.getelementptr %env_ptr[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %n, %n_slot : i32, !llvm.ptr

// After: 1 line!
%closure = funlang.closure @lambda_adder, %n : !funlang.closure
```

**이득:**

- **간결성**: 12줄 → 1줄
- **타입 안전성**: `!funlang.closure` (dedicated type)
- **의도 명확**: "클로저를 만든다"라는 의미가 즉시 보임
- **컴파일러 단순화**: GEP 인덱스 계산 불필요
- **최적화 가능**: 클로저 특화 pass 작성 가능 (escape analysis, inlining)

### TableGen 정의: FunLang_ClosureOp

`FunLangOps.td` 파일에 다음과 같이 정의한다:

```tablegen
//===- FunLangOps.td - FunLang dialect operations ---------*- tablegen -*-===//
//
// FunLang Dialect Operations
//
//===----------------------------------------------------------------------===//

#ifndef FUNLANG_OPS
#define FUNLANG_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "FunLangDialect.td"
include "FunLangTypes.td"

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

def FunLang_ClosureOp : FunLang_Op<"closure", [Pure]> {
  let summary = "Create a closure with captured environment";

  let description = [{
    Creates a closure by combining a function reference with captured values.

    Syntax:
    ```
    %closure = funlang.closure @func_name, %arg1, %arg2, ... : !funlang.closure
    ```

    This operation abstracts the low-level closure creation pattern:
    - Allocate environment (GC_malloc)
    - Store function pointer (env[0])
    - Store captured values (env[1..n])

    Example:
    ```
    // Create closure: fun x -> x + n
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    ```

    Lowering to LLVM dialect:
    - Compute environment size: 8 (fn ptr) + sizeof(captured values)
    - Call GC_malloc
    - Store function pointer at slot 0
    - Store captured values at slots 1..n
    - Return environment pointer
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$capturedValues
  );

  let results = (outs FunLang_ClosureType:$result);

  let assemblyFormat = [{
    $callee (`,` $capturedValues^)? attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "mlir::FlatSymbolRefAttr":$callee,
                   "mlir::ValueRange":$capturedValues), [{
      build($_builder, $_state,
            FunLangClosureType::get($_builder.getContext()),
            callee, capturedValues);
    }]>
  ];
}

#endif // FUNLANG_OPS
```

### TableGen 상세 설명

#### 1. Operation 이름과 Traits

```tablegen
def FunLang_ClosureOp : FunLang_Op<"closure", [Pure]> {
```

**구성 요소:**

- `FunLang_ClosureOp`: C++ 클래스 이름 (ClosureOp 생성)
- `"closure"`: MLIR assembly에서의 operation 이름 (funlang.closure)
- `[Pure]`: Operation traits 리스트

**Pure Trait:**

`Pure` trait는 operation이 **side-effect free**임을 선언한다:

```cpp
// Pure operation의 의미:
// 1. 같은 입력 → 항상 같은 출력
// 2. 메모리 읽기/쓰기 없음 (pure function)
// 3. 외부 상태에 영향 없음
```

**왜 funlang.closure가 Pure인가?**

"GC_malloc을 호출하는데 Pure라고?"라는 의문이 들 수 있다. 여기서 Pure는 **FunLang dialect 수준**에서의 의미다:

- **FunLang 수준**: 클로저 생성은 pure (같은 인자 → 같은 클로저 값)
- **Lowering 후**: GC_malloc 호출 (side effect 있음)

Progressive lowering의 핵심: **각 dialect 수준에서 독립적인 의미론을 가진다.**

Pure trait의 이점:

```mlir
// CSE (Common Subexpression Elimination) 가능
%c1 = funlang.closure @lambda_add, %n : !funlang.closure
%c2 = funlang.closure @lambda_add, %n : !funlang.closure
// CSE pass가 %c2를 %c1로 대체 가능 (Pure이므로)
```

#### 2. Summary와 Description

```tablegen
let summary = "Create a closure with captured environment";
```

- **summary**: 한 줄 설명 (IDE tooltip, 문서 생성에 사용)
- **description**: 상세 설명 (Markdown 포맷 지원)

Description에 포함할 내용:

1. **Syntax**: 사용 방법
2. **Semantics**: 의미론 (무엇을 하는가)
3. **Example**: 구체적 예시
4. **Lowering**: LLVM dialect으로의 변환 방법

#### 3. Arguments (입력)

```tablegen
let arguments = (ins
  FlatSymbolRefAttr:$callee,
  Variadic<AnyType>:$capturedValues
);
```

**FlatSymbolRefAttr:$callee**

- **타입**: Symbol reference (함수 이름)
- **이름**: `callee` (호출할 함수)
- **FlatSymbolRefAttr**: 같은 모듈 내 심볼 참조

```mlir
// FlatSymbolRefAttr 예시
funlang.closure @lambda_adder, %n  // @lambda_adder가 FlatSymbolRefAttr
```

**왜 StrAttr이 아니라 FlatSymbolRefAttr인가?**

| StrAttr | FlatSymbolRefAttr |
|---------|-------------------|
| 단순 문자열 | 심볼 테이블 참조 |
| 검증 없음 | 컴파일 타임 검증 (심볼 존재 여부) |
| 최적화 불가 | 최적화 가능 (인라이닝, DCE) |
| 타입 정보 없음 | 타입 정보 있음 (함수 시그니처) |

```tablegen
// 잘못된 정의
let arguments = (ins StrAttr:$callee, ...);
// 문제: "@lambda_adder"가 존재하는지 검증 불가

// 올바른 정의
let arguments = (ins FlatSymbolRefAttr:$callee, ...);
// MLIR이 심볼 테이블에서 @lambda_adder 검증
```

**Variadic<AnyType>:$capturedValues**

- **Variadic**: 가변 길이 인자 (0개 이상)
- **AnyType**: 어떤 타입이든 허용
- **이름**: `capturedValues`

```mlir
// 캡처 변수 0개
%closure0 = funlang.closure @const_fn : !funlang.closure

// 캡처 변수 1개
%closure1 = funlang.closure @add_n, %n : !funlang.closure

// 캡처 변수 3개
%closure3 = funlang.closure @lambda_xyz, %x, %y, %z : !funlang.closure
```

**AnyType의 Trade-off:**

장점:
- 유연성: i32, f64, !llvm.ptr 등 모든 타입 허용
- 간단한 정의

단점:
- 타입 안전성 감소
- Verifier에서 추가 검증 필요

Alternative (더 엄격한 타입):
```tablegen
// 특정 타입만 허용
let arguments = (ins
  FlatSymbolRefAttr:$callee,
  Variadic<AnyTypeOf<[I32, F64, LLVM_AnyPointer]>>:$capturedValues
);
```

Phase 5에서는 단순성을 위해 AnyType을 사용한다.

#### 4. Results (출력)

```tablegen
let results = (outs FunLang_ClosureType:$result);
```

- **outs**: 출력 값들
- **FunLang_ClosureType**: 커스텀 타입 (FunLangTypes.td에 정의)
- **$result**: 결과 값 이름

단일 결과 operation이므로 `outs` 안에 하나만 선언한다.

**FunLang_ClosureType은 어디서 정의되는가?**

`FunLangTypes.td` 파일에 다음과 같이 정의한다:

```tablegen
//===- FunLangTypes.td - FunLang dialect types ------------*- tablegen -*-===//

#ifndef FUNLANG_TYPES
#define FUNLANG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "FunLangDialect.td"

//===----------------------------------------------------------------------===//
// FunLang Type Definitions
//===----------------------------------------------------------------------===//

class FunLang_Type<string name, string typeMnemonic>
    : TypeDef<FunLang_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def FunLang_ClosureType : FunLang_Type<"Closure", "closure"> {
  let summary = "FunLang closure type";
  let description = [{
    Represents a closure value (function + captured environment).

    Syntax: `!funlang.closure`

    Opaque type (no type parameters).
    Lowering: !funlang.closure -> !llvm.ptr
  }];
}

#endif // FUNLANG_TYPES
```

#### 5. Assembly Format (Parser/Printer)

```tablegen
let assemblyFormat = [{
  $callee (`,` $capturedValues^)? attr-dict `:` type($result)
}];
```

**구문 분석:**

- `$callee`: 심볼 참조 (필수)
- `(`,` $capturedValues^)?`: 캡처 변수들 (선택, 쉼표로 구분)
  - `^`: anchor (variadic의 첫 요소에만 `,` 붙음)
  - `?`: 선택 (캡처 변수 없으면 생략)
- `attr-dict`: 추가 속성들 (location 등)
- `:`: 타입 구분자
- `type($result)`: 결과 타입 (`:!funlang.closure`)

**생성되는 Assembly:**

```mlir
// 캡처 변수 없음
%c0 = funlang.closure @const_fn : !funlang.closure

// 캡처 변수 1개
%c1 = funlang.closure @add_n, %n : !funlang.closure

// 캡처 변수 3개
%c3 = funlang.closure @lambda_xyz, %x, %y, %z : !funlang.closure
```

**TableGen이 자동 생성:**

- **Parser**: assembly → C++ operation
- **Printer**: C++ operation → assembly

수동 구현과 비교:

```cpp
// 수동 구현 (100+ lines)
class ClosureOp : public Op<...> {
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

// TableGen 자동 생성 (1 line in .td)
let assemblyFormat = [{...}];
```

#### 6. Builders (생성자)

```tablegen
let builders = [
  OpBuilder<(ins "mlir::FlatSymbolRefAttr":$callee,
                 "mlir::ValueRange":$capturedValues), [{
    build($_builder, $_state,
          FunLangClosureType::get($_builder.getContext()),
          callee, capturedValues);
  }]>
];
```

**Builder의 역할:**

C++ 코드에서 operation을 생성할 때 사용하는 헬퍼 함수:

```cpp
// C++ 코드에서 사용
auto calleeAttr = mlir::FlatSymbolRefAttr::get(context, "lambda_adder");
SmallVector<mlir::Value> captured = {nValue};
auto closure = builder.create<FunLang::ClosureOp>(loc, calleeAttr, captured);
```

**Builder 파라미터:**

- `$_builder`: OpBuilder 인스턴스
- `$_state`: OperationState (operation 생성 중간 상태)
- `callee`: 함수 심볼
- `capturedValues`: 캡처된 변수들

**자동 타입 추론:**

Builder 내부에서 결과 타입을 자동으로 설정한다:

```cpp
FunLangClosureType::get($_builder.getContext())
// 항상 !funlang.closure 타입
```

### 생성되는 C++ 클래스

TableGen은 `FunLangOps.td`를 읽고 다음 C++ 코드를 생성한다:

**Generated: FunLangOps.h.inc**

```cpp
namespace mlir {
namespace funlang {

class ClosureOp : public Op<ClosureOp,
                             OpTrait::ZeroRegions,
                             OpTrait::OneResult,
                             OpTrait::Pure> {
public:
  using Op::Op;

  static StringRef getOperationName() {
    return "funlang.closure";
  }

  // Accessors
  FlatSymbolRefAttr getCalleeAttr() { return /*...*/ ; }
  StringRef getCallee() { return getCalleeAttr().getValue(); }

  OperandRange getCapturedValues() { return /*...*/ ; }

  FunLangClosureType getType() { return /*...*/ ; }

  // Builder
  static void build(OpBuilder &builder, OperationState &state,
                    FlatSymbolRefAttr callee,
                    ValueRange capturedValues);

  // Parser/Printer
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);

  // Verifier (default)
  LogicalResult verify();
};

} // namespace funlang
} // namespace mlir
```

**자동 생성되는 기능:**

1. **Accessors**: `getCallee()`, `getCapturedValues()` (argument 접근)
2. **Builder**: `create<ClosureOp>(...)` (operation 생성)
3. **Parser**: assembly → operation (assemblyFormat 기반)
4. **Printer**: operation → assembly (assemblyFormat 기반)
5. **Verifier**: 기본 검증 (타입 일치, operand 개수)

### C API Shim 구현

F#에서 ClosureOp를 생성하려면 C API shim이 필요하다.

**FunLangCAPI.h:**

```c
//===- FunLangCAPI.h - C API for FunLang dialect --------------------------===//

#ifndef FUNLANG_CAPI_H
#define FUNLANG_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// FunLang Types
//===----------------------------------------------------------------------===//

/// Create a FunLang closure type.
MLIR_CAPI_EXPORTED MlirType mlirFunLangClosureTypeGet(MlirContext ctx);

/// Check if a type is a FunLang closure type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAFunLangClosureType(MlirType type);

//===----------------------------------------------------------------------===//
// FunLang Operations
//===----------------------------------------------------------------------===//

/// Create a funlang.closure operation.
///
/// Arguments:
///   ctx: MLIR context
///   loc: Source location
///   callee: Symbol reference to the function (FlatSymbolRefAttr)
///   numCaptured: Number of captured values
///   capturedValues: Array of captured SSA values
///
/// Returns: The created operation (as MlirOperation)
MLIR_CAPI_EXPORTED MlirOperation mlirFunLangClosureOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirAttribute callee,
    intptr_t numCaptured,
    MlirValue *capturedValues);

/// Get the callee attribute from a funlang.closure operation.
MLIR_CAPI_EXPORTED MlirAttribute mlirFunLangClosureOpGetCallee(MlirOperation op);

/// Get the number of captured values from a funlang.closure operation.
MLIR_CAPI_EXPORTED intptr_t mlirFunLangClosureOpGetNumCapturedValues(MlirOperation op);

/// Get a captured value by index from a funlang.closure operation.
MLIR_CAPI_EXPORTED MlirValue mlirFunLangClosureOpGetCapturedValue(
    MlirOperation op, intptr_t index);

#ifdef __cplusplus
}
#endif

#endif // FUNLANG_CAPI_H
```

**FunLangCAPI.cpp:**

```cpp
//===- FunLangCAPI.cpp - C API for FunLang dialect ------------------------===//

#include "FunLangCAPI.h"
#include "FunLang/FunLangDialect.h"
#include "FunLang/FunLangOps.h"
#include "FunLang/FunLangTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace mlir::funlang;

//===----------------------------------------------------------------------===//
// Type API
//===----------------------------------------------------------------------===//

MlirType mlirFunLangClosureTypeGet(MlirContext ctx) {
  return wrap(FunLangClosureType::get(unwrap(ctx)));
}

bool mlirTypeIsAFunLangClosureType(MlirType type) {
  return unwrap(type).isa<FunLangClosureType>();
}

//===----------------------------------------------------------------------===//
// Operation API
//===----------------------------------------------------------------------===//

MlirOperation mlirFunLangClosureOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirAttribute callee,
    intptr_t numCaptured,
    MlirValue *capturedValues) {

  MLIRContext *context = unwrap(ctx);
  Location location = unwrap(loc);

  // Verify callee is a FlatSymbolRefAttr
  auto calleeAttr = unwrap(callee).dyn_cast<FlatSymbolRefAttr>();
  assert(calleeAttr && "callee must be a FlatSymbolRefAttr");

  // Build captured values range
  SmallVector<Value, 4> captured;
  for (intptr_t i = 0; i < numCaptured; ++i) {
    captured.push_back(unwrap(capturedValues[i]));
  }

  // Create operation using OpBuilder
  OpBuilder builder(context);
  auto op = builder.create<ClosureOp>(location, calleeAttr, captured);

  return wrap(op.getOperation());
}

MlirAttribute mlirFunLangClosureOpGetCallee(MlirOperation op) {
  auto closureOp = llvm::cast<ClosureOp>(unwrap(op));
  return wrap(closureOp.getCalleeAttr());
}

intptr_t mlirFunLangClosureOpGetNumCapturedValues(MlirOperation op) {
  auto closureOp = llvm::cast<ClosureOp>(unwrap(op));
  return closureOp.getCapturedValues().size();
}

MlirValue mlirFunLangClosureOpGetCapturedValue(MlirOperation op, intptr_t index) {
  auto closureOp = llvm::cast<ClosureOp>(unwrap(op));
  return wrap(closureOp.getCapturedValues()[index]);
}
```

**wrap/unwrap Pattern:**

MLIR C API의 핵심 패턴:

| Direction | Function | Purpose |
|-----------|----------|---------|
| C → C++ | `unwrap(MlirX)` | C handle을 C++ pointer로 변환 |
| C++ → C | `wrap(X*)` | C++ pointer를 C handle로 변환 |

```cpp
// unwrap: C handle -> C++ pointer
MLIRContext *context = unwrap(ctx);          // MlirContext -> MLIRContext*
Location location = unwrap(loc);             // MlirLocation -> Location
Value value = unwrap(capturedValues[i]);     // MlirValue -> Value

// wrap: C++ pointer -> C handle
MlirOperation result = wrap(op.getOperation());  // Operation* -> MlirOperation
MlirType resultType = wrap(closure_type);         // Type -> MlirType
```

### F# P/Invoke 바인딩

`FunLangBindings.fs`:

```fsharp
namespace Mlir.FunLang

open System.Runtime.InteropServices
open Mlir.Core

/// FunLang dialect P/Invoke bindings
module FunLangBindings =

    //==========================================================================
    // Types
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangClosureTypeGet(MlirContext ctx)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirTypeIsAFunLangClosureType(MlirType ty)

    //==========================================================================
    // Operations
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirFunLangClosureOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirAttribute callee,
        nativeint numCaptured,
        MlirValue[] capturedValues)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirFunLangClosureOpGetCallee(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirFunLangClosureOpGetNumCapturedValues(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirFunLangClosureOpGetCapturedValue(
        MlirOperation op,
        nativeint index)

/// High-level F# wrappers for FunLang operations
type FunLangOps =

    /// Create !funlang.closure type
    static member ClosureType(context: MlirContext) : MlirType =
        FunLangBindings.mlirFunLangClosureTypeGet(context)

    /// Check if type is !funlang.closure
    static member IsClosureType(ty: MlirType) : bool =
        FunLangBindings.mlirTypeIsAFunLangClosureType(ty)

    /// Create funlang.closure operation
    static member CreateClosure(
        context: MlirContext,
        location: MlirLocation,
        callee: string,
        capturedValues: MlirValue list) : MlirOperation =

        // Convert function name to FlatSymbolRefAttr
        use calleeStrRef = MlirStringRef.FromString(callee)
        let calleeAttr =
            mlirFlatSymbolRefAttrGet(context, calleeStrRef)

        // Convert F# list to array
        let capturedArray = List.toArray capturedValues
        let numCaptured = nativeint capturedArray.Length

        // Call C API
        FunLangBindings.mlirFunLangClosureOpCreate(
            context, location, calleeAttr, numCaptured, capturedArray)

    /// Get callee name from funlang.closure operation
    static member GetClosureCallee(op: MlirOperation) : string =
        let attr = FunLangBindings.mlirFunLangClosureOpGetCallee(op)
        let strRef = mlirFlatSymbolRefAttrGetValue(attr)
        MlirStringRef.ToString(strRef)

    /// Get captured values from funlang.closure operation
    static member GetClosureCapturedValues(op: MlirOperation) : MlirValue list =
        let count = FunLangBindings.mlirFunLangClosureOpGetNumCapturedValues(op)
        [ for i in 0n .. (count - 1n) do
            yield FunLangBindings.mlirFunLangClosureOpGetCapturedValue(op, i) ]
```

**F# Wrapper 설계 패턴:**

1. **Low-level bindings**: `FunLangBindings` 모듈에 extern 선언
2. **High-level wrappers**: `FunLangOps` 타입에 static member
3. **타입 변환**: F# list ↔ C array, string ↔ MlirStringRef
4. **Resource 관리**: `use` 키워드로 자동 해제

### 사용 예시: F#에서 funlang.closure 생성

**Before (Phase 4): Low-level LLVM Operations**

```fsharp
// Phase 4: 12줄의 LLVM dialect 코드
let compileLambda (builder: OpBuilder) (param: string) (body: Expr) (freeVars: (string * MlirValue) list) =
    let context = builder.Context
    let loc = builder.Location

    // 1. 환경 크기 계산
    let fnPtrSize = 8L
    let varSize = 4L
    let totalSize = fnPtrSize + (int64 freeVars.Length) * varSize
    let sizeConst = builder.CreateI64Const(totalSize)

    // 2. GC_malloc 호출
    let envPtr = builder.CreateCall("GC_malloc", [sizeConst])

    // 3. 함수 포인터 저장
    let lambdaName = freshLambdaName()
    let fnAddr = builder.CreateAddressOf(lambdaName)
    let fnSlot = builder.CreateGEP(envPtr, 0)
    builder.CreateStore(fnAddr, fnSlot)

    // 4. 캡처된 변수들 저장
    freeVars |> List.iteri (fun i (name, value) ->
        let slot = builder.CreateGEP(envPtr, i + 1)
        builder.CreateStore(value, slot)
    )

    // 5. 환경 포인터 반환
    envPtr
```

**After (Phase 5): FunLang Dialect**

```fsharp
// Phase 5: 1줄!
let compileLambda (builder: OpBuilder) (param: string) (body: Expr) (freeVars: (string * MlirValue) list) =
    let context = builder.Context
    let loc = builder.Location

    // 1. 람다 함수 생성 (lifted function)
    let lambdaName = freshLambdaName()
    createLiftedFunction builder lambdaName param body freeVars

    // 2. 캡처된 변수 값들 추출
    let capturedValues = freeVars |> List.map snd

    // 3. funlang.closure 생성 (1 line!)
    let closureOp = FunLangOps.CreateClosure(context, loc, lambdaName, capturedValues)
    let closureValue = mlirOperationGetResult(closureOp, 0)
    closureValue
```

**코드 비교:**

| Aspect | Phase 4 | Phase 5 | Improvement |
|--------|---------|---------|-------------|
| 줄 수 | ~20 lines | ~10 lines | 50% 감소 |
| GEP 패턴 | 수동 (인덱스 관리) | 없음 | 오류 가능성 제거 |
| 타입 | `!llvm.ptr` | `!funlang.closure` | 타입 안전성 향상 |
| 가독성 | 저수준 메모리 조작 | 고수준 의미 표현 | 명확성 향상 |

### Phase 4 vs Phase 5 코드 비교: 완전한 예시

**테스트 프로그램:**

```fsharp
// FunLang source
let make_adder n =
    fun x -> x + n

let add5 = make_adder 5
let result = add5 10
// result = 15
```

**Phase 4 Generated MLIR (LLVM Dialect):**

```mlir
module {
  // GC_malloc 선언
  llvm.func @GC_malloc(i64) -> !llvm.ptr

  // lambda_adder lifted function
  func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // n 로드 (env[1])
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32

    // x + n
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  // make_adder 함수
  func.func @make_adder(%n: i32) -> !llvm.ptr {
    // 환경 크기: 8 (fn ptr) + 4 (n) = 12 bytes
    %c12 = arith.constant 12 : i64
    %env = llvm.call @GC_malloc(%c12) : (i64) -> !llvm.ptr

    // 함수 포인터 저장 (env[0])
    %fn_addr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %c0 = arith.constant 0 : i64
    %fn_slot = llvm.getelementptr %env[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr

    // n 저장 (env[1])
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr

    func.return %env : !llvm.ptr
  }

  // main 함수
  func.func @main() -> i32 {
    // add5 = make_adder 5
    %c5 = arith.constant 5 : i32
    %add5 = func.call @make_adder(%c5) : (i32) -> !llvm.ptr

    // result = add5 10 (간접 호출)
    %c10 = arith.constant 10 : i32
    %c0 = arith.constant 0 : i64
    %fn_ptr_addr = llvm.getelementptr %add5[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr(%add5, %c10) : (!llvm.ptr, i32) -> i32

    func.return %result : i32
  }
}
```

**Phase 5 Generated MLIR (FunLang Dialect):**

```mlir
module {
  // lambda_adder lifted function (동일)
  func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  // make_adder 함수 (funlang.closure 사용!)
  func.func @make_adder(%n: i32) -> !funlang.closure {
    // 클로저 생성: 1줄!
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    func.return %closure : !funlang.closure
  }

  // main 함수 (funlang.apply는 다음 섹션에서)
  func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32
    %add5 = func.call @make_adder(%c5) : (i32) -> !funlang.closure

    // 간접 호출 (Chapter 15 Part 2에서 funlang.apply로 대체)
    %c10 = arith.constant 10 : i32
    // ... (임시로 Phase 4 패턴 유지)

    func.return %result : i32
  }
}
```

**줄 수 비교 (make_adder 함수만):**

- Phase 4: 12 lines (GC_malloc + store 패턴)
- Phase 5: 2 lines (funlang.closure)
- **Reduction: 83%**

---

## Part 2: funlang.apply Operation

### Phase 4 간접 호출 패턴 분석

Chapter 13에서 클로저를 호출할 때, 8줄의 LLVM dialect 코드가 필요했다:

```mlir
func.func @apply(%f: !llvm.ptr, %x: i32) -> i32 {
    // Step 1: 함수 포인터 추출 (env[0])
    %c0 = arith.constant 0 : i64
    %fn_ptr_addr = llvm.getelementptr %f[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr

    // Step 2: 간접 호출 (환경 + 인자)
    %result = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32

    // Step 3: 결과 반환
    func.return %result : i32
}
```

**패턴 분석:**

1. **상수 0 생성**: 함수 포인터 슬롯 인덱스
2. **GEP**: 환경 포인터의 0번 슬롯 주소 계산
3. **Load**: 함수 포인터 로드
4. **간접 호출**: `llvm.call %fn_ptr(...)`
   - 첫 인자: 환경 포인터 (클로저 자체)
   - 나머지 인자: 실제 함수 인자들
5. **타입 시그니처**: 수동 지정 필요

**문제점:**

- **반복 코드**: 모든 클로저 호출마다 동일한 패턴
- **인덱스 하드코딩**: `%c0` (함수 포인터는 항상 슬롯 0)
- **타입 안전성 부족**: 간접 호출 시그니처 수동 관리
- **환경 전달 실수**: `llvm.call %fn_ptr(%x)` (환경 누락 버그)

**해결책: funlang.apply Operation**

```mlir
// Before: 8 lines
%c0 = arith.constant 0 : i64
%fn_ptr_addr = llvm.getelementptr %f[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
%fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
%result = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32

// After: 1 line!
%result = funlang.apply %f(%x) : (i32) -> i32
```

### TableGen 정의: FunLang_ApplyOp

`FunLangOps.td`에 추가:

```tablegen
//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

def FunLang_ApplyOp : FunLang_Op<"apply", []> {
  let summary = "Apply a closure to arguments";

  let description = [{
    Applies a closure (function + environment) to arguments via indirect call.

    Syntax:
    ```
    %result = funlang.apply %closure(%arg1, %arg2, ...) : (T1, T2, ...) -> Tresult
    ```

    This operation abstracts the indirect call pattern:
    - Load function pointer from closure (env[0])
    - Call function pointer with environment + args

    Example:
    ```
    // Call closure: %f(10)
    %result = funlang.apply %f(%c10) : (i32) -> i32
    ```

    Lowering to LLVM dialect:
    - %fn_ptr_addr = llvm.getelementptr %closure[0]
    - %fn_ptr = llvm.load %fn_ptr_addr
    - %result = llvm.call %fn_ptr(%closure, %args...)
  }];

  let arguments = (ins
    FunLang_ClosureType:$closure,
    Variadic<AnyType>:$args
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $closure `(` $args `)` attr-dict `:` functional-type($args, $result)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$closure,
                   "mlir::ValueRange":$args,
                   "mlir::Type":$resultType), [{
      build($_builder, $_state, resultType, closure, args);
    }]>
  ];
}
```

### TableGen 상세 설명

#### 1. Operation 이름과 Traits

```tablegen
def FunLang_ApplyOp : FunLang_Op<"apply", []> {
```

**Traits가 비어있는 이유:**

`funlang.apply`는 **side-effect가 있다** (간접 호출):

- 호출되는 함수가 무엇을 할지 모름 (메모리 쓰기, I/O 등)
- Pure trait 불가
- 최적화 제한 (CSE 불가, DCE 불가)

**Alternative: MemoryEffects Trait**

Phase 6 이후에는 더 정밀한 trait를 추가할 수 있다:

```tablegen
def FunLang_ApplyOp : FunLang_Op<"apply", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  // ...
}
```

이를 통해 "메모리 읽기만 한다" 등의 정보를 제공할 수 있다.

#### 2. Arguments

```tablegen
let arguments = (ins
  FunLang_ClosureType:$closure,
  Variadic<AnyType>:$args
);
```

**FunLang_ClosureType:$closure**

- **타입**: `!funlang.closure` (커스텀 타입)
- **이름**: `closure`
- **필수**: 단일 값 (variadic 아님)

**ClosureOp와의 차이:**

| ClosureOp | ApplyOp |
|-----------|---------|
| `FlatSymbolRefAttr:$callee` | `FunLang_ClosureType:$closure` |
| 심볼 (함수 이름) | SSA 값 (클로저) |
| 컴파일 타임 해석 | 런타임 값 |

```mlir
// ClosureOp: callee는 심볼
%c = funlang.closure @lambda_add, %n : !funlang.closure

// ApplyOp: closure는 SSA 값
%result = funlang.apply %c(%x) : (i32) -> i32
```

**Variadic<AnyType>:$args**

- **가변 길이 인자**: 0개 이상
- **AnyType**: 타입 제약 없음

```mlir
// 인자 0개
%result0 = funlang.apply %const_fn() : () -> i32

// 인자 1개
%result1 = funlang.apply %add_n(%x) : (i32) -> i32

// 인자 2개
%result2 = funlang.apply %mul(%x, %y) : (i32, i32) -> i32
```

#### 3. Results

```tablegen
let results = (outs AnyType:$result);
```

**AnyType을 사용하는 이유:**

클로저가 반환하는 타입은 **런타임에 결정**된다:

```mlir
// 클로저 A: i32 반환
%r1 = funlang.apply %closure_a(%x) : (i32) -> i32

// 클로저 B: f64 반환
%r2 = funlang.apply %closure_b(%y) : (f64) -> f64

// 클로저 C: 클로저 반환 (HOF)
%r3 = funlang.apply %closure_c(%z) : (i32) -> !funlang.closure
```

**타입 추론:**

Verifier에서 다음을 검증해야 한다:
- 호출 시그니처 (`(T1, ...) -> Tresult`)
- 클로저의 실제 타입과 일치하는지

Phase 5에서는 단순화를 위해 AnyType을 사용하고, 기본 검증만 수행한다.

#### 4. Assembly Format

```tablegen
let assemblyFormat = [{
  $closure `(` $args `)` attr-dict `:` functional-type($args, $result)
}];
```

**구문 분석:**

- `$closure`: 클로저 값 (필수)
- `(` `)`: 괄호 (인자 구분)
- `$args`: 인자들 (쉼표로 자동 구분, 0개 가능)
- `:`: 타입 구분자
- `functional-type($args, $result)`: 함수 타입 `(T1, ...) -> Tresult`

**functional-type이란?**

함수 시그니처 표기법:

```mlir
// functional-type 예시
(i32) -> i32              // 1 arg, 1 result
(i32, i32) -> i32         // 2 args, 1 result
() -> i32                 // 0 args, 1 result
(i32) -> !funlang.closure // HOF (클로저 반환)
```

**생성되는 Assembly:**

```mlir
// 다양한 호출 예시
%r1 = funlang.apply %f() : () -> i32
%r2 = funlang.apply %f(%x) : (i32) -> i32
%r3 = funlang.apply %f(%x, %y) : (i32, i32) -> i32
%r4 = funlang.apply %compose(%f, %g) : (!funlang.closure, !funlang.closure) -> !funlang.closure
```

#### 5. Builders

```tablegen
let builders = [
  OpBuilder<(ins "mlir::Value":$closure,
                 "mlir::ValueRange":$args,
                 "mlir::Type":$resultType), [{
    build($_builder, $_state, resultType, closure, args);
  }]>
];
```

**Builder 파라미터:**

- `closure`: 클로저 SSA 값
- `args`: 인자들 (가변 길이)
- `resultType`: 결과 타입 (명시적 지정 필요)

**C++ 사용 예시:**

```cpp
// C++ code
Value closureVal = /*...*/;
SmallVector<Value> args = {xValue};
Type resultType = builder.getI32Type();

auto applyOp = builder.create<FunLang::ApplyOp>(
    loc, closureVal, args, resultType);
Value result = applyOp.getResult();
```

### C API Shim 구현

`FunLangCAPI.h`에 추가:

```c
//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

/// Create a funlang.apply operation.
///
/// Arguments:
///   ctx: MLIR context
///   loc: Source location
///   closure: Closure value to apply
///   numArgs: Number of arguments
///   args: Array of argument SSA values
///   resultType: Type of the result
///
/// Returns: The created operation (as MlirOperation)
MLIR_CAPI_EXPORTED MlirOperation mlirFunLangApplyOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue closure,
    intptr_t numArgs,
    MlirValue *args,
    MlirType resultType);

/// Get the closure value from a funlang.apply operation.
MLIR_CAPI_EXPORTED MlirValue mlirFunLangApplyOpGetClosure(MlirOperation op);

/// Get the number of arguments from a funlang.apply operation.
MLIR_CAPI_EXPORTED intptr_t mlirFunLangApplyOpGetNumArgs(MlirOperation op);

/// Get an argument by index from a funlang.apply operation.
MLIR_CAPI_EXPORTED MlirValue mlirFunLangApplyOpGetArg(
    MlirOperation op, intptr_t index);

/// Get the result type from a funlang.apply operation.
MLIR_CAPI_EXPORTED MlirType mlirFunLangApplyOpGetResultType(MlirOperation op);
```

`FunLangCAPI.cpp`에 추가:

```cpp
MlirOperation mlirFunLangApplyOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue closure,
    intptr_t numArgs,
    MlirValue *args,
    MlirType resultType) {

  MLIRContext *context = unwrap(ctx);
  Location location = unwrap(loc);
  Value closureVal = unwrap(closure);
  Type resType = unwrap(resultType);

  // Build args range
  SmallVector<Value, 4> argVals;
  for (intptr_t i = 0; i < numArgs; ++i) {
    argVals.push_back(unwrap(args[i]));
  }

  // Create operation
  OpBuilder builder(context);
  auto op = builder.create<ApplyOp>(location, closureVal, argVals, resType);

  return wrap(op.getOperation());
}

MlirValue mlirFunLangApplyOpGetClosure(MlirOperation op) {
  auto applyOp = llvm::cast<ApplyOp>(unwrap(op));
  return wrap(applyOp.getClosure());
}

intptr_t mlirFunLangApplyOpGetNumArgs(MlirOperation op) {
  auto applyOp = llvm::cast<ApplyOp>(unwrap(op));
  return applyOp.getArgs().size();
}

MlirValue mlirFunLangApplyOpGetArg(MlirOperation op, intptr_t index) {
  auto applyOp = llvm::cast<ApplyOp>(unwrap(op));
  return wrap(applyOp.getArgs()[index]);
}

MlirType mlirFunLangApplyOpGetResultType(MlirOperation op) {
  auto applyOp = llvm::cast<ApplyOp>(unwrap(op));
  return wrap(applyOp.getResult().getType());
}
```

### F# P/Invoke 바인딩

`FunLangBindings.fs`에 추가:

```fsharp
module FunLangBindings =
    // (이전 ClosureOp 바인딩...)

    //==========================================================================
    // ApplyOp
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirFunLangApplyOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue closure,
        nativeint numArgs,
        MlirValue[] args,
        MlirType resultType)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirFunLangApplyOpGetClosure(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirFunLangApplyOpGetNumArgs(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirFunLangApplyOpGetArg(MlirOperation op, nativeint index)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangApplyOpGetResultType(MlirOperation op)

type FunLangOps =
    // (이전 ClosureType, CreateClosure...)

    /// Create funlang.apply operation
    static member CreateApply(
        context: MlirContext,
        location: MlirLocation,
        closure: MlirValue,
        args: MlirValue list,
        resultType: MlirType) : MlirValue =

        // Convert F# list to array
        let argsArray = List.toArray args
        let numArgs = nativeint argsArray.Length

        // Call C API
        let op = FunLangBindings.mlirFunLangApplyOpCreate(
            context, location, closure, numArgs, argsArray, resultType)

        // Extract result SSA value
        mlirOperationGetResult(op, 0)

    /// Get closure from funlang.apply operation
    static member GetApplyClosure(op: MlirOperation) : MlirValue =
        FunLangBindings.mlirFunLangApplyOpGetClosure(op)

    /// Get arguments from funlang.apply operation
    static member GetApplyArgs(op: MlirOperation) : MlirValue list =
        let count = FunLangBindings.mlirFunLangApplyOpGetNumArgs(op)
        [ for i in 0n .. (count - 1n) do
            yield FunLangBindings.mlirFunLangApplyOpGetArg(op, i) ]
```

### Closure + Apply 조합 예시

**완전한 makeAdder 예시:**

```mlir
module {
  // Lifted function
  func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // (환경에서 n 로드 - Phase 5에서도 여전히 저수준)
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32

    // x + n 계산
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  // make_adder: funlang.closure 사용
  func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    func.return %closure : !funlang.closure
  }

  // apply: funlang.apply 사용
  func.func @apply(%f: !funlang.closure, %x: i32) -> i32 {
    %result = funlang.apply %f(%x) : (i32) -> i32
    func.return %result : i32
  }

  // main: 전체 조합
  func.func @main() -> i32 {
    // add5 = make_adder 5
    %c5 = arith.constant 5 : i32
    %add5 = func.call @make_adder(%c5) : (i32) -> !funlang.closure

    // result = apply add5 10
    %c10 = arith.constant 10 : i32
    %result = func.call @apply(%add5, %c10) : (!funlang.closure, i32) -> i32

    func.return %result : i32
  }
}
```

**Phase 4 vs Phase 5 비교 (main 함수):**

| Operation | Phase 4 | Phase 5 |
|-----------|---------|---------|
| 클로저 생성 | `func.call @make_adder` → `!llvm.ptr` | `func.call @make_adder` → `!funlang.closure` |
| 클로저 호출 | GEP + load + llvm.call (8 lines) | `funlang.apply` (1 line) |
| 타입 | `!llvm.ptr` (opaque) | `!funlang.closure` (typed) |

**apply 함수 비교:**

```mlir
// Phase 4: 8 lines
func.func @apply(%f: !llvm.ptr, %x: i32) -> i32 {
    %c0 = arith.constant 0 : i64
    %fn_ptr_addr = llvm.getelementptr %f[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32
    func.return %result : i32
}

// Phase 5: 2 lines
func.func @apply(%f: !funlang.closure, %x: i32) -> i32 {
    %result = funlang.apply %f(%x) : (i32) -> i32
    func.return %result : i32
}
```

**Reduction: 75% (8 lines → 2 lines)**

---

## Part 3: funlang.match Operation (Phase 6 Preview)

### 패턴 매칭 개념

**Pattern matching**은 함수형 언어의 핵심 기능이다:

```fsharp
// FunLang pattern matching (Phase 6)
let rec sum_list lst =
    match lst with
    | [] -> 0
    | head :: tail -> head + sum_list tail
```

**두 가지 구성 요소:**

1. **Scrutinee**: 패턴을 검사할 값 (`lst`)
2. **Cases**: 각 패턴과 해당 동작
   - `[]`: nil case (빈 리스트)
   - `head :: tail`: cons case (head와 tail로 분해)

### Why Region-Based Operation?

**나쁜 설계: Block-based (scf.if 스타일)**

```mlir
// 가상의 잘못된 설계
%result = funlang.match %list
    then ^nil_block
    else ^cons_block

^nil_block:
    %zero = arith.constant 0 : i32
    br ^merge_block(%zero : i32)

^cons_block:
    // ... head/tail 분해 ...
    br ^merge_block(%sum : i32)

^merge_block(%result: i32):
    func.return %result : i32
```

**문제점:**

- **블록들이 함수 레벨**: 다른 operation과 섞임
- **결과 타입 검증 어려움**: 각 블록이 독립적
- **가독성 저하**: 패턴과 동작이 분리됨

**좋은 설계: Region-based**

```mlir
%result = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32
  ^cons(%head: i32, %tail: !funlang.list<i32>):
    %sum_tail = /* recursive call */
    %sum = arith.addi %head, %sum_tail : i32
    funlang.yield %sum : i32
}
```

**장점:**

- **각 case가 별도 region**: operation 내부에 encapsulated
- **Block arguments**: 패턴 변수를 직접 표현 (head, tail)
- **Unified terminator**: 모든 case가 `funlang.yield`로 종료
- **타입 검증 간단**: 모든 yield가 같은 타입 반환해야 함

### TableGen 정의: FunLang_MatchOp

```tablegen
//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

def FunLang_MatchOp : FunLang_Op<"match", [
    RecursiveSideEffects,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Pattern matching expression";

  let description = [{
    Pattern matches on a value (scrutinee) with multiple cases.
    Each case is a separate region with optional block arguments.

    Syntax:
    ```
    %result = funlang.match %scrutinee : Tin -> Tout {
      ^case1:
        funlang.yield %val1 : Tout
      ^case2(%arg: T):
        funlang.yield %val2 : Tout
    }
    ```

    Example (list pattern matching):
    ```
    %sum = funlang.match %list : !funlang.list<i32> -> i32 {
      ^nil:
        %zero = arith.constant 0 : i32
        funlang.yield %zero : i32
      ^cons(%head: i32, %tail: !funlang.list<i32>):
        // ... compute sum ...
        funlang.yield %sum : i32
    }
    ```

    Constraints:
    - Each region must have exactly one block
    - Each region must end with funlang.yield
    - All yields must have the same result type

    Lowering (Phase 6):
    - Scrutinee tag check
    - Branch to corresponding case
    - Extract pattern variables (block arguments)
    - Execute case body
  }];

  let arguments = (ins AnyType:$scrutinee);
  let results = (outs AnyType:$result);
  let regions = (region VariadicRegion<SizedRegion<1>>:$cases);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}
```

### Region-Based Operation 설명

#### 1. Traits

```tablegen
def FunLang_MatchOp : FunLang_Op<"match", [
    RecursiveSideEffects,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
```

**RecursiveSideEffects:**

- Match operation의 side effect는 **각 case에 의존**한다
- Case body가 Pure면 match도 Pure
- Case body가 side effect 있으면 match도 side effect 있음

```mlir
// Pure match
%result = funlang.match %x : i32 -> i32 {
  ^case1:
    %c1 = arith.constant 1 : i32
    funlang.yield %c1 : i32  // Pure
  ^case2:
    %c2 = arith.constant 2 : i32
    funlang.yield %c2 : i32  // Pure
}
// 전체 match가 Pure

// Side effect match
%result = funlang.match %x : i32 -> i32 {
  ^case1:
    func.call @print(%c1) : (i32) -> ()  // Side effect!
    funlang.yield %c1 : i32
  ^case2:
    funlang.yield %c2 : i32
}
// 전체 match가 side effect 있음
```

**SingleBlockImplicitTerminator<"YieldOp">:**

- 각 region이 **정확히 하나의 block**을 가짐
- 각 block이 **YieldOp로 종료**됨 (implicit terminator)
- Parser가 자동으로 검증

```mlir
// 올바른 match
%r = funlang.match %x : i32 -> i32 {
  ^case1:
    %val = arith.constant 42 : i32
    funlang.yield %val : i32  // OK: YieldOp terminator
}

// 잘못된 match
%r = funlang.match %x : i32 -> i32 {
  ^case1:
    %val = arith.constant 42 : i32
    func.return %val : i32  // ERROR: Wrong terminator
}
```

#### 2. Regions

```tablegen
let regions = (region VariadicRegion<SizedRegion<1>>:$cases);
```

**VariadicRegion:**

- 가변 개수의 region (case 개수에 따라)
- 최소 1개 이상

**SizedRegion<1>:**

- 각 region이 정확히 **1개의 block**을 가짐
- 다중 block 불가 (control flow는 block 내에서만)

```mlir
// 2개 case
%r = funlang.match %x : i32 -> i32 {
  ^case1: funlang.yield %c1 : i32
  ^case2: funlang.yield %c2 : i32
}

// 3개 case
%r = funlang.match %x : i32 -> i32 {
  ^case1: funlang.yield %c1 : i32
  ^case2: funlang.yield %c2 : i32
  ^case3: funlang.yield %c3 : i32
}
```

**Region vs Block:**

| Concept | Definition | Example |
|---------|------------|---------|
| Region | Operation의 내부 범위 | scf.if의 then/else |
| Block | Region 내의 명령 시퀀스 | 기본 블록 (CFG 노드) |

```mlir
// scf.if: 2 regions, 각 region은 1+ blocks
scf.if %cond {
  // Then region
  %val = arith.constant 1 : i32
  scf.yield %val : i32
} else {
  // Else region
  %val = arith.constant 2 : i32
  scf.yield %val : i32
}

// funlang.match: N regions, 각 region은 정확히 1 block
funlang.match %x : i32 -> i32 {
  // Case 1 region (1 block)
  ^case1:
    funlang.yield %c1 : i32
  // Case 2 region (1 block)
  ^case2:
    funlang.yield %c2 : i32
}
```

#### 3. 각 Case가 별도 Region인 이유

**이유 1: 독립적인 스코프**

각 case는 독립적인 변수 바인딩을 가진다:

```mlir
%result = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil:
    // 이 region에는 변수 없음
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32

  ^cons(%head: i32, %tail: !funlang.list<i32>):
    // 이 region에는 head, tail 변수 있음
    // %head, %tail은 block arguments
    funlang.yield %head : i32
}
```

**이유 2: 타입 안전성**

모든 case의 yield 타입을 검증할 수 있다:

```mlir
// 올바른 match (모든 yield가 i32)
%r = funlang.match %x : i32 -> i32 {
  ^case1: funlang.yield %c1 : i32  // OK
  ^case2: funlang.yield %c2 : i32  // OK
}

// 잘못된 match (타입 불일치)
%r = funlang.match %x : i32 -> i32 {
  ^case1: funlang.yield %c1 : i32       // OK
  ^case2: funlang.yield %f : f64        // ERROR: Type mismatch
}
```

**이유 3: Lowering 간소화**

각 region을 독립적인 블록으로 lowering:

```mlir
// Before lowering
%r = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil: funlang.yield %zero : i32
  ^cons(%h, %t): funlang.yield %h : i32
}

// After lowering (pseudo-code)
%tag = funlang.list_tag %list : i32  // 0 = nil, 1 = cons
cf.switch %tag [
  case 0: ^nil_block
  case 1: ^cons_block
]

^nil_block:
  %zero = arith.constant 0 : i32
  cf.br ^merge(%zero : i32)

^cons_block:
  %h = funlang.list_head %list : i32
  %t = funlang.list_tail %list : !funlang.list<i32>
  cf.br ^merge(%h : i32)

^merge(%result: i32):
  // ...
```

#### 4. Verifier 필요성

```tablegen
let hasVerifier = 1;
```

TableGen 기본 검증만으로는 부족하다. 추가 검증 필요:

**검증 사항:**

1. **모든 yield 타입 일치**: 각 case의 yield 타입 == match 결과 타입
2. **Case 개수 검증**: 최소 1개 이상
3. **Block arguments 타입 검증**: Pattern 변수 타입이 valid한지
4. **Terminator 검증**: 모든 block이 YieldOp로 종료

**C++ Verifier 구현 (Phase 6):**

```cpp
LogicalResult MatchOp::verify() {
  auto resultType = getResult().getType();

  // Check all cases
  for (auto &region : getCases()) {
    Block &block = region.front();

    // Check terminator
    auto yieldOp = dyn_cast<YieldOp>(block.getTerminator());
    if (!yieldOp)
      return emitOpError("case must end with funlang.yield");

    // Check yield type
    auto yieldType = yieldOp.getValue().getType();
    if (yieldType != resultType)
      return emitOpError("yield type mismatch: expected ")
             << resultType << ", got " << yieldType;
  }

  return success();
}
```

### C API Shim 구현 패턴 (Region 생성 포함)

**Region-based operation의 C API는 복잡하다.** Phase 6에서 완전 구현하지만, 패턴을 미리 소개한다.

**FunLangCAPI.h (Preview):**

```c
//===----------------------------------------------------------------------===//
// MatchOp (Phase 6 Preview)
//===----------------------------------------------------------------------===//

/// Create a funlang.match operation.
///
/// Arguments:
///   ctx: MLIR context
///   loc: Source location
///   scrutinee: Value to pattern match on
///   numCases: Number of cases
///   resultType: Type of the result
///
/// Returns: The created operation (caller must build case regions)
MLIR_CAPI_EXPORTED MlirOperation mlirFunLangMatchOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue scrutinee,
    intptr_t numCases,
    MlirType resultType);

/// Get a case region by index from a funlang.match operation.
MLIR_CAPI_EXPORTED MlirRegion mlirFunLangMatchOpGetCaseRegion(
    MlirOperation op, intptr_t index);

/// Create a block in a region with block arguments.
MLIR_CAPI_EXPORTED MlirBlock mlirRegionAppendBlockWithArgs(
    MlirRegion region,
    intptr_t numArgs,
    MlirType *argTypes);

/// Create a funlang.yield operation.
MLIR_CAPI_EXPORTED MlirOperation mlirFunLangYieldOpCreate(
    MlirContext ctx,
    MlirLocation loc,
    MlirValue value);
```

**사용 패턴 (F# pseudo-code):**

```fsharp
// 1. MatchOp 생성 (빈 regions)
let matchOp = FunLangBindings.mlirFunLangMatchOpCreate(
    context, loc, scrutinee, 2, resultType)

// 2. 각 case region 가져오기
let nilRegion = FunLangBindings.mlirFunLangMatchOpGetCaseRegion(matchOp, 0)
let consRegion = FunLangBindings.mlirFunLangMatchOpGetCaseRegion(matchOp, 1)

// 3. Nil case 구축
let nilBlock = FunLangBindings.mlirRegionAppendBlockWithArgs(
    nilRegion, 0, [||])  // No block arguments
builder.SetInsertionPointToEnd(nilBlock)
let zero = builder.CreateI32Const(0)
FunLangBindings.mlirFunLangYieldOpCreate(context, loc, zero)

// 4. Cons case 구축
let consBlock = FunLangBindings.mlirRegionAppendBlockWithArgs(
    consRegion, 2, [| i32Type; listType |])  // head, tail
builder.SetInsertionPointToEnd(consBlock)
let head = mlirBlockGetArgument(consBlock, 0)
let tail = mlirBlockGetArgument(consBlock, 1)
// ... compute with head, tail ...
FunLangBindings.mlirFunLangYieldOpCreate(context, loc, result)
```

**Phase 6에서 완전 구현한다.** Phase 5에서는 MatchOp 정의만 포함한다.

### Phase 6에서의 사용 예시

**FunLang source:**

```fsharp
// Phase 6: List pattern matching
let rec length lst =
    match lst with
    | [] -> 0
    | head :: tail -> 1 + length tail

let test = length [1; 2; 3]
// test = 3
```

**Generated MLIR (Phase 6):**

```mlir
module {
  // length 함수
  func.func @length(%lst: !funlang.list<i32>) -> i32 {
    %result = funlang.match %lst : !funlang.list<i32> -> i32 {
      // Nil case
      ^nil:
        %zero = arith.constant 0 : i32
        funlang.yield %zero : i32

      // Cons case
      ^cons(%head: i32, %tail: !funlang.list<i32>):
        // 1 + length tail
        %one = arith.constant 1 : i32
        %tail_length = func.call @length(%tail) : (!funlang.list<i32>) -> i32
        %result = arith.addi %one, %tail_length : i32
        funlang.yield %result : i32
    }
    func.return %result : i32
  }

  // test = length [1, 2, 3]
  func.func @test() -> i32 {
    // Build list [1, 2, 3]
    %nil = funlang.nil : !funlang.list<i32>
    %c3 = arith.constant 3 : i32
    %lst1 = funlang.cons %c3, %nil : !funlang.list<i32>
    %c2 = arith.constant 2 : i32
    %lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>
    %c1 = arith.constant 1 : i32
    %lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>

    // Call length
    %len = func.call @length(%lst3) : (!funlang.list<i32>) -> i32
    func.return %len : i32
  }
}
```

**Chapter 15에서는 MatchOp의 정의와 구조만 다룬다. 실제 구현과 사용은 Chapter 17 (Phase 6)에서 완성한다.**

---

## Part 4: FunLang Custom Types

### FunLang_ClosureType 상세

Chapter 15 Part 1에서 `!funlang.closure` 타입을 간단히 소개했다. 이제 상세히 다룬다.

**FunLangTypes.td:**

```tablegen
//===- FunLangTypes.td - FunLang dialect types ------------*- tablegen -*-===//

#ifndef FUNLANG_TYPES
#define FUNLANG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "FunLangDialect.td"

//===----------------------------------------------------------------------===//
// FunLang Type Definitions
//===----------------------------------------------------------------------===//

class FunLang_Type<string name, string typeMnemonic>
    : TypeDef<FunLang_Dialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ClosureType
//===----------------------------------------------------------------------===//

def FunLang_ClosureType : FunLang_Type<"Closure", "closure"> {
  let summary = "FunLang closure type";

  let description = [{
    Represents a closure value: a combination of function pointer and
    captured environment.

    Syntax: `!funlang.closure`

    This is an opaque type (no type parameters). The internal representation
    is hidden from the FunLang dialect level.

    Lowering:
    - FunLang dialect: !funlang.closure
    - LLVM dialect: !llvm.ptr

    The lowering pass converts !funlang.closure to !llvm.ptr, exposing the
    internal representation (function pointer + environment data).
  }];

  let extraClassDeclaration = [{
    // No extra methods needed for opaque type
  }];
}

#endif // FUNLANG_TYPES
```

### Opaque Type vs Parameterized Type

**Opaque Type (Phase 5 선택):**

```tablegen
def FunLang_ClosureType : FunLang_Type<"Closure", "closure"> {
  // No parameters
}
```

**MLIR Assembly:**

```mlir
%closure = funlang.closure @lambda_add, %n : !funlang.closure
// 타입 파라미터 없음
```

**장점:**

- **단순성**: 정의와 사용이 간단
- **구현 숨김**: 내부 표현을 dialect 레벨에서 감춤
- **Lowering 유연성**: 표현 방식을 나중에 변경 가능

**단점:**

- **타입 정보 부족**: 함수 시그니처를 타입에서 알 수 없음
- **검증 제한**: 타입 레벨에서 인자/결과 타입 검증 불가

**Parameterized Type (Alternative):**

```tablegen
def FunLang_ClosureType : FunLang_Type<"Closure", "closure"> {
  let parameters = (ins "FunctionType":$funcType);
  let assemblyFormat = "`<` $funcType `>`";
}
```

**MLIR Assembly:**

```mlir
// 파라미터화된 타입
%closure = funlang.closure @lambda_add, %n : !funlang.closure<(i32) -> i32>
//                                          함수 시그니처 ^^^^^^^^^^^
```

**장점:**

- **타입 안전성 향상**: 함수 시그니처가 타입에 포함됨
- **검증 가능**: apply operation에서 인자 타입 검증 가능
- **문서화**: 타입만 봐도 클로저 시그니처 알 수 있음

**단점:**

- **복잡성 증가**: 타입 파라미터 관리 필요
- **Lowering 복잡도**: 타입 변환 시 파라미터 제거 필요

**Phase 5 설계 결정:**

Opaque type을 사용한다:
1. **단순성 우선**: Phase 5는 dialect 도입이 목표
2. **Phase 6 고려**: 리스트 타입은 parameterized (필수)
3. **점진적 복잡도**: 나중에 파라미터 추가 가능

### FunLang_ListType (Phase 6 Preview)

Phase 6에서는 리스트를 위한 **parameterized type**이 필요하다:

```tablegen
//===----------------------------------------------------------------------===//
// ListType (Phase 6)
//===----------------------------------------------------------------------===//

def FunLang_ListType : FunLang_Type<"List", "list"> {
  let summary = "FunLang immutable list type";

  let description = [{
    Represents an immutable linked list.

    Syntax: `!funlang.list<T>`

    Type parameter:
    - T: Element type (any MLIR type)

    Examples:
    - !funlang.list<i32>: List of integers
    - !funlang.list<f64>: List of floats
    - !funlang.list<!funlang.closure>: List of closures

    Lowering:
    - FunLang dialect: !funlang.list<T>
    - LLVM dialect: !llvm.ptr (cons cell pointer)

    Internal representation (after lowering):
    - Nil: nullptr
    - Cons: struct { T head; !llvm.ptr tail }
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";

  let extraClassDeclaration = [{
    // Get element type
    Type getElementType() { return getImpl()->elementType; }
  }];
}
```

**Parameterized Type의 필요성:**

리스트는 **다양한 원소 타입**을 지원해야 한다:

```mlir
// 정수 리스트
%int_list = funlang.nil : !funlang.list<i32>
%int_list2 = funlang.cons %x, %int_list : !funlang.list<i32>

// 클로저 리스트
%closure_list = funlang.nil : !funlang.list<!funlang.closure>
%closure_list2 = funlang.cons %f, %closure_list : !funlang.list<!funlang.closure>
```

타입 파라미터 없이는 **타입 안전성**을 보장할 수 없다:

```mlir
// 잘못된 설계 (opaque list type)
%list = funlang.nil : !funlang.list  // 어떤 타입의 리스트?
%list2 = funlang.cons %x, %list : !funlang.list  // i32? f64?

// 타입 체커가 다음을 검증할 수 없음:
// - cons의 head 타입이 list의 원소 타입과 일치하는지
// - match에서 추출한 head의 타입이 무엇인지
```

### 타입의 LLVM Lowering

Progressive lowering에서 타입도 변환된다:

**FunLang Dialect → LLVM Dialect:**

| FunLang Type | LLVM Type | Internal Representation |
|--------------|-----------|------------------------|
| `!funlang.closure` | `!llvm.ptr` | `struct { fn_ptr, var1, var2, ... }` |
| `!funlang.list<T>` | `!llvm.ptr` | `struct { T head; ptr tail }` or `nullptr` |

**Lowering Pass (Phase 6):**

```cpp
// FunLangToLLVM type converter
class FunLangTypeConverter : public TypeConverter {
public:
  FunLangTypeConverter() {
    // !funlang.closure -> !llvm.ptr
    addConversion([](FunLangClosureType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    // !funlang.list<T> -> !llvm.ptr
    addConversion([](FunLangListType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    // Pass through other types (i32, f64, etc.)
    addConversion([](Type type) { return type; });
  }
};
```

**Lowering 예시:**

```mlir
// Before lowering (FunLang dialect)
func.func @make_adder(%n: i32) -> !funlang.closure {
  %closure = funlang.closure @lambda_add, %n : !funlang.closure
  func.return %closure : !funlang.closure
}

// After lowering (LLVM dialect)
func.func @make_adder(%n: i32) -> !llvm.ptr {
  %env_size = arith.constant 16 : i64
  %env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
  %fn_addr = llvm.mlir.addressof @lambda_add : !llvm.ptr
  %fn_slot = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr
  %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %n, %n_slot : i32, !llvm.ptr
  func.return %env : !llvm.ptr
}
```

**타입 변환과 operation 변환의 관계:**

- **Operation 변환**: `funlang.closure` → `GC_malloc + store` 패턴
- **Type 변환**: `!funlang.closure` → `!llvm.ptr`
- **동시 적용**: Lowering pass가 두 변환을 함께 수행

### C++ Type 클래스 (Generated)

TableGen이 생성하는 C++ 코드:

**Generated: FunLangTypes.h.inc**

```cpp
namespace mlir {
namespace funlang {

class FunLangClosureType : public Type::TypeBase<
    FunLangClosureType,
    Type,
    detail::FunLangClosureTypeStorage> {
public:
  using Base::Base;

  static FunLangClosureType get(MLIRContext *context);

  static constexpr StringLiteral name = "funlang.closure";
};

class FunLangListType : public Type::TypeBase<
    FunLangListType,
    Type,
    detail::FunLangListTypeStorage,
    TypeTrait::HasTypeParameter> {
public:
  using Base::Base;

  static FunLangListType get(Type elementType);

  Type getElementType() const;

  static constexpr StringLiteral name = "funlang.list";
};

} // namespace funlang
} // namespace mlir
```

**사용 예시 (C++):**

```cpp
MLIRContext *context = /*...*/;

// Create !funlang.closure type
auto closureType = FunLangClosureType::get(context);

// Create !funlang.list<i32> type
auto i32Type = IntegerType::get(context, 32);
auto listType = FunLangListType::get(i32Type);

// Get element type
Type elemType = listType.getElementType();
// elemType == i32Type
```

---

## Part 5: Complete F# Integration Module

이제 모든 요소를 통합해 **완전한 F# 래퍼**를 작성한다.

### Mlir.FunLang.fs 모듈 전체 구조

```fsharp
namespace Mlir.FunLang

open System
open System.Runtime.InteropServices
open Mlir.Core

//==============================================================================
// Low-level P/Invoke Bindings
//==============================================================================

module FunLangBindings =

    //==========================================================================
    // Types
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangClosureTypeGet(MlirContext ctx)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirTypeIsAFunLangClosureType(MlirType ty)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangListTypeGet(MlirContext ctx, MlirType elementType)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirTypeIsAFunLangListType(MlirType ty)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangListTypeGetElementType(MlirType ty)

    //==========================================================================
    // Operations - ClosureOp
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirFunLangClosureOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirAttribute callee,
        nativeint numCaptured,
        MlirValue[] capturedValues)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirAttribute mlirFunLangClosureOpGetCallee(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirFunLangClosureOpGetNumCapturedValues(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirFunLangClosureOpGetCapturedValue(
        MlirOperation op,
        nativeint index)

    //==========================================================================
    // Operations - ApplyOp
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirFunLangApplyOpCreate(
        MlirContext ctx,
        MlirLocation loc,
        MlirValue closure,
        nativeint numArgs,
        MlirValue[] args,
        MlirType resultType)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirFunLangApplyOpGetClosure(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint mlirFunLangApplyOpGetNumArgs(MlirOperation op)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirValue mlirFunLangApplyOpGetArg(MlirOperation op, nativeint index)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangApplyOpGetResultType(MlirOperation op)

//==============================================================================
// High-level F# Wrappers
//==============================================================================

/// FunLang dialect operations wrapper
type FunLangDialect(context: MlirContext) =

    /// MLIR context
    member val Context = context

    //==========================================================================
    // Type Creation
    //==========================================================================

    /// Create !funlang.closure type
    member this.ClosureType() : MlirType =
        FunLangBindings.mlirFunLangClosureTypeGet(this.Context)

    /// Check if type is !funlang.closure
    member this.IsClosureType(ty: MlirType) : bool =
        FunLangBindings.mlirTypeIsAFunLangClosureType(ty)

    /// Create !funlang.list<T> type
    member this.ListType(elementType: MlirType) : MlirType =
        FunLangBindings.mlirFunLangListTypeGet(this.Context, elementType)

    /// Check if type is !funlang.list
    member this.IsListType(ty: MlirType) : bool =
        FunLangBindings.mlirTypeIsAFunLangListType(ty)

    /// Get element type from !funlang.list<T>
    member this.ListElementType(ty: MlirType) : MlirType =
        if not (this.IsListType(ty)) then
            invalidArg "ty" "Expected !funlang.list type"
        FunLangBindings.mlirFunLangListTypeGetElementType(ty)

    //==========================================================================
    // Operation Creation
    //==========================================================================

    /// Create funlang.closure operation
    ///
    /// Returns the operation (caller extracts result value via getResult(0))
    member this.CreateClosureOp(
        location: MlirLocation,
        callee: string,
        capturedValues: MlirValue list) : MlirOperation =

        // Convert function name to FlatSymbolRefAttr
        use calleeStrRef = MlirStringRef.FromString(callee)
        let calleeAttr = mlirFlatSymbolRefAttrGet(this.Context, calleeStrRef)

        // Convert F# list to array
        let capturedArray = List.toArray capturedValues
        let numCaptured = nativeint capturedArray.Length

        // Call C API
        FunLangBindings.mlirFunLangClosureOpCreate(
            this.Context, location, calleeAttr, numCaptured, capturedArray)

    /// Create funlang.closure operation and return result value
    member this.CreateClosure(
        location: MlirLocation,
        callee: string,
        capturedValues: MlirValue list) : MlirValue =

        let op = this.CreateClosureOp(location, callee, capturedValues)
        mlirOperationGetResult(op, 0)

    /// Create funlang.apply operation
    ///
    /// Returns the operation (caller extracts result value via getResult(0))
    member this.CreateApplyOp(
        location: MlirLocation,
        closure: MlirValue,
        args: MlirValue list,
        resultType: MlirType) : MlirOperation =

        // Convert F# list to array
        let argsArray = List.toArray args
        let numArgs = nativeint argsArray.Length

        // Call C API
        FunLangBindings.mlirFunLangApplyOpCreate(
            this.Context, location, closure, numArgs, argsArray, resultType)

    /// Create funlang.apply operation and return result value
    member this.CreateApply(
        location: MlirLocation,
        closure: MlirValue,
        args: MlirValue list,
        resultType: MlirType) : MlirValue =

        let op = this.CreateApplyOp(location, closure, args, resultType)
        mlirOperationGetResult(op, 0)

//==============================================================================
// OpBuilder Extension Methods
//==============================================================================

/// Extension methods for OpBuilder to work with FunLang dialect
[<AutoOpen>]
module OpBuilderExtensions =

    type OpBuilder with

        /// Create funlang.closure operation
        member this.CreateFunLangClosure(
            callee: string,
            capturedValues: MlirValue list) : MlirValue =

            let funlang = FunLangDialect(this.Context)
            funlang.CreateClosure(this.Location, callee, capturedValues)

        /// Create funlang.apply operation
        member this.CreateFunLangApply(
            closure: MlirValue,
            args: MlirValue list,
            resultType: MlirType) : MlirValue =

            let funlang = FunLangDialect(this.Context)
            funlang.CreateApply(this.Location, closure, args, resultType)

        /// Create !funlang.closure type
        member this.FunLangClosureType() : MlirType =
            let funlang = FunLangDialect(this.Context)
            funlang.ClosureType()

        /// Create !funlang.list<T> type
        member this.FunLangListType(elementType: MlirType) : MlirType =
            let funlang = FunLangDialect(this.Context)
            funlang.ListType(elementType)
```

### F# Wrapper 클래스 설계

**설계 원칙:**

1. **Low-level과 High-level 분리**
   - `FunLangBindings` 모듈: extern 선언 (P/Invoke)
   - `FunLangDialect` 클래스: 타입 안전 래퍼

2. **Builder 패턴**
   - `CreateClosureOp`: MlirOperation 반환 (유연성)
   - `CreateClosure`: MlirValue 반환 (편의성)

3. **OpBuilder Extension**
   - `this.CreateFunLangClosure(...)`: 간결한 사용
   - Context와 Location 자동 전달

4. **타입 안전성**
   - F# 타입 시스템 활용 (list, string)
   - Runtime 검증 (`IsClosureType`, `IsListType`)

### Builder 패턴으로 Operation 생성

**패턴 1: Direct Operation Creation**

```fsharp
// 명시적 operation 생성
let funlang = FunLangDialect(context)
let op = funlang.CreateClosureOp(location, "lambda_add", [nValue])
let closure = mlirOperationGetResult(op, 0)

// Use cases:
// - Operation에 추가 속성 설정
// - Operation을 블록에 수동 삽입
```

**패턴 2: Direct Value Creation**

```fsharp
// 결과 값만 필요
let funlang = FunLangDialect(context)
let closure = funlang.CreateClosure(location, "lambda_add", [nValue])

// Use cases:
// - 대부분의 일반적인 사용
// - Operation 자체에는 관심 없음
```

**패턴 3: OpBuilder Extension**

```fsharp
// OpBuilder를 통한 생성 (가장 간결)
let closure = builder.CreateFunLangClosure("lambda_add", [nValue])

// Use cases:
// - Compiler.fs에서 compileExpr 내부
// - Location과 Context 자동 전달
// - 코드 가독성 최대화
```

### 타입 안전성 보장

**컴파일 타임 안전성:**

F# 타입 시스템이 다음을 보장:

```fsharp
// 올바른 사용
let values: MlirValue list = [v1; v2; v3]
let closure = builder.CreateFunLangClosure("lambda", values)

// 컴파일 에러
let wrong: int list = [1; 2; 3]
let closure = builder.CreateFunLangClosure("lambda", wrong)
// ERROR: Expected MlirValue list, got int list
```

**런타임 안전성:**

추가 검증 함수 제공:

```fsharp
// 타입 검증
let ty = mlirValueGetType(someValue)
if funlang.IsClosureType(ty) then
    // someValue는 !funlang.closure 타입
    let result = funlang.CreateApply(location, someValue, [arg], i32Type)
else
    failwith "Expected closure type"
```

### 사용 예시: makeAdder를 FunLang Dialect로 컴파일

**Phase 4 Compiler.fs (Before):**

```fsharp
let rec compileExpr (builder: OpBuilder) (env: Map<string, MlirValue>) (expr: Expr) : MlirValue =
    match expr with
    | Lambda(param, body) ->
        // Free variables analysis
        let freeVars = Set.difference (freeVarsExpr body) (Set.singleton param)
        let freeVarList = Set.toList freeVars

        // Create lifted function
        let lambdaName = freshLambdaName()
        createLiftedFunction builder lambdaName param body freeVarList env

        // Environment size: 8 (fn ptr) + 4 * |freeVars|
        let fnPtrSize = 8L
        let varSize = 4L
        let totalSize = fnPtrSize + (int64 freeVarList.Length) * varSize
        let sizeConst = builder.CreateI64Const(totalSize)

        // GC_malloc
        let envPtr = builder.CreateCall("GC_malloc", [sizeConst])

        // Store function pointer at env[0]
        let fnAddr = builder.CreateAddressOf(lambdaName)
        let fnSlot = builder.CreateGEP(envPtr, 0L)
        builder.CreateStore(fnAddr, fnSlot)

        // Store captured values at env[1..n]
        freeVarList |> List.iteri (fun i varName ->
            let value = env.[varName]
            let slot = builder.CreateGEP(envPtr, int64 (i + 1))
            builder.CreateStore(value, slot)
        )

        envPtr  // Return closure (environment pointer)

    | App(funcExpr, argExpr) ->
        // Compile function and argument
        let closureVal = compileExpr builder env funcExpr
        let argVal = compileExpr builder env argExpr

        // Indirect call: GEP + load + llvm.call
        let c0 = builder.CreateI64Const(0L)
        let fnPtrAddr = builder.CreateGEP(closureVal, 0L)
        let fnPtr = builder.CreateLoad(fnPtrAddr, builder.PtrType())
        let result = builder.CreateLLVMCall(fnPtr, [closureVal; argVal], builder.IntType(32))
        result

    // ... other cases ...
```

**Phase 5 Compiler.fs (After):**

```fsharp
let rec compileExpr (builder: OpBuilder) (env: Map<string, MlirValue>) (expr: Expr) : MlirValue =
    match expr with
    | Lambda(param, body) ->
        // Free variables analysis (same)
        let freeVars = Set.difference (freeVarsExpr body) (Set.singleton param)
        let freeVarList = Set.toList freeVars

        // Create lifted function (same)
        let lambdaName = freshLambdaName()
        createLiftedFunction builder lambdaName param body freeVarList env

        // Create closure with FunLang dialect (1 line!)
        let capturedValues = freeVarList |> List.map (fun v -> env.[v])
        builder.CreateFunLangClosure(lambdaName, capturedValues)

    | App(funcExpr, argExpr) ->
        // Compile function and argument (same)
        let closureVal = compileExpr builder env funcExpr
        let argVal = compileExpr builder env argExpr

        // Apply closure with FunLang dialect (1 line!)
        let resultType = builder.IntType(32)  // Assume i32 for now
        builder.CreateFunLangApply(closureVal, [argVal], resultType)

    // ... other cases ...
```

**코드 비교:**

| Aspect | Phase 4 | Phase 5 | Improvement |
|--------|---------|---------|-------------|
| Lambda body | ~15 lines | ~5 lines | 67% 감소 |
| GC_malloc + GEP | 명시적 | 숨김 | 추상화 |
| App body | ~5 lines | ~3 lines | 40% 감소 |
| 타입 | `!llvm.ptr` | `!funlang.closure` | 타입 안전성 |
| 가독성 | 저수준 | 고수준 | 의도 명확 |

---

## Part 6: Refactoring Chapter 12-13 with Custom Dialect

Phase 4 코드를 Phase 5 코드로 리팩토링하는 **구체적인 예시**를 제공한다.

### Before: Chapter 12 Phase 4 구현

**Compiler.fs (Phase 4):**

```fsharp
module Compiler

open Mlir.Core
open AST

// Counter for fresh lambda names
let mutable lambdaCounter = 0
let freshLambdaName() =
    lambdaCounter <- lambdaCounter + 1
    sprintf "lambda_%d" lambdaCounter

// Free variables analysis
let rec freeVarsExpr (expr: Expr) : Set<string> =
    match expr with
    | Int _ -> Set.empty
    | Var x -> Set.singleton x
    | Add(e1, e2) -> Set.union (freeVarsExpr e1) (freeVarsExpr e2)
    | Lambda(param, body) -> Set.remove param (freeVarsExpr body)
    | App(e1, e2) -> Set.union (freeVarsExpr e1) (freeVarsExpr e2)

// Create lifted function
let createLiftedFunction
    (builder: OpBuilder)
    (name: string)
    (param: string)
    (body: Expr)
    (freeVars: string list)
    (outerEnv: Map<string, MlirValue>) : unit =

    // Function type: (!llvm.ptr, i32) -> i32
    let envType = builder.PtrType()
    let paramType = builder.IntType(32)
    let resultType = builder.IntType(32)
    let funcType = builder.FunctionType([envType; paramType], [resultType])

    // Create function
    let func = builder.CreateFunction(name, funcType)

    // Build function body
    let entryBlock = builder.GetFunctionEntryBlock(func)
    builder.SetInsertionPointToEnd(entryBlock)

    let envParam = mlirBlockGetArgument(entryBlock, 0)
    let xParam = mlirBlockGetArgument(entryBlock, 1)

    // Build environment for body: {param -> xParam, freeVars -> loads}
    let mutable innerEnv = Map.ofList [(param, xParam)]

    freeVars |> List.iteri (fun i varName ->
        // Load from env[i+1]
        let idx = int64 (i + 1)
        let slot = builder.CreateGEP(envParam, idx)
        let value = builder.CreateLoad(slot, paramType)
        innerEnv <- Map.add varName value innerEnv
    )

    // Compile body
    let resultVal = compileExpr builder innerEnv body
    builder.CreateReturn(resultVal)

// Compile expression
and compileExpr (builder: OpBuilder) (env: Map<string, MlirValue>) (expr: Expr) : MlirValue =
    match expr with
    | Int n ->
        builder.CreateI32Const(n)

    | Var x ->
        env.[x]

    | Add(e1, e2) ->
        let v1 = compileExpr builder env e1
        let v2 = compileExpr builder env e2
        builder.CreateArithBinaryOp(ArithOp.Addi, v1, v2)

    | Lambda(param, body) ->
        // Phase 4: 12+ lines of low-level code
        let freeVars = freeVarsExpr body |> Set.toList

        let lambdaName = freshLambdaName()
        createLiftedFunction builder lambdaName param body freeVars env

        // Calculate environment size
        let fnPtrSize = 8L
        let varSize = 4L
        let totalSize = fnPtrSize + (int64 freeVars.Length) * varSize
        let sizeConst = builder.CreateI64Const(totalSize)

        // Allocate environment
        let envPtr = builder.CreateCall("GC_malloc", [sizeConst])

        // Store function pointer at env[0]
        let fnAddr = builder.CreateAddressOf(lambdaName)
        let fnSlot = builder.CreateGEP(envPtr, 0L)
        builder.CreateStore(fnAddr, fnSlot)

        // Store captured variables at env[1..n]
        freeVars |> List.iteri (fun i varName ->
            let value = env.[varName]
            let slot = builder.CreateGEP(envPtr, int64 (i + 1))
            builder.CreateStore(value, slot)
        )

        envPtr

    | App(funcExpr, argExpr) ->
        // Phase 4: 8+ lines of indirect call
        let closureVal = compileExpr builder env funcExpr
        let argVal = compileExpr builder env argExpr

        // Load function pointer from closure[0]
        let c0 = builder.CreateI64Const(0L)
        let fnPtrAddr = builder.CreateGEP(closureVal, 0L)
        let fnPtr = builder.CreateLoad(fnPtrAddr, builder.PtrType())

        // Indirect call: fn_ptr(closure, arg)
        let resultType = builder.IntType(32)
        builder.CreateLLVMCall(fnPtr, [closureVal; argVal], resultType)

// Main compile function
let compile (expr: Expr) : MlirModule =
    use context = new MlirContext()
    context.LoadDialect("builtin")
    context.LoadDialect("func")
    context.LoadDialect("arith")
    context.LoadDialect("llvm")

    use mlirModule = MlirModule.Create(context, "main_module")
    use builder = new OpBuilder(context)
    builder.SetInsertionPointToEnd(mlirModule.Body)

    // Declare GC_malloc
    let i64Type = builder.IntType(64)
    let ptrType = builder.PtrType()
    let gcMallocType = builder.FunctionType([i64Type], [ptrType])
    builder.CreateFunctionDecl("GC_malloc", gcMallocType)

    // Compile main function
    let i32Type = builder.IntType(32)
    let mainType = builder.FunctionType([], [i32Type])
    let mainFunc = builder.CreateFunction("main", mainType)

    let entryBlock = builder.GetFunctionEntryBlock(mainFunc)
    builder.SetInsertionPointToEnd(entryBlock)

    let resultVal = compileExpr builder Map.empty expr
    builder.CreateReturn(resultVal)

    mlirModule
```

### After: Chapter 15 Phase 5 구현

**Compiler.fs (Phase 5):**

```fsharp
module Compiler

open Mlir.Core
open Mlir.FunLang  // Add FunLang dialect
open AST

// (freshLambdaName, freeVarsExpr - same as Phase 4)

// Create lifted function (same as Phase 4)
let createLiftedFunction
    (builder: OpBuilder)
    (name: string)
    (param: string)
    (body: Expr)
    (freeVars: string list)
    (outerEnv: Map<string, MlirValue>) : unit =
    // ... (same implementation) ...

// Compile expression
and compileExpr (builder: OpBuilder) (env: Map<string, MlirValue>) (expr: Expr) : MlirValue =
    match expr with
    | Int n -> builder.CreateI32Const(n)
    | Var x -> env.[x]
    | Add(e1, e2) ->
        let v1 = compileExpr builder env e1
        let v2 = compileExpr builder env e2
        builder.CreateArithBinaryOp(ArithOp.Addi, v1, v2)

    | Lambda(param, body) ->
        // Phase 5: 5 lines with FunLang dialect!
        let freeVars = freeVarsExpr body |> Set.toList

        let lambdaName = freshLambdaName()
        createLiftedFunction builder lambdaName param body freeVars env

        // Create closure (1 line!)
        let capturedValues = freeVars |> List.map (fun v -> env.[v])
        builder.CreateFunLangClosure(lambdaName, capturedValues)

    | App(funcExpr, argExpr) ->
        // Phase 5: 3 lines with FunLang dialect!
        let closureVal = compileExpr builder env funcExpr
        let argVal = compileExpr builder env argExpr

        // Apply closure (1 line!)
        let resultType = builder.IntType(32)
        builder.CreateFunLangApply(closureVal, [argVal], resultType)

// Main compile function
let compile (expr: Expr) : MlirModule =
    use context = new MlirContext()
    context.LoadDialect("builtin")
    context.LoadDialect("func")
    context.LoadDialect("arith")
    context.LoadDialect("llvm")
    context.LoadDialect("funlang")  // Add FunLang dialect!

    use mlirModule = MlirModule.Create(context, "main_module")
    use builder = new OpBuilder(context)
    builder.SetInsertionPointToEnd(mlirModule.Body)

    // Declare GC_malloc (same)
    let i64Type = builder.IntType(64)
    let ptrType = builder.PtrType()
    let gcMallocType = builder.FunctionType([i64Type], [ptrType])
    builder.CreateFunctionDecl("GC_malloc", gcMallocType)

    // Compile main function (same)
    let i32Type = builder.IntType(32)
    let mainType = builder.FunctionType([], [i32Type])
    let mainFunc = builder.CreateFunction("main", mainType)

    let entryBlock = builder.GetFunctionEntryBlock(mainFunc)
    builder.SetInsertionPointToEnd(entryBlock)

    let resultVal = compileExpr builder Map.empty expr
    builder.CreateReturn(resultVal)

    mlirModule
```

### 코드 줄 수 비교

**Lambda case:**

| Version | Lines | Key Operations |
|---------|-------|----------------|
| Phase 4 | ~20 | Size calculation, GC_malloc, GEP loop, stores |
| Phase 5 | ~5 | CreateFunLangClosure |
| **Reduction** | **75%** | **15 lines eliminated** |

**App case:**

| Version | Lines | Key Operations |
|---------|-------|----------------|
| Phase 4 | ~8 | GEP, load, llvm.call |
| Phase 5 | ~3 | CreateFunLangApply |
| **Reduction** | **63%** | **5 lines eliminated** |

**Overall (compileExpr function):**

| Version | Total Lines | Lambda Lines | App Lines |
|---------|-------------|--------------|-----------|
| Phase 4 | ~50 | ~20 | ~8 |
| Phase 5 | ~25 | ~5 | ~3 |
| **Reduction** | **50%** | **75%** | **63%** |

### compileExpr 함수 변경점 요약

**추가된 import:**

```fsharp
open Mlir.FunLang  // FunLang dialect wrapper
```

**변경된 dialect 로딩:**

```fsharp
context.LoadDialect("funlang")  // FunLang dialect 추가
```

**Lambda case 변경:**

```fsharp
// Before: 12+ lines (GC_malloc + GEP loop)
let totalSize = ...
let envPtr = builder.CreateCall("GC_malloc", [sizeConst])
// ... GEP loop ...

// After: 1 line
let capturedValues = freeVars |> List.map (fun v -> env.[v])
builder.CreateFunLangClosure(lambdaName, capturedValues)
```

**App case 변경:**

```fsharp
// Before: 5+ lines (GEP + load + llvm.call)
let fnPtrAddr = builder.CreateGEP(closureVal, 0L)
let fnPtr = builder.CreateLoad(fnPtrAddr, ...)
builder.CreateLLVMCall(fnPtr, [closureVal; argVal], ...)

// After: 1 line
builder.CreateFunLangApply(closureVal, [argVal], resultType)
```

### Generated MLIR 비교

**Test program:**

```fsharp
// FunLang AST
let test =
    Let("make_adder",
        Lambda("n",
            Lambda("x",
                Add(Var "x", Var "n"))),
        App(App(Var "make_adder", Int 5), Int 10))
```

**Phase 4 Generated MLIR:**

```mlir
module {
  llvm.func @GC_malloc(i64) -> !llvm.ptr

  func.func @lambda_1(%env: !llvm.ptr, %x: i32) -> i32 {
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  func.func @lambda_0(%env: !llvm.ptr, %n: i32) -> !llvm.ptr {
    %c12 = arith.constant 12 : i64
    %inner_env = llvm.call @GC_malloc(%c12) : (i64) -> !llvm.ptr
    %fn_addr = llvm.mlir.addressof @lambda_1 : !llvm.ptr
    %c0 = arith.constant 0 : i64
    %fn_slot = llvm.getelementptr %inner_env[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %inner_env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %n, %n_slot : i32, !llvm.ptr
    func.return %inner_env : !llvm.ptr
  }

  func.func @main() -> i32 {
    %c12 = arith.constant 12 : i64
    %outer_env = llvm.call @GC_malloc(%c12) : (i64) -> !llvm.ptr
    %fn_addr = llvm.mlir.addressof @lambda_0 : !llvm.ptr
    %c0 = arith.constant 0 : i64
    %fn_slot = llvm.getelementptr %outer_env[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.store %fn_addr, %fn_slot : !llvm.ptr, !llvm.ptr

    %c5 = arith.constant 5 : i32
    %fn_ptr_addr = llvm.getelementptr %outer_env[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %add5 = llvm.call %fn_ptr(%outer_env, %c5) : (!llvm.ptr, i32) -> !llvm.ptr

    %c10 = arith.constant 10 : i32
    %fn_ptr_addr2 = llvm.getelementptr %add5[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %fn_ptr2 = llvm.load %fn_ptr_addr2 : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr2(%add5, %c10) : (!llvm.ptr, i32) -> i32

    func.return %result : i32
  }
}
```

**Phase 5 Generated MLIR:**

```mlir
module {
  llvm.func @GC_malloc(i64) -> !llvm.ptr

  func.func @lambda_1(%env: !llvm.ptr, %x: i32) -> i32 {
    %c1 = arith.constant 1 : i64
    %n_slot = llvm.getelementptr %env[%c1] : (!llvm.ptr, i64) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  func.func @lambda_0(%env: !llvm.ptr, %n: i32) -> !funlang.closure {
    // Closure creation: 1 line!
    %inner_closure = funlang.closure @lambda_1, %n : !funlang.closure
    func.return %inner_closure : !funlang.closure
  }

  func.func @main() -> i32 {
    // Outer closure
    %make_adder = funlang.closure @lambda_0 : !funlang.closure

    // Apply make_adder 5
    %c5 = arith.constant 5 : i32
    %add5 = funlang.apply %make_adder(%c5) : (i32) -> !funlang.closure

    // Apply add5 10
    %c10 = arith.constant 10 : i32
    %result = funlang.apply %add5(%c10) : (i32) -> i32

    func.return %result : i32
  }
}
```

**MLIR Line Count:**

| Function | Phase 4 | Phase 5 | Reduction |
|----------|---------|---------|-----------|
| lambda_0 | 11 lines | 3 lines | 73% |
| main | 14 lines | 8 lines | 43% |
| **Total** | **~35 lines** | **~18 lines** | **49%** |

---

## Part 7: Common Errors

FunLang dialect 사용 시 흔히 발생하는 오류들과 해결 방법을 다룬다.

### Error 1: Missing Dialect Registration

**증상:**

```
ERROR: Dialect 'funlang' not found in context
```

**원인:**

FunLang dialect을 context에 로드하지 않았다.

**잘못된 코드:**

```fsharp
use context = new MlirContext()
context.LoadDialect("builtin")
context.LoadDialect("func")
// funlang dialect 누락!

let builder = new OpBuilder(context)
let closure = builder.CreateFunLangClosure("lambda", [])
// ERROR: funlang dialect not registered
```

**올바른 코드:**

```fsharp
use context = new MlirContext()
context.LoadDialect("builtin")
context.LoadDialect("func")
context.LoadDialect("funlang")  // FunLang dialect 로드!

let builder = new OpBuilder(context)
let closure = builder.CreateFunLangClosure("lambda", [])
// OK
```

**체크리스트:**

- [ ] `context.LoadDialect("funlang")` 호출했는가?
- [ ] FunLang dialect 라이브러리를 링크했는가? (`-lMLIR-FunLang-CAPI`)
- [ ] Dialect 초기화 함수를 호출했는가? (C++ 프로젝트에서만 필요)

### Error 2: Wrong Attribute Type for Callee

**증상:**

```
ERROR: Expected FlatSymbolRefAttr, got StringAttr
```

**원인:**

함수 이름을 일반 문자열 대신 SymbolRefAttr로 전달하지 않았다.

**잘못된 코드:**

```fsharp
// F# string을 직접 전달 (wrong!)
let nameAttr = mlirStringAttrGet(context, MlirStringRef.FromString("lambda"))
let op = FunLangBindings.mlirFunLangClosureOpCreate(
    context, loc, nameAttr, 0n, [||])
// ERROR: StringAttr is not FlatSymbolRefAttr
```

**올바른 코드:**

```fsharp
// FlatSymbolRefAttr로 변환
use nameStrRef = MlirStringRef.FromString("lambda")
let calleeAttr = mlirFlatSymbolRefAttrGet(context, nameStrRef)
let op = FunLangBindings.mlirFunLangClosureOpCreate(
    context, loc, calleeAttr, 0n, [||])
// OK
```

**또는 High-level wrapper 사용:**

```fsharp
// FunLangDialect wrapper가 변환 처리
let funlang = FunLangDialect(context)
let closure = funlang.CreateClosure(loc, "lambda", [])
// OK: "lambda" string is converted to FlatSymbolRefAttr internally
```

**Why FlatSymbolRefAttr?**

- **Symbol table 검증**: MLIR이 `@lambda` 함수 존재 여부 확인
- **최적화 지원**: Inlining, DCE 등에서 심볼 참조 추적
- **타입 정보**: 함수 시그니처 접근 가능

### Error 3: Type Mismatch in Variadic Arguments

**증상:**

```
ERROR: funlang.closure expects all captured values to be SSA values
```

**원인:**

캡처된 변수 배열에 잘못된 값을 전달했다 (예: null, 초기화되지 않은 값).

**잘못된 코드:**

```fsharp
// 빈 MlirValue 배열 생성 (uninitialized)
let capturedArray : MlirValue[] = Array.zeroCreate 3
// capturedArray[0..2] are default (uninitialized)

let op = FunLangBindings.mlirFunLangClosureOpCreate(
    context, loc, calleeAttr, 3n, capturedArray)
// ERROR: Invalid MlirValue
```

**올바른 코드:**

```fsharp
// F# list에서 변환
let capturedList = [v1; v2; v3]
let capturedArray = List.toArray capturedList

let op = FunLangBindings.mlirFunLangClosureOpCreate(
    context, loc, calleeAttr, nativeint capturedArray.Length, capturedArray)
// OK: All values are valid SSA values
```

**또는 High-level wrapper 사용:**

```fsharp
// FunLangDialect wrapper가 변환 처리
let funlang = FunLangDialect(context)
let closure = funlang.CreateClosure(loc, "lambda", [v1; v2; v3])
// OK: F# list is safely converted to array
```

**디버깅 팁:**

MlirValue의 유효성을 검증:

```fsharp
// MlirValue가 유효한지 확인
let isValidValue (v: MlirValue) : bool =
    v.ptr <> 0n  // nativeint 0은 null pointer

// 사용 전 검증
if not (isValidValue v1) then
    failwith "v1 is invalid MlirValue"
```

### Error 4: Forgetting to Declare Dependent Dialects

**증상:**

```
ERROR: Operation 'func.call' not found
ERROR: Operation 'arith.addi' not found
```

**원인:**

FunLang dialect은 다른 dialect (func, arith, llvm)에 의존한다. 이들을 로드하지 않으면 lifted function 내부에서 오류 발생.

**잘못된 코드:**

```fsharp
use context = new MlirContext()
context.LoadDialect("funlang")  // FunLang만 로드

let builder = new OpBuilder(context)
let closure = builder.CreateFunLangClosure("lambda", [])
// ERROR: lifted function uses arith.addi, but arith dialect not loaded
```

**올바른 코드:**

```fsharp
use context = new MlirContext()
context.LoadDialect("builtin")   // Module, FuncOp
context.LoadDialect("func")      // func.func, func.call, func.return
context.LoadDialect("arith")     // arith.constant, arith.addi
context.LoadDialect("llvm")      // llvm.ptr, llvm.getelementptr
context.LoadDialect("funlang")   // funlang.closure, funlang.apply

// 이제 모든 operations 사용 가능
```

**Dialect 의존성 체인:**

```
FunLang dialect
  ├── depends on Func dialect (func.func, func.return)
  ├── depends on Arith dialect (arith.constant, arith.addi)
  └── depends on LLVM dialect (!llvm.ptr, llvm.getelementptr)
```

**TableGen 선언 (FunLangDialect.td):**

```tablegen
def FunLang_Dialect : Dialect {
  let name = "funlang";
  let summary = "FunLang functional language dialect";
  let description = [{...}];
  let cppNamespace = "::mlir::funlang";

  // Dependent dialects
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}
```

### Error 5: Incorrect Result Type in funlang.apply

**증상:**

```
ERROR: funlang.apply result type does not match function signature
```

**원인:**

`funlang.apply`에 지정한 결과 타입이 실제 클로저 함수의 반환 타입과 다르다.

**잘못된 코드:**

```fsharp
// lambda_add 함수: (i32) -> i32
%closure = funlang.closure @lambda_add, %n : !funlang.closure

// 잘못된 결과 타입 (f64)
%result = funlang.apply %closure(%x) : (i32) -> f64
// ERROR: lambda_add returns i32, not f64
```

**올바른 코드:**

```fsharp
// lambda_add 함수: (i32) -> i32
%closure = funlang.closure @lambda_add, %n : !funlang.closure

// 올바른 결과 타입 (i32)
%result = funlang.apply %closure(%x) : (i32) -> i32
// OK
```

**F# 컴파일러에서의 해결:**

타입 추론을 통해 자동으로 올바른 타입 지정:

```fsharp
// 컴파일러가 resultType를 추론
let resultType =
    match exprType funcExpr with
    | FunctionType(argTypes, retType) -> retType
    | _ -> failwith "Expected function type"

builder.CreateFunLangApply(closureVal, [argVal], resultType)
```

### Error 6: Using funlang.closure with Non-Existent Function

**증상:**

```
ERROR: Symbol '@lambda_99' not found in module
```

**원인:**

`funlang.closure @lambda_99`를 생성했지만, `@lambda_99` 함수를 정의하지 않았다.

**잘못된 코드:**

```fsharp
// 클로저 생성
let closure = builder.CreateFunLangClosure("lambda_99", [])

// 하지만 lambda_99 함수는 정의되지 않음!
// ERROR: Symbol not found
```

**올바른 코드:**

```fsharp
// 1. 먼저 lifted function 생성
createLiftedFunction builder "lambda_99" "x" bodyExpr [] env

// 2. 그 다음 클로저 생성
let closure = builder.CreateFunLangClosure("lambda_99", [])
// OK: lambda_99 exists
```

**순서 보장:**

```fsharp
// Lambda case in compileExpr
| Lambda(param, body) ->
    let lambdaName = freshLambdaName()

    // Step 1: Create lifted function FIRST
    createLiftedFunction builder lambdaName param body freeVars env

    // Step 2: Create closure AFTER function exists
    let capturedValues = freeVars |> List.map (fun v -> env.[v])
    builder.CreateFunLangClosure(lambdaName, capturedValues)
```

---

## Summary

### Chapter 15에서 배운 것

**1. funlang.closure Operation**
- Phase 4의 12줄 클로저 생성 코드를 1줄로 압축
- TableGen ODS로 선언적 정의
- Pure trait로 최적화 가능
- FlatSymbolRefAttr로 타입 안전 함수 참조
- C API shim으로 F# 통합

**2. funlang.apply Operation**
- Phase 4의 8줄 간접 호출 코드를 1줄로 압축
- 클로저 타입을 인자로 받음 (!funlang.closure)
- Side effect 고려 (trait 없음)
- Functional-type syntax로 명확한 시그니처

**3. funlang.match Operation (Phase 6 Preview)**
- Region-based operation 구조
- VariadicRegion<SizedRegion<1>>로 각 case 독립
- SingleBlockImplicitTerminator<"YieldOp">로 통일된 종료
- Verifier로 타입 안전성 보장
- Block arguments로 패턴 변수 표현

**4. FunLang Custom Types**
- !funlang.closure: Opaque type (단순성 우선)
- !funlang.list<T>: Parameterized type (타입 안전성 필수)
- Lowering: FunLang types → !llvm.ptr

**5. Complete F# Integration**
- Low-level bindings (FunLangBindings 모듈)
- High-level wrappers (FunLangDialect 클래스)
- OpBuilder extensions (CreateFunLangClosure/Apply)
- Type-safe API (F# list, string 자동 변환)

**6. Code Reduction**
- Lambda: 20 lines → 5 lines (75% 감소)
- App: 8 lines → 3 lines (63% 감소)
- Overall: 50% 코드 감소
- 타입 안전성 향상 (!llvm.ptr → !funlang.closure)

### 핵심 패턴

**TableGen ODS:**
```tablegen
def FunLang_ClosureOp : FunLang_Op<"closure", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$capturedValues);
  let results = (outs FunLang_ClosureType:$result);
  let assemblyFormat = [...];
}
```

**C API Shim:**
```cpp
MlirOperation mlirFunLangClosureOpCreate(...) {
  MLIRContext *ctx = unwrap(mlirCtx);
  OpBuilder builder(ctx);
  auto op = builder.create<ClosureOp>(...);
  return wrap(op.getOperation());
}
```

**F# High-level Wrapper:**
```fsharp
type FunLangDialect(context: MlirContext) =
    member this.CreateClosure(loc, callee, captured) =
        // Handle string → FlatSymbolRefAttr conversion
        // Handle F# list → C array conversion
        // Call C API
        // Return MlirValue
```

### Chapter 16 Preview

**Chapter 16: Lowering Passes**

다음 장에서는 FunLang dialect을 LLVM dialect으로 lowering하는 pass를 구현한다:

1. **FunLangToLLVM Lowering Pass**
   - funlang.closure → GC_malloc + store 패턴
   - funlang.apply → GEP + load + llvm.call 패턴
   - !funlang.closure → !llvm.ptr 타입 변환

2. **Pass Infrastructure**
   - Pass registration (PassManager)
   - ConversionTarget 설정
   - TypeConverter 구현
   - RewritePattern 작성

3. **Testing**
   - FileCheck 테스트 작성
   - Before/After IR 비교
   - 실행 테스트 (JIT)

4. **Optimization Opportunities**
   - Closure inlining
   - Escape analysis
   - Dead closure elimination

**Progressive Lowering 완성:**

```
FunLang AST
  ↓ (Compiler.fs)
FunLang Dialect (funlang.closure, funlang.apply)
  ↓ (Chapter 16: FunLangToLLVM pass)
LLVM Dialect (llvm.call @GC_malloc, llvm.getelementptr)
  ↓ (MLIR built-in passes)
LLVM IR
  ↓ (LLVM backend)
Native Code
```

**Phase 5의 목표 달성:**

- [x] Custom dialect 정의 (Chapter 14 theory, Chapter 15 implementation)
- [x] Operations 구현 (closure, apply, match preview)
- [x] Types 구현 (closure, list preview)
- [x] F# 통합 (C API shim + bindings)
- [x] Compiler 리팩토링 (Phase 4 코드 50% 감소)
- [ ] Lowering pass 구현 (Chapter 16)
- [ ] 테스트와 검증 (Chapter 16)

**다음: Chapter 16 - Lowering Passes로 Phase 5를 완성한다!**
