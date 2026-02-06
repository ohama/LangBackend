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

(Chapter 계속... Part 4: FunLang Custom Types로 이어집니다)
