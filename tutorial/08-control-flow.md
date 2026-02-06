# Chapter 08: 제어 흐름과 Block Arguments

## 소개

프로그래밍에서 조건부 실행은 필수다. 조건에 따라 다른 코드 경로를 실행하는 능력은 모든 실용적인 프로그램의 핵심이다.

함수형 언어에서 **if/then/else는 표현식(expression)**이다. 명령형 언어의 문(statement)이 아니라, 값을 생성하는 표현식이다:

```fsharp
// 함수형 스타일 - if는 값을 반환한다
let result = if condition then 42 else 0

// 명령형 스타일과 대비
int result;
if (condition) {
    result = 42;
} else {
    result = 0;
}
```

함수형 스타일에서 if 표현식은 **값을 생성**한다. 두 분기(then/else) 중 하나가 실행되고, 그 결과가 if 표현식의 값이 된다.

**컴파일 도전과제:** SSA 형태에서 두 분기가 어떻게 하나의 값으로 합쳐지는가?

```fsharp
let x = if condition then 10 else 20 in
x + x
```

조건이 true면 `x = 10`, false면 `x = 20`이다. 하지만 SSA 형태에서 `x`는 단일 SSA value여야 한다. 두 분기의 값을 어떻게 합칠까?

**MLIR의 우아한 해답:** **Block Arguments**

전통적인 SSA는 PHI 노드를 사용하지만, MLIR은 더 깔끔한 방식을 제공한다. 이 장에서 MLIR의 block arguments와 `scf.if` 연산을 배운다.

이 장을 마치면:
- if/then/else 표현식을 네이티브 바이너리로 컴파일할 수 있다
- Block arguments와 PHI 노드의 차이를 이해한다
- MLIR의 `scf.if` 연산과 `scf.yield` 종결자를 사용할 수 있다
- 제어 흐름 합류 지점에서 SSA 값이 어떻게 병합되는지 안다

> **중요:** Block arguments는 MLIR의 핵심 혁신이다. PHI 노드의 복잡성을 제거하고 SSA 형태를 더 명확하게 만든다.

## PHI 노드 문제

### 전통적인 SSA: PHI 노드

LLVM IR과 전통적인 SSA 형태는 **PHI 노드**를 사용하여 제어 흐름 합류 지점에서 값을 병합한다.

**LLVM IR 예시:**

```llvm
define i32 @example(i1 %cond) {
entry:
  br i1 %cond, label %then, label %else

then:
  %a = add i32 10, 1
  br label %merge

else:
  %b = add i32 20, 1
  br label %merge

merge:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}
```

**동작 설명:**

1. `entry` 블록에서 조건 분기 (`br i1 %cond`)
2. `then` 블록: `%a = 11` 계산 후 `merge`로 이동
3. `else` 블록: `%b = 21` 계산 후 `merge`로 이동
4. `merge` 블록: **PHI 노드**가 선택
   - `%then` 블록에서 왔으면 `%a` 사용
   - `%else` 블록에서 왔으면 `%b` 사용

PHI 노드는 "어느 블록에서 왔는가"에 따라 값을 선택한다. 표기법: `phi type [ value1, pred1 ], [ value2, pred2 ]`

### PHI 노드의 문제점

#### 1. 블록 시작 위치 제약

PHI 노드는 **반드시 블록의 시작**에 있어야 한다:

```llvm
merge:
  %result = phi i32 [ %a, %then ], [ %b, %else ]  ; PHI는 여기!
  %x = add i32 %result, 1                          ; 일반 연산은 PHI 뒤
  ; PHI를 여기에 추가할 수 없다 - 순서 규칙 위반
```

이 제약은 코드 생성을 복잡하게 만든다. PHI 노드를 먼저 모으고, 일반 연산을 뒤에 배치해야 한다.

#### 2. Lost Copy Problem

PHI 노드의 의미는 "블록 진입 시" 값을 선택하는 것이다. 하지만 실제 구현에서는 **선행 블록의 끝**에서 값을 복사한다:

```llvm
then:
  %a = add i32 10, 1
  ; 실제로는 여기서 %a를 %result로 복사
  br label %merge

merge:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ; %result는 이미 복사된 값을 가진다
```

이것이 **lost copy problem**이다:
- PHI 노드는 "merge 블록 진입 시" 선택하는 것처럼 보인다
- 실제 구현은 "선행 블록 종료 시" 복사한다
- 의미론과 구현의 불일치

#### 3. Dominance Frontier 계산

PHI 노드를 올바르게 배치하려면 **dominance frontier** 알고리즘이 필요하다:

```
// 어디에 PHI 노드를 삽입해야 할까?
// 복잡한 제어 흐름에서는 자명하지 않다
if (cond1) {
  x = 10;
} else if (cond2) {
  x = 20;
} else {
  x = 30;
}
// 여기서 x에 PHI 노드가 필요하다
// 하지만 몇 개의 선행 블록이 있는가?
```

Dominance frontier는 "변수가 재정의되는 모든 블록의 지배 경계"를 계산한다. 알고리즘이 복잡하고 구현이 어렵다.

#### 4. 가독성 문제

PHI 노드는 직관적이지 않다:

```llvm
%result = phi i32 [ %a, %then ], [ %b, %else ]
; 이것이 무엇을 의미하는가?
; "then에서 왔으면 %a, else에서 왔으면 %b"
; 함수 호출처럼 보이지만 실제로는 특별한 의미를 가진다
```

초보자가 PHI 노드를 이해하기 어렵다. 특별한 규칙(블록 시작, 순서 지정, edge 의미론)을 배워야 한다.

### PHI 노드 요약

**PHI 노드의 특징:**
- 제어 흐름 합류 지점에서 값을 병합한다
- 블록 시작에 위치해야 한다 (특별한 위치 규칙)
- Lost copy problem - 의미론과 구현의 불일치
- Dominance frontier 계산 필요
- 가독성이 낮다

**MLIR의 해답:** Block Arguments - PHI 노드를 대체하는 더 깔끔한 방식

## Block Arguments in MLIR

MLIR은 PHI 노드 대신 **block arguments**를 사용한다.

### Block Arguments 개념

**핵심 아이디어:** 기본 블록(basic block)도 함수처럼 파라미터를 받을 수 있다.

함수는 인자를 받는다:

```fsharp
let add(x: int, y: int) = x + y
```

**MLIR에서는 블록도 인자를 받는다:**

```mlir
^myblock(%arg0: i32, %arg1: i32):
  %sum = arith.addi %arg0, %arg1 : i32
  ...
```

`^myblock`은 두 개의 i32 인자를 받는다. 블록으로 분기할 때 값을 전달한다:

```mlir
cf.br ^myblock(%value1, %value2 : i32, i32)
```

이것은 함수 호출과 유사하다: `myblock(value1, value2)`

### Block Arguments vs PHI Nodes

같은 예시를 block arguments로 작성하면:

**MLIR with Block Arguments:**

```mlir
func.func @example(%cond: i1) -> i32 {
  cf.cond_br %cond, ^then, ^else

^then:
  %a = arith.constant 11 : i32
  cf.br ^merge(%a : i32)

^else:
  %b = arith.constant 21 : i32
  cf.br ^merge(%b : i32)

^merge(%result: i32):
  func.return %result : i32
}
```

**차이점 분석:**

| 측면 | PHI 노드 (LLVM) | Block Arguments (MLIR) |
|------|-----------------|------------------------|
| **값 전달** | `phi i32 [ %a, %then ], [ %b, %else ]` | `cf.br ^merge(%a : i32)` |
| **의미론** | "어느 블록에서 왔는가" | "블록 호출 시 인자 전달" |
| **위치 제약** | 블록 시작에만 가능 | 블록 인자로 선언 (일반 파라미터) |
| **가독성** | 특별한 문법, edge 리스트 | 함수 호출과 유사 |

**핵심 통찰력:**

- **PHI 노드:** "merge 블록이 선행 블록을 검사하여 값을 선택"
- **Block Arguments:** "선행 블록이 merge 블록에 값을 전달" (함수 호출처럼)

Block arguments는 제어의 역전(inversion of control)이다:
- PHI: pull 방식 (merge 블록이 값을 가져온다)
- Block Arguments: push 방식 (선행 블록이 값을 전달한다)

### Block Arguments의 장점

#### 1. 통일된 의미론

함수 인자와 블록 인자가 같은 개념이다:

```mlir
// 함수 인자
func.func @foo(%arg: i32) -> i32 {
  ...
}

// 블록 인자 (동일한 문법!)
^myblock(%arg: i32):
  ...
```

배울 것이 하나다. 함수를 이해하면 블록도 이해한다.

#### 2. Lost Copy Problem 해결

Block arguments는 의미론과 구현이 일치한다:

```mlir
^then:
  %a = arith.constant 11 : i32
  cf.br ^merge(%a : i32)  ; 명시적으로 %a 전달
```

"분기할 때 값을 전달한다"는 의미가 명확하다. Lost copy problem이 없다.

#### 3. 위치 제약 없음

Block arguments는 블록 파라미터다. 블록 내 어디서든 일반 value처럼 사용할 수 있다:

```mlir
^merge(%result: i32):
  %x = arith.constant 1 : i32
  %y = arith.addi %result, %x : i32  ; %result 사용
  func.return %y : i32
```

특별한 위치 규칙이 없다. 블록 파라미터는 블록 내 모든 곳에서 유효하다.

#### 4. 가독성

코드가 더 명확하다:

```mlir
cf.br ^merge(%a : i32)  ; "merge 블록을 %a와 함께 호출"
^merge(%result: i32):   ; "merge 블록은 %result 파라미터를 받는다"
```

함수 호출 비유가 자연스럽다. 초보자가 쉽게 이해한다.

### Block Arguments 예시

**복잡한 제어 흐름:**

```mlir
func.func @complex(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32

  %cond1 = arith.cmpi slt, %x, %c0 : i32
  cf.cond_br %cond1, ^negative, ^nonnegative

^negative:
  %neg = arith.constant -1 : i32
  cf.br ^merge(%neg : i32)

^nonnegative:
  %cond2 = arith.cmpi sgt, %x, %c10 : i32
  cf.cond_br %cond2, ^large, ^small

^large:
  %l = arith.constant 1 : i32
  cf.br ^merge(%l : i32)

^small:
  cf.br ^merge(%c0 : i32)

^merge(%result: i32):
  func.return %result : i32
}
```

**동작:**
- `x < 0`: ^negative → ^merge(-1)
- `x > 10`: ^nonnegative → ^large → ^merge(1)
- `0 ≤ x ≤ 10`: ^nonnegative → ^small → ^merge(0)

`^merge` 블록은 세 곳에서 호출된다. 각 선행 블록이 값을 전달한다. Block argument `%result`가 전달된 값을 받는다.

**PHI 노드로 작성했다면:**

```llvm
merge:
  %result = phi i32 [ %neg, %negative ], [ %l, %large ], [ %c0, %small ]
```

어느 쪽이 더 명확한가? Block arguments가 push 방식으로 값을 전달하므로 추적하기 쉽다.

### Block Arguments 요약

**Block Arguments:**
- 기본 블록이 함수처럼 파라미터를 받는다
- 분기 시 값을 전달: `cf.br ^block(%value : type)`
- 블록 선언에서 파라미터 정의: `^block(%arg: type):`

**장점:**
- 함수 인자와 통일된 의미론
- Lost copy problem 해결
- 위치 제약 없음
- 가독성 향상

**PHI 노드 대비:**
- PHI는 pull (merge가 선택), Block Arguments는 push (선행이 전달)
- PHI는 특별한 규칙, Block Arguments는 일반 파라미터

> **다음 섹션:** MLIR의 고수준 제어 흐름인 `scf.if` 연산을 배운다!

## scf.if: 고수준 제어 흐름

Block arguments를 직접 사용하는 것은 저수준(low-level) 방식이다. MLIR은 **구조화된 제어 흐름(Structured Control Flow)**을 위한 `scf` dialect를 제공한다.

### scf Dialect 소개

**scf (Structured Control Flow) dialect:**
- 고수준 제어 흐름 연산 제공
- `scf.if`, `scf.for`, `scf.while` 등
- 구조화된 방식으로 제어 흐름 표현
- 나중에 저수준 `cf` dialect로 lowering된다

**Progressive Lowering 철학:**

```
scf.if (high-level)
  ↓ lowering pass
cf.cond_br (low-level branches)
  ↓ lowering pass
llvm.cond_br (LLVM IR)
```

사용자는 고수준 `scf.if`를 사용한다. 컴파일러가 자동으로 저수준 분기로 변환한다.

### scf.if 문법

**기본 형태:**

```mlir
%result = scf.if %condition -> (result_type) {
  // then region
  scf.yield %then_value : result_type
} else {
  // else region
  scf.yield %else_value : result_type
}
```

**구성 요소:**

1. **%condition**: i1 타입의 boolean 값
2. **-> (result_type)**: 반환할 타입 선언
3. **then region**: 조건이 true일 때 실행
4. **else region**: 조건이 false일 때 실행
5. **scf.yield**: 각 region의 종결자, 값을 반환

**중요:** 양쪽 region이 **같은 타입**을 yield해야 한다!

### scf.if 예시

**간단한 예시:**

```mlir
func.func @example(%cond: i1) -> i32 {
  %result = scf.if %cond -> (i32) {
    %c42 = arith.constant 42 : i32
    scf.yield %c42 : i32
  } else {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  }
  func.return %result : i32
}
```

**동작:**
- `%cond`가 true: then region 실행 → `%c42` yield → `%result = 42`
- `%cond`가 false: else region 실행 → `%c0` yield → `%result = 0`

**핵심:** `scf.if`는 **표현식**이다. 값을 반환한다 (`%result`). if/then/else의 함수형 의미론!

### scf.yield 종결자

**scf.yield의 역할:**

```mlir
scf.yield %value : type
```

- Region의 **종결자(terminator)**다
- Region을 종료하고 값을 반환한다
- 함수의 `return`과 유사하지만, region에서 사용한다

**중요 규칙:**

1. **모든 region은 종결자가 필요하다**
   ```mlir
   scf.if %cond -> (i32) {
     %c42 = arith.constant 42 : i32
     // 에러! scf.yield 누락
   }
   ```

2. **yield 타입이 일치해야 한다**
   ```mlir
   // 에러! then은 i32, else는 i1
   scf.if %cond -> (i32) {
     %c42 = arith.constant 42 : i32
     scf.yield %c42 : i32
   } else {
     %true = arith.constant 1 : i1
     scf.yield %true : i1  // 타입 불일치!
   }
   ```

3. **선언된 결과 타입과 일치해야 한다**
   ```mlir
   // 에러! -> (i32) 선언했지만 i64 yield
   %result = scf.if %cond -> (i32) {
     %c42 = arith.constant 42 : i64
     scf.yield %c42 : i64  // 타입 불일치!
   }
   ```

### scf.if의 장점

#### 1. 타입 안전성

결과 타입을 미리 선언한다 (`-> (i32)`). 컴파일러가 양쪽 region을 검증한다.

```mlir
%result = scf.if %cond -> (i32) {
  scf.yield %then_val : i32
} else {
  scf.yield %else_val : i32
}
// 컴파일러: "양쪽 모두 i32를 yield하는가?" ✓
```

#### 2. 구조화된 형태

`scf.if`는 블록 구조가 명확하다:
- then region
- else region
- 둘 다 명확한 시작과 끝

저수준 분기(`cf.cond_br`)는 임의의 블록으로 점프할 수 있다 (덜 구조화됨).

#### 3. 변환 용이성

고수준 구조는 최적화와 분석이 쉽다:
- Dead branch elimination
- Condition hoisting
- Pattern matching

저수준 분기는 제어 흐름 그래프(CFG) 분석이 필요하다.

### scf.if에서 cf.cond_br로 Lowering

`scf.if`는 나중에 `cf.cond_br`와 block arguments로 변환된다.

**High-level (scf.if):**

```mlir
%result = scf.if %cond -> (i32) {
  %c42 = arith.constant 42 : i32
  scf.yield %c42 : i32
} else {
  %c0 = arith.constant 0 : i32
  scf.yield %c0 : i32
}
func.return %result : i32
```

**Lowering 후 (cf.cond_br + block arguments):**

```mlir
cf.cond_br %cond, ^then, ^else

^then:
  %c42 = arith.constant 42 : i32
  cf.br ^merge(%c42 : i32)

^else:
  %c0 = arith.constant 0 : i32
  cf.br ^merge(%c0 : i32)

^merge(%result: i32):
  func.return %result : i32
```

**변환 과정:**

1. `scf.if`의 then region → `^then` 블록
2. `scf.if`의 else region → `^else` 블록
3. `scf.yield` → `cf.br ^merge(value)`
4. `scf.if`의 결과 → `^merge` 블록의 block argument

**자동 변환:** `--convert-scf-to-cf` pass가 이 변환을 수행한다. 사용자는 신경 쓰지 않아도 된다!

### Multiple Results

`scf.if`는 여러 값을 반환할 수 있다:

```mlir
%x, %y = scf.if %cond -> (i32, i32) {
  %a = arith.constant 10 : i32
  %b = arith.constant 20 : i32
  scf.yield %a, %b : i32, i32
} else {
  %c = arith.constant 30 : i32
  %d = arith.constant 40 : i32
  scf.yield %c, %d : i32, i32
}
// %x, %y는 (10, 20) 또는 (30, 40)
```

**Lowering 후:**

```mlir
^merge(%x: i32, %y: i32):
  // %x, %y는 block arguments
```

Block arguments도 여러 개 가질 수 있다. `scf.if`의 유연성이 그대로 lowering된다.

### scf.if 요약

**scf.if 연산:**
- 고수준 구조화된 제어 흐름
- 결과 타입 선언: `-> (type)`
- 양쪽 region이 같은 타입 yield
- `scf.yield` 종결자로 값 반환

**장점:**
- 타입 안전성
- 구조화된 형태
- 최적화 용이성
- Progressive lowering: scf → cf → llvm

**다음:** F# P/Invoke 바인딩을 추가하여 `scf.if`와 `scf.yield`를 생성한다!

## P/Invoke 바인딩: SCF Dialect

이제 F#에서 SCF dialect 연산을 사용할 수 있도록 P/Invoke 바인딩을 추가한다.

### MLIR C API for SCF

MLIR C API는 `mlir-c/Dialect/SCF.h` 헤더에서 SCF dialect 지원을 제공한다.

**주요 함수:**

```c
// mlir-c/Dialect/SCF.h

// scf.if operation 생성
MlirOperation mlirSCFIfCreate(
    MlirLocation location,
    MlirValue condition,
    bool hasElse
);

// scf.yield operation 생성
MlirOperation mlirSCFYieldCreate(
    MlirLocation location,
    intptr_t nResults,
    MlirValue const *results
);

// scf.if의 then/else region 접근
MlirRegion mlirSCFIfGetThenRegion(MlirOperation ifOp);
MlirRegion mlirSCFIfGetElseRegion(MlirOperation ifOp);
```

**Note:** 실제 MLIR C API에서 SCF dialect 지원은 제한적일 수 있다. 필요한 함수가 없으면 C++ shim을 작성한다 (Appendix 참조).

### F# P/Invoke 바인딩

**MlirBindings.fs에 추가:**

```fsharp
namespace MlirBindings

open System
open System.Runtime.InteropServices

module MlirNative =
    // ... 기존 바인딩 ...

    // ===== SCF Dialect Operations =====

    /// scf.if operation 생성
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirSCFIfCreate(
        MlirLocation location,
        MlirValue condition,
        bool hasElse
    )

    /// scf.yield operation 생성
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirSCFYieldCreate(
        MlirLocation location,
        nativeint nResults,
        MlirValue[] results
    )

    /// scf.if의 then region 가져오기
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirSCFIfGetThenRegion(MlirOperation ifOp)

    /// scf.if의 else region 가져오기
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirRegion mlirSCFIfGetElseRegion(MlirOperation ifOp)

    /// operation의 결과 개수 설정 (scf.if 결과 타입용)
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirOperationSetResultTypes(
        MlirOperation operation,
        nativeint nTypes,
        MlirType[] types
    )
```

**바인딩 설명:**

1. **mlirSCFIfCreate**: `scf.if` operation 생성
   - `location`: operation 위치
   - `condition`: i1 타입 boolean 값
   - `hasElse`: else region 포함 여부 (true면 then/else, false면 then만)

2. **mlirSCFYieldCreate**: `scf.yield` operation 생성
   - `nResults`: yield할 값 개수
   - `results`: yield할 값 배열

3. **mlirSCFIfGetThenRegion/ElseRegion**: region 접근
   - `scf.if`는 내부에 then/else region을 가진다
   - Region에 블록을 추가하고 연산을 작성한다

### C API 제약과 대안

MLIR C API의 SCF dialect 지원은 완전하지 않을 수 있다. 특히:

- `scf.if` 결과 타입 설정 API가 명확하지 않을 수 있다
- Region builder API가 제한적일 수 있다

**대안 1: Operation State Builder 사용**

MLIR C API의 일반 operation builder를 사용:

```fsharp
let createScfIf (builder: OpBuilder) (condition: MlirValue) (resultTypes: MlirType[]) (location: MlirLocation) =
    let opName = MlirHelpers.fromString("scf.if")
    let state = MlirNative.mlirOperationStateGet(opName, location)

    // operand 추가 (condition)
    MlirNative.mlirOperationStateAddOperands(state, 1n, [| condition |])

    // 결과 타입 추가
    MlirNative.mlirOperationStateAddResults(state, nativeint resultTypes.Length, resultTypes)

    // region 추가 (then, else)
    MlirNative.mlirOperationStateAddOwnedRegions(state, 2n, [| thenRegion; elseRegion |])

    // operation 생성
    MlirNative.mlirOperationCreate(state)
```

**대안 2: C++ Shim 작성**

Appendix (Chapter 01-03에서 다룬 C++ dialect wrapper 패턴)에 따라 C++ shim을 작성:

```cpp
// mlir_scf_wrapper.cpp
extern "C" {

MlirOperation mlirCreateSCFIf(
    MlirLocation location,
    MlirValue condition,
    MlirType* resultTypes,
    intptr_t numResults,
    bool hasElse
) {
    // C++ MLIR API 사용
    mlir::OpBuilder builder(...);
    auto ifOp = builder.create<mlir::scf::IfOp>(
        unwrap(location),
        llvm::ArrayRef<mlir::Type>(...),
        unwrap(condition),
        hasElse
    );
    return wrap(ifOp.getOperation());
}

} // extern "C"
```

이 shim을 컴파일하여 F#에서 호출한다.

**권장 방안:** 먼저 C API를 시도하고, 부족하면 C++ shim을 작성한다. Chapter 01 Appendix가 이미 패턴을 확립했다.

### OpBuilder 헬퍼 메서드

고수준 래퍼를 `OpBuilder` 클래스에 추가한다:

**MlirWrapper.fs에 추가:**

```fsharp
type OpBuilder(context: Context) =
    // ... 기존 메서드 ...

    /// scf.if operation 생성
    member this.CreateScfIf(condition: MlirValue, resultTypes: MlirType[], location: MlirLocation) : MlirOperation =
        let ifOp = MlirNative.mlirSCFIfCreate(location, condition, true)

        // 결과 타입 설정 (C API 함수 사용)
        MlirNative.mlirOperationSetResultTypes(ifOp, nativeint resultTypes.Length, resultTypes)

        ifOp

    /// scf.if의 then region에 블록 추가
    member this.GetThenBlock(ifOp: MlirOperation) : MlirBlock =
        let thenRegion = MlirNative.mlirSCFIfGetThenRegion(ifOp)
        let block = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
        MlirNative.mlirRegionAppendOwnedBlock(thenRegion, block)
        block

    /// scf.if의 else region에 블록 추가
    member this.GetElseBlock(ifOp: MlirOperation) : MlirBlock =
        let elseRegion = MlirNative.mlirSCFIfGetElseRegion(ifOp)
        let block = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
        MlirNative.mlirRegionAppendOwnedBlock(elseRegion, block)
        block

    /// scf.yield operation 생성
    member this.CreateScfYield(results: MlirValue[], location: MlirLocation) : MlirOperation =
        MlirNative.mlirSCFYieldCreate(location, nativeint results.Length, results)
```

**사용 예시:**

```fsharp
// scf.if operation 생성
let i32Type = builder.I32Type()
let ifOp = builder.CreateScfIf(condition, [| i32Type |], location)

// then region 작성
let thenBlock = builder.GetThenBlock(ifOp)
// ... thenBlock에 연산 추가 ...
let thenYield = builder.CreateScfYield([| thenValue |], location)
MlirNative.mlirBlockAppendOwnedOperation(thenBlock, thenYield)

// else region 작성
let elseBlock = builder.GetElseBlock(ifOp)
// ... elseBlock에 연산 추가 ...
let elseYield = builder.CreateScfYield([| elseValue |], location)
MlirNative.mlirBlockAppendOwnedOperation(elseBlock, elseYield)
```

### Dialect 로딩

SCF dialect를 사용하려면 context에 로드해야 한다:

```fsharp
let ctx = new Context()
ctx.LoadDialect("arith")
ctx.LoadDialect("func")
ctx.LoadDialect("scf")  // SCF dialect 로드!
```

이것으로 `scf.if`와 `scf.yield` 연산을 사용할 준비가 완료되었다!

### P/Invoke 바인딩 요약

**추가한 바인딩:**
- `mlirSCFIfCreate`: scf.if operation 생성
- `mlirSCFYieldCreate`: scf.yield operation 생성
- `mlirSCFIfGetThenRegion/ElseRegion`: region 접근

**OpBuilder 헬퍼:**
- `CreateScfIf`: scf.if 생성 + 결과 타입 설정
- `GetThenBlock/GetElseBlock`: region에 블록 추가
- `CreateScfYield`: scf.yield 생성

**C API 제약:**
- C API가 불완전하면 C++ shim 작성 (Appendix 패턴 따름)
- Operation State Builder를 일반 대안으로 사용

**다음 섹션:** AST에 If 케이스를 추가하고, 코드 생성을 구현한다!

## AST 확장: If 표현식과 Boolean 리터럴

이제 AST에 if 표현식과 boolean 리터럴을 추가한다.

### Expr 타입 확장

**Ast.fs 수정:**

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
    | Let of name: string * binding: Expr * body: Expr
    | Var of name: string
    // NEW: If 표현식과 Boolean 리터럴
    | If of condition: Expr * thenBranch: Expr * elseBranch: Expr
    | Bool of bool

/// 최상위 프로그램
type Program =
    { expr: Expr }
```

**새로운 케이스 설명:**

### If of condition * thenBranch * elseBranch

```fsharp
| If of condition: Expr * thenBranch: Expr * elseBranch: Expr
```

**의미:** `if {condition} then {thenBranch} else {elseBranch}`

**필드:**
- `condition`: 조건 표현식 (i1 boolean 값을 생성해야 함)
- `thenBranch`: 조건이 true일 때 실행하는 표현식
- `elseBranch`: 조건이 false일 때 실행하는 표현식

**타입 제약:**
- `condition`은 i1 타입을 생성해야 한다
- `thenBranch`와 `elseBranch`는 **같은 타입**을 생성해야 한다

**예시:**

```fsharp
// FunLang: if 5 < 10 then 42 else 0
If(
  Comparison(LessThan, IntLiteral 5, IntLiteral 10),
  IntLiteral 42,
  IntLiteral 0
)
```

### Bool of bool

```fsharp
| Bool of bool
```

**의미:** Boolean 리터럴 - `true` 또는 `false`

**필드:**
- `bool`: F# boolean 값 (true 또는 false)

**예시:**

```fsharp
// FunLang: if true then 1 else 0
If(
  Bool true,
  IntLiteral 1,
  IntLiteral 0
)
```

**MLIR로 컴파일:** `Bool true` → `arith.constant 1 : i1`, `Bool false` → `arith.constant 0 : i1`

### AST 예시

**간단한 if:**

```fsharp
// FunLang: if true then 42 else 0
If(Bool true, IntLiteral 42, IntLiteral 0)
```

**비교 조건:**

```fsharp
// FunLang: if 5 < 10 then 1 else 0
If(
  Comparison(LessThan, IntLiteral 5, IntLiteral 10),
  IntLiteral 1,
  IntLiteral 0
)
```

**let 바인딩과 결합:**

```fsharp
// FunLang: let x = 5 in if x > 0 then x * 2 else 0
Let("x",
  IntLiteral 5,
  If(
    Comparison(GreaterThan, Var "x", IntLiteral 0),
    BinaryOp(Multiply, Var "x", IntLiteral 2),
    IntLiteral 0
  )
)
```

## Boolean 표현식

Boolean 값은 MLIR에서 **i1 타입** (1-bit integer)으로 표현된다.

### Boolean 타입: i1

MLIR은 boolean을 위한 전용 타입이 없다. 대신 1-bit integer (`i1`)를 사용한다:

```mlir
%true = arith.constant 1 : i1    // Boolean true
%false = arith.constant 0 : i1   // Boolean false
```

**i1의 값:**
- `1`: true
- `0`: false

### Boolean 리터럴 컴파일

`Bool` 케이스를 i1 상수로 컴파일한다:

```fsharp
| Bool(value) ->
    let i1Type = builder.Context.GetIntegerType(1)  // 1-bit integer
    let intValue = if value then 1L else 0L
    let attr = builder.Context.GetIntegerAttr(i1Type, intValue)
    let constOp = builder.CreateConstant(attr, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, constOp)
    builder.GetResult(constOp, 0)
```

**생성된 MLIR IR:**

```mlir
// Bool true
%true = arith.constant 1 : i1

// Bool false
%false = arith.constant 0 : i1
```

### 비교 연산은 이미 i1을 반환한다

Chapter 06에서 구현한 비교 연산 (`arith.cmpi`)은 i1을 반환한다:

```mlir
%c5 = arith.constant 5 : i32
%c10 = arith.constant 10 : i32
%cond = arith.cmpi slt, %c5, %c10 : i32  // 결과는 i1
```

**중요:** if 조건으로 비교 연산을 사용할 때, i1 → i32 확장(`arith.extui`)을 제거해야 한다!

Chapter 06에서는 main 함수 반환을 위해 i1을 i32로 확장했다:

```fsharp
// Chapter 06 코드 (비교 결과를 i32로 확장)
| Comparison(compareOp, lhs, rhs) ->
    let lhsVal = compileExpr builder block location lhs env
    let rhsVal = compileExpr builder block location rhs env
    let cmpOp = builder.CreateArithCompare(compareOp, lhsVal, rhsVal, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, cmpOp)
    let cmpVal = builder.GetResult(cmpOp, 0)  // i1 값
    // i1 -> i32 확장
    let i32Type = builder.I32Type()
    let extOp = builder.CreateArithExtUI(cmpVal, i32Type, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, extOp)
    builder.GetResult(extOp, 0)  // i32 반환
```

**문제:** if 조건은 i1이 필요한데, 위 코드는 i32를 반환한다!

**해결 방안:** 컨텍스트에 따라 확장 여부를 결정한다:
- if 조건: i1 그대로 사용
- main 함수 반환: i32로 확장

**간단한 접근:** Comparison 케이스가 i1을 반환하도록 하고, main 함수에서만 확장한다.

**수정된 Comparison 케이스:**

```fsharp
| Comparison(compareOp, lhs, rhs) ->
    let lhsVal = compileExpr builder block location lhs env
    let rhsVal = compileExpr builder block location rhs env
    let cmpOp = builder.CreateArithCompare(compareOp, lhsVal, rhsVal, location)
    MlirNative.mlirBlockAppendOwnedOperation(block, cmpOp)
    builder.GetResult(cmpOp, 0)  // i1 반환 (확장 안 함)
```

**main 함수에서 확장:**

```fsharp
let resultValue = compileExpr builder entryBlock loc program.expr env

// 결과가 i1이면 i32로 확장 (main 함수 반환용)
let resultType = MlirNative.mlirValueGetType(resultValue)
let finalResult =
    if MlirNative.mlirTypeIsI1(resultType) then
        let i32Type = builder.I32Type()
        let extOp = builder.CreateArithExtUI(resultValue, i32Type, loc)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, extOp)
        builder.GetResult(extOp, 0)
    else
        resultValue
```

### Boolean 연산 (선택 사항)

Boolean 값에 논리 연산을 적용할 수 있다:

**AND:**

```mlir
%a = arith.constant 1 : i1
%b = arith.constant 0 : i1
%result = arith.andi %a, %b : i1  // 결과: 0 (false)
```

**OR:**

```mlir
%result = arith.ori %a, %b : i1  // 결과: 1 (true)
```

**NOT (XOR with 1):**

```mlir
%c1 = arith.constant 1 : i1
%result = arith.xori %a, %c1 : i1  // a의 반대
```

**AST 추가 (나중에):**

Phase 2에서는 boolean 연산을 추가하지 않는다. if/then/else만으로 충분하다. 필요하면 나중에 추가한다.

## If/Then/Else 코드 생성

이제 If 케이스를 scf.if로 컴파일한다.

### If 케이스 구현

**CodeGen.fs에 추가:**

```fsharp
| If(condition, thenExpr, elseExpr) ->
    // 1. 조건 표현식 컴파일 (i1 타입 필요)
    let condVal = compileExpr builder block location condition env

    // 2. 결과 타입 결정 (thenBranch의 타입을 사용)
    // 실제로는 타입 추론이 필요하지만, 지금은 i32로 가정
    let i32Type = builder.I32Type()
    let resultTypes = [| i32Type |]

    // 3. scf.if operation 생성
    let ifOp = builder.CreateScfIf(condVal, resultTypes, location)

    // 4. Then region 작성
    let thenBlock = builder.GetThenBlock(ifOp)
    let thenVal = compileExpr builder thenBlock location thenExpr env
    let thenYield = builder.CreateScfYield([| thenVal |], location)
    MlirNative.mlirBlockAppendOwnedOperation(thenBlock, thenYield)

    // 5. Else region 작성
    let elseBlock = builder.GetElseBlock(ifOp)
    let elseVal = compileExpr builder elseBlock location elseExpr env
    let elseYield = builder.CreateScfYield([| elseVal |], location)
    MlirNative.mlirBlockAppendOwnedOperation(elseBlock, elseYield)

    // 6. scf.if를 블록에 추가
    MlirNative.mlirBlockAppendOwnedOperation(block, ifOp)

    // 7. scf.if의 결과 반환
    builder.GetResult(ifOp, 0)
```

**동작 설명:**

1. **조건 컴파일:** `condition` 표현식을 컴파일하여 i1 값을 얻는다
2. **결과 타입:** if 표현식의 결과 타입 (여기서는 i32로 가정)
3. **scf.if 생성:** `CreateScfIf`로 operation 생성
4. **Then region:** thenBranch 컴파일 → scf.yield로 값 반환
5. **Else region:** elseBranch 컴파일 → scf.yield로 값 반환
6. **Operation 추가:** scf.if를 부모 블록에 추가
7. **결과 사용:** scf.if의 결과 (SSA value)를 반환

**핵심:** 각 region에서 `compileExpr`를 호출할 때 **해당 region의 블록**을 전달한다. 이렇게 하면 연산이 올바른 region에 추가된다.

### 예시: if true then 42 else 0

**AST:**

```fsharp
If(Bool true, IntLiteral 42, IntLiteral 0)
```

**컴파일 과정:**

1. `Bool true` 컴파일: `%true = arith.constant 1 : i1`
2. `scf.if` 생성
3. Then region:
   - `IntLiteral 42` 컴파일: `%c42 = arith.constant 42 : i32`
   - `scf.yield %c42`
4. Else region:
   - `IntLiteral 0` 컴파일: `%c0 = arith.constant 0 : i32`
   - `scf.yield %c0`
5. scf.if 결과: `%result`

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %true = arith.constant 1 : i1
    %result = scf.if %true -> (i32) {
      %c42 = arith.constant 42 : i32
      scf.yield %c42 : i32
    } else {
      %c0 = arith.constant 0 : i32
      scf.yield %c0 : i32
    }
    func.return %result : i32
  }
}
```

**실행:**

```bash
$ ./program
$ echo $?
42
```

조건이 true이므로 42를 반환한다!

### 예시: if 5 < 10 then 1 else 0

**AST:**

```fsharp
If(
  Comparison(LessThan, IntLiteral 5, IntLiteral 10),
  IntLiteral 1,
  IntLiteral 0
)
```

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %cond = arith.cmpi slt, %c5, %c10 : i32  // i1 결과
    %result = scf.if %cond -> (i32) {
      %c1 = arith.constant 1 : i32
      scf.yield %c1 : i32
    } else {
      %c0 = arith.constant 0 : i32
      scf.yield %c0 : i32
    }
    func.return %result : i32
  }
}
```

**실행:**

```bash
$ ./program
$ echo $?
1
```

5 < 10이 true이므로 1을 반환한다!

## Lowering Pass 업데이트

SCF dialect를 사용하므로 lowering pass에 `--convert-scf-to-cf`를 추가해야 한다.

### Pass Manager 순서

**Lowering.fs 수정:**

```fsharp
namespace FunLangCompiler

module Lowering =

    /// MLIR module을 LLVM dialect로 lowering
    let lowerToLLVMDialect (mlirMod: Module) =
        let ctx = mlirMod.Context
        let pm = MlirNative.mlirPassManagerCreate(ctx.Handle)

        // 1. SCF -> CF 변환 (구조화된 제어 흐름 -> 분기)
        let scfToCfPass = MlirNative.mlirCreateConversionConvertSCFToCFPass()
        MlirNative.mlirPassManagerAddOwnedPass(pm, scfToCfPass)

        // 2. Arith -> LLVM 변환
        let arithToLLVMPass = MlirNative.mlirCreateConversionConvertArithToLLVMPass()
        MlirNative.mlirPassManagerAddOwnedPass(pm, arithToLLVMPass)

        // 3. Func -> LLVM 변환
        let funcToLLVMPass = MlirNative.mlirCreateConversionConvertFuncToLLVMPass()
        MlirNative.mlirPassManagerAddOwnedPass(pm, funcToLLVMPass)

        // 4. Unrealized casts 정리
        let reconcilePass = MlirNative.mlirCreateConversionReconcileUnrealizedCastsPass()
        MlirNative.mlirPassManagerAddOwnedPass(pm, reconcilePass)

        // Pass 실행
        let result = MlirNative.mlirPassManagerRun(pm, mlirMod.Handle)
        if not (MlirNative.mlirLogicalResultIsSuccess(result)) then
            failwith "Failed to run lowering passes"

        MlirNative.mlirPassManagerDestroy(pm)

    /// MLIR module을 LLVM IR로 변환
    let translateToLLVMIR (mlirMod: Module) : string =
        let llvmIR = MlirNative.mlirTranslateModuleToLLVMIR(mlirMod.Handle)
        MlirHelpers.toString llvmIR
```

**Pass 순서 설명:**

1. **SCF → CF:** `scf.if` → `cf.cond_br` + block arguments
2. **Arith → LLVM:** `arith.constant`, `arith.addi` 등 → `llvm.mlir.constant`, `llvm.add` 등
3. **Func → LLVM:** `func.func`, `func.return` → `llvm.func`, `llvm.return`
4. **Reconcile:** 중간 cast 연산 정리

### MlirBindings.fs에 Pass 추가

**MlirBindings.fs에 추가:**

```fsharp
/// SCF to CF 변환 pass 생성
[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirPass mlirCreateConversionConvertSCFToCFPass()
```

**Note:** 함수 이름은 MLIR C API 버전에 따라 다를 수 있다. `mlir-c/Conversion/Passes.h` 헤더를 확인한다.

### Lowering 후 MLIR IR

**scf.if lowering 전:**

```mlir
func.func @main() -> i32 {
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32
  %cond = arith.cmpi slt, %c5, %c10 : i32
  %result = scf.if %cond -> (i32) {
    %c1 = arith.constant 1 : i32
    scf.yield %c1 : i32
  } else {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  }
  func.return %result : i32
}
```

**scf.if lowering 후 (cf dialect):**

```mlir
func.func @main() -> i32 {
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32
  %cond = arith.cmpi slt, %c5, %c10 : i32
  cf.cond_br %cond, ^then, ^else

^then:
  %c1 = arith.constant 1 : i32
  cf.br ^merge(%c1 : i32)

^else:
  %c0 = arith.constant 0 : i32
  cf.br ^merge(%c0 : i32)

^merge(%result: i32):
  func.return %result : i32
}
```

**핵심:**
- `scf.if` → `cf.cond_br` + 블록
- `scf.yield` → `cf.br ^merge(value)`
- Block argument `%result`가 PHI 역할

## Let 바인딩과 If 결합

Let 바인딩과 if 표현식을 결합한 예시를 보자.

**FunLang 소스:**

```fsharp
let x = 5 in
if x > 0 then x * 2 else 0
```

**AST:**

```fsharp
Let("x",
  IntLiteral 5,
  If(
    Comparison(GreaterThan, Var "x", IntLiteral 0),
    BinaryOp(Multiply, Var "x", IntLiteral 2),
    IntLiteral 0
  )
)
```

**컴파일 과정:**

1. `Let("x", IntLiteral 5, ...)`
   - `IntLiteral 5` 컴파일: `%c5 = arith.constant 5 : i32`
   - `env' = env.Add("x", %c5)`
   - Body 컴파일 (env' 사용)

2. `If(...)` (env'에서)
   - Condition: `Comparison(GreaterThan, Var "x", IntLiteral 0)`
     - `Var "x"`: env'에서 조회 → %c5
     - `IntLiteral 0`: `%c0 = arith.constant 0 : i32`
     - `%cond = arith.cmpi sgt, %c5, %c0 : i32`
   - Then: `BinaryOp(Multiply, Var "x", IntLiteral 2)`
     - `Var "x"`: env'에서 조회 → %c5
     - `IntLiteral 2`: `%c2 = arith.constant 2 : i32`
     - `%then_val = arith.muli %c5, %c2 : i32`
   - Else: `IntLiteral 0`
     - `%else_val = arith.constant 0 : i32`

**생성된 MLIR IR:**

```mlir
module {
  func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32          // let x = 5
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %c5, %c0 : i32  // x > 0
    %result = scf.if %cond -> (i32) {
      %c2 = arith.constant 2 : i32
      %then_val = arith.muli %c5, %c2 : i32  // x * 2
      scf.yield %then_val : i32
    } else {
      %else_val = arith.constant 0 : i32
      scf.yield %else_val : i32
    }
    func.return %result : i32
  }
}
```

**실행:**

```bash
$ ./program
$ echo $?
10
```

x = 5, x > 0이 true, x * 2 = 10!

### 중첩된 If

if 안에 if를 넣을 수도 있다:

```fsharp
// FunLang: if x > 0 then (if x < 10 then 1 else 2) else 0
If(
  Comparison(GreaterThan, Var "x", IntLiteral 0),
  If(
    Comparison(LessThan, Var "x", IntLiteral 10),
    IntLiteral 1,
    IntLiteral 2
  ),
  IntLiteral 0
)
```

**생성된 MLIR IR:**

```mlir
%outer_cond = arith.cmpi sgt, %x, %c0 : i32
%result = scf.if %outer_cond -> (i32) {
  %inner_cond = arith.cmpi slt, %x, %c10 : i32
  %inner_result = scf.if %inner_cond -> (i32) {
    %c1 = arith.constant 1 : i32
    scf.yield %c1 : i32
  } else {
    %c2 = arith.constant 2 : i32
    scf.yield %c2 : i32
  }
  scf.yield %inner_result : i32
} else {
  %c0 = arith.constant 0 : i32
  scf.yield %c0 : i32
}
```

중첩된 scf.if가 올바르게 생성된다!

## 공통 에러

### 에러 1: 조건이 i32인데 i1이 필요

**증상:**

```
MLIR verification failed:
'scf.if' op operand #0 must be 1-bit signless integer, but got 'i32'
```

**원인:**

if 조건에 i32 값을 전달했다.

**해결:**

조건은 반드시 i1 타입이어야 한다:
- Boolean 리터럴: `Bool true` → `arith.constant 1 : i1`
- 비교 연산: `arith.cmpi` → i1 결과
- i32를 i1로 변환하지 말고, 비교 연산을 사용한다

```fsharp
// WRONG: i32를 조건으로 사용
let x = IntLiteral 5
If(x, ..., ...)  // 에러! x는 i32

// CORRECT: 비교 연산 사용
If(Comparison(GreaterThan, x, IntLiteral 0), ..., ...)
```

### 에러 2: scf.yield 타입 불일치

**증상:**

```
MLIR verification failed:
'scf.yield' op types mismatch between then and else regions
```

**원인:**

then region과 else region이 다른 타입을 yield했다.

**해결:**

양쪽 region이 같은 타입을 yield해야 한다:

```fsharp
// WRONG: then은 i32, else는 i1
If(cond,
  IntLiteral 42,        // i32
  Bool true)            // i1 - 타입 불일치!

// CORRECT: 둘 다 i32
If(cond,
  IntLiteral 42,        // i32
  IntLiteral 0)         // i32
```

### 에러 3: scf.yield 누락

**증상:**

```
MLIR verification failed:
Region does not have a terminator
```

**원인:**

then 또는 else region에 scf.yield를 추가하지 않았다.

**해결:**

모든 region은 종결자가 필요하다. 코드 생성 시 항상 scf.yield를 추가한다:

```fsharp
// 올바른 코드 생성 패턴
let thenBlock = builder.GetThenBlock(ifOp)
let thenVal = compileExpr builder thenBlock location thenExpr env
let thenYield = builder.CreateScfYield([| thenVal |], location)
MlirNative.mlirBlockAppendOwnedOperation(thenBlock, thenYield)  // 필수!
```

### 에러 4: --convert-scf-to-cf pass 누락

**증상:**

```
Failed to translate MLIR to LLVM IR:
Unhandled operation: scf.if
```

**원인:**

Lowering pass에서 SCF → CF 변환을 실행하지 않았다.

**해결:**

Pass manager에 `--convert-scf-to-cf`를 추가한다:

```fsharp
let scfToCfPass = MlirNative.mlirCreateConversionConvertSCFToCFPass()
MlirNative.mlirPassManagerAddOwnedPass(pm, scfToCfPass)
```

Pass 순서: SCF → CF → Arith → Func → Reconcile

## 장 요약

이 장에서 다음을 성취했다:

1. **PHI 노드 문제 이해**: 위치 제약, lost copy problem, dominance frontier 계산
2. **Block Arguments 학습**: MLIR의 우아한 대안, 함수 인자와 통일된 의미론
3. **scf.if 연산 사용**: 고수준 구조화된 제어 흐름, scf.yield 종결자
4. **P/Invoke 바인딩 추가**: SCF dialect 지원 (mlirSCFIfCreate, mlirSCFYieldCreate)
5. **AST 확장**: If 표현식과 Bool 리터럴 추가
6. **Boolean 타입**: i1 (1-bit integer), true = 1, false = 0
7. **코드 생성 구현**: If 케이스를 scf.if로 컴파일
8. **Lowering pass 업데이트**: SCF → CF 변환 추가
9. **완전한 예제**: if/then/else와 let 바인딩 결합

**독자가 할 수 있는 것:**

- `if true then 42 else 0` 컴파일 → 네이티브 바이너리 → 결과: 42 ✓
- `if 5 < 10 then 1 else 0` 컴파일 → 결과: 1 ✓
- `let x = 5 in if x > 0 then x * 2 else 0` 컴파일 → 결과: 10 ✓
- Block arguments vs PHI 노드 차이 이해 ✓
- scf.if lowering 과정 이해 ✓
- Boolean 타입 (i1) 사용 ✓
- 타입 불일치 에러 디버깅 ✓

**핵심 개념:**

- **Block Arguments > PHI 노드**: 깔끔한 의미론, push vs pull
- **scf.if = 표현식**: 값을 반환, 함수형 의미론
- **scf.yield = 종결자**: Region에서 값 반환, return과 유사
- **i1 타입 = Boolean**: 1 = true, 0 = false
- **Progressive Lowering**: scf → cf → llvm

**다음 장 미리보기:**

Chapter 09에서는 **메모리 관리**를 다룬다:

- Stack vs Heap 할당
- `memref.alloca` (stack allocation)
- `memref.alloc` (heap allocation)
- **Boehm GC 통합** (garbage collection)

Phase 2의 마지막 장이다. Phase 3에서는 함수와 클로저를 구현할 것이다!

---

**이제 독자는 if/then/else 제어 흐름을 컴파일하고, block arguments와 scf.if를 이해한다!**
