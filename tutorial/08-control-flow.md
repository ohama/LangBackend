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
