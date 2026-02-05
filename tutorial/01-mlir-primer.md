# Chapter 01: MLIR 입문

## 소개

이전 챕터에서 LLVM/MLIR을 소스에서 빌드하고 .NET SDK를 설정했습니다. 필요한 도구는 모두 설치되었습니다. 하지만 MLIR을 생성하는 F# 코드를 작성하기 전에, MLIR이 무엇이고 프로그램을 어떻게 표현하는지 이해해야 합니다.

MLIR은 전통적인 중간 표현(intermediate representation)과 다릅니다. 단순히 "하나의 IR"이 아니라, 서로 상호 운용할 수 있는 여러 IR(dialect이라고 부릅니다)을 구축하기 위한 프레임워크입니다. 이 다단계(multi-level) 철학이 MLIR을 컴파일러 개발에 강력하게 만드는 핵심입니다. 고수준 함수형 언어를 매우 저수준인 LLVM IR로 직접 변환하도록 강제하는 대신, MLIR은 언어의 의미론(semantics)을 필요한 만큼 보존하는 중간 표현을 정의한 다음, 단계적으로 점진적 하강(progressive lowering)할 수 있게 해줍니다.

FunLang의 컴파일 파이프라인은 다음과 같습니다:

```
FunLang Typed AST
    ↓
High-Level MLIR (arith, func, scf dialects)
    ↓
Low-Level MLIR (LLVM dialect)
    ↓
LLVM IR
    ↓
Native Machine Code
```

이 챕터에서는 MLIR IR을 이해하기 위한 멘탈 모델을 제공합니다. 다섯 가지 핵심 개념 — **dialect**, **operation**, **region**, **block**, 그리고 **SSA form** — 을 구체적인 예제를 통해 배우게 됩니다. 챕터를 마치면 MLIR 텍스트 IR을 읽고, FunLang 프로그램이 MLIR 구조에 어떻게 매핑되는지 이해할 수 있을 것입니다.

## MLIR IR 구조

완전한 MLIR 프로그램을 보면서 각 부분을 분석해 보겠습니다. 다음은 두 개의 32비트 정수를 더하는 간단한 함수입니다:

```mlir
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
```

한 줄씩 분석해 보겠습니다:

- **`module { ... }`**: 모든 MLIR 프로그램은 module에 포함됩니다. module은 모든 코드를 담는 최상위 컨테이너로, C의 컴파일 단위(compilation unit)나 .NET의 어셈블리와 유사합니다.

- **`func.func @add(...) -> i32 { ... }`**: `func` dialect의 operation으로, `@add`라는 이름의 함수를 정의합니다. `@` 접두사는 심볼(함수 이름)을 나타냅니다. 이 함수는 두 개의 인자를 받아 `i32`(32비트 정수)를 반환합니다.

- **`%arg0: i32, %arg1: i32`**: 함수 매개변수입니다. 각 매개변수는 타입 어노테이션(`: i32`)을 가진 SSA 값(`%`로 시작)입니다. 이것이 함수의 입력입니다.

- **`%result = arith.addi %arg0, %arg1 : i32`**: `arith` dialect의 산술 덧셈 operation입니다. 두 피연산자(`%arg0`과 `%arg1`)를 받아 더한 후, 새로운 SSA 값 `%result`를 생성합니다. `: i32` 접미사는 결과 타입을 지정합니다.

- **`return %result : i32`**: 함수의 return operation입니다. `%result` 값을 호출자에게 반환합니다. `: i32` 타입 어노테이션은 타입 안전성을 보장합니다.

MLIR의 모든 요소에는 목적과 타입이 있습니다. 암시적 변환이나 정의되지 않은 동작(undefined behavior)은 없습니다. 이러한 엄격함이 MLIR이 공격적인 최적화와 검증을 수행할 수 있게 해주는 것입니다.

## Dialect

**Dialect**은 관련된 operation, 타입, attribute를 그룹화하는 네임스페이스입니다. Dialect은 MLIR의 확장성 메커니즘입니다 — 모든 가능한 operation을 하나의 거대한 IR에 넣는 대신, MLIR은 도메인에 맞는 커스텀 dialect을 정의할 수 있게 해줍니다.

### 사용할 내장 Dialect

FunLang 컴파일러에서는 주로 다음 표준 dialect들을 사용합니다:

1. **`arith`** — 산술 연산
   - `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi` (부호 있는 정수 산술)
   - `arith.cmpi` (정수 비교: `<`, `>`, `==` 등)
   - `arith.constant` (정수 및 부동소수점 상수)

2. **`func`** — 함수 정의 및 호출
   - `func.func` (함수 정의)
   - `func.call` (함수 호출)
   - `func.return` (함수에서 반환)

3. **`scf`** — 구조적 제어 흐름(Structured Control Flow)
   - `scf.if` (조건부 실행)
   - `scf.for` (카운트 루프)
   - `scf.while` (조건 루프)

4. **`llvm`** — LLVM dialect (lowering 대상)
   - `llvm.func`, `llvm.call`, `llvm.add` 등
   - 이 dialect은 LLVM IR 구성 요소와 1:1로 매핑됩니다

### 커스텀 Dialect

이 튜토리얼 시리즈의 후반부(Chapter 10-11)에서는 다음과 같은 operation을 가진 **FunLang dialect**을 정의하게 됩니다:

- `funlang.closure` (클로저 생성)
- `funlang.apply` (클로저에 인자를 적용)
- `funlang.match` (패턴 매칭)

커스텀 dialect을 사용하면 컴파일 과정에서 고수준 의미론을 보존할 수 있습니다. FunLang 클로저를 즉시 저수준 구조체 할당과 함수 포인터로 변환하는 대신, 고수준 `funlang.closure` operation으로 표현합니다. 이렇게 하면 최적화를 작성하고 이해하기가 더 쉬워집니다.

### Dialect 명명 규칙

Operation은 항상 자신이 속한 dialect 이름을 접두사로 가지며, 점(.)으로 구분됩니다:

```mlir
arith.addi   // "arith" dialect의 "addi" operation
func.call    // "func" dialect의 "call" operation
llvm.load    // "llvm" dialect의 "load" operation
```

이를 통해 이름 충돌을 방지합니다. `arith` dialect의 `addi`는 가상의 `mydialect.addi`와 구별됩니다.

## Operation

**Operation**은 MLIR IR의 기본 단위입니다. MLIR에서는 함수 정의, 산술 명령어, 제어 흐름 등 모든 것이 operation으로 표현됩니다. 심지어 타입과 attribute도 operation에 첨부됩니다.

### Operation의 구조

텍스트 형식에서 operation은 다음과 같은 구조를 가집니다:

```mlir
%results = dialect.opname(%operands) {attributes} : (types) -> result_type
```

덧셈 예제에서 각 구성 요소를 살펴보겠습니다:

```mlir
%result = arith.addi %arg0, %arg1 : i32
```

- **`%result`**: 이 operation이 생성하는 SSA 값입니다. 이 값은 이후 operation에서 사용할 수 있습니다. `%` 접두사는 SSA 값을 심볼(`@function_name`)과 구별합니다.

- **`arith.addi`**: operation 이름(dialect + opname)입니다.

- **`%arg0, %arg1`**: 피연산자(operation의 입력)입니다. 이전에 정의된 SSA 값(이 경우 함수 인자)입니다.

- **`: i32`**: 타입 제약 조건입니다. 이 operation은 32비트 정수에 대해 동작합니다.

모든 operation이 결과를 생성하는 것은 아닙니다. 예를 들어, `return`은 함수를 종료하는 operation이지만 이후에 사용할 값을 생성하지는 않습니다:

```mlir
return %result : i32
```

### 복수 결과를 가진 Operation

일부 operation은 여러 값을 생성합니다. 예를 들어, 몫과 나머지를 모두 반환하는 나눗셈 operation이 있습니다:

```mlir
%quot, %rem = arith.divrem %dividend, %divisor : i32
```

이제 `%quot`과 `%rem` 모두 사용 가능한 SSA 값입니다.

### Attribute를 가진 Operation

Attribute는 컴파일 타임 상수 메타데이터를 제공합니다. 예를 들어, 정수 상수는 다음과 같습니다:

```mlir
%zero = arith.constant 0 : i32
```

`0`은 attribute(상수 값)이고, `i32`는 타입입니다. Attribute는 런타임 값이 아니라 컴파일 타임에 IR에 내장되는 것입니다.

## Region과 Block

MLIR operation은 **region**을 포함할 수 있고, region은 **block**을 포함합니다. 이것이 MLIR이 중첩된 스코프와 제어 흐름을 표현하는 방식입니다.

### Region

**Region**은 block의 목록입니다. 함수 본문은 region입니다. `scf.if`와 같은 제어 흐름 operation에는 "then"과 "else" 분기를 위한 region이 있습니다.

다음은 하나의 region에 하나의 block을 포함하는 함수입니다:

```mlir
func.func @example() -> i32 {
  %one = arith.constant 1 : i32
  return %one : i32
}
```

중괄호 `{ ... }`가 함수의 region을 구분합니다. region 내부에는 두 개의 operation(상수와 return)을 가진 하나의 block이 있습니다.

### Block

**Block**은 선형적으로 실행되는 operation의 시퀀스입니다. 모든 block은 **terminator** operation — 제어를 다른 곳으로 이전하는 operation(return, branch 등) — 으로 끝나야 합니다. Block을 "통과(fall through)"할 수 없습니다.

제어 흐름이 있을 때 block이 필수적이 됩니다. 다음은 두 개의 block을 가진 함수입니다:

```mlir
func.func @conditional(%cond: i1, %a: i32, %b: i32) -> i32 {
  cf.cond_br %cond, ^then_block, ^else_block

^then_block:
  return %a : i32

^else_block:
  return %b : i32
}
```

분석해 보겠습니다:

- **`cf.cond_br %cond, ^then_block, ^else_block`**: 조건 분기 operation(`cf` control-flow dialect)입니다. `%cond`가 참이면 `^then_block`으로, 그렇지 않으면 `^else_block`으로 점프합니다. 이것이 entry block의 terminator입니다.

- **`^then_block:`**: block 레이블입니다. `^` 접두사는 block을 나타냅니다. block 이름은 함수 내에서 로컬입니다.

- **`return %a : i32`**: `^then_block`의 terminator입니다. `%a`를 호출자에게 반환합니다.

- **`^else_block:`**: 또 다른 block 레이블입니다.

- **`return %b : i32`**: `^else_block`의 terminator입니다. `%b`를 반환합니다.

### Block 인자 (MLIR의 Phi Node 처리 방식)

MLIR은 LLVM의 phi node 대신 **block 인자**를 사용합니다. LLVM IR에서는 여러 선행 block의 값을 병합하기 위해 phi node를 사용합니다. MLIR에서는 block으로 분기할 때 값을 인자로 전달합니다.

다음은 두 값을 병합하는 예제입니다:

```mlir
func.func @merge_example(%cond: i1, %a: i32, %b: i32) -> i32 {
  cf.cond_br %cond, ^merge(%a : i32), ^merge(%b : i32)

^merge(%result: i32):
  return %result : i32
}
```

무슨 일이 일어나는지 살펴보겠습니다:

- **`cf.cond_br %cond, ^merge(%a : i32), ^merge(%b : i32)`**: `^merge` block으로 분기하되, 조건이 참이면 `%a`를, 거짓이면 `%b`를 전달합니다.

- **`^merge(%result: i32):`**: `^merge` block은 `i32` 타입의 인자 하나를 기대한다고 선언합니다. 어느 분기가 선택되든, 전달된 값이 이 block 내에서 `%result`가 됩니다.

이 방식은 LLVM의 phi node보다 깔끔합니다. 데이터 흐름이 분기 지점에서 명시적으로 표현되며, 사후에 재구성할 필요가 없기 때문입니다.

## SSA Form (Static Single Assignment)

MLIR은 **SSA form**을 사용합니다. 즉, 모든 값은 정확히 한 번만 정의되고 절대 변경되지 않습니다. `%x`를 정의하면 다시 할당할 수 없습니다. 이 속성 덕분에 "지금 어떤 버전의 변수를 보고 있는 것인가?"를 추적할 필요가 없어 최적화가 단순해집니다.

### SSA 동작 예시

다음 FunLang 코드를 살펴보겠습니다:

```fsharp
let x = 5
let y = x + 3
let z = y * 2
z
```

MLIR SSA form에서 각 let 바인딩은 새로운 SSA 값이 됩니다:

```mlir
func.func @example() -> i32 {
  %x = arith.constant 5 : i32
  %three = arith.constant 3 : i32
  %y = arith.addi %x, %three : i32
  %two = arith.constant 2 : i32
  %z = arith.muli %y, %two : i32
  return %z : i32
}
```

주목할 점:

- 각 `let` 바인딩은 새로운 SSA 값(`%x`, `%y`, `%z`)이 됩니다.
- 상수는 값을 생성하는 operation(`arith.constant`)입니다.
- 어떤 값도 재할당되지 않습니다.

### SSA와 가변성(Mutability)

FunLang은 불변(immutable)이므로 SSA와 자연스럽게 매핑됩니다. 하지만 변이(mutation)가 있는 명령형 코드는 어떨까요? 다음을 살펴보겠습니다:

```c
int x = 1;
x = x + 1;
return x;
```

SSA에서는 `x`를 변경할 수 없습니다. 대신, 새로운 버전을 생성합니다:

```mlir
%x0 = arith.constant 1 : i32
%one = arith.constant 1 : i32
%x1 = arith.addi %x0, %one : i32
return %x1 : i32
```

각 "변이"는 새로운 SSA 값(`%x0`, `%x1` 등)을 생성합니다. 이 변환을 **SSA conversion**이라고 하며, 명령형 언어의 컴파일러에서 자동으로 처리됩니다.

FunLang은 함수형이므로 이 작업은 필요하지 않습니다 — 모든 `let` 바인딩이 이미 새로운 이름을 도입하기 때문입니다.

### 핵심 통찰: SSA는 최적화를 가능하게 한다

SSA form은 많은 컴파일러 최적화를 간단하게 만들어 줍니다. 예를 들어:

- **Dead code elimination(죽은 코드 제거):** SSA 값이 정의되었지만 사용되지 않으면, 해당 값을 정의하는 operation을 삭제합니다.
- **Constant propagation(상수 전파):** `%x`가 `arith.constant 5`로 정의되었다면, `%x`의 모든 사용을 `5`로 대체합니다.
- **Common subexpression elimination(공통 하위 표현식 제거):** 두 operation이 같은 값을 계산하면, 하나를 재사용하고 다른 하나를 삭제합니다.

이 모든 최적화는 값이 정의 후 절대 변경되지 않는다는 보장에 의존합니다.

## MLIR의 타입

MLIR은 강타입(strongly typed)입니다. 모든 SSA 값, operation, 함수에는 타입이 있습니다. 타입 시스템은 확장 가능하며(dialect이 커스텀 타입을 정의할 수 있음), 다음은 사용하게 될 내장 타입입니다:

### 정수 타입

- `i1` — 1비트 정수 (boolean)
- `i32` — 32비트 부호 있는 정수
- `i64` — 64비트 부호 있는 정수
- `i8`, `i16`, `i128` 등 — 임의 비트 너비 정수

### 부동소수점 타입

- `f32` — 32비트 IEEE 754 float
- `f64` — 64비트 IEEE 754 double

### Index 타입

- `index` — 배열 인덱싱을 위한 플랫폼 의존 정수 (대상 아키텍처에 따라 일반적으로 32비트 또는 64비트)

### 메모리 타입

- `memref<4xi32>` — 메모리상의 4개 `i32` 값 배열에 대한 참조
- `memref<*xf64>` — `f64` 값에 대한 unranked(동적) 메모리 참조

### 함수 타입

- `(i32, i32) -> i32` — 두 개의 `i32` 인자를 받아 `i32`를 반환하는 함수

### FunLang 타입 매핑

FunLang 타입이 MLIR 타입에 어떻게 매핑되는지 정리하면 다음과 같습니다:

| FunLang 타입 | MLIR 타입 | 비고 |
|--------------|-----------|-------|
| `Int` | `i64` | FunLang 정수는 인터프리터에서 임의 정밀도이지만, 64비트로 컴파일합니다 |
| `Bool` | `i1` | True = 1, False = 0 |
| `String` | `!llvm.ptr` (LLVM dialect 포인터) | 문자열은 힙에 할당된 null 종료 C 문자열입니다 |
| `Float` | `f64` | 배정밀도 부동소수점 |
| `List<'a>` | `!llvm.ptr` | 리스트는 힙에 할당된 연결 구조입니다 |
| `Tuple<'a, 'b, ...>` | `!llvm.struct<...>` | 튜플은 LLVM struct로 컴파일됩니다 |

`!` 접두사는 dialect에서 정의된 타입을 나타냅니다 (예: `!llvm.ptr`는 LLVM dialect의 포인터 타입).

## Progressive Lowering

MLIR의 강력함은 **progressive lowering**에 있습니다: 한 번에 크게 변환하는 대신, 고수준 operation을 여러 단계에 걸쳐 저수준 operation으로 변환하는 방식입니다.

### FunLang 컴파일 파이프라인

이 튜토리얼에서 구축할 파이프라인은 다음과 같습니다:

```
Stage 1: AST → High-Level MLIR
    FunLang AST (타입 검사기에서 전달)
    ↓
    arith, func, scf dialect을 사용하여 MLIR로 변환
    예: `let x = 1 + 2`는 `%x = arith.addi ...`가 됩니다

Stage 2: High-Level MLIR → LLVM Dialect
    `arith.addi` 같은 operation이 `llvm.add`로 lowering됩니다
    구조적 제어 흐름(`scf.if`)은 basic block과 branch로 lowering됩니다

Stage 3: LLVM Dialect → LLVM IR
    MLIR의 LLVM dialect이 텍스트 LLVM IR로 변환됩니다

Stage 4: LLVM IR → Native Code
    LLVM 백엔드(llc)가 대상 플랫폼의 머신 코드로 컴파일합니다
```

각 lowering 단계는 **pass** — IR을 재작성하는 변환 — 입니다. MLIR은 pass 정의, 패턴 기반 재작성, 각 단계 후 검증을 위한 인프라를 제공합니다.

### Progressive Lowering이 중요한 이유

FunLang의 패턴 매칭을 컴파일하는 경우를 생각해 보겠습니다. LLVM IR로 직접 lowering해야 한다면, 즉시 basic block, phi node, 메모리 로드로 이루어진 복잡한 결정 트리로 확장해야 합니다. 하지만 progressive lowering을 사용하면:

1. **고수준:** 패턴 매칭을 구조를 보존하는 `funlang.match` operation으로 표현합니다.
2. **중간 수준:** `funlang.match`를 `scf.if`와 `scf.while`(구조적 제어 흐름)로 lowering합니다.
3. **저수준:** `scf.if`를 LLVM basic block과 branch로 lowering합니다.

각 단계에서 해당 추상화 수준에 맞는 최적화를 수행할 수 있습니다. 패턴 매칭 최적화(중복 검사 제거)는 고수준에서 이루어지고, LLVM 수준 최적화(레지스터 할당, 명령어 스케줄링)는 저수준에서 이루어집니다.

## 종합 예제

여러 개념을 함께 사용하는 좀 더 현실적인 MLIR 예제를 살펴보겠습니다:

```mlir
module {
  func.func @factorial(%n: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %n, %c0 : i64
    cf.cond_br %is_zero, ^base_case, ^recursive_case

  ^base_case:
    return %c1 : i64

  ^recursive_case:
    %n_minus_1 = arith.subi %n, %c1 : i64
    %rec_result = func.call @factorial(%n_minus_1) : (i64) -> i64
    %result = arith.muli %n, %rec_result : i64
    return %result : i64
  }
}
```

이 코드를 추적해 보겠습니다:

1. **`func.func @factorial(%n: i64) -> i64`**: 하나의 64비트 정수를 받아 64비트 정수를 반환하는 `@factorial` 함수를 정의합니다.

2. **`%c0 = arith.constant 0 : i64`**: 상수 `0`을 생성합니다.

3. **`%c1 = arith.constant 1 : i64`**: 상수 `1`을 생성합니다.

4. **`%is_zero = arith.cmpi eq, %n, %c0 : i64`**: `%n`과 `0`을 동등성 비교합니다. 결과는 `i1`(boolean)입니다.

5. **`cf.cond_br %is_zero, ^base_case, ^recursive_case`**: 참이면 `^base_case`로, 아니면 `^recursive_case`로 분기합니다.

6. **`^base_case:`**: n == 0이면 1을 반환합니다.

7. **`^recursive_case:`**: n > 0이면 `n * factorial(n - 1)`을 계산합니다:
   - `%n_minus_1 = arith.subi %n, %c1`: `n - 1`을 계산합니다.
   - `%rec_result = func.call @factorial(%n_minus_1)`: 재귀 호출입니다.
   - `%result = arith.muli %n, %rec_result`: `n`과 재귀 결과를 곱합니다.
   - `return %result`: 결과를 반환합니다.

이 예제는 다음을 보여줍니다:

- **SSA form:** 모든 값(`%c0`, `%n_minus_1` 등)이 한 번만 정의됩니다.
- **Operation:** 상수, 비교, 산술, 함수 호출.
- **Region과 block:** 함수 본문은 세 개의 block(entry, `^base_case`, `^recursive_case`)을 가진 region입니다.
- **Terminator:** 모든 block이 terminator(`cf.cond_br` 또는 `return`)로 끝납니다.
- **Dialect:** `arith`, `func`, `cf` dialect을 사용합니다.

## 학습 내용 정리

이제 MLIR의 다섯 가지 핵심 개념을 이해하게 되었습니다:

1. **Dialect:** operation, 타입, attribute의 네임스페이스 (예: `arith`, `func`, `llvm`).
2. **Operation:** MLIR IR의 기본 단위 (예: `arith.addi`, `func.call`).
3. **Region:** block의 목록 (예: 함수 본문).
4. **Block:** terminator로 끝나는 operation 시퀀스 (예: 제어 흐름의 basic block).
5. **SSA form:** 모든 값이 정확히 한 번만 정의되며 불변.

구체적인 예제(산술, 제어 흐름, 재귀)를 통해 이 개념들이 어떻게 함께 작동하는지 살펴보았습니다. 또한 progressive lowering — IR을 한 번에 큰 점프가 아닌 단계적으로 변환하는 철학 — 을 이해하게 되었습니다.

## 다음 단계

다음 챕터에서는 MLIR IR을 생성하는 첫 번째 F# 프로그램을 작성합니다. P/Invoke를 사용하여 MLIR의 C API를 호출하고, 컴파일러의 "Hello, World"인 상수 정수를 반환하는 프로그램을 생성할 것입니다.

**Chapter 02: Hello MLIR from F#**로 계속됩니다 (작성 예정).

## 참고 자료

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) — MLIR의 텍스트 형식, dialect, 의미론에 대한 공식 사양.
- [Understanding MLIR IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) — operation, region, block에 대한 심층 분석.
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) — MLIR을 사용하여 "Toy" 언어의 컴파일러를 구축하는 완전한 튜토리얼.
- [Dialects Documentation](https://mlir.llvm.org/docs/Dialects/) — 내장 dialect(arith, func, scf, llvm 등)에 대한 참조 문서.
