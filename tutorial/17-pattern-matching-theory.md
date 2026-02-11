# Chapter 17: Pattern Matching Theory (Pattern Matching Theory)

## 소개

**Phase 6이 시작된다.** Phase 5에서 커스텀 MLIR dialect를 구축했다. `funlang.closure`와 `funlang.apply`로 클로저를 추상화했고, lowering pass로 LLVM dialect로 변환했다. 이제 **함수형 언어의 핵심 기능**을 추가할 시간이다: **패턴 매칭(pattern matching)**과 **데이터 구조(data structures)**.

### Phase 6 로드맵

**Phase 6: Pattern Matching & Data Structures**

이번 phase에서 구현할 내용:

1. **Chapter 17 (현재)**: Pattern matching 이론 - Decision tree 알고리즘
2. **Chapter 18**: List operations - `funlang.nil`, `funlang.cons` 구현
3. **Chapter 19**: Match compilation - `funlang.match` operation과 lowering
4. **Chapter 20**: Functional programs - 실전 예제 (map, filter, fold)

**왜 이 순서인가?**

- **이론 먼저:** Decision tree 알고리즘을 이해해야 MLIR 구현이 명확해진다
- **데이터 구조 다음:** List operations가 있어야 패턴 매칭할 대상이 생긴다
- **매칭 구현:** `funlang.match` operation으로 decision tree를 MLIR로 표현한다
- **실전 활용:** 지금까지 배운 모든 기능을 종합해서 함수형 프로그램을 작성한다

### Phase 5 복습: 왜 패턴 매칭이 필요한가?

Phase 5까지 우리는 이런 코드를 작성할 수 있게 되었다:

```fsharp
// F# compiler input
let make_adder n =
    fun x -> x + n

let add_5 = make_adder 5
let result = add_5 10  // 15
```

```mlir
// Phase 5 MLIR output (FunLang dialect)
%closure = funlang.closure @lambda_adder, %n : !funlang.closure
%result = funlang.apply %closure(%x) : (i32) -> i32
```

**하지만 함수형 언어의 진짜 힘은 데이터 구조와 패턴 매칭의 조합이다:**

```fsharp
// F#에서 list 패턴 매칭
let rec sum_list lst =
    match lst with
    | [] -> 0
    | head :: tail -> head + sum_list tail

sum_list [1; 2; 3]  // 6
```

```ocaml
(* OCaml에서 패턴 매칭 *)
let rec length = function
  | [] -> 0
  | _ :: tail -> 1 + length tail
```

**패턴 매칭이 제공하는 것:**

1. **구조적 분해(structural decomposition)**: 데이터를 한 번에 분해하고 변수에 바인딩
2. **Exhaustiveness checking**: 컴파일러가 모든 경우를 다뤘는지 검증
3. **효율적인 분기**: 각 subterm을 최대 한 번만 테스트하는 코드 생성
4. **가독성**: if-else 체인보다 선언적이고 명확한 코드

### Pattern Matching Compilation의 도전

**Naive한 접근:**

```mlir
// 잘못된 방법: if-else 체인으로 번역
%is_nil = // list가 nil인지 테스트
scf.if %is_nil {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
} else {
    %is_cons = // list가 cons인지 테스트 (중복!)
    scf.if %is_cons {
        %head = // head 추출
        %tail = // tail 추출
        %sum_tail = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
        %result = arith.addi %head, %sum_tail : i32
        scf.yield %result : i32
    }
}
```

**문제점:**

1. **중복 테스트**: Nil 테스트 실패 후 Cons 테스트는 중복이다 (list는 Nil 아니면 Cons)
2. **비효율적 코드**: Nested patterns에서 exponential blowup 발생
3. **Exhaustiveness 검증 어려움**: 모든 case를 다뤘는지 확인이 복잡하다

**올바른 접근: Decision Tree Compilation**

Luc Maranget의 decision tree 알고리즘 (2008)을 사용하면:

- 각 subterm을 **최대 한 번**만 테스트
- Pattern matrix representation으로 체계적 변환
- Exhaustiveness checking이 자연스럽게 통합됨
- 최적화된 분기 코드 생성

### Chapter 17 목표

이 장을 마치면:

- [ ] **Pattern matrix 표현법**을 이해한다
- [ ] **Decision tree 알고리즘**의 동작 원리를 안다
- [ ] **Specialization**과 **defaulting** 연산을 설명할 수 있다
- [ ] **Exhaustiveness checking**이 어떻게 동작하는지 안다
- [ ] Chapter 18-19에서 MLIR 구현을 시작할 준비가 된다

**이론 중심 장(theory-focused chapter):**

이 장은 구현 코드가 없다. **알고리즘 설명과 예제**에 집중한다. 왜냐하면:

1. Decision tree 알고리즘은 MLIR과 독립적이다 (OCaml, Haskell, Rust 등 모든 함수형 언어에서 사용)
2. 알고리즘을 먼저 이해하면 MLIR lowering 구현이 명확해진다
3. Pattern matrix 표기법은 Chapter 19의 `funlang.match` operation 설계 기반이 된다

### 성공 기준

이 장을 이해했다면:

1. Pattern matrix에서 rows/columns가 무엇을 의미하는지 설명할 수 있다
2. Specialization 연산이 pattern을 어떻게 분해하는지 예시를 들 수 있다
3. Default 연산이 wildcard rows를 어떻게 처리하는지 설명할 수 있다
4. Empty pattern matrix가 왜 non-exhaustive match를 의미하는지 안다
5. Decision tree가 if-else chain보다 효율적인 이유를 설명할 수 있다

**Let's begin.**

---

## Pattern Matching 문제 정의

패턴 매칭 컴파일의 핵심 문제를 정의하자.

### ML 계열 언어의 패턴 매칭

**OCaml/F# syntax:**

```ocaml
(* OCaml *)
match expr with
| pattern1 -> action1
| pattern2 -> action2
| pattern3 -> action3
```

```fsharp
// F#
match expr with
| pattern1 -> action1
| pattern2 -> action2
| pattern3 -> action3
```

**Example: List length function**

```ocaml
let rec length lst =
  match lst with
  | [] -> 0
  | _ :: tail -> 1 + length tail
```

**구성 요소:**

1. **Scrutinee** (`lst`): 매칭 대상 expression
2. **Patterns** (`[]`, `_ :: tail`): 구조 템플릿
3. **Actions** (`0`, `1 + length tail`): 패턴이 매칭되면 실행할 코드
4. **Pattern variables** (`tail`): 패턴 내부에서 값을 바인딩

### FunLang 패턴 매칭 구문

**FunLang의 제안 syntax** (Phase 6 구현 목표):

```fsharp
// match expression
match list with
| Nil -> 0
| Cons(head, tail) -> head + sum tail
```

**Pattern types:**

1. **Wildcard pattern** (`_`): 모든 값과 매칭, 변수 바인딩 없음
2. **Variable pattern** (`x`, `tail`): 모든 값과 매칭, 변수에 바인딩
3. **Constructor pattern** (`Nil`, `Cons(x, xs)`): 특정 constructor와 매칭
4. **Literal pattern** (`0`, `true`): 특정 상수 값과 매칭

**Constructor patterns의 subpatterns:**

```fsharp
// Nested constructor patterns
match list with
| Nil -> "empty"
| Cons(x, Nil) -> "singleton"  // tail is Nil
| Cons(x, Cons(y, rest)) -> "at least two elements"
```

`Cons(x, Nil)`에서:
- `Cons`는 constructor
- `x`는 head subpattern (variable)
- `Nil`은 tail subpattern (constructor)

### 컴파일 문제: Patterns → Efficient Branching Code

**Input:** Pattern clauses (scrutinee, patterns, actions)

```fsharp
match list with
| Nil -> 0
| Cons(head, tail) -> head + sum tail
```

**Output:** Efficient branching code (MLIR IR)

```mlir
%tag = llvm.extractvalue %list[0] : !llvm.struct<(i32, ptr)>
%is_nil = arith.cmpi eq, %tag, %c0 : i32

%result = scf.if %is_nil -> (i32) {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
} else {
    // Cons case: extract head and tail
    %data = llvm.extractvalue %list[1] : !llvm.struct<(i32, ptr)>
    %head = llvm.load %data[0] : !llvm.ptr -> i32
    %tail = llvm.load %data[1] : !llvm.ptr -> !llvm.struct<(i32, ptr)>

    %sum_tail = func.call @sum(%tail) : (!funlang.list<i32>) -> i32
    %result_val = arith.addi %head, %sum_tail : i32
    scf.yield %result_val : i32
}
```

**핵심 요구사항:**

1. **Correctness**: 패턴 순서를 존중 (첫 번째 매칭 패턴이 선택됨)
2. **Efficiency**: 각 subterm을 최대 한 번만 테스트
3. **Exhaustiveness**: 모든 가능한 값이 처리되는지 검증
4. **Optimization**: 불필요한 테스트 제거 (Nil 아니면 자동으로 Cons)

### Naive 컴파일의 문제점

**If-else chain으로 직접 번역하면?**

```mlir
// Pattern 1: Nil
%is_nil = // test if tag == 0
scf.if %is_nil {
    scf.yield %zero : i32
} else {
    // Pattern 2: Cons(head, tail)
    %is_cons = // test if tag == 1 (redundant!)
    scf.if %is_cons {
        // extract head, tail, compute result
    } else {
        // No more patterns -> error!
    }
}
```

**문제 1: 중복 테스트**

Nil 테스트가 false면 자동으로 Cons다 (list는 Nil 또는 Cons만 존재). 하지만 naive 번역은 다시 Cons를 테스트한다.

**문제 2: Nested patterns의 exponential blowup**

```fsharp
match (list1, list2) with
| (Nil, Nil) -> 0
| (Nil, Cons(_, _)) -> 1
| (Cons(_, _), Nil) -> 2
| (Cons(x, _), Cons(y, _)) -> x + y
```

두 개의 scrutinee를 독립적으로 테스트하면:

```
list1 test -> list2 test (중복!)
            -> list2 test (중복!)
-> list1 test (중복!)
            -> list2 test (중복!)
            -> list2 test (중복!)
```

4개의 패턴이 8번의 테스트를 발생시킨다. Patterns이 늘어나면 `2^n` 테스트가 필요하다.

**문제 3: Exhaustiveness 검증 복잡도**

If-else tree를 분석해서 모든 경로가 종료되는지 확인해야 한다. 복잡한 중첩 패턴에서는 거의 불가능하다.

### 해결책: Decision Tree Compilation

**Key insight (Maranget 2008):**

> "패턴 매칭은 **search problem**이다. Pattern clauses를 **structured representation** (pattern matrix)로 변환하면, systematic하게 optimal decision tree를 구성할 수 있다."

**Decision tree 특징:**

1. **각 internal node는 하나의 test** (constructor tag, literal value)
2. **각 edge는 test outcome** (Nil vs Cons, 0 vs 1 vs 2)
3. **각 leaf는 action** (실행할 코드)
4. **Root에서 leaf까지 경로는 unique test sequence**

**장점:**

- 각 subterm을 최대 한 번만 테스트 (no redundancy)
- Test 순서를 최적화 가능 (heuristic으로 선택)
- Exhaustiveness checking이 자연스러움 (leaf가 없는 경로 = missing pattern)

**다음 섹션에서:** Pattern matrix 표기법과 decision tree 구성 알고리즘을 자세히 살펴본다.

---

## Pattern Matrix 표현법

Decision tree 알고리즘의 핵심은 **pattern matrix**라는 structured representation이다.

### Pattern Matrix 정의

**Pattern matrix는 2차원 테이블이다:**

- **Rows**: Pattern clauses (각 row는 하나의 `pattern -> action`)
- **Columns**: Scrutinees (매칭 대상 values)
- **Cells**: Patterns (wildcard, constructor, literal)

**Notation:**

```
P = | p11  p12  ...  p1m  →  a1
    | p21  p22  ...  p2m  →  a2
    | ...
    | pn1  pn2  ...  pnm  →  an
```

- `P`: Pattern matrix (n rows × m columns)
- `pij`: Row i, column j의 pattern
- `ai`: Row i의 action
- `m`: Scrutinee 개수
- `n`: Pattern clause 개수

### Example 1: 단일 Scrutinee (List Length)

**FunLang code:**

```fsharp
match list with
| Nil -> 0
| Cons(head, tail) -> 1 + length tail
```

**Pattern matrix:**

```
Scrutinee: [list]

Matrix:
| Nil             →  0
| Cons(head, tail) →  1 + length tail
```

**설명:**

- 1개의 scrutinee column: `list`
- 2개의 pattern rows:
  - Row 1: `Nil` pattern → action은 `0`
  - Row 2: `Cons(head, tail)` pattern → action은 `1 + length tail`

**Constructor patterns의 subpatterns:**

`Cons(head, tail)`은 2개의 subpatterns를 가진다:
- `head`: variable pattern (head 값에 바인딩)
- `tail`: variable pattern (tail 값에 바인딩)

나중에 이 subpatterns가 **새로운 columns**로 확장된다 (specialization).

### Example 2: 다중 Scrutinee (Pair Matching)

**FunLang code:**

```fsharp
match (list1, list2) with
| (Nil, Nil) -> 0
| (Nil, Cons(x, _)) -> 1
| (Cons(_, _), Nil) -> 2
| (Cons(x, _), Cons(y, _)) -> x + y
```

**Pattern matrix:**

```
Scrutinee: [list1, list2]

Matrix:
| Nil         Nil          →  0
| Nil         Cons(x, _)   →  1
| Cons(_, _)  Nil          →  2
| Cons(x, _)  Cons(y, _)   →  x + y
```

**설명:**

- 2개의 scrutinee columns: `list1`, `list2`
- 4개의 pattern rows
- 각 cell은 해당 scrutinee의 pattern

**Wildcard pattern `_`:**

값을 바인딩하지 않는 pattern. 모든 값과 매칭된다.

**Variable pattern `x`, `y`:**

값을 변수에 바인딩하는 pattern. 모든 값과 매칭되지만 이름을 부여한다.

> **Wildcard vs Variable**: Semantically 둘 다 모든 값과 매칭된다. Variable은 추가로 바인딩을 생성한다. Pattern matrix 관점에서는 동일하게 취급된다 (irrefutable pattern).

### Example 3: Nested Pattern (List Prefix)

**FunLang code:**

```fsharp
match list with
| Nil -> "empty"
| Cons(x, Nil) -> "singleton"
| Cons(x, Cons(y, rest)) -> "at least two"
```

**Initial pattern matrix:**

```
Scrutinee: [list]

Matrix:
| Nil                   →  "empty"
| Cons(x, Nil)          →  "singleton"
| Cons(x, Cons(y, rest)) →  "at least two"
```

**Nested constructor `Cons(y, rest)`:**

Row 3의 tail subpattern `Cons(y, rest)`는 또 다른 constructor pattern이다. 이게 **nested pattern**이다.

**Compilation strategy:**

1. 먼저 `list`의 constructor (Nil vs Cons) 테스트
2. Cons인 경우, head와 tail 추출
3. 이제 `tail`에 대해 다시 pattern matching (Nil vs Cons)

**Specialization 후 matrix는 확장된다** (나중에 자세히 설명).

### Occurrence Vectors

**Pattern matrix와 함께 occurrence vectors를 유지한다.**

**Occurrence vector (π):**

Scrutinee values에 **어떻게 접근하는지** 나타내는 경로(path) 목록.

**Initial occurrences:**

```
π = [o1, o2, ..., om]
```

- `o1`: First scrutinee (예: `list`)
- `o2`: Second scrutinee (예: `list2`)

**Example: Single scrutinee**

```
π = [list]
```

**Example: Pair of scrutinees**

```
π = [list1, list2]
```

**Specialization 시 occurrences 확장:**

Constructor pattern `Cons(x, xs)`를 specialize하면:

```
π = [list]
  → [list.head, list.tail]
```

`list.head`와 `list.tail`은 **subterm access path**를 의미한다 (MLIR에서는 `llvm.extractvalue` operations).

**왜 occurrence vectors가 필요한가?**

Decision tree를 생성할 때, 각 test가 **어느 값을 검사하는지** 알아야 한다.

- Initial: `list` 자체를 테스트
- After specialization: `list.head`, `list.tail`을 테스트

Occurrence vectors는 **code generation의 기반**이다.

### Pattern Matrix Properties

**Irrefutable row:**

Row의 모든 patterns가 wildcard 또는 variable이면 **irrefutable**이다 (항상 매칭).

```
| _  _  _  →  action  // Irrefutable
```

**Exhaustive matrix:**

Matrix가 **exhaustive**하면 모든 가능한 input values가 어떤 row와 매칭된다.

**Non-exhaustive matrix:**

어떤 input value도 매칭되지 않는 경우가 있으면 **non-exhaustive**.

**Empty matrix (P = ∅):**

Row가 하나도 없는 matrix. **항상 non-exhaustive**다.

**Example: Non-exhaustive pattern**

```fsharp
match list with
| Nil -> 0
// Missing: Cons case!
```

Matrix:
```
| Nil  →  0
```

Input `Cons(1, Nil)`은 어떤 row와도 매칭 안 됨 → **non-exhaustive**.

### Pattern Matrix Compilation Goal

**Compilation algorithm의 목표:**

Pattern matrix `P`와 occurrence vector `π`를 입력받아서:

1. **Decision tree를 생성**한다 (efficient branching code)
2. **Exhaustiveness를 검증**한다 (empty matrix 체크)
3. **Optimal test sequence**를 선택한다 (heuristic)

**Next section:** Decision tree의 구조와 pattern matrix의 관계를 살펴본다.

---

## Decision Tree 개념

Pattern matrix를 compile하면 **decision tree**가 생성된다. 이 섹션에서 decision tree의 구조와 특징을 이해한다.

### Decision Tree 구조

**Decision tree는 다음 요소로 구성된다:**

1. **Internal nodes (decision nodes)**: Test operations
   - Constructor test: "Is this value Nil or Cons?"
   - Literal test: "Is this value 0 or 1 or 2?"
2. **Edges**: Test outcomes (branches)
   - Constructor edges: Nil branch, Cons branch
   - Literal edges: 0 branch, 1 branch, default branch
3. **Leaf nodes**: Actions
   - Success leaf: Execute action (return value)
   - Failure leaf: Match failure (non-exhaustive error)

**Tree traversal:**

- Root에서 시작
- 각 internal node에서 test 실행
- Test outcome에 따라 edge 선택
- Leaf에 도달하면 종료 (action 실행 또는 failure)

### Example: List Length Decision Tree

**Pattern matrix:**

```
| Nil             →  a1 (return 0)
| Cons(head, tail) →  a2 (return 1 + length tail)
```

**Decision tree:**

```
       [list]
         |
    Test: constructor
       /   \
     Nil   Cons
     /       \
   Leaf     [head, tail]
   a1          |
             Leaf
              a2
```

**Tree 설명:**

1. **Root node**: `list`의 constructor 테스트
2. **Nil edge**: Nil constructor → Leaf (action a1)
3. **Cons edge**: Cons constructor → Intermediate node (head, tail 추출)
4. **Cons leaf**: Action a2 실행

**왜 [head, tail] node가 필요한가?**

Cons pattern `Cons(head, tail)`은 subpatterns를 가진다. Cons case에서:
- `head` 값을 추출해서 변수 `head`에 바인딩
- `tail` 값을 추출해서 변수 `tail`에 바인딩

이 바인딩들이 action a2에서 사용된다.

**Simplified view (bindings 생략):**

```
       [list]
         |
    Test: constructor
       /   \
     Nil   Cons
     /       \
   a1        a2
```

구현에서는 Cons branch에서 head/tail 추출 코드를 삽입한다.

### Example: Nested Pattern Decision Tree

**Pattern matrix:**

```
| Nil                   →  a1 ("empty")
| Cons(x, Nil)          →  a2 ("singleton")
| Cons(x, Cons(y, rest)) →  a3 ("at least two")
```

**Decision tree:**

```
          [list]
            |
       Test: constructor
         /   \
       Nil   Cons
       /       \
     a1      [head, tail]
                |
          Test: tail constructor
              /   \
            Nil   Cons
            /       \
          a2      [y, rest]
                     |
                    a3
```

**Tree traversal example:**

Input: `Cons(1, Cons(2, Nil))`

1. Root: Test `list` constructor → Cons
2. Extract `head = 1`, `tail = Cons(2, Nil)`
3. Test `tail` constructor → Cons
4. Extract `y = 2`, `rest = Nil`
5. Leaf a3 ("at least two")

**Key property: 각 subterm을 한 번만 테스트**

- `list` constructor: 1번 테스트
- `tail` constructor: 1번 테스트

Naive if-else chain은 `list` constructor를 여러 번 테스트할 수 있다.

### Comparison: Decision Tree vs If-Else Chain

**If-Else chain (naive compilation):**

```mlir
// Pattern 1: Nil
%is_nil = arith.cmpi eq, %tag, %c0 : i32
scf.if %is_nil {
    scf.yield %a1 : i32
} else {
    // Pattern 2: Cons(x, Nil)
    %is_cons = arith.cmpi eq, %tag, %c1 : i32  // Redundant test!
    scf.if %is_cons {
        %tail = // extract tail
        %tail_tag = llvm.extractvalue %tail[0] : !llvm.struct<(i32, ptr)>
        %tail_is_nil = arith.cmpi eq, %tail_tag, %c0 : i32
        scf.if %tail_is_nil {
            scf.yield %a2 : i32
        } else {
            // Pattern 3: Cons(x, Cons(y, rest))
            // ... (more tests)
        }
    }
}
```

**문제:**

1. `%is_cons` test는 중복 (Nil이 아니면 자동으로 Cons)
2. Nested if-else는 depth가 깊어진다
3. 각 level에서 동일한 값을 반복 테스트

**Decision tree (optimal compilation):**

```mlir
// Test list constructor once
%tag = llvm.extractvalue %list[0] : !llvm.struct<(i32, ptr)>
%result = scf.index_switch %tag : i32 -> i32
case 0 {  // Nil
    scf.yield %a1 : i32
}
case 1 {  // Cons
    %data = llvm.extractvalue %list[1] : !llvm.struct<(i32, ptr)>
    %head = llvm.load %data[0] : !llvm.ptr -> i32
    %tail_ptr = llvm.getelementptr %data[1] : (!llvm.ptr) -> !llvm.ptr
    %tail = llvm.load %tail_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>

    // Test tail constructor once
    %tail_tag = llvm.extractvalue %tail[0] : !llvm.struct<(i32, ptr)>
    %tail_result = scf.index_switch %tail_tag : i32 -> i32
    case 0 {  // Nil
        scf.yield %a2 : i32
    }
    case 1 {  // Cons
        %tail_data = llvm.extractvalue %tail[1] : !llvm.struct<(i32, ptr)>
        %y = llvm.load %tail_data[0] : !llvm.ptr -> i32
        %rest_ptr = llvm.getelementptr %tail_data[1] : (!llvm.ptr) -> !llvm.ptr
        %rest = llvm.load %rest_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>
        scf.yield %a3 : i32
    }
    scf.yield %tail_result : i32
}
```

**장점:**

1. 각 constructor tag를 정확히 한 번만 테스트 (`scf.index_switch`)
2. 불필요한 비교 연산 제거
3. Structured control flow (SCF dialect)로 최적화 기회 제공

### Decision Tree Benefits

**1. Efficiency: O(d) tests (d = pattern depth)**

Nested pattern의 depth가 d면, 최대 d번의 test만 필요하다.

- Flat pattern (`Nil`, `Cons(_, _)`): 1번 test
- Nested pattern (`Cons(_, Cons(_, _))`): 2번 test (outer, inner)

If-else chain은 worst case O(n × d) tests (n = pattern 개수).

**2. Exhaustiveness checking: Leaf coverage**

모든 가능한 input이 어떤 leaf에 도달하면 exhaustive.

Leaf에 도달하지 않는 경로가 있으면 non-exhaustive.

**Example: Non-exhaustive detection**

```
Pattern matrix:
| Nil  →  a1
// Missing Cons case
```

Decision tree:
```
    [list]
      |
  Test: constructor
    /   \
  Nil   Cons
  /       \
a1      FAILURE  // No action for Cons
```

Cons branch가 Failure leaf로 이어진다 → Compile error: "non-exhaustive match"

**3. Optimization opportunities**

Decision tree는 structured representation이라서:

- Common subexpression elimination (같은 test를 여러 번 안 함)
- Dead code elimination (도달 불가능한 patterns 제거)
- Branch prediction hints (frequent cases 먼저 테스트)

### Relationship: Pattern Matrix → Decision Tree

**Compilation function:**

```
compile : PatternMatrix × OccurrenceVector → DecisionTree
```

**Input:**

- Pattern matrix `P` (n rows × m columns)
- Occurrence vector `π` (m elements)

**Output:**

- Decision tree `T`

**Recursive algorithm:**

```
function compile(P, π):
    if P is empty:
        return Failure  // Non-exhaustive

    if first row is irrefutable:
        return Success(action)  // Found match

    column = select_column(P)
    constructors = get_constructors(P, column)

    branches = {}
    for each constructor c:
        P_c = specialize(P, column, c)
        π_c = specialize_occurrences(π, column, c)
        branches[c] = compile(P_c, π_c)

    P_default = default(P, column)
    π_default = default_occurrences(π, column)
    default_branch = compile(P_default, π_default)

    return Switch(π[column], branches, default_branch)
```

**핵심 operations:**

1. **`select_column`**: 어느 column을 먼저 테스트할지 선택 (heuristic)
2. **`specialize`**: Constructor와 매칭되는 rows만 남기고, subpatterns 확장
3. **`default`**: Wildcard rows만 남기고, 테스트한 column 제거

**Next sections:** Specialization과 defaulting을 자세히 설명한다.

---

## Specialization 연산

Specialization은 decision tree 알고리즘의 **핵심 operation**이다. Constructor test가 성공했을 때 pattern matrix를 어떻게 변환하는지 정의한다.

### Specialization 정의

**Specialization (S):**

```
S(c, i, P) = Specialized pattern matrix
```

**Parameters:**

- `c`: Constructor (예: `Cons`, `Nil`)
- `i`: Column index (어느 scrutinee를 테스트하는가)
- `P`: Original pattern matrix

**Operation:**

1. Column `i`의 pattern이 constructor `c`와 **호환되는** rows만 유지
2. 호환되는 patterns를 **subpatterns로 확장** (constructor decomposition)
3. Column `i`를 제거하고 subpattern columns를 삽입

### Example 1: Simple List Specialization (Cons)

**Original pattern matrix:**

```
Column: [list]

| Nil             →  a1
| Cons(head, tail) →  a2
| _               →  a3
```

**Specialize on column 0, constructor Cons:**

`S(Cons, 0, P)`:

**Step 1: Filter compatible rows**

- Row 1 (`Nil`): Incompatible with Cons → 제거
- Row 2 (`Cons(head, tail)`): Compatible → 유지
- Row 3 (`_`): Wildcard, compatible → 유지

**Step 2: Decompose patterns**

- Row 2: `Cons(head, tail)` → expand to `[head, tail]`
- Row 3: `_` → expand to `[_, _]` (wildcard for each subpattern)

**Step 3: Replace column 0 with subpattern columns**

```
Columns: [head, tail]

| head  tail  →  a2
| _     _     →  a3
```

**Occurrence vector update:**

```
Before: π = [list]
After:  π = [list.head, list.tail]
```

### Example 2: Specialization on Nil

**Original pattern matrix:**

```
Column: [list]

| Nil             →  a1
| Cons(head, tail) →  a2
| _               →  a3
```

**Specialize on column 0, constructor Nil:**

`S(Nil, 0, P)`:

**Step 1: Filter compatible rows**

- Row 1 (`Nil`): Compatible → 유지
- Row 2 (`Cons(head, tail)`): Incompatible with Nil → 제거
- Row 3 (`_`): Wildcard, compatible → 유지

**Step 2: Decompose patterns**

Nil constructor는 **subpatterns가 없다** (nullary constructor).

- Row 1: `Nil` → no subpatterns
- Row 3: `_` → no subpatterns

**Step 3: Remove column 0 (no subpatterns to add)**

```
Columns: [] (empty)

| →  a1
| →  a3
```

**Occurrence vector update:**

```
Before: π = [list]
After:  π = [] (empty)
```

Empty occurrence vector는 **모든 tests가 완료**되었음을 의미. 이제 첫 번째 row의 action을 선택한다.

### Example 3: Nested Pattern Specialization

**Original pattern matrix:**

```
Column: [list]

| Cons(x, Nil)          →  a1
| Cons(x, Cons(y, rest)) →  a2
```

**Specialize on column 0, constructor Cons:**

`S(Cons, 0, P)`:

**Step 1: Filter compatible rows**

Both rows have `Cons` → 둘 다 유지

**Step 2: Decompose patterns**

- Row 1: `Cons(x, Nil)` → subpatterns `[x, Nil]`
- Row 2: `Cons(x, Cons(y, rest))` → subpatterns `[x, Cons(y, rest)]`

**Step 3: Replace column 0 with subpattern columns**

```
Columns: [head, tail]

| x  Nil              →  a1
| x  Cons(y, rest)    →  a2
```

**Occurrence vector update:**

```
Before: π = [list]
After:  π = [list.head, list.tail]
```

**이제 column 1 (tail)에 대해 다시 specialization:**

Matrix after first specialization:
```
| x  Nil              →  a1
| x  Cons(y, rest)    →  a2
```

Specialize on column 1, constructor Nil:

```
Columns: [head]

| x  →  a1
```

Specialize on column 1, constructor Cons:

```
Columns: [head, y, rest]

| x  y  rest  →  a2
```

**Nested patterns는 여러 번의 specialization으로 처리된다.**

### Wildcard Expansion Rule

**Wildcard pattern `_`의 specialization:**

Constructor `c`가 arity `n` (subpatterns 개수)를 가지면:

```
_ → [_, _, ..., _]  (n개의 wildcards)
```

**Example: Cons constructor (arity 2)**

```
_ → [_, _]  // head wildcard, tail wildcard
```

**Example: Nil constructor (arity 0)**

```
_ → []  // No subpatterns
```

**왜 wildcard를 확장하는가?**

Wildcard는 "모든 값과 매칭"을 의미한다. Constructor `c`와 매칭되면, `c`의 모든 subpatterns도 wildcard로 매칭된다.

```fsharp
// Original pattern
| _ -> action

// After specialization on Cons
// Equivalent to:
| Cons(_, _) -> action
```

### Variable Pattern Specialization

**Variable pattern `x`의 specialization:**

Variable은 wildcard와 동일하게 확장되지만, **binding name을 유지**한다.

```
x → [_, _, ..., _]  // Subpatterns, 하지만 x는 여전히 전체 값에 바인딩됨
```

**Example:**

```fsharp
match list with
| xs -> length xs  // xs는 전체 list에 바인딩
```

Specialize on Cons:

```
Columns: [head, tail]

| _  _  →  length (Cons head tail)
```

`xs` 바인딩은 **original occurrence**에 남는다. Specialization 후에도 `xs`는 사용 가능하다.

> **Implementation note:** Variable bindings는 pattern matrix에 직접 저장되지 않고, **occurrence vector와 함께 관리**된다. Action에서 variable을 사용할 때 occurrence path로 접근한다.

### Specialization Pseudocode

**Algorithm: specialize(P, column, constructor)**

```python
def specialize(P, column, constructor):
    """
    P: Pattern matrix (n rows × m columns)
    column: Column index to specialize
    constructor: Constructor to match (e.g., Cons, Nil)

    Returns: Specialized matrix
    """
    result_rows = []
    arity = get_arity(constructor)  // Subpattern 개수

    for row in P:
        pattern = row[column]

        if matches_constructor(pattern, constructor):
            # Compatible pattern
            if pattern.is_constructor and pattern.name == constructor:
                # Extract subpatterns
                subpatterns = pattern.subpatterns  // e.g., [head, tail]
            elif pattern.is_wildcard or pattern.is_variable:
                # Expand to wildcard subpatterns
                subpatterns = [Wildcard] * arity  // e.g., [_, _]
            else:
                # Incompatible (different constructor)
                continue  # Skip this row

            # Build new row: columns before + subpatterns + columns after
            new_row = (
                row[:column] +
                subpatterns +
                row[column+1:]
            )
            result_rows.append((new_row, row.action))

    return PatternMatrix(result_rows)

def matches_constructor(pattern, constructor):
    """Check if pattern is compatible with constructor"""
    if pattern.is_wildcard or pattern.is_variable:
        return True  # Wildcard matches everything
    if pattern.is_constructor and pattern.name == constructor:
        return True  # Same constructor
    return False  # Different constructor
```

### Visual Example: Specialization Flow

**Original:**

```
   [list]
     |
| Nil        →  a1
| Cons(x, y) →  a2
| _          →  a3
```

**After S(Cons, 0, P):**

```
   [x, y]  (head, tail)
     |
| x  y  →  a2  (from Cons(x, y))
| _  _  →  a3  (from _)
```

Row 1 (`Nil`) 제거됨 (incompatible).

**After S(Nil, 0, P) on original:**

```
   []  (no occurrences)
    |
| →  a1  (from Nil)
| →  a3  (from _)
```

Rows 2 (`Cons`) 제거됨 (incompatible).

### Key Insight: Specialization = Assumption + Decomposition

**Specialization의 의미:**

> "Column `i`의 constructor가 `c`라고 **가정**하면, pattern matrix는 어떻게 변하는가?"

**Assumption:**

- Constructor test가 성공했다 (e.g., `list`가 `Cons`)
- 이제 `c`의 subpatterns에 접근 가능 (e.g., `head`, `tail`)

**Decomposition:**

- 호환되지 않는 rows 제거 (Nil patterns)
- 호환되는 rows의 patterns를 subpatterns로 확장

**Next:** Defaulting 연산은 반대 상황을 다룬다 (constructor test 실패).

---

## Defaulting 연산

Defaulting은 specialization의 **complement**다. Constructor test가 **실패**했을 때 (또는 테스트하지 않고 default case로 가려 할 때) pattern matrix를 어떻게 변환하는지 정의한다.

### Defaulting 정의

**Defaulting (D):**

```
D(i, P) = Default pattern matrix
```

**Parameters:**

- `i`: Column index
- `P`: Original pattern matrix

**Operation:**

1. Column `i`의 pattern이 **wildcard 또는 variable**인 rows만 유지
2. Column `i`를 **제거** (더 이상 테스트 안 함)
3. 나머지 columns는 유지

**의미:**

> "Column `i`에 대한 모든 constructor tests가 실패했다. Wildcard rows만 남는다."

### Example 1: Simple List Defaulting

**Original pattern matrix:**

```
Column: [list]

| Nil             →  a1
| Cons(head, tail) →  a2
| _               →  a3
```

**Default on column 0:**

`D(0, P)`:

**Step 1: Filter wildcard rows**

- Row 1 (`Nil`): Constructor pattern → 제거
- Row 2 (`Cons(head, tail)`): Constructor pattern → 제거
- Row 3 (`_`): Wildcard → 유지

**Step 2: Remove column 0**

```
Columns: [] (empty)

| →  a3
```

**Occurrence vector update:**

```
Before: π = [list]
After:  π = [] (empty)
```

Empty matrix with one row → Irrefutable → Select action a3.

### Example 2: Empty Default Matrix

**Original pattern matrix:**

```
Column: [list]

| Nil             →  a1
| Cons(head, tail) →  a2
```

**Default on column 0:**

`D(0, P)`:

**Step 1: Filter wildcard rows**

- Row 1 (`Nil`): Constructor pattern → 제거
- Row 2 (`Cons(head, tail)`): Constructor pattern → 제거

**Result: Empty matrix**

```
Columns: []

(no rows)
```

**의미: Non-exhaustive match!**

모든 rows가 constructor patterns이면, defaulting은 empty matrix를 생성한다. 즉, wildcard case가 없다 → Non-exhaustive.

**Compiler action:**

Empty default matrix는 **compile error**를 발생시킨다:

```
Error: Non-exhaustive pattern match
Missing case: (other constructors or wildcard)
```

### Example 3: Multiple Columns Defaulting

**Original pattern matrix:**

```
Columns: [list1, list2]

| Nil         Nil          →  a1
| Nil         Cons(x, _)   →  a2
| Cons(_, _)  Nil          →  a3
| Cons(x, _)  Cons(y, _)   →  a4
| _           _            →  a5
```

**Default on column 0:**

`D(0, P)`:

**Step 1: Filter wildcard rows on column 0**

- Row 1 (`Nil`): Constructor → 제거
- Row 2 (`Nil`): Constructor → 제거
- Row 3 (`Cons(_, _)`): Constructor → 제거
- Row 4 (`Cons(x, _)`): Constructor → 제거
- Row 5 (`_`): Wildcard → 유지

**Step 2: Remove column 0**

```
Columns: [list2]

| _  →  a5
```

**Occurrence vector update:**

```
Before: π = [list1, list2]
After:  π = [list2]
```

**이제 column 0 (이전 list2)에 대해 specialization 또는 defaulting을 계속할 수 있다.**

### Defaulting vs Specialization: When to Use

**Specialization:**

Constructor test가 **성공**했을 때.

```
if (tag == CONS) {
    // Specialize on Cons
    S(Cons, 0, P)
}
```

**Defaulting:**

모든 constructor tests가 **실패**했을 때.

```
if (tag == NIL) {
    S(Nil, 0, P)
} else if (tag == CONS) {
    S(Cons, 0, P)
} else {
    // Default case
    D(0, P)
}
```

**하지만 list는 Nil 또는 Cons만 존재한다!**

완전한 constructor set (Nil, Cons)을 모두 테스트하면 default case는 unreachable이다.

**Defaulting이 필요한 경우:**

1. **Extensible constructors**: Open constructor sets (예: integers)
2. **Incomplete specialization**: 일부 constructors만 테스트
3. **Wildcard-only rows**: 모든 constructors 후 남은 wildcard 처리

**List의 경우 (closed constructor set):**

```
if (tag == NIL) {
    S(Nil, 0, P)
} else {
    // Must be CONS (only two constructors)
    S(Cons, 0, P)
}
```

Default branch는 필요 없다. 하지만 algorithm에서는 여전히 defaulting을 계산해서 **exhaustiveness를 체크**한다.

### Defaulting Empty Matrix Detection

**Defaulting의 중요한 역할: Exhaustiveness checking**

**Case 1: Non-empty default matrix**

```
Pattern matrix:
| Cons(x, xs)  →  a1
| _            →  a2  // Wildcard exists
```

Default on column 0:
```
| →  a2  // Non-empty
```

**Result: Exhaustive** (wildcard catches everything)

**Case 2: Empty default matrix**

```
Pattern matrix:
| Cons(x, xs)  →  a1
// No wildcard
```

Default on column 0:
```
(empty matrix)
```

**Result: Non-exhaustive** (missing Nil case and wildcard)

**Compiler error:**

```
Error: Non-exhaustive pattern match
Missing case: Nil
```

### Defaulting Pseudocode

**Algorithm: default(P, column)**

```python
def default(P, column):
    """
    P: Pattern matrix (n rows × m columns)
    column: Column index to default

    Returns: Default matrix (wildcard rows only, column removed)
    """
    result_rows = []

    for row in P:
        pattern = row[column]

        if pattern.is_wildcard or pattern.is_variable:
            # Wildcard row: keep it, remove column
            new_row = row[:column] + row[column+1:]
            result_rows.append((new_row, row.action))
        else:
            # Constructor pattern: remove this row
            continue

    return PatternMatrix(result_rows)
```

**Simplicity:**

Defaulting은 specialization보다 간단하다:
- No subpattern expansion
- Just filter wildcard rows and remove column

### Visual Example: Defaulting Flow

**Original:**

```
   [list]
     |
| Nil        →  a1
| Cons(x, y) →  a2
| _          →  a3
```

**After D(0, P):**

```
   []  (no occurrences)
    |
| →  a3  (from _)
```

Rows 1 (`Nil`) and 2 (`Cons`) 제거됨 (constructor patterns).

**Empty default example:**

```
   [list]
     |
| Nil        →  a1
| Cons(x, y) →  a2
```

**After D(0, P):**

```
   []
    |
(empty - no wildcard rows)
```

**Compiler:** "Error: Non-exhaustive match"

### Key Insight: Defaulting = Catch-All Case

**Defaulting의 의미:**

> "모든 명시적 constructor tests가 실패했다. 남은 rows는 wildcard만 있다. Wildcard는 **catch-all**이다."

**Properties:**

1. **Default matrix는 항상 wildcards만 포함** (constructors 제거됨)
2. **Empty default matrix = non-exhaustive** (catch-all 없음)
3. **Default 후 irrefutable row가 남으면 항상 매칭** (first wildcard row 선택)

**Next:** Specialization과 defaulting을 결합해서 complete compilation algorithm을 만든다.

