# Chapter 20: Functional Programs (Functional Programs)

## 소개

**Phase 6의 여정을 복습하자:**

**Chapter 17: Pattern Matching Theory**에서는 패턴 매칭의 이론적 기초를 다뤘다:
- Decision tree 알고리즘 (Maranget 2008)
- Pattern matrix와 specialization/defaulting 연산
- Exhaustiveness checking과 unreachable case detection
- 컴파일 시간에 패턴을 분석하여 최적의 decision tree 생성

**Chapter 18: List Operations**에서는 패턴 매칭이 작동할 데이터 구조를 구현했다:
- `!funlang.list<T>` parameterized type으로 타입 안전한 리스트 표현
- `funlang.nil`과 `funlang.cons` operations으로 리스트 생성
- TypeConverter로 tagged union `!llvm.struct<(i32, ptr)>` 변환
- NilOpLowering과 ConsOpLowering patterns로 LLVM dialect 생성

**Chapter 19: Match Compilation**에서는 모든 것을 종합했다:
- `funlang.match` operation으로 패턴 매칭 표현
- Multi-stage lowering: FunLang → SCF → CF → LLVM
- IRMapping으로 block argument remapping
- 실행 가능한 코드 생성

**Chapter 20에서는 이 모든 것을 사용하여 실제 함수형 프로그램을 작성한다.**

### Phase 6의 최종 목표: 완전한 함수형 프로그래밍

Phase 4에서 우리는 클로저를 구현했다:

```fsharp
// Phase 4: 클로저
let makeAdder n = fun x -> x + n
let add5 = makeAdder 5
let result = add5 10  // 15
```

Phase 5에서 우리는 커스텀 FunLang dialect를 만들었다:

```mlir
// Phase 5: FunLang operations
%closure = funlang.closure @add_impl(%n) : (i32) -> ((i32) -> i32)
%result = funlang.apply %closure(%x) : ((i32) -> i32, i32) -> i32
```

Phase 6에서 우리는 리스트와 패턴 매칭을 구현했다:

```mlir
// Phase 6: Lists and pattern matching
%list = funlang.cons %head, %tail : (i32, !funlang.list<i32>) -> !funlang.list<i32>
%result = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil:
    funlang.yield %zero : i32
  ^cons(%h: i32, %t: !funlang.list<i32>):
    funlang.yield %h : i32
}
```

**이제 이 세 가지를 조합하면 강력한 함수형 프로그래밍이 가능하다:**

```fsharp
// Phase 6 Complete: 클로저 + 리스트 + 패턴 매칭
let map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)

let double x = x * 2
let result = map double [1, 2, 3]  // [2, 4, 6]
```

### Chapter 20의 목표: 실전 함수형 프로그램

이 장을 마치면 다음과 같은 **실제 함수형 프로그램을 컴파일하고 실행**할 수 있다:

**1. map: 리스트의 각 원소에 함수를 적용**

```fsharp
let map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)

map (fun x -> x * 2) [1, 2, 3]  // [2, 4, 6]
```

**2. filter: 조건을 만족하는 원소만 남기기**

```fsharp
let filter pred lst =
  match lst with
  | [] -> []
  | head :: tail ->
      if pred head then
        head :: filter pred tail
      else
        filter pred tail

filter (fun x -> x > 2) [1, 2, 3, 4]  // [3, 4]
```

**3. fold: 리스트를 하나의 값으로 축약**

```fsharp
let fold f acc lst =
  match lst with
  | [] -> acc
  | head :: tail -> fold f (f acc head) tail

fold (+) 0 [1, 2, 3, 4, 5]  // 15
```

**4. 조합: 복잡한 프로그램**

```fsharp
// 제곱의 합: [1, 2, 3] -> 14
let sum_of_squares lst =
  fold (+) 0 (map (fun x -> x * x) lst)

sum_of_squares [1, 2, 3]  // 1 + 4 + 9 = 14
```

### 성공 기준: 완전한 컴파일 파이프라인

각 함수형 프로그램에 대해 다음을 보여준다:

1. **FunLang 소스 코드**: F# 스타일의 함수형 문법
2. **FunLang dialect MLIR**: 커스텀 operations 사용
3. **SCF dialect MLIR**: 제어 흐름으로 변환
4. **LLVM dialect MLIR**: 최종 lowering
5. **실행 결과**: JIT으로 실행하여 결과 확인

**이것이 바로 "실전 컴파일러"다:**
- 교과서의 toy 예제가 아니라 실제 사용 가능한 프로그램
- 모든 단계를 추적 가능하고 검증 가능
- Phase 7 (최적화)로 이어지는 기반

### Chapter 20의 구성

**Part 1: Map and Filter (이번 섹션)**
1. FunLang에서 리스트 구축하기
2. map 함수: 소스, 컴파일, 실행
3. filter 함수: 중첩 제어 흐름
4. Helper 함수: length, append

**Part 2: Fold and Complete Pipeline**
1. fold 함수: 일반적인 리스트 combinator
2. 완전한 예제: sum_of_squares
3. 성능 고려사항
4. 완전한 컴파일러 통합
5. Phase 6 요약과 Phase 7 미리보기

이 장을 마치면 **Phase 6가 완료**되며, Phase 7 (최적화)로 넘어갈 준비가 된다.

## FunLang에서 리스트 구축하기

### FunLang AST 확장: List Expressions

지금까지 우리는 MLIR operations로 리스트를 직접 구축했다:

```mlir
// 직접 MLIR 작성
%nil = funlang.nil : !funlang.list<i32>
%three = arith.constant 3 : i32
%l1 = funlang.cons %three, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
```

하지만 사용자는 **FunLang 언어**로 작성하고 싶어한다:

```fsharp
// 사용자가 원하는 문법
let empty = []
let list = [1, 2, 3]
let consed = 1 :: [2, 3]
```

**AST 확장이 필요하다.**

### FunLang AST Type Definition

`Ast.fs`에 리스트 표현식을 추가한다:

```fsharp
// Ast.fs
type Expr =
    | Int of int
    | Var of string
    | Add of Expr * Expr
    | Let of string * Expr * Expr
    | If of Expr * Expr * Expr
    | Fun of string * Expr              // Phase 4: lambda
    | App of Expr * Expr                // Phase 4: application

    // Phase 6: List expressions
    | Nil                                // []
    | Cons of Expr * Expr                // head :: tail
    | List of Expr list                  // [1, 2, 3] - syntactic sugar
    | Match of Expr * (Pattern * Expr) list  // match expr with cases

and Pattern =
    | PVar of string                     // x (bind any value)
    | PNil                               // [] (empty list)
    | PCons of Pattern * Pattern         // head :: tail
    | PWild                              // _ (wildcard)
```

**설계 결정:**

1. **`Nil`**: Empty list `[]`는 zero-argument constructor
2. **`Cons`**: Binary operator `::` (head와 tail)
3. **`List`**: List literal `[1, 2, 3]`는 syntactic sugar (연속된 Cons로 desugaring)
4. **`Match`**: Pattern matching expression

### List Literal Desugaring

List literal은 syntactic sugar다:

```fsharp
// 사용자 작성
[1, 2, 3]

// Desugaring
1 :: (2 :: (3 :: []))

// AST 표현
Cons(Int 1, Cons(Int 2, Cons(Int 3, Nil)))
```

Desugaring 함수:

```fsharp
// Parser.fs or Desugar.fs
let rec desugarList (exprs: Expr list) : Expr =
    match exprs with
    | [] -> Nil
    | head :: tail -> Cons(head, desugarList tail)

// Usage
let ast = List [Int 1; Int 2; Int 3]
let desugared = desugarList [Int 1; Int 2; Int 3]
// Result: Cons(Int 1, Cons(Int 2, Cons(Int 3, Nil)))
```

**왜 desugaring인가?**

1. **간단한 컴파일**: 컴파일러는 `Cons`와 `Nil`만 처리하면 된다
2. **중복 제거**: `List` literal과 `Cons` operator가 같은 representation을 공유
3. **확장성**: 새로운 syntactic sugar 추가 시 desugaring만 변경

### 컴파일러 통합: compileExpr 확장

`Compiler.fs`의 `compileExpr` 함수를 확장하여 리스트를 처리한다:

```fsharp
// Compiler.fs
let rec compileExpr (builder: OpBuilder) (expr: Expr) (symbolTable: Map<string, Value>) : Value =
    match expr with
    | Int n ->
        let ty = builder.GetI32Type()
        builder.CreateConstantInt(ty, n)

    | Var name ->
        symbolTable.[name]

    | Add (left, right) ->
        let lhs = compileExpr builder left symbolTable
        let rhs = compileExpr builder right symbolTable
        builder.CreateAddI(lhs, rhs)

    // ... (Phase 3-4 cases)

    // Phase 6: Nil case
    | Nil ->
        // funlang.nil : !funlang.list<T>
        // Type inference: 주변 context에서 element type 결정
        let elemTy = inferElementType expr  // e.g., i32
        let listTy = builder.GetListType(elemTy)
        builder.CreateNil(listTy)

    // Phase 6: Cons case
    | Cons (head, tail) ->
        // funlang.cons %head, %tail : (T, !funlang.list<T>) -> !funlang.list<T>
        let headVal = compileExpr builder head symbolTable
        let tailVal = compileExpr builder tail symbolTable
        let headTy = headVal.GetType()
        let listTy = builder.GetListType(headTy)
        builder.CreateCons(headVal, tailVal, listTy)

    // Phase 6: Match case (covered later in this chapter)
    | Match (scrutinee, cases) ->
        compileMatch builder scrutinee cases symbolTable
```

**Type inference 예제:**

```fsharp
// FunLang source
let list = 1 :: 2 :: []

// Type inference
// - 1 is i32, so head is i32
// - Cons expects (i32, !funlang.list<i32>)
// - [] must be !funlang.list<i32>

// Compiled MLIR
%c1 = arith.constant 1 : i32
%c2 = arith.constant 2 : i32
%nil = funlang.nil : !funlang.list<i32>
%tail = funlang.cons %c2, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
%list = funlang.cons %c1, %tail : (i32, !funlang.list<i32>) -> !funlang.list<i32>
```

### 예제: 리스트 컴파일

**Example 1: Empty list**

```fsharp
// FunLang
let empty = []
```

Compiled MLIR:

```mlir
func.func @example1() -> !funlang.list<i32> {
  %empty = funlang.nil : !funlang.list<i32>
  return %empty : !funlang.list<i32>
}
```

**Example 2: Single element**

```fsharp
// FunLang
let single = [42]

// Desugared
let single = 42 :: []
```

Compiled MLIR:

```mlir
func.func @example2() -> !funlang.list<i32> {
  %c42 = arith.constant 42 : i32
  %nil = funlang.nil : !funlang.list<i32>
  %single = funlang.cons %c42, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  return %single : !funlang.list<i32>
}
```

**Example 3: Multiple elements**

```fsharp
// FunLang
let list = [1, 2, 3]

// Desugared
let list = 1 :: (2 :: (3 :: []))
```

Compiled MLIR:

```mlir
func.func @example3() -> !funlang.list<i32> {
  // Build from inside out: 3 :: []
  %c3 = arith.constant 3 : i32
  %nil = funlang.nil : !funlang.list<i32>
  %l3 = funlang.cons %c3, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  // 2 :: [3]
  %c2 = arith.constant 2 : i32
  %l2 = funlang.cons %c2, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  // 1 :: [2, 3]
  %c1 = arith.constant 1 : i32
  %l1 = funlang.cons %c1, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  return %l1 : !funlang.list<i32>
}
```

**Example 4: Cons operator**

```fsharp
// FunLang
let list = 1 :: 2 :: 3 :: []
```

Compiled MLIR (same as Example 3):

```mlir
func.func @example4() -> !funlang.list<i32> {
  %c3 = arith.constant 3 : i32
  %nil = funlang.nil : !funlang.list<i32>
  %l3 = funlang.cons %c3, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  %c2 = arith.constant 2 : i32
  %l2 = funlang.cons %c2, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  %c1 = arith.constant 1 : i32
  %l1 = funlang.cons %c1, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  return %l1 : !funlang.list<i32>
}
```

**Type safety:**

FunLang의 타입 시스템은 heterogeneous list를 방지한다:

```fsharp
// Type error: element type mismatch
let bad = [1, "hello", 3]
// Error: Expected i32, found string
```

MLIR type은 element type을 명시한다:
- `!funlang.list<i32>`: 32비트 정수 리스트
- `!funlang.list<f64>`: 64비트 부동소수점 리스트
- `!funlang.list<!funlang.closure<(i32) -> i32>>`: 클로저 리스트 (고차 함수)

이제 우리는 리스트를 구축할 수 있다. 다음은 **리스트를 조작하는 함수**를 작성할 차례다.

## map 함수: 리스트 변환

### map의 개념

`map`은 함수형 프로그래밍의 가장 기본적인 고차 함수다:

```fsharp
// map의 타입
map : (a -> b) -> [a] -> [b]

// map의 의미
map f [x1, x2, ..., xn] = [f x1, f x2, ..., f xn]
```

**예제:**

```fsharp
let double x = x * 2
map double [1, 2, 3]  // [2, 4, 6]

let inc x = x + 1
map inc [10, 20, 30]  // [11, 21, 31]

map (fun x -> x * x) [1, 2, 3, 4]  // [1, 4, 9, 16]
```

### FunLang 소스 코드

`map` 함수를 FunLang으로 작성한다:

```fsharp
let rec map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)
```

**동작 원리:**

1. **Base case**: Empty list → return empty list
2. **Recursive case**:
   - Apply `f` to `head` → transformed head
   - Recursively map over `tail`
   - Cons the results

**실행 trace:**

```fsharp
map double [1, 2, 3]
→ double 1 :: map double [2, 3]
→ 2 :: (double 2 :: map double [3])
→ 2 :: (4 :: (double 3 :: map double []))
→ 2 :: (4 :: (6 :: []))
→ [2, 4, 6]
```

### FunLang AST 표현

FunLang AST로 표현하면:

```fsharp
// let rec map f lst = ...
Let("map",
    Fun("f",
        Fun("lst",
            Match(Var "lst",
                [ (PNil, Nil)
                ; (PCons(PVar "head", PVar "tail"),
                   Cons(App(Var "f", Var "head"),
                        App(App(Var "map", Var "f"), Var "tail")))
                ]))),
    // ... body that uses map ...
)
```

**구조 분석:**

1. **Outer Let**: `map` 정의를 scope에 바인딩
2. **Curried function**: `f`와 `lst` 두 개의 중첩 lambda
3. **Match expression**: `lst`에 대한 패턴 매칭
4. **Patterns**: `[]` (PNil)과 `head :: tail` (PCons)
5. **Recursive call**: `map f tail`에서 `map` 자기 자신 호출

### 컴파일된 MLIR: FunLang Dialect

`compileExpr`가 위 AST를 컴파일하면 다음 MLIR이 생성된다:

```mlir
// map : (T -> U) -> !funlang.list<T> -> !funlang.list<U>
func.func @map(%f: !funlang.closure<(i32) -> i32>,
               %lst: !funlang.list<i32>) -> !funlang.list<i32> {
  // match lst with ...
  %result = funlang.match %lst : !funlang.list<i32> -> !funlang.list<i32> {
    // Case 1: [] -> []
    ^nil:
      %empty = funlang.nil : !funlang.list<i32>
      funlang.yield %empty : !funlang.list<i32>

    // Case 2: head :: tail -> (f head) :: (map f tail)
    ^cons(%head: i32, %tail: !funlang.list<i32>):
      // f head
      %transformed = funlang.apply %f(%head) : (!funlang.closure<(i32) -> i32>, i32) -> i32

      // map f tail (recursive call)
      %mapped_tail = func.call @map(%f, %tail)
        : (!funlang.closure<(i32) -> i32>, !funlang.list<i32>) -> !funlang.list<i32>

      // transformed :: mapped_tail
      %new_list = funlang.cons %transformed, %mapped_tail
        : (i32, !funlang.list<i32>) -> !funlang.list<i32>

      funlang.yield %new_list : !funlang.list<i32>
  }

  return %result : !funlang.list<i32>
}
```

**핵심 포인트:**

1. **`funlang.match`**: 리스트를 검사하는 control flow
2. **`funlang.apply`**: 클로저 간접 호출 (`f head`)
3. **`func.call @map`**: 재귀 호출 (named function)
4. **`funlang.cons`**: 결과 리스트 구축
5. **Type safety**: 모든 operations가 타입 정보를 유지

### Lowering Stage 1: FunLang → SCF

`FunLangToSCFPass`가 실행되면 `funlang.match`가 `scf.if`로 lowering된다:

```mlir
func.func @map(%f: !funlang.closure<(i32) -> i32>,
               %lst: !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)> {
  // Extract tag: lst->tag
  %tag_ptr = llvm.getelementptr %lst[0, 0] : (!llvm.struct<(i32, ptr)>) -> !llvm.ptr
  %tag = llvm.load %tag_ptr : !llvm.ptr -> i32

  // Check if tag == 0 (Nil)
  %c0 = arith.constant 0 : i32
  %is_nil = arith.cmpi eq, %tag, %c0 : i32

  // if (is_nil) then ... else ...
  %result = scf.if %is_nil -> !llvm.struct<(i32, ptr)> {
    // Nil case: return empty list
    %nil_tag = arith.constant 0 : i32
    %null_ptr = llvm.mlir.null : !llvm.ptr
    %empty = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %empty1 = llvm.insertvalue %nil_tag, %empty[0] : !llvm.struct<(i32, ptr)>
    %empty2 = llvm.insertvalue %null_ptr, %empty1[1] : !llvm.struct<(i32, ptr)>
    scf.yield %empty2 : !llvm.struct<(i32, ptr)>
  } else {
    // Cons case: extract head and tail
    %cons_tag = arith.constant 1 : i32
    %payload_ptr = llvm.getelementptr %lst[0, 1] : (!llvm.struct<(i32, ptr)>) -> !llvm.ptr
    %payload = llvm.load %payload_ptr : !llvm.ptr -> !llvm.ptr

    // Cast payload to ConsCell: struct { head: i32, tail: list }
    %head_ptr = llvm.getelementptr %payload[0, 0] : (!llvm.ptr) -> !llvm.ptr
    %head = llvm.load %head_ptr : !llvm.ptr -> i32

    %tail_ptr = llvm.getelementptr %payload[0, 1] : (!llvm.ptr) -> !llvm.ptr
    %tail = llvm.load %tail_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>

    // Apply closure: f head
    %transformed = funlang.apply %f(%head) : (!funlang.closure<(i32) -> i32>, i32) -> i32

    // Recursive call: map f tail
    %mapped_tail = func.call @map(%f, %tail)
      : (!funlang.closure<(i32) -> i32>, !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)>

    // Build cons cell: transformed :: mapped_tail
    %cell_size = llvm.mlir.constant(16 : i64) : i64  // sizeof(ConsCell)
    %cell = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr

    %cell_head_ptr = llvm.getelementptr %cell[0, 0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %transformed, %cell_head_ptr : i32, !llvm.ptr

    %cell_tail_ptr = llvm.getelementptr %cell[0, 1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %mapped_tail, %cell_tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr

    // Build list struct: {tag=1, payload=cell}
    %cons_tag_val = arith.constant 1 : i32
    %new_list = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %new_list1 = llvm.insertvalue %cons_tag_val, %new_list[0] : !llvm.struct<(i32, ptr)>
    %new_list2 = llvm.insertvalue %cell, %new_list1[1] : !llvm.struct<(i32, ptr)>

    scf.yield %new_list2 : !llvm.struct<(i32, ptr)>
  }

  return %result : !llvm.struct<(i32, ptr)>
}
```

**변환 내용:**

1. **`funlang.match` → `scf.if`**: Binary choice (Nil vs Cons)
2. **Tag extraction**: `llvm.getelementptr` + `llvm.load`로 tag field 읽기
3. **Comparison**: `arith.cmpi eq`로 tag 검사
4. **Block arguments → loads**: Cons case의 `%head`, `%tail`을 payload에서 추출
5. **GC allocation**: `GC_malloc`으로 새 cons cell 할당

### Lowering Stage 2: SCF → CF + LLVM

`SCFToControlFlowPass`가 실행되면 `scf.if`가 `cf.br`, `cf.cond_br`로 lowering된다:

```mlir
func.func @map(%f: !funlang.closure<(i32) -> i32>,
               %lst: !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)> {
^entry:
  // Extract tag
  %tag_ptr = llvm.getelementptr %lst[0, 0] : (!llvm.struct<(i32, ptr)>) -> !llvm.ptr
  %tag = llvm.load %tag_ptr : !llvm.ptr -> i32

  %c0 = arith.constant 0 : i32
  %is_nil = arith.cmpi eq, %tag, %c0 : i32

  // Conditional branch
  cf.cond_br %is_nil, ^nil_case, ^cons_case

^nil_case:
  // Return empty list
  %nil_tag = arith.constant 0 : i32
  %null_ptr = llvm.mlir.null : !llvm.ptr
  %empty = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
  %empty1 = llvm.insertvalue %nil_tag, %empty[0] : !llvm.struct<(i32, ptr)>
  %empty2 = llvm.insertvalue %null_ptr, %empty1[1] : !llvm.struct<(i32, ptr)>
  cf.br ^exit(%empty2 : !llvm.struct<(i32, ptr)>)

^cons_case:
  // Extract head and tail
  %payload_ptr = llvm.getelementptr %lst[0, 1] : (!llvm.struct<(i32, ptr)>) -> !llvm.ptr
  %payload = llvm.load %payload_ptr : !llvm.ptr -> !llvm.ptr

  %head_ptr = llvm.getelementptr %payload[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %head = llvm.load %head_ptr : !llvm.ptr -> i32

  %tail_ptr = llvm.getelementptr %payload[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %tail = llvm.load %tail_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>

  // Apply closure
  %transformed = funlang.apply %f(%head) : (!funlang.closure<(i32) -> i32>, i32) -> i32

  // Recursive call
  %mapped_tail = func.call @map(%f, %tail)
    : (!funlang.closure<(i32) -> i32>, !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)>

  // Allocate cons cell
  %cell_size = llvm.mlir.constant(16 : i64) : i64
  %cell = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr

  %cell_head_ptr = llvm.getelementptr %cell[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %transformed, %cell_head_ptr : i32, !llvm.ptr

  %cell_tail_ptr = llvm.getelementptr %cell[0, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %mapped_tail, %cell_tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr

  // Build result
  %cons_tag = arith.constant 1 : i32
  %new_list = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
  %new_list1 = llvm.insertvalue %cons_tag, %new_list[0] : !llvm.struct<(i32, ptr)>
  %new_list2 = llvm.insertvalue %cell, %new_list1[1] : !llvm.struct<(i32, ptr)>

  cf.br ^exit(%new_list2 : !llvm.struct<(i32, ptr)>)

^exit(%result: !llvm.struct<(i32, ptr)>):
  return %result : !llvm.struct<(i32, ptr)>
}
```

**CFG 구조:**

```
       [entry]
          |
       (is_nil?)
        /    \
    [nil]  [cons]
       \    /
       [exit]
```

### 테스트 프로그램: map (fun x -> x * 2) [1, 2, 3]

완전한 프로그램을 컴파일하고 실행해보자:

```fsharp
// FunLang source
let double = fun x -> x * 2

let rec map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)

let result = map double [1, 2, 3]
// Expected: [2, 4, 6]
```

**Compiled MLIR (simplified):**

```mlir
module {
  // Helper: double function as closure implementation
  func.func @double_impl(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %result = arith.muli %x, %c2 : i32
    return %result : i32
  }

  // map function (as defined above)
  func.func @map(%f: !funlang.closure<(i32) -> i32>,
                 %lst: !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)> {
    // ... (as shown in previous section)
  }

  // Main entry point
  func.func @main() -> !llvm.struct<(i32, ptr)> {
    // Create closure: double
    %double_fn = llvm.mlir.addressof @double_impl : !llvm.ptr
    %null_env = llvm.mlir.null : !llvm.ptr  // no captures
    %closure_size = llvm.mlir.constant(16 : i64) : i64
    %closure_mem = llvm.call @GC_malloc(%closure_size) : (i64) -> !llvm.ptr

    %fn_ptr_field = llvm.getelementptr %closure_mem[0, 0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %double_fn, %fn_ptr_field : !llvm.ptr, !llvm.ptr

    %env_ptr_field = llvm.getelementptr %closure_mem[0, 1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %null_env, %env_ptr_field : !llvm.ptr, !llvm.ptr

    %double = llvm.load %closure_mem : !llvm.ptr -> !funlang.closure<(i32) -> i32>

    // Create list: [1, 2, 3]
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    %nil = funlang.nil : !funlang.list<i32>
    %l3 = funlang.cons %c3, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
    %l2 = funlang.cons %c2, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
    %l1 = funlang.cons %c1, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

    // Call map
    %result = func.call @map(%double, %l1)
      : (!funlang.closure<(i32) -> i32>, !funlang.list<i32>) -> !funlang.list<i32>

    return %result : !llvm.struct<(i32, ptr)>
  }
}
```

**실행 trace:**

```
map double [1, 2, 3]
→ double 1 :: map double [2, 3]
→ 2 :: (double 2 :: map double [3])
→ 2 :: (4 :: (double 3 :: map double []))
→ 2 :: (4 :: (6 :: []))
→ [2, 4, 6]
```

**Memory layout (heap):**

```
Closure (double):
  +0: fn_ptr    -> @double_impl
  +8: env_ptr   -> NULL

List [2, 4, 6]:
  +0: tag=1, payload -> ConsCell1

  ConsCell1:
    +0: head=2
    +8: tail -> {tag=1, payload -> ConsCell2}

  ConsCell2:
    +0: head=4
    +8: tail -> {tag=1, payload -> ConsCell3}

  ConsCell3:
    +0: head=6
    +8: tail -> {tag=0, payload=NULL}  // Nil
```

### 검증: JIT 실행

```fsharp
// Compiler.fs
let testMapDouble() =
    let ctx = MLIRContext.Create()
    let module = compileProgram ctx mapDoubleSource

    // Apply lowering passes
    let pm = PassManager.Create(ctx)
    pm.AddPass("convert-funlang-to-scf")
    pm.AddPass("convert-scf-to-cf")
    pm.AddPass("convert-funlang-to-llvm")
    pm.Run(module)

    // JIT execute
    let engine = ExecutionEngine.Create(module)
    let result = engine.Invoke("main", [||])

    // Verify result: [2, 4, 6]
    let list = result :?> ListValue
    assert (list.Count = 3)
    assert (list.[0] = 2)
    assert (list.[1] = 4)
    assert (list.[2] = 6)

    printfn "map double [1, 2, 3] = [2, 4, 6] ✓"
```

**Output:**

```
map double [1, 2, 3] = [2, 4, 6] ✓
```

성공! `map` 함수가 완전히 작동한다.

## filter 함수: 조건부 리스트 필터링

### filter의 개념

`filter`는 조건을 만족하는 원소만 남긴다:

```fsharp
// filter의 타입
filter : (a -> bool) -> [a] -> [a]

// filter의 의미
filter pred [x1, x2, ..., xn] = [xi | pred xi = true]
```

**예제:**

```fsharp
let is_positive x = x > 0
filter is_positive [-2, -1, 0, 1, 2]  // [1, 2]

let is_even x = x % 2 == 0
filter is_even [1, 2, 3, 4, 5, 6]  // [2, 4, 6]

filter (fun x -> x > 2) [1, 2, 3, 4]  // [3, 4]
```

### FunLang 소스 코드

`filter` 함수를 FunLang으로 작성한다:

```fsharp
let rec filter pred lst =
  match lst with
  | [] -> []
  | head :: tail ->
      if pred head then
        head :: filter pred tail
      else
        filter pred tail
```

**동작 원리:**

1. **Base case**: Empty list → return empty list
2. **Recursive case**:
   - 조건 검사: `pred head`
   - True이면: `head`를 결과에 포함
   - False이면: `head`를 건너뛰고 tail만 재귀 처리

**실행 trace:**

```fsharp
filter (fun x -> x > 2) [1, 2, 3, 4]
→ (1 > 2)? No → filter pred [2, 3, 4]
→ (2 > 2)? No → filter pred [3, 4]
→ (3 > 2)? Yes → 3 :: filter pred [4]
→ (4 > 2)? Yes → 3 :: (4 :: filter pred [])
→ 3 :: (4 :: [])
→ [3, 4]
```

### map vs filter 비교

| 특성 | map | filter |
|-----|-----|--------|
| 타입 | `(a -> b) -> [a] -> [b]` | `(a -> bool) -> [a] -> [a]` |
| 결과 크기 | Input과 동일 | Input 이하 |
| 조건 분기 | 없음 (항상 변환) | 있음 (if-else) |
| 원소 변환 | 있음 (`f x`) | 없음 (원소 그대로) |
| MLIR 복잡도 | Moderate | Higher (nested control flow) |

### 컴파일된 MLIR: FunLang Dialect

```mlir
// filter : (T -> i1) -> !funlang.list<T> -> !funlang.list<T>
func.func @filter(%pred: !funlang.closure<(i32) -> i1>,
                  %lst: !funlang.list<i32>) -> !funlang.list<i32> {
  // match lst with ...
  %result = funlang.match %lst : !funlang.list<i32> -> !funlang.list<i32> {
    // Case 1: [] -> []
    ^nil:
      %empty = funlang.nil : !funlang.list<i32>
      funlang.yield %empty : !funlang.list<i32>

    // Case 2: head :: tail -> if (pred head) then ... else ...
    ^cons(%head: i32, %tail: !funlang.list<i32>):
      // pred head
      %should_keep = funlang.apply %pred(%head)
        : (!funlang.closure<(i32) -> i1>, i32) -> i1

      // Recursive call (always needed)
      %filtered_tail = func.call @filter(%pred, %tail)
        : (!funlang.closure<(i32) -> i1>, !funlang.list<i32>) -> !funlang.list<i32>

      // if should_keep then head :: filtered_tail else filtered_tail
      %new_list = scf.if %should_keep -> !funlang.list<i32> {
        // Keep head
        %kept = funlang.cons %head, %filtered_tail
          : (i32, !funlang.list<i32>) -> !funlang.list<i32>
        scf.yield %kept : !funlang.list<i32>
      } else {
        // Skip head
        scf.yield %filtered_tail : !funlang.list<i32>
      }

      funlang.yield %new_list : !funlang.list<i32>
  }

  return %result : !funlang.list<i32>
}
```

**핵심 포인트:**

1. **Nested control flow**: `funlang.match` 안에 `scf.if`
2. **Predicate 호출**: `funlang.apply %pred(%head)`는 boolean 반환
3. **Conditional cons**: True일 때만 `funlang.cons`
4. **Recursive call position**: if 밖에서 호출 (항상 필요)

### Nested Control Flow 분석

`filter`는 두 단계의 제어 흐름을 가진다:

**Level 1: Pattern matching**

```
match lst:
  Nil  → []
  Cons → [Level 2]
```

**Level 2: Conditional inclusion**

```
if pred head:
  True  → head :: filtered_tail
  False → filtered_tail
```

**Combined CFG:**

```
        [entry]
           |
       (is_nil?)
        /    \
    [nil]  [cons]
       |      |
       |   (pred head?)
       |    /      \
       | [keep]  [skip]
       |    \      /
       |   [merge]
        \    /
        [exit]
```

### Lowering Stage 1: FunLang → SCF

```mlir
func.func @filter(%pred: !funlang.closure<(i32) -> i1>,
                  %lst: !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)> {
  // Extract tag
  %tag_ptr = llvm.getelementptr %lst[0, 0] : (!llvm.struct<(i32, ptr)>) -> !llvm.ptr
  %tag = llvm.load %tag_ptr : !llvm.ptr -> i32

  %c0 = arith.constant 0 : i32
  %is_nil = arith.cmpi eq, %tag, %c0 : i32

  // Level 1: match
  %result = scf.if %is_nil -> !llvm.struct<(i32, ptr)> {
    // Nil case
    %nil_tag = arith.constant 0 : i32
    %null_ptr = llvm.mlir.null : !llvm.ptr
    %empty = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %empty1 = llvm.insertvalue %nil_tag, %empty[0] : !llvm.struct<(i32, ptr)>
    %empty2 = llvm.insertvalue %null_ptr, %empty1[1] : !llvm.struct<(i32, ptr)>
    scf.yield %empty2 : !llvm.struct<(i32, ptr)>
  } else {
    // Cons case: extract head and tail
    %payload_ptr = llvm.getelementptr %lst[0, 1] : (!llvm.struct<(i32, ptr)>) -> !llvm.ptr
    %payload = llvm.load %payload_ptr : !llvm.ptr -> !llvm.ptr

    %head_ptr = llvm.getelementptr %payload[0, 0] : (!llvm.ptr) -> !llvm.ptr
    %head = llvm.load %head_ptr : !llvm.ptr -> i32

    %tail_ptr = llvm.getelementptr %payload[0, 1] : (!llvm.ptr) -> !llvm.ptr
    %tail = llvm.load %tail_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>

    // Apply predicate
    %should_keep = funlang.apply %pred(%head)
      : (!funlang.closure<(i32) -> i1>, i32) -> i1

    // Recursive call
    %filtered_tail = func.call @filter(%pred, %tail)
      : (!funlang.closure<(i32) -> i1>, !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)>

    // Level 2: if pred
    %new_list = scf.if %should_keep -> !llvm.struct<(i32, ptr)> {
      // Keep: allocate cons cell
      %cell_size = llvm.mlir.constant(16 : i64) : i64
      %cell = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr

      %cell_head_ptr = llvm.getelementptr %cell[0, 0] : (!llvm.ptr) -> !llvm.ptr
      llvm.store %head, %cell_head_ptr : i32, !llvm.ptr

      %cell_tail_ptr = llvm.getelementptr %cell[0, 1] : (!llvm.ptr) -> !llvm.ptr
      llvm.store %filtered_tail, %cell_tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr

      %cons_tag = arith.constant 1 : i32
      %kept = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
      %kept1 = llvm.insertvalue %cons_tag, %kept[0] : !llvm.struct<(i32, ptr)>
      %kept2 = llvm.insertvalue %cell, %kept1[1] : !llvm.struct<(i32, ptr)>

      scf.yield %kept2 : !llvm.struct<(i32, ptr)>
    } else {
      // Skip: return filtered_tail directly
      scf.yield %filtered_tail : !llvm.struct<(i32, ptr)>
    }

    scf.yield %new_list : !llvm.struct<(i32, ptr)>
  }

  return %result : !llvm.struct<(i32, ptr)>
}
```

**Nested `scf.if` analysis:**

1. **Outer if**: 리스트가 empty인지 검사
2. **Inner if**: Head를 keep할지 skip할지 결정
3. **Region nesting**: Inner if는 outer if의 else branch 안에 있다
4. **Type consistency**: 모든 branch가 같은 타입 반환

### 테스트 프로그램: filter (fun x -> x > 2) [1, 2, 3, 4]

```fsharp
// FunLang source
let is_greater_than_2 = fun x -> x > 2

let rec filter pred lst =
  match lst with
  | [] -> []
  | head :: tail ->
      if pred head then
        head :: filter pred tail
      else
        filter pred tail

let result = filter is_greater_than_2 [1, 2, 3, 4]
// Expected: [3, 4]
```

**Compiled MLIR (main function):**

```mlir
func.func @main() -> !llvm.struct<(i32, ptr)> {
  // Create predicate closure: fun x -> x > 2
  %pred_fn = llvm.mlir.addressof @is_greater_than_2_impl : !llvm.ptr
  %null_env = llvm.mlir.null : !llvm.ptr
  %closure_size = llvm.mlir.constant(16 : i64) : i64
  %closure_mem = llvm.call @GC_malloc(%closure_size) : (i64) -> !llvm.ptr

  %fn_ptr_field = llvm.getelementptr %closure_mem[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %pred_fn, %fn_ptr_field : !llvm.ptr, !llvm.ptr

  %env_ptr_field = llvm.getelementptr %closure_mem[0, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %null_env, %env_ptr_field : !llvm.ptr, !llvm.ptr

  %pred = llvm.load %closure_mem : !llvm.ptr -> !funlang.closure<(i32) -> i1>

  // Create list: [1, 2, 3, 4]
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32

  %nil = funlang.nil : !funlang.list<i32>
  %l4 = funlang.cons %c4, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l3 = funlang.cons %c3, %l4 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l2 = funlang.cons %c2, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l1 = funlang.cons %c1, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  // Call filter
  %result = func.call @filter(%pred, %l1)
    : (!funlang.closure<(i32) -> i1>, !funlang.list<i32>) -> !funlang.list<i32>

  return %result : !llvm.struct<(i32, ptr)>
}

// Predicate implementation
func.func @is_greater_than_2_impl(%x: i32) -> i1 {
  %c2 = arith.constant 2 : i32
  %result = arith.cmpi sgt, %x, %c2 : i32
  return %result : i1
}
```

**실행 trace:**

```
filter pred [1, 2, 3, 4]
→ (1 > 2)? No → filter pred [2, 3, 4]
→ (2 > 2)? No → filter pred [3, 4]
→ (3 > 2)? Yes → 3 :: filter pred [4]
→ (4 > 2)? Yes → 3 :: (4 :: filter pred [])
→ 3 :: (4 :: [])
→ [3, 4]
```

**검증:**

```fsharp
let testFilterGreaterThan2() =
    let ctx = MLIRContext.Create()
    let module = compileProgram ctx filterSource

    let pm = PassManager.Create(ctx)
    pm.AddPass("convert-funlang-to-scf")
    pm.AddPass("convert-scf-to-cf")
    pm.AddPass("convert-funlang-to-llvm")
    pm.Run(module)

    let engine = ExecutionEngine.Create(module)
    let result = engine.Invoke("main", [||])

    let list = result :?> ListValue
    assert (list.Count = 2)
    assert (list.[0] = 3)
    assert (list.[1] = 4)

    printfn "filter (fun x -> x > 2) [1, 2, 3, 4] = [3, 4] ✓"
```

**Output:**

```
filter (fun x -> x > 2) [1, 2, 3, 4] = [3, 4] ✓
```

성공! `filter` 함수도 완전히 작동한다.

## Helper 함수: length와 append

`map`과 `filter` 외에도 유용한 리스트 함수가 많다. 두 가지 기본 helper를 구현한다.

### length 함수

**FunLang 소스:**

```fsharp
let rec length lst =
  match lst with
  | [] -> 0
  | head :: tail -> 1 + length tail
```

**타입:** `[a] -> int`

**예제:**

```fsharp
length []           // 0
length [1]          // 1
length [1, 2, 3]    // 3
```

**Compiled MLIR:**

```mlir
func.func @length(%lst: !funlang.list<i32>) -> i32 {
  %result = funlang.match %lst : !funlang.list<i32> -> i32 {
    ^nil:
      %zero = arith.constant 0 : i32
      funlang.yield %zero : i32

    ^cons(%head: i32, %tail: !funlang.list<i32>):
      %tail_len = func.call @length(%tail) : (!funlang.list<i32>) -> i32
      %one = arith.constant 1 : i32
      %len = arith.addi %one, %tail_len : i32
      funlang.yield %len : i32
  }
  return %result : i32
}
```

**특징:**

- `head` 값은 무시 (타입만 필요)
- 재귀 호출로 tail length 계산
- 결과: `1 + tail_length`

### append 함수

**FunLang 소스:**

```fsharp
let rec append xs ys =
  match xs with
  | [] -> ys
  | head :: tail -> head :: (append tail ys)
```

**타입:** `[a] -> [a] -> [a]`

**예제:**

```fsharp
append [] [1, 2]         // [1, 2]
append [1, 2] []         // [1, 2]
append [1, 2] [3, 4]     // [1, 2, 3, 4]
```

**Compiled MLIR:**

```mlir
func.func @append(%xs: !funlang.list<i32>,
                  %ys: !funlang.list<i32>) -> !funlang.list<i32> {
  %result = funlang.match %xs : !funlang.list<i32> -> !funlang.list<i32> {
    ^nil:
      // Base case: [] ++ ys = ys
      funlang.yield %ys : !funlang.list<i32>

    ^cons(%head: i32, %tail: !funlang.list<i32>):
      // Recursive case: (h :: t) ++ ys = h :: (t ++ ys)
      %appended = func.call @append(%tail, %ys)
        : (!funlang.list<i32>, !funlang.list<i32>) -> !funlang.list<i32>
      %new_list = funlang.cons %head, %appended
        : (i32, !funlang.list<i32>) -> !funlang.list<i32>
      funlang.yield %new_list : !funlang.list<i32>
  }
  return %result : !funlang.list<i32>
}
```

**특징:**

- Base case: 첫 번째 리스트가 empty이면 두 번째 리스트 반환
- Recursive case: 첫 번째 리스트의 head를 보존하고 tail 재귀 처리
- 시간 복잡도: O(|xs|) - 첫 번째 리스트 길이에 비례

**실행 trace:**

```fsharp
append [1, 2] [3, 4]
→ 1 :: append [2] [3, 4]
→ 1 :: (2 :: append [] [3, 4])
→ 1 :: (2 :: [3, 4])
→ [1, 2, 3, 4]
```

### 테스트: Helper 함수

```fsharp
let testHelpers() =
    // Test length
    let len1 = length []            // 0
    let len2 = length [1]           // 1
    let len3 = length [1, 2, 3]     // 3

    assert (len1 = 0)
    assert (len2 = 1)
    assert (len3 = 3)
    printfn "length tests passed ✓"

    // Test append
    let app1 = append [] [1, 2]         // [1, 2]
    let app2 = append [1, 2] []         // [1, 2]
    let app3 = append [1, 2] [3, 4]     // [1, 2, 3, 4]

    assert (listEqual app1 [1, 2])
    assert (listEqual app2 [1, 2])
    assert (listEqual app3 [1, 2, 3, 4])
    printfn "append tests passed ✓"
```

**Output:**

```
length tests passed ✓
append tests passed ✓
```

이제 우리는 **기본 함수형 프로그래밍 toolkit**을 갖췄다:
- `map`: 변환
- `filter`: 필터링
- `length`: 크기 측정
- `append`: 결합

다음 섹션에서는 가장 강력한 combinator인 **`fold`**를 구현한다.

## fold 함수: 일반적인 리스트 Combinator

### fold의 개념

`fold` (또는 `reduce`)는 리스트를 하나의 값으로 축약하는 가장 일반적인 combinator다:

```fsharp
// fold의 타입
fold : (acc -> a -> acc) -> acc -> [a] -> acc

// fold의 의미
fold f acc [x1, x2, ..., xn] = f (... (f (f acc x1) x2) ...) xn
```

**fold는 모든 리스트 연산의 기초다:**

```fsharp
// sum: 모든 원소의 합
let sum lst = fold (+) 0 lst
sum [1, 2, 3, 4, 5]  // 15

// product: 모든 원소의 곱
let product lst = fold (*) 1 lst
product [1, 2, 3, 4]  // 24

// length: map과 filter도 fold로 구현 가능
let length lst = fold (fun acc _ -> acc + 1) 0 lst
length [1, 2, 3]  // 3
```

**왜 fold가 가장 강력한가?**

| 함수 | fold로 구현 가능? | 예제 |
|------|-----------------|------|
| `sum` | ✓ | `fold (+) 0` |
| `product` | ✓ | `fold (*) 1` |
| `length` | ✓ | `fold (fun acc _ -> acc + 1) 0` |
| `map` | ✓ | `fold (fun acc x -> acc ++ [f x]) []` |
| `filter` | ✓ | `fold (fun acc x -> if p x then acc ++ [x] else acc) []` |
| `reverse` | ✓ | `fold (fun acc x -> x :: acc) []` |

**fold는 universal list combinator다.** 다른 모든 리스트 함수를 fold로 표현할 수 있다.

### FunLang 소스 코드

`fold` 함수를 FunLang으로 작성한다:

```fsharp
let rec fold f acc lst =
  match lst with
  | [] -> acc
  | head :: tail -> fold f (f acc head) tail
```

**동작 원리:**

1. **Base case**: Empty list → return accumulator (결과)
2. **Recursive case**:
   - Apply `f` to `acc` and `head` → new accumulator
   - Recursively fold over `tail` with new accumulator

**실행 trace:**

```fsharp
fold (+) 0 [1, 2, 3, 4, 5]
→ fold (+) (0 + 1) [2, 3, 4, 5]
→ fold (+) 1 [2, 3, 4, 5]
→ fold (+) (1 + 2) [3, 4, 5]
→ fold (+) 3 [3, 4, 5]
→ fold (+) (3 + 3) [4, 5]
→ fold (+) 6 [4, 5]
→ fold (+) (6 + 4) [5]
→ fold (+) 10 [5]
→ fold (+) (10 + 5) []
→ fold (+) 15 []
→ 15
```

**Accumulator 패턴:**

Accumulator는 중간 결과를 저장하는 변수다:
- **초기값**: `acc = 0` (sum의 경우)
- **갱신**: `acc = f acc head` (각 원소마다 업데이트)
- **최종값**: 리스트가 empty일 때 accumulator 반환

### fold vs map/filter 비교

| 특성 | map | filter | fold |
|------|-----|--------|------|
| 타입 | `(a -> b) -> [a] -> [b]` | `(a -> bool) -> [a] -> [a]` | `(acc -> a -> acc) -> acc -> [a] -> acc` |
| 입력 | 리스트 | 리스트 | 리스트 + 초기값 |
| 출력 | 리스트 | 리스트 | 단일 값 |
| 함수 인자 | 1개 (원소) | 1개 (원소) | 2개 (누적값, 원소) |
| 일반성 | 특수 | 특수 | 일반 (map/filter 구현 가능) |

### 컴파일된 MLIR: FunLang Dialect

```mlir
// fold : (acc -> T -> acc) -> acc -> !funlang.list<T> -> acc
func.func @fold(%f: !funlang.closure<(i32, i32) -> i32>,
                %acc: i32,
                %lst: !funlang.list<i32>) -> i32 {
  // match lst with ...
  %result = funlang.match %lst : !funlang.list<i32> -> i32 {
    // Case 1: [] -> acc
    ^nil:
      funlang.yield %acc : i32

    // Case 2: head :: tail -> fold f (f acc head) tail
    ^cons(%head: i32, %tail: !funlang.list<i32>):
      // f acc head
      %new_acc = funlang.apply %f(%acc, %head)
        : (!funlang.closure<(i32, i32) -> i32>, i32, i32) -> i32

      // fold f new_acc tail (tail recursion!)
      %final = func.call @fold(%f, %new_acc, %tail)
        : (!funlang.closure<(i32, i32) -> i32>, i32, !funlang.list<i32>) -> i32

      funlang.yield %final : i32
  }

  return %result : i32
}
```

**핵심 포인트:**

1. **Three arguments**: 클로저 `f`, 누적값 `acc`, 리스트 `lst`
2. **Binary closure**: `f`는 두 인자 (`acc`, `head`)를 받는다
3. **Tail recursion**: 재귀 호출이 함수의 마지막 operation (최적화 가능!)
4. **Accumulator threading**: `acc` → `new_acc` → `final`로 흐름

### Tail Recursion 분석

`fold`는 **tail recursive**다:

```fsharp
// Tail recursive (good)
let rec fold f acc lst =
  match lst with
  | [] -> acc
  | head :: tail -> fold f (f acc head) tail
  // ^^^ Recursive call is the LAST operation

// NOT tail recursive (map, filter)
let rec map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)
  // ^^^ Recursive call is NOT the last (cons follows)
```

**Tail recursion의 장점:**

1. **Stack frame 재사용**: 각 재귀 호출이 새 stack frame을 생성하지 않음
2. **메모리 효율**: O(1) stack space (vs O(n) for non-tail)
3. **컴파일러 최적화**: Loop로 변환 가능

**LLVM optimization pass가 tail call을 감지하면:**

```mlir
// Before optimization (recursive)
%result = func.call @fold(%f, %new_acc, %tail) : (...) -> i32

// After optimization (loop)
// Stack frame 재사용, jump로 변환
```

### Common Fold Patterns

**1. Sum (합계)**

```fsharp
let sum lst = fold (fun acc x -> acc + x) 0 lst
// Or simply: fold (+) 0 lst

sum [1, 2, 3, 4, 5]  // 15
```

Compiled MLIR:

```mlir
func.func @sum(%lst: !funlang.list<i32>) -> i32 {
  // Create add closure
  %add = funlang.closure @add_impl() : () -> ((i32, i32) -> i32)

  // Initial accumulator
  %zero = arith.constant 0 : i32

  // Call fold
  %result = func.call @fold(%add, %zero, %lst)
    : (!funlang.closure<(i32, i32) -> i32>, i32, !funlang.list<i32>) -> i32

  return %result : i32
}

func.func @add_impl(%acc: i32, %x: i32) -> i32 {
  %result = arith.addi %acc, %x : i32
  return %result : i32
}
```

**2. Product (곱셈)**

```fsharp
let product lst = fold (*) 1 lst

product [1, 2, 3, 4]  // 24
```

**3. Length (길이)**

```fsharp
let length lst = fold (fun acc _ -> acc + 1) 0 lst

length [1, 2, 3]  // 3
```

이전에 재귀로 구현한 `length`와 같은 결과지만, fold를 사용하면 더 일반적이다.

**4. Reverse (역순)**

```fsharp
let reverse lst = fold (fun acc x -> x :: acc) [] lst

reverse [1, 2, 3]  // [3, 2, 1]
```

**Trace:**

```
fold cons [] [1, 2, 3]
→ fold cons (1 :: []) [2, 3]
→ fold cons [1] [2, 3]
→ fold cons (2 :: [1]) [3]
→ fold cons [2, 1] [3]
→ fold cons (3 :: [2, 1]) []
→ fold cons [3, 2, 1] []
→ [3, 2, 1]
```

**5. Maximum (최댓값)**

```fsharp
let max_list lst =
  match lst with
  | [] -> error "empty list"
  | head :: tail -> fold (fun acc x -> if x > acc then x else acc) head tail

max_list [3, 1, 4, 1, 5, 9, 2]  // 9
```

### 테스트 프로그램: fold (+) 0 [1, 2, 3, 4, 5]

```fsharp
// FunLang source
let add = fun acc x -> acc + x

let rec fold f acc lst =
  match lst with
  | [] -> acc
  | head :: tail -> fold f (f acc head) tail

let result = fold add 0 [1, 2, 3, 4, 5]
// Expected: 15
```

**Compiled MLIR (main function):**

```mlir
func.func @main() -> i32 {
  // Create add closure
  %add_fn = llvm.mlir.addressof @add_impl : !llvm.ptr
  %null_env = llvm.mlir.null : !llvm.ptr
  %closure_size = llvm.mlir.constant(16 : i64) : i64
  %closure_mem = llvm.call @GC_malloc(%closure_size) : (i64) -> !llvm.ptr

  %fn_ptr_field = llvm.getelementptr %closure_mem[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %add_fn, %fn_ptr_field : !llvm.ptr, !llvm.ptr

  %env_ptr_field = llvm.getelementptr %closure_mem[0, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %null_env, %env_ptr_field : !llvm.ptr, !llvm.ptr

  %add = llvm.load %closure_mem : !llvm.ptr -> !funlang.closure<(i32, i32) -> i32>

  // Initial accumulator
  %zero = arith.constant 0 : i32

  // Create list: [1, 2, 3, 4, 5]
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32

  %nil = funlang.nil : !funlang.list<i32>
  %l5 = funlang.cons %c5, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l4 = funlang.cons %c4, %l5 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l3 = funlang.cons %c3, %l4 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l2 = funlang.cons %c2, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
  %l1 = funlang.cons %c1, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

  // Call fold
  %result = func.call @fold(%add, %zero, %l1)
    : (!funlang.closure<(i32, i32) -> i32>, i32, !funlang.list<i32>) -> i32

  return %result : i32
}

func.func @add_impl(%acc: i32, %x: i32) -> i32 {
  %result = arith.addi %acc, %x : i32
  return %result : i32
}
```

**검증:**

```fsharp
let testFoldSum() =
    let ctx = MLIRContext.Create()
    let module = compileProgram ctx foldSumSource

    let pm = PassManager.Create(ctx)
    pm.AddPass("convert-funlang-to-scf")
    pm.AddPass("convert-scf-to-cf")
    pm.AddPass("convert-funlang-to-llvm")
    pm.Run(module)

    let engine = ExecutionEngine.Create(module)
    let result = engine.Invoke("main", [||])

    assert (result = 15)
    printfn "fold (+) 0 [1, 2, 3, 4, 5] = 15 ✓"
```

**Output:**

```
fold (+) 0 [1, 2, 3, 4, 5] = 15 ✓
```

성공! `fold` 함수도 완전히 작동한다.

## 완전한 예제: Sum of Squares

이제 모든 것을 조합하여 **실전 함수형 프로그램**을 작성한다.

### 문제 정의

주어진 숫자 리스트의 **제곱의 합**을 계산한다:

```
sum_of_squares [1, 2, 3] = 1² + 2² + 3² = 1 + 4 + 9 = 14
```

### FunLang 소스 코드

```fsharp
// Helper: square function
let square = fun x -> x * x

// Helper: add function
let add = fun acc x -> acc + x

// map: transform each element
let rec map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)

// fold: reduce to single value
let rec fold f acc lst =
  match lst with
  | [] -> acc
  | head :: tail -> fold f (f acc head) tail

// Composition: sum of squares
let sum_of_squares lst =
  fold add 0 (map square lst)

// Test
let result = sum_of_squares [1, 2, 3]
// Expected: 14
```

**함수 조합 분석:**

```
[1, 2, 3]
  ↓ map square
[1, 4, 9]
  ↓ fold add 0
14
```

**이것이 바로 함수형 프로그래밍의 핵심이다:**
- 작은 함수들 (`square`, `add`, `map`, `fold`)
- 조합하여 복잡한 동작 (`sum_of_squares`)
- 선언적 스타일: "무엇을" 계산할지 명확

### 전체 컴파일 파이프라인 (9 단계)

이 프로그램을 end-to-end로 컴파일하는 과정을 모두 추적한다.

**Stage 1: FunLang Source (사용자 작성)**

```fsharp
let sum_of_squares lst =
  fold add 0 (map square lst)
```

**Stage 2: FunLang AST (Parser 출력)**

```fsharp
Let("sum_of_squares",
    Fun("lst",
        App(App(App(Var "fold", Var "add"),
                Int 0),
            App(App(Var "map", Var "square"),
                Var "lst"))),
    ...)
```

**Stage 3: FunLang MLIR (Compiler.fs 출력)**

```mlir
func.func @sum_of_squares(%lst: !funlang.list<i32>) -> i32 {
  // square closure (defined elsewhere)
  %square = ... : !funlang.closure<(i32) -> i32>

  // add closure (defined elsewhere)
  %add = ... : !funlang.closure<(i32, i32) -> i32>

  // map square lst
  %squared_list = func.call @map(%square, %lst)
    : (!funlang.closure<(i32) -> i32>, !funlang.list<i32>) -> !funlang.list<i32>

  // fold add 0 squared_list
  %zero = arith.constant 0 : i32
  %result = func.call @fold(%add, %zero, %squared_list)
    : (!funlang.closure<(i32, i32) -> i32>, i32, !funlang.list<i32>) -> i32

  return %result : i32
}
```

**Stage 4: FunLang → SCF Lowering (FunLangToSCFPass)**

`funlang.match` operations이 `scf.if`로 변환된다:

```mlir
// @map function (simplified)
func.func @map(...) -> ... {
  %is_nil = ... : i1
  %result = scf.if %is_nil -> ... {
    // Nil case
    scf.yield %empty : ...
  } else {
    // Cons case
    %transformed = funlang.apply %f(%head) : ...
    %mapped_tail = func.call @map(...) : ...
    %new_list = funlang.cons %transformed, %mapped_tail : ...
    scf.yield %new_list : ...
  }
  return %result : ...
}
```

**Stage 5: FunLang Ops → LLVM (FunLangToLLVMPass)**

`funlang.cons`, `funlang.nil`, `funlang.apply` 등이 LLVM operations로 변환:

```mlir
// funlang.cons lowering
%cell_size = llvm.mlir.constant(16 : i64) : i64
%cell = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr
%head_ptr = llvm.getelementptr %cell[0, 0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %head, %head_ptr : i32, !llvm.ptr
%tail_ptr = llvm.getelementptr %cell[0, 1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %tail, %tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr

%cons_tag = arith.constant 1 : i32
%list = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%list1 = llvm.insertvalue %cons_tag, %list[0] : !llvm.struct<(i32, ptr)>
%list2 = llvm.insertvalue %cell, %list1[1] : !llvm.struct<(i32, ptr)>
```

**Stage 6: SCF → CF Lowering (SCFToControlFlowPass)**

`scf.if` → `cf.cond_br`, `cf.br`:

```mlir
func.func @map(...) -> ... {
^entry:
  %is_nil = ... : i1
  cf.cond_br %is_nil, ^nil_case, ^cons_case

^nil_case:
  %empty = ...
  cf.br ^exit(%empty : ...)

^cons_case:
  %transformed = ...
  %mapped_tail = func.call @map(...) : ...
  %new_list = ...
  cf.br ^exit(%new_list : ...)

^exit(%result: ...):
  return %result : ...
}
```

**Stage 7: Func → LLVM (ConvertFuncToLLVMPass)**

`func.func` → `llvm.func`, `func.call` → `llvm.call`:

```mlir
llvm.func @map(%f: !llvm.ptr, %lst: !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)> {
  ...
  %result = llvm.call @map(%f, %tail) : (!llvm.ptr, !llvm.struct<(i32, ptr)>) -> !llvm.struct<(i32, ptr)>
  ...
}
```

**Stage 8: LLVM Dialect → LLVM IR (Translate to LLVM IR)**

MLIR LLVM dialect를 실제 LLVM IR로 변환:

```llvm
define { i32, i8* } @map({ i8*, i8* }* %f, { i32, i8* } %lst) {
entry:
  %0 = extractvalue { i32, i8* } %lst, 0
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %nil_case, label %cons_case

nil_case:
  %2 = insertvalue { i32, i8* } undef, i32 0, 0
  %3 = insertvalue { i32, i8* } %2, i8* null, 1
  br label %exit

cons_case:
  %4 = extractvalue { i32, i8* } %lst, 1
  %5 = bitcast i8* %4 to { i32, { i32, i8* } }*
  %6 = getelementptr { i32, { i32, i8* } }, { i32, { i32, i8* } }* %5, i32 0, i32 0
  %7 = load i32, i32* %6
  %8 = getelementptr { i32, { i32, i8* } }, { i32, { i32, i8* } }* %5, i32 0, i32 1
  %9 = load { i32, i8* }, { i32, i8* }* %8
  ; ... (apply closure, recursive call, cons)
  br label %exit

exit:
  %result = phi { i32, i8* } [ %3, %nil_case ], [ %new_list, %cons_case ]
  ret { i32, i8* } %result
}
```

**Stage 9: LLVM IR → Machine Code (JIT 또는 AOT)**

LLVM backend가 target architecture의 machine code 생성:

```asm
; x86-64 assembly (simplified)
map:
    push    rbp
    mov     rbp, rsp
    ; Extract tag
    mov     eax, dword ptr [rsi]
    test    eax, eax
    je      .LBB0_1        ; Nil case
    ; Cons case
    mov     rdi, qword ptr [rsi + 8]
    mov     ecx, dword ptr [rdi]     ; head
    mov     rsi, qword ptr [rdi + 8]  ; tail
    ; ... (apply f, recursive call)
    jmp     .LBB0_2
.LBB0_1:
    ; Return empty list
    xor     eax, eax
    xor     edx, edx
.LBB0_2:
    pop     rbp
    ret
```

### 실행 및 검증

```fsharp
let testSumOfSquares() =
    let ctx = MLIRContext.Create()
    let module = compileProgram ctx sumOfSquaresSource

    // Apply all passes
    let pm = PassManager.Create(ctx)
    pm.AddPass("convert-funlang-to-scf")
    pm.AddPass("convert-scf-to-cf")
    pm.AddPass("convert-funlang-to-llvm")
    pm.AddPass("convert-func-to-llvm")
    pm.Run(module)

    // JIT compile and execute
    let engine = ExecutionEngine.Create(module)
    let result = engine.Invoke("main", [||])

    // Verify
    assert (result = 14)
    printfn "sum_of_squares [1, 2, 3] = 14 ✓"

    // Detailed trace
    printfn "Pipeline trace:"
    printfn "  [1, 2, 3]"
    printfn "  → map square"
    printfn "  [1, 4, 9]"
    printfn "  → fold add 0"
    printfn "  14 ✓"
```

**Output:**

```
sum_of_squares [1, 2, 3] = 14 ✓
Pipeline trace:
  [1, 2, 3]
  → map square
  [1, 4, 9]
  → fold add 0
  14 ✓
```

**완전한 컴파일러가 작동한다!**

9단계의 변환을 거쳐 FunLang 소스 코드가 실행 가능한 machine code가 되었다.

## 성능 고려사항

### Stack Usage in Recursive List Functions

리스트 함수는 재귀적이므로 stack 사용량이 중요하다.

**Stack depth by function:**

| 함수 | Stack depth | 이유 |
|------|-------------|------|
| `map` | O(n) | Non-tail recursive (cons 후에 return) |
| `filter` | O(n) | Non-tail recursive (cons 후에 return) |
| `fold` | **O(1)** | **Tail recursive (최적화 가능)** |
| `length` | O(n) | Non-tail recursive |
| `append` | O(n) | Non-tail recursive |

**Non-tail recursion example (map):**

```fsharp
let rec map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (map f tail)
  // ^^^ Cons operation AFTER recursive call
  // Stack frame must be preserved until map returns
```

Call stack for `map square [1, 2, 3]`:

```
map [1, 2, 3]
  map [2, 3]
    map [3]
      map []
      return []
    cons 9 []
    return [9]
  cons 4 [9]
  return [4, 9]
cons 1 [4, 9]
return [1, 4, 9]
```

**각 frame은 다음을 저장해야 한다:**
- Return address
- `head` value (cons를 위해)
- `tail` pointer

**Tail recursion example (fold):**

```fsharp
let rec fold f acc lst =
  match lst with
  | [] -> acc
  | head :: tail -> fold f (f acc head) tail
  // ^^^ Recursive call is LAST operation
  // Stack frame can be REUSED
```

Call stack for `fold add 0 [1, 2, 3]`:

```
fold 0 [1, 2, 3]
fold 1 [2, 3]      // Same stack frame, acc updated
fold 3 [3]         // Same stack frame, acc updated
fold 6 []          // Same stack frame, acc updated
return 6
```

**Only ONE stack frame!**

### Tail Call Optimization (TCO)

LLVM은 tail call을 감지하여 최적화할 수 있다.

**Before TCO:**

```llvm
define i32 @fold(...) {
  ; ...
  %new_acc = add i32 %acc, %head
  %result = call i32 @fold(..., %new_acc, %tail)
  ret i32 %result
}
```

**After TCO:**

```llvm
define i32 @fold(...) {
entry:
  br label %loop

loop:
  ; ...
  %new_acc = add i32 %acc, %head
  ; Update arguments and jump (no new stack frame)
  br label %loop
}
```

**TCO 활성화:**

```fsharp
// PassManager.fs
let pm = PassManager.Create(ctx)

// Add standard LLVM optimization passes
pm.AddPass("inline")              // Inline small functions
pm.AddPass("simplifycfg")         // Simplify control flow
pm.AddPass("tailcallelim")        // Tail call elimination
pm.AddPass("mem2reg")             // Promote memory to registers
pm.Run(module)
```

**결과:**

- `fold`는 loop로 변환되어 O(1) stack 사용
- 큰 리스트 (100,000+ elements)도 stack overflow 없이 처리 가능

### GC Pressure

리스트 연산은 많은 메모리를 할당한다.

**Allocation counts:**

```fsharp
// Create list [1, 2, 3]
// - 3 cons cells = 3 * 16 bytes = 48 bytes

// map square [1, 2, 3]
// - Input: 3 cells (48 bytes)
// - Output: 3 NEW cells (48 bytes)
// - Total alive: 96 bytes (both lists live)

// fold add 0 (map square [1, 2, 3])
// - Input: 3 cells (48 bytes) from map
// - Output: i32 (4 bytes) - no new list!
// - GC can collect input list after fold
```

**Allocation pattern by function:**

| 함수 | 할당량 | 설명 |
|------|--------|------|
| `map` | O(n) cons cells | 새 리스트 생성 |
| `filter` | O(k) cons cells (k ≤ n) | 조건 만족하는 원소만 |
| `fold` | **O(1)** | 단일 값만 반환 |
| `append` | O(n) cons cells | 첫 번째 리스트 복사 |

**GC optimization:**

```fsharp
// BAD: 중간 리스트가 메모리에 남는다
let result1 = map f1 lst
let result2 = map f2 result1
let result3 = map f3 result2
// result1, result2, result3 모두 메모리에 존재

// GOOD: Fusion으로 중간 리스트 제거 (Phase 7에서 다룸)
let result = map (f3 << f2 << f1) lst
// 단일 pass, 중간 리스트 없음
```

### Phase 7 Preview: Optimization Opportunities

Phase 7에서 다룰 최적화:

**1. List Fusion**

```fsharp
// Before: 두 번 순회
map f (map g lst)

// After fusion: 한 번만 순회
map (f << g) lst
```

**2. Deforestation**

```fsharp
// Before: 중간 리스트 생성
fold h z (map f lst)

// After deforestation: 직접 계산
fold (fun acc x -> h acc (f x)) z lst
```

**3. Tail Recursion Modulo Cons**

```fsharp
// map을 tail recursive로 변환
let map f lst =
  let rec loop acc lst =
    match lst with
    | [] -> reverse acc
    | head :: tail -> loop ((f head) :: acc) tail
  loop [] lst
```

**4. Parallel Map**

큰 리스트에 대해 map을 병렬화:

```mlir
// Sequential
%result = scf.for %i = 0 to %n step 1 iter_args(%acc = %init) -> ... {
  %elem = load %lst[%i]
  %transformed = apply %f(%elem)
  ...
}

// Parallel (MLIR scf.parallel)
scf.parallel (%i) = (0) to (%n) step (1) {
  %elem = load %lst[%i]
  %transformed = apply %f(%elem)
  store %transformed, %result[%i]
}
```

이러한 최적화는 Phase 7에서 MLIR transformation passes로 구현할 것이다.

## 완전한 컴파일러 통합

이제 모든 것을 통합하여 **완전한 FunLang 컴파일러**를 구축한다.

### FunLang AST Type Extensions

최종 AST 정의:

```fsharp
// Ast.fs
module Ast

type Expr =
    // Phase 1-2: Basics
    | Int of int
    | Float of float
    | Bool of bool
    | Var of string
    | Add of Expr * Expr
    | Sub of Expr * Expr
    | Mul of Expr * Expr
    | Div of Expr * Expr
    | Lt of Expr * Expr
    | Gt of Expr * Expr
    | Eq of Expr * Expr

    // Phase 3: Control flow and functions
    | Let of string * Expr * Expr
    | If of Expr * Expr * Expr
    | LetRec of string * Expr * Expr

    // Phase 4: Closures and higher-order functions
    | Fun of string * Expr              // lambda
    | App of Expr * Expr                // application

    // Phase 6: Lists and pattern matching
    | Nil                                // []
    | Cons of Expr * Expr                // head :: tail
    | List of Expr list                  // [1, 2, 3] (syntactic sugar)
    | Match of Expr * (Pattern * Expr) list

and Pattern =
    | PVar of string                     // x (variable binding)
    | PNil                               // [] (empty list)
    | PCons of Pattern * Pattern         // head :: tail (cons pattern)
    | PWild                              // _ (wildcard)
    | PInt of int                        // 42 (literal match)
    | PBool of bool                      // true/false

type Program = Expr
```

### Compiler.fs: compileExpr Complete Implementation

```fsharp
// Compiler.fs
module Compiler

open MLIR
open Ast

let rec compileExpr (builder: OpBuilder) (expr: Expr) (symbolTable: Map<string, Value>) : Value =
    match expr with
    // Phase 1-2: Arithmetic
    | Int n ->
        let ty = builder.GetI32Type()
        builder.CreateConstantInt(ty, n)

    | Float f ->
        let ty = builder.GetF64Type()
        builder.CreateConstantFloat(ty, f)

    | Bool b ->
        let ty = builder.GetI1Type()
        builder.CreateConstantBool(ty, b)

    | Var name ->
        symbolTable.[name]

    | Add (left, right) ->
        let lhs = compileExpr builder left symbolTable
        let rhs = compileExpr builder right symbolTable
        builder.CreateAddI(lhs, rhs)

    | Mul (left, right) ->
        let lhs = compileExpr builder left symbolTable
        let rhs = compileExpr builder right symbolTable
        builder.CreateMulI(lhs, rhs)

    // ... (other arithmetic ops)

    // Phase 3: Let and If
    | Let (name, value, body) ->
        let val_result = compileExpr builder value symbolTable
        let newSymbolTable = symbolTable.Add(name, val_result)
        compileExpr builder body newSymbolTable

    | If (cond, thenExpr, elseExpr) ->
        let condVal = compileExpr builder cond symbolTable
        let resultTy = inferType thenExpr symbolTable
        builder.CreateScfIf(condVal, resultTy, fun thenBuilder ->
            let thenResult = compileExpr thenBuilder thenExpr symbolTable
            thenBuilder.CreateScfYield(thenResult)
        , fun elseBuilder ->
            let elseResult = compileExpr elseBuilder elseExpr symbolTable
            elseBuilder.CreateScfYield(elseResult)
        )

    | LetRec (name, func, body) ->
        // Create named function for recursion
        let funcName = sprintf "_%s" name
        let funcOp = compileFunctionDefinition builder funcName func symbolTable
        let funcRef = builder.CreateFuncRef(funcOp)
        let newSymbolTable = symbolTable.Add(name, funcRef)
        compileExpr builder body newSymbolTable

    // Phase 4: Closures
    | Fun (param, body) ->
        // Analyze free variables
        let freeVars = analyzeFreeVars (Fun(param, body)) symbolTable

        // Create closure implementation function
        let implName = sprintf "_lambda_%d" (freshId())
        let implFunc = createClosureImpl builder implName param body freeVars symbolTable

        // Capture environment
        let captures = freeVars |> List.map (fun v -> symbolTable.[v])

        // Create closure object
        builder.CreateClosure(implFunc, captures)

    | App (func, arg) ->
        let funcVal = compileExpr builder func symbolTable
        let argVal = compileExpr builder arg symbolTable
        builder.CreateApply(funcVal, argVal)

    // Phase 6: Lists
    | Nil ->
        let elemTy = inferElementType expr symbolTable
        let listTy = builder.GetListType(elemTy)
        builder.CreateNil(listTy)

    | Cons (head, tail) ->
        let headVal = compileExpr builder head symbolTable
        let tailVal = compileExpr builder tail symbolTable
        let headTy = headVal.GetType()
        let listTy = builder.GetListType(headTy)
        builder.CreateCons(headVal, tailVal, listTy)

    | List exprs ->
        // Desugar to nested Cons
        let desugared = desugarList exprs
        compileExpr builder desugared symbolTable

    | Match (scrutinee, cases) ->
        compileMatch builder scrutinee cases symbolTable

and compileMatch (builder: OpBuilder) (scrutinee: Expr) (cases: (Pattern * Expr) list) (symbolTable: Map<string, Value>) : Value =
    let scrutineeVal = compileExpr builder scrutinee symbolTable
    let resultTy = inferType (snd cases.[0]) symbolTable

    // Create funlang.match operation
    builder.CreateMatch(scrutineeVal, resultTy, fun matchBuilder ->
        cases |> List.map (fun (pattern, body) ->
            match pattern with
            | PNil ->
                // Nil case: no block arguments
                matchBuilder.CreateNilCase(fun caseBuilder ->
                    let result = compileExpr caseBuilder body symbolTable
                    caseBuilder.CreateYield(result)
                )

            | PCons (PVar headName, PVar tailName) ->
                // Cons case: bind head and tail
                let headTy = inferPatternType pattern symbolTable
                let listTy = builder.GetListType(headTy)
                matchBuilder.CreateConsCase(headTy, listTy, fun caseBuilder headArg tailArg ->
                    let newSymbolTable =
                        symbolTable
                            .Add(headName, headArg)
                            .Add(tailName, tailArg)
                    let result = compileExpr caseBuilder body newSymbolTable
                    caseBuilder.CreateYield(result)
                )

            | _ -> failwith "Unsupported pattern"
        )
    )

and desugarList (exprs: Expr list) : Expr =
    match exprs with
    | [] -> Nil
    | head :: tail -> Cons(head, desugarList tail)
```

### Type Inference for List Types

리스트 타입 추론:

```fsharp
// TypeInfer.fs
let rec inferType (expr: Expr) (symbolTable: Map<string, Value>) : MLIRType =
    match expr with
    | Int _ -> builder.GetI32Type()
    | Float _ -> builder.GetF64Type()
    | Bool _ -> builder.GetI1Type()

    | Var name ->
        let value = symbolTable.[name]
        value.GetType()

    | Nil ->
        // Need context to infer element type
        // If context is unavailable, default to i32
        builder.GetListType(builder.GetI32Type())

    | Cons (head, tail) ->
        let headTy = inferType head symbolTable
        builder.GetListType(headTy)

    | List (head :: _) ->
        let headTy = inferType head symbolTable
        builder.GetListType(headTy)

    | Match (scrutinee, cases) ->
        // Result type is the type of first case body
        inferType (snd cases.[0]) symbolTable

    | Fun (param, body) ->
        // Function type: paramTy -> returnTy
        // Need type annotation or inference
        let paramTy = inferParamType param
        let returnTy = inferType body symbolTable
        builder.GetFunctionType(paramTy, returnTy)

    | _ -> failwith "Type inference not implemented"
```

### End-to-End Compilation Function

```fsharp
// Pipeline.fs
let compileProgram (source: string) : MLIRModule =
    // 1. Parse
    let ast = Parser.parse source

    // 2. Desugar
    let desugared = Desugar.desugar ast

    // 3. Type check
    TypeChecker.check desugared

    // 4. Compile to MLIR
    let ctx = MLIRContext.Create()
    let module = MLIRModule.Create(ctx)
    let builder = OpBuilder.Create(ctx)

    let mainFunc = builder.CreateFunc("main", [], inferType desugared Map.empty, fun funcBuilder ->
        let result = Compiler.compileExpr funcBuilder desugared Map.empty
        funcBuilder.CreateReturn(result)
    )

    module.AddFunction(mainFunc)

    // 5. Apply lowering passes
    let pm = PassManager.Create(ctx)
    pm.AddPass("convert-funlang-to-scf")
    pm.AddPass("convert-scf-to-cf")
    pm.AddPass("convert-funlang-to-llvm")
    pm.AddPass("convert-func-to-llvm")
    pm.Run(module)

    module

// Execute
let execute (module: MLIRModule) : obj =
    let engine = ExecutionEngine.Create(module)
    engine.Invoke("main", [||])

// Complete pipeline
let run (source: string) : obj =
    let module = compileProgram source
    execute module
```

### Example Usage

```fsharp
// Main.fs
[<EntryPoint>]
let main argv =
    let source = """
        let square = fun x -> x * x
        let add = fun acc x -> acc + x

        let rec map f lst =
          match lst with
          | [] -> []
          | head :: tail -> (f head) :: (map f tail)

        let rec fold f acc lst =
          match lst with
          | [] -> acc
          | head :: tail -> fold f (f acc head) tail

        let sum_of_squares lst =
          fold add 0 (map square lst)

        sum_of_squares [1, 2, 3]
    """

    let result = Pipeline.run source
    printfn "Result: %A" result  // Result: 14

    0
```

**Output:**

```
Result: 14
```

**완전한 컴파일러가 작동한다!**

## Common Errors and Debugging

함수형 프로그램 작성 시 자주 발생하는 오류와 해결 방법.

### 1. Infinite Recursion

**오류:**

```fsharp
let rec bad_map f lst =
  match lst with
  | [] -> []
  | head :: tail -> (f head) :: (bad_map f lst)  // BUG: lst instead of tail
```

**증상:**

```
Stack overflow
Segmentation fault
Infinite loop
```

**해결:**

- 재귀 호출이 "smaller" input을 사용하는지 확인
- Base case가 반드시 도달 가능한지 확인

```fsharp
// Correct
| head :: tail -> (f head) :: (map f tail)  // ✓ tail is smaller
```

### 2. Type Mismatch

**오류:**

```fsharp
let bad_fold f acc lst =
  match lst with
  | [] -> 0  // BUG: should return acc, not 0
  | head :: tail -> fold f (f acc head) tail
```

**증상:**

```
Type error: Expected i32, found i64
Type mismatch in match branches
```

**해결:**

- 모든 match branch가 같은 타입 반환하는지 확인
- Accumulator 타입이 일관되는지 확인

```fsharp
// Correct
| [] -> acc  // ✓ Same type as recursive case
```

### 3. Wrong Accumulator Type

**오류:**

```fsharp
// Want to reverse a list
let reverse lst = fold (fun acc x -> acc :: x) [] lst  // BUG: wrong cons order
```

**증상:**

```
Type error: Cannot cons list to element
Expected: element :: list
Found: list :: element
```

**해결:**

- Cons operator는 `element :: list` 순서
- Accumulator 타입 확인

```fsharp
// Correct
let reverse lst = fold (fun acc x -> x :: acc) [] lst  // ✓ x :: acc
```

### 4. Stack Overflow

**오류:**

```fsharp
// Large list
let big_list = [1..100000]
let result = map square big_list  // Stack overflow!
```

**증상:**

```
Segmentation fault (core dumped)
Stack overflow at recursion depth 100000
```

**해결:**

- Tail recursive 버전 사용
- TCO 활성화
- Iteration으로 변환 (Phase 7)

```fsharp
// Tail recursive version
let map_tailrec f lst =
  let rec loop acc lst =
    match lst with
    | [] -> reverse acc
    | head :: tail -> loop ((f head) :: acc) tail
  loop [] lst
```

### 5. Debugging Strategies

**전략 1: Trace execution**

```fsharp
let rec map f lst =
  printfn "map called with list of length %d" (length lst)
  match lst with
  | [] ->
      printfn "  -> returning []"
      []
  | head :: tail ->
      printfn "  -> transforming %A" head
      let transformed = f head
      printfn "  -> recursing on tail"
      let mapped_tail = map f tail
      printfn "  -> cons %A onto result" transformed
      transformed :: mapped_tail
```

**전략 2: Unit tests**

```fsharp
let test_map() =
    assert (map square [] = [])
    assert (map square [1] = [1])
    assert (map square [1, 2] = [1, 4])
    assert (map square [1, 2, 3] = [1, 4, 9])
    printfn "map tests passed ✓"
```

**전략 3: MLIR inspection**

```fsharp
let module = compileProgram source
printfn "%s" (module.ToString())  // Print MLIR before lowering

let pm = PassManager.Create(ctx)
pm.EnableIRPrinting()  // Print after each pass
pm.AddPass("convert-funlang-to-scf")
pm.Run(module)
```

**전략 4: GDB debugging**

```bash
# Compile with debug info
mlir-opt --debug-only=funlang-to-scf input.mlir

# Run under GDB
gdb --args mlir-opt ...
(gdb) break FunLangToSCFPass::runOnOperation
(gdb) run
```

## 리터럴 패턴 예제: fizzbuzz

지금까지 리스트에 대한 constructor pattern (Nil, Cons)을 다뤘다. 이제 **리터럴 패턴**을 사용하는 실전 예제를 살펴본다.

### FizzBuzz 문제

**FizzBuzz 규칙:**

- 3의 배수: "Fizz"
- 5의 배수: "Buzz"
- 15의 배수: "FizzBuzz"
- 그 외: 숫자 그대로

**FunLang 구현:**

```fsharp
let fizzbuzz n =
    match (n % 3, n % 5) with
    | (0, 0) -> "FizzBuzz"
    | (0, _) -> "Fizz"
    | (_, 0) -> "Buzz"
    | (_, _) -> string_of_int n
```

**패턴 분석:**

| Row | n % 3 | n % 5 | Result |
|-----|-------|-------|--------|
| 1 | 0 | 0 | "FizzBuzz" |
| 2 | 0 | _ | "Fizz" |
| 3 | _ | 0 | "Buzz" |
| 4 | _ | _ | n |

### 컴파일된 MLIR: 리터럴 패턴

```mlir
func.func @fizzbuzz(%n: i32) -> !llvm.ptr<i8> {
  // Compute remainders
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %c0 = arith.constant 0 : i32

  %mod3 = arith.remsi %n, %c3 : i32
  %mod5 = arith.remsi %n, %c5 : i32

  // Pattern matching: sequential arith.cmpi chain
  %is_div3 = arith.cmpi eq, %mod3, %c0 : i32
  %result = scf.if %is_div3 -> !llvm.ptr<i8> {
    // First column is 0 (n % 3 == 0)
    %is_div5 = arith.cmpi eq, %mod5, %c0 : i32
    %inner = scf.if %is_div5 -> !llvm.ptr<i8> {
      // Case (0, 0): FizzBuzz
      scf.yield %fizzbuzz_str : !llvm.ptr<i8>
    } else {
      // Case (0, _): Fizz
      scf.yield %fizz_str : !llvm.ptr<i8>
    }
    scf.yield %inner : !llvm.ptr<i8>
  } else {
    // First column is not 0 (n % 3 != 0)
    %is_div5_2 = arith.cmpi eq, %mod5, %c0 : i32
    %inner2 = scf.if %is_div5_2 -> !llvm.ptr<i8> {
      // Case (_, 0): Buzz
      scf.yield %buzz_str : !llvm.ptr<i8>
    } else {
      // Case (_, _): n as string
      %str = func.call @int_to_string(%n) : (i32) -> !llvm.ptr<i8>
      scf.yield %str : !llvm.ptr<i8>
    }
    scf.yield %inner2 : !llvm.ptr<i8>
  }

  return %result : !llvm.ptr<i8>
}
```

**핵심 관찰:**

1. **`arith.cmpi eq`**: 리터럴 0과의 비교
2. **Nested `scf.if`**: Decision tree 구조
3. **Wildcard `_`**: else branch로 fallthrough (테스트 없음)

### classify 함수: 숫자 분류

**숫자를 여러 카테고리로 분류하는 예제:**

```fsharp
let classify n =
    match n with
    | 0 -> "zero"
    | 1 -> "one"
    | 2 -> "two"
    | _ -> if n < 0 then "negative" else "many"
```

**컴파일된 MLIR:**

```mlir
func.func @classify(%n: i32) -> !llvm.ptr<i8> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32

  // Sequential literal comparisons
  %is_zero = arith.cmpi eq, %n, %c0 : i32
  %result = scf.if %is_zero -> !llvm.ptr<i8> {
    scf.yield %zero_str : !llvm.ptr<i8>
  } else {
    %is_one = arith.cmpi eq, %n, %c1 : i32
    %r1 = scf.if %is_one -> !llvm.ptr<i8> {
      scf.yield %one_str : !llvm.ptr<i8>
    } else {
      %is_two = arith.cmpi eq, %n, %c2 : i32
      %r2 = scf.if %is_two -> !llvm.ptr<i8> {
        scf.yield %two_str : !llvm.ptr<i8>
      } else {
        // Default case with guard
        %is_neg = arith.cmpi slt, %n, %c0 : i32
        %r3 = scf.if %is_neg -> !llvm.ptr<i8> {
          scf.yield %negative_str : !llvm.ptr<i8>
        } else {
          scf.yield %many_str : !llvm.ptr<i8>
        }
        scf.yield %r3 : !llvm.ptr<i8>
      }
      scf.yield %r2 : !llvm.ptr<i8>
    }
    scf.yield %r1 : !llvm.ptr<i8>
  }

  return %result : !llvm.ptr<i8>
}
```

### 최적화: Dense Range Switch

리터럴이 0, 1, 2 연속일 때 `scf.index_switch` 최적화 가능:

```mlir
// Optimized: range check + index_switch
%in_range = arith.cmpi ult, %n, %c3 : i32
%result = scf.if %in_range -> !llvm.ptr<i8> {
  %idx = arith.index_cast %n : i32 to index
  %r = scf.index_switch %idx : index -> !llvm.ptr<i8>
  case 0 { scf.yield %zero_str : !llvm.ptr<i8> }
  case 1 { scf.yield %one_str : !llvm.ptr<i8> }
  case 2 { scf.yield %two_str : !llvm.ptr<i8> }
  default { scf.yield %unreachable : !llvm.ptr<i8> }
  scf.yield %r : !llvm.ptr<i8>
} else {
  // n >= 3: check if negative
  %is_neg = arith.cmpi slt, %n, %c0 : i32
  %r2 = scf.if %is_neg -> !llvm.ptr<i8> {
    scf.yield %negative_str : !llvm.ptr<i8>
  } else {
    scf.yield %many_str : !llvm.ptr<i8>
  }
  scf.yield %r2 : !llvm.ptr<i8>
}
```

**최적화 효과:**

- **Before:** O(n) sequential comparisons
- **After:** O(1) jump table for dense range

### Wildcard Default Case 최적화

**Wildcard `_`는 테스트를 생성하지 않는다:**

```fsharp
match x with
| 0 -> handle_zero()
| 1 -> handle_one()
| _ -> handle_default()  // No comparison needed!
```

```mlir
%is_zero = arith.cmpi eq, %x, %c0 : i32
scf.if %is_zero {
  // case 0
} else {
  %is_one = arith.cmpi eq, %x, %c1 : i32
  scf.if %is_one {
    // case 1
  } else {
    // _ case: NO arith.cmpi, just fallthrough
    // All other cases exhausted, this is the default
  }
}
```

**핵심 원칙:**

- 마지막 else branch는 이전 모든 테스트가 실패한 경우
- 추가 비교 없이 바로 default 코드 실행
- 이것이 wildcard의 **zero-cost abstraction**

### 리터럴 + Constructor 혼합 예제

**리스트와 숫자를 함께 매칭:**

```fsharp
let take_first_n lst n =
    match (lst, n) with
    | (_, 0) -> []
    | ([], _) -> []
    | (head :: tail, n) -> head :: take_first_n tail (n - 1)
```

**컴파일된 MLIR:**

```mlir
func.func @take_first_n(%lst: !funlang.list<i32>, %n: i32) -> !funlang.list<i32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // Check n == 0 first (literal pattern)
  %is_n_zero = arith.cmpi eq, %n, %c0 : i32
  %result = scf.if %is_n_zero -> !funlang.list<i32> {
    // Case (_, 0): return empty
    %empty = funlang.nil : !funlang.list<i32>
    scf.yield %empty : !funlang.list<i32>
  } else {
    // Check list constructor (constructor pattern)
    %struct = builtin.unrealized_conversion_cast %lst : ... to !llvm.struct<(i32, ptr)>
    %tag = llvm.extractvalue %struct[0] : !llvm.struct<(i32, ptr)>
    %tag_index = arith.index_cast %tag : i32 to index

    %inner = scf.index_switch %tag_index : index -> !funlang.list<i32>
    case 0 {
      // Case ([], _): return empty
      %empty = funlang.nil : !funlang.list<i32>
      scf.yield %empty : !funlang.list<i32>
    }
    case 1 {
      // Case (head :: tail, n): recursive
      %data = llvm.extractvalue %struct[1] : !llvm.struct<(i32, ptr)>
      %head = llvm.load %data : !llvm.ptr -> i32
      %tail_ptr = llvm.getelementptr %data[1] : (!llvm.ptr) -> !llvm.ptr
      %tail = llvm.load %tail_ptr : !llvm.ptr -> !funlang.list<i32>

      %n_minus_1 = arith.subi %n, %c1 : i32
      %rest = func.call @take_first_n(%tail, %n_minus_1) : (...) -> !funlang.list<i32>
      %new_list = funlang.cons %head, %rest : ...
      scf.yield %new_list : !funlang.list<i32>
    }
    default { scf.yield %unreachable : !funlang.list<i32> }

    scf.yield %inner : !funlang.list<i32>
  }

  return %result : !funlang.list<i32>
}
```

**혼합 패턴 lowering 전략:**

1. **Literal column first**: `arith.cmpi` + `scf.if`
2. **Constructor column inside**: `scf.index_switch`
3. **Wildcard**: test 없이 fallthrough

### 검증 및 테스트

```fsharp
let testFizzBuzz() =
    // Test fizzbuzz
    assert (fizzbuzz 3 = "Fizz")
    assert (fizzbuzz 5 = "Buzz")
    assert (fizzbuzz 15 = "FizzBuzz")
    assert (fizzbuzz 7 = "7")
    printfn "fizzbuzz tests passed"

    // Test classify
    assert (classify 0 = "zero")
    assert (classify 1 = "one")
    assert (classify 2 = "two")
    assert (classify 42 = "many")
    assert (classify (-5) = "negative")
    printfn "classify tests passed"

    // Test take_first_n
    assert (take_first_n [1, 2, 3, 4, 5] 3 = [1, 2, 3])
    assert (take_first_n [1, 2, 3] 0 = [])
    assert (take_first_n [] 5 = [])
    printfn "take_first_n tests passed"
```

**Output:**

```
fizzbuzz tests passed
classify tests passed
take_first_n tests passed
```

### Key Takeaways

1. **리터럴 패턴**: `arith.cmpi eq` + `scf.if` chain
2. **Constructor 패턴**: `scf.index_switch`로 O(1) dispatch
3. **Wildcard**: else branch로 fallthrough (테스트 없음)
4. **Dense range**: `scf.index_switch`로 최적화 가능
5. **혼합 패턴**: 각 column의 패턴 타입에 맞는 dispatch 사용

---

## Phase 6 Complete Summary

**축하한다! Phase 6를 완료했다.**

### Chapter 17-20 복습

**Chapter 17: Pattern Matching Theory**
- Decision tree 알고리즘으로 패턴 매칭을 효율적으로 컴파일
- Exhaustiveness checking으로 빠진 case 감지
- Unreachable case detection으로 중복 제거

**Chapter 18: List Operations**
- `!funlang.list<T>` parameterized type
- Tagged union representation: `!llvm.struct<(i32, ptr)>`
- `funlang.nil`과 `funlang.cons` operations
- TypeConverter와 lowering patterns

**Chapter 19: Match Compilation**
- `funlang.match` operation 정의
- Multi-stage lowering: FunLang → SCF → CF → LLVM
- IRMapping으로 block argument remapping
- Region-based IR structure

**Chapter 20: Functional Programs (this chapter)**
- FunLang AST extensions for lists
- Compiler integration (compileExpr, type inference)
- Core list functions: map, filter, fold, length, append
- Complete example: sum_of_squares
- End-to-end compilation pipeline (9 stages)
- Performance analysis and optimization preview

### What You Can Now Compile

**Phase 6 종료 시점에 컴파일 가능한 프로그램:**

```fsharp
// 1. List construction
let list = [1, 2, 3, 4, 5]

// 2. Pattern matching
let rec sum lst =
  match lst with
  | [] -> 0
  | head :: tail -> head + sum tail

// 3. Higher-order functions
let map f lst = ...
let filter pred lst = ...
let fold combiner acc lst = ...

// 4. Function composition
let sum_of_squares lst =
  fold (+) 0 (map (fun x -> x * x) lst)

// 5. Complex functional programs
let process data =
  data
  |> filter is_valid
  |> map transform
  |> fold aggregate initial

// 6. Nested data structures
let nested = [[1, 2], [3, 4], [5, 6]]
let flattened = fold append [] nested
```

**이것은 실제 함수형 언어와 동등한 표현력이다!**

### Technical Achievements

**Phase 6에서 구현한 기술:**

1. **Parameterized types**: `!funlang.list<T>` with element type parameter
2. **Tagged unions**: Efficient runtime representation of ADTs
3. **Pattern matching**: Decision tree compilation for performance
4. **Multi-stage lowering**: Progressive refinement through dialects
5. **Type conversion**: Consistent type mapping across lowering stages
6. **Region-based IR**: Structured control flow with scoped bindings
7. **Tail recursion**: Optimization opportunity for fold
8. **GC integration**: Automatic memory management for lists
9. **Complete pipeline**: Source → AST → MLIR → LLVM IR → Machine code

### Phase 7 Preview: Optimization

Phase 7에서 다룰 내용:

**1. List Fusion**

중간 리스트 제거:

```fsharp
// Before
map f (map g lst)  // Two passes, intermediate list

// After fusion
map (f << g) lst   // One pass, no intermediate
```

**2. Deforestation**

Tree 구조 중간 생성 제거:

```fsharp
// Before
fold h z (map f lst)  // Creates intermediate list

// After deforestation
fold (fun acc x -> h acc (f x)) z lst  // Direct
```

**3. Inlining**

Small 함수 inline:

```mlir
// Before
%result = func.call @square(%x) : (i32) -> i32

// After inlining
%result = arith.muli %x, %x : i32
```

**4. Loop Unrolling**

재귀를 explicit loop로 변환:

```mlir
// Before (recursive)
func.func @map(...) {
  %result = funlang.match %lst : ... {
    ^nil: ...
    ^cons(...): %mapped = func.call @map(...) ...
  }
}

// After (loop)
func.func @map(...) {
  scf.for %i = 0 to %n step 1 iter_args(%acc = %init) -> ... {
    %elem = load %lst[%i]
    %transformed = apply %f(%elem)
    ...
  }
}
```

**5. Parallel Map**

데이터 병렬성 활용:

```mlir
scf.parallel (%i) = (0) to (%n) step (1) {
  %elem = load %lst[%i]
  %result = apply %f(%elem)
  store %result, %output[%i]
}
```

**6. Constant Folding**

컴파일 시간에 계산:

```fsharp
// Before
let result = sum [1, 2, 3, 4, 5]

// After constant folding
let result = 15  // Computed at compile time
```

이러한 최적화는 MLIR의 **transformation passes**로 구현되며, Phase 7에서 자세히 다룬다.

### Congratulations!

**Phase 6 완료를 축하한다!**

이제 여러분은:
- ✓ 완전한 함수형 프로그래밍 언어를 컴파일할 수 있다
- ✓ 리스트, 패턴 매칭, 고차 함수를 지원한다
- ✓ Multi-stage lowering pipeline을 이해한다
- ✓ End-to-end 컴파일 (source to machine code)을 할 수 있다
- ✓ 성능 특성과 최적화 기회를 안다

**다음 단계:** Phase 7 (Optimization)에서 더 빠르고 효율적인 코드 생성을 배운다.

Happy functional programming! 🎉
