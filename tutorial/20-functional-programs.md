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
