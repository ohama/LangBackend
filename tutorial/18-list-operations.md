# Chapter 18: List Operations (List Operations)

## 소개

**Chapter 17**에서는 패턴 매칭 컴파일의 **이론적 기반**을 다뤘다:
- Decision tree 알고리즘 (Maranget 2008)
- Pattern matrix 표현법
- Specialization과 defaulting 연산
- Exhaustiveness checking

**Chapter 18**에서는 패턴 매칭이 작동할 **데이터 구조**를 구현한다. FunLang dialect에 list operations를 추가하여 불변 리스트를 만들고 조작할 수 있게 한다.

### Chapter 17 복습: 왜 List Operations가 먼저인가?

Chapter 17에서 우리는 decision tree 알고리즘을 배웠다:

```fsharp
// F# 패턴 매칭 예제
let rec sum_list lst =
    match lst with
    | [] -> 0                           // Nil pattern
    | head :: tail -> head + sum_list tail  // Cons pattern

sum_list [1; 2; 3]  // 6
```

Decision tree 컴파일 과정:

1. **Pattern matrix 구성**: `[[]; [Cons(head, tail)]]`
2. **Specialization**: Nil case, Cons case 분리
3. **Code generation**: 각 case에 대한 MLIR 코드 생성

**하지만 MLIR로 변환하려면 무엇이 필요한가?**

```mlir
// 목표: 이런 MLIR을 생성하고 싶다
%result = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32
  ^cons(%head: i32, %tail: !funlang.list<i32>):
    // ... recursive call ...
    funlang.yield %sum : i32
}
```

**필요한 요소들:**

1. **List data structure**: `!funlang.list<T>` 타입으로 리스트 표현
2. **List construction**: `funlang.nil`, `funlang.cons`로 리스트 생성
3. **Pattern matching**: `funlang.match`로 리스트 분해 (Chapter 19)

**왜 이 순서인가?**

- 데이터 구조 없이는 패턴 매칭할 대상이 없다
- `funlang.match`는 `!funlang.list` 타입을 입력으로 받는다
- List operations를 먼저 구현하면 Chapter 19에서 `funlang.match`만 집중할 수 있다

### Chapter 18의 목표

**이 장에서 구현할 것:**

1. **List Representation Design**
   - Tagged union으로 Nil/Cons 구분
   - GC-allocated cons cells
   - Immutable shared structure

2. **FunLang List Type**
   - `!funlang.list<T>` parameterized type
   - TableGen 정의, C API shim, F# bindings

3. **funlang.nil Operation**
   - Empty list 생성
   - Constant representation (no allocation)

4. **funlang.cons Operation**
   - Cons cell 생성 (head :: tail)
   - GC allocation for cell

5. **TypeConverter for Lists**
   - `!funlang.list<T>` → `!llvm.struct<(i32, ptr)>` 변환
   - Extending FunLangTypeConverter from Chapter 16

6. **Lowering Patterns**
   - NilOpLowering: struct construction
   - ConsOpLowering: GC_malloc + store operations

### Before vs After: List Operations의 위력

**Before (만약 list operations 없이 직접 구현한다면):**

```mlir
// Empty list: 수동으로 struct 구성
%tag_zero = arith.constant 0 : i32
%null_ptr = llvm.mlir.zero : !llvm.ptr
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag_zero, %undef[0] : !llvm.struct<(i32, ptr)>
%empty = llvm.insertvalue %null_ptr, %s1[1] : !llvm.struct<(i32, ptr)>

// Cons cell: 8줄 이상의 GC_malloc + store 패턴
%cell_size = arith.constant 16 : i64
%cell_ptr = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr
%head_ptr = llvm.getelementptr %cell_ptr[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %head_val, %head_ptr : i32, !llvm.ptr
%tail_ptr = llvm.getelementptr %cell_ptr[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %tail_val, %tail_ptr : !llvm.ptr, !llvm.ptr
%tag_one = arith.constant 1 : i32
%s1 = llvm.insertvalue %tag_one, %undef[0] : !llvm.struct<(i32, ptr)>
%list = llvm.insertvalue %cell_ptr, %s1[1] : !llvm.struct<(i32, ptr)>
```

**After (Chapter 18 구현 후):**

```mlir
// Empty list: 1줄!
%empty = funlang.nil : !funlang.list<i32>

// Cons cell: 1줄!
%list = funlang.cons %head, %tail : !funlang.list<i32>

// Building [1, 2, 3]: 4줄
%nil = funlang.nil : !funlang.list<i32>
%lst1 = funlang.cons %c3, %nil : !funlang.list<i32>
%lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>
%lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>
```

**개선 효과:**

- **코드 줄 수**: 15+ 줄 → 1-2줄 (90%+ 감소!)
- **가독성**: 저수준 struct 조작 제거, 의도 명확
- **타입 안전성**: `!funlang.list<T>` parameterized type으로 element type 검증
- **최적화 가능성**: Empty list sharing, cons cell inlining

### Chapter 15 복습: Custom Operations 패턴

Chapter 15에서 우리는 `funlang.closure`와 `funlang.apply`를 구현하며 custom operations 패턴을 배웠다:

**1. TableGen ODS 정의**

```tablegen
def FunLang_ClosureOp : FunLang_Op<"closure", [Pure]> {
  let summary = "Create closure";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$captures);
  let results = (outs FunLang_ClosureType:$result);
  let assemblyFormat = "$fn `,` $captures attr-dict `:` type($result)";
}
```

**2. C API Shim**

```cpp
extern "C" MlirOperation mlirFunLangClosureOpCreate(
    MlirLocation loc, MlirAttribute fn, MlirValue *captures, intptr_t nCaptures) {
  return wrap(builder.create<funlang::ClosureOp>(loc, fn, ValueRange));
}
```

**3. F# Bindings**

```fsharp
member this.CreateClosure(fn: string, captures: MlirValue list) : MlirValue =
    let op = funlang.CreateClosureOp(loc, fn, captures)
    GetOperationResult(op, 0)
```

**Chapter 18에서도 동일한 패턴을 적용한다:**

- `funlang.nil` ← TableGen → C API → F# bindings
- `funlang.cons` ← TableGen → C API → F# bindings
- `!funlang.list<T>` ← TableGen → C API → F# bindings

### Chapter 18 로드맵

**Part 1 (현재 섹션):**
- List representation design
- `!funlang.list<T>` parameterized type
- `funlang.nil` operation
- `funlang.cons` operation

**Part 2 (다음 섹션):**
- TypeConverter for `!funlang.list<T>`
- NilOpLowering pattern
- ConsOpLowering pattern
- Complete lowering pass update

### 성공 기준

이 장을 완료하면:

- [ ] List의 메모리 표현(tagged union)을 이해한다
- [ ] `!funlang.list<T>` 타입을 TableGen으로 정의할 수 있다
- [ ] `funlang.nil`과 `funlang.cons`의 동작 원리를 안다
- [ ] TypeConverter로 FunLang → LLVM 타입 변환을 구현할 수 있다
- [ ] Lowering pattern으로 operation을 LLVM dialect로 변환할 수 있다
- [ ] Chapter 19에서 `funlang.match` 구현을 시작할 준비가 된다

**Let's build the foundation for pattern matching—list data structures!**

---

## List Representation Design

함수형 언어에서 리스트는 가장 기본적인 데이터 구조다. **Immutable linked list**는 다음 특징을 가진다:

- **Immutability**: 한번 생성되면 변경 불가 (functional purity)
- **Structural sharing**: 서브리스트를 공유하여 메모리 효율적
- **Recursive structure**: Nil (empty) 또는 Cons (head, tail)

### List는 Algebraic Data Type이다

함수형 언어에서 리스트는 **sum type** (tagged union)으로 정의된다:

```fsharp
// F#
type List<'T> =
    | Nil
    | Cons of 'T * List<'T>

// 예제
let empty = Nil
let one = Cons(1, Nil)               // [1]
let three = Cons(1, Cons(2, Cons(3, Nil)))  // [1; 2; 3]
```

```ocaml
(* OCaml *)
type 'a list =
  | []
  | (::) of 'a * 'a list

(* 예제 *)
let empty = []
let one = 1 :: []
let three = 1 :: 2 :: 3 :: []
```

```haskell
-- Haskell
data List a = Nil | Cons a (List a)

-- 예제
empty = Nil
one = Cons 1 Nil
three = Cons 1 (Cons 2 (Cons 3 Nil))
```

**공통 패턴:**

1. **Two constructors**: Nil (empty), Cons (non-empty)
2. **Type parameter**: `'T`, `'a`, `a` (element type)
3. **Recursive definition**: Cons의 tail은 List 자체

### Tagged Union Representation

LLVM에서 sum type을 표현하는 일반적인 방법:

**Discriminator tag + Data pointer**

```
struct TaggedUnion {
    i32 tag;        // 0 = Nil, 1 = Cons, 2 = OtherVariant, ...
    ptr data;       // variant-specific data
}
```

**List의 경우:**

```
!llvm.struct<(i32, ptr)>

- tag = 0: Nil (data = null)
- tag = 1: Cons (data = pointer to {head, tail})
```

**메모리 레이아웃:**

```
Nil representation:
┌─────┬──────┐
│  0  │ null │
└─────┴──────┘
  tag   data

Cons representation:
┌─────┬──────┐        ┌────────┬──────────┐
│  1  │ ptr  │───────>│  head  │   tail   │
└─────┴──────┘        └────────┴──────────┘
  tag   data            element   ptr/struct
```

### Cons Cell Memory Layout

Cons cell은 heap에 할당되는 구조체다:

```
Cons Cell = struct {
    element: T,           // head value
    tail: !llvm.struct<(i32, ptr)>  // tail as tagged union
}
```

**예제: 리스트 [1, 2, 3]의 메모리 구조**

```
%lst3 = Cons(1, Cons(2, Cons(3, Nil)))

Stack (list values as tagged unions):
%lst3: {1, ptr_to_cell1}
%lst2: {1, ptr_to_cell2}
%lst1: {1, ptr_to_cell3}
%nil:  {0, null}

Heap (cons cells):
cell1: {1, %lst2}
       ↑   ↓
     head  tail

cell2: {2, %lst1}
       ↑   ↓
     head  tail

cell3: {3, %nil}
       ↑   ↓
     head  tail (= {0, null})
```

**Visual representation:**

```
%lst3               cell1              %lst2              cell2              %lst1              cell3              %nil
┌───┬────┐          ┌───┬──────┐       ┌───┬────┐         ┌───┬──────┐       ┌───┬────┐         ┌───┬──────┐       ┌───┬──────┐
│ 1 │ ●──┼─────────>│ 1 │ ●────┼──────>│ 1 │ ●──┼────────>│ 2 │ ●────┼──────>│ 1 │ ●──┼────────>│ 3 │ ●────┼──────>│ 0 │ null │
└───┴────┘          └───┴──────┘       └───┴────┘         └───┴──────┘       └───┴────┘         └───┴──────┘       └───┴──────┘
```

### GC Allocation for Cons Cells

Cons cell은 항상 **heap에 할당**된다:

**이유:**

1. **Escape analysis**: 리스트는 함수 반환값으로 사용됨 (upward funarg)
2. **Sharing**: 여러 리스트가 같은 tail을 공유할 수 있음
3. **Lifetime**: 리스트의 lifetime은 생성 함수보다 길 수 있음

**Allocation strategy:**

```mlir
// funlang.cons %head, %tail

// Lowering:
%cell_size = arith.constant 16 : i64  // sizeof(element) + sizeof(ptr)
%cell_ptr = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr

// Store head
%head_offset = llvm.getelementptr %cell_ptr[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %head, %head_offset : i32, !llvm.ptr

// Store tail
%tail_offset = llvm.getelementptr %cell_ptr[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %tail, %tail_offset : !llvm.struct<(i32, ptr)>, !llvm.ptr

// Build tagged union
%tag = arith.constant 1 : i32
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%result = llvm.insertvalue %cell_ptr, %s1[1] : !llvm.struct<(i32, ptr)>
```

**GC의 역할:**

- Cons cells는 명시적으로 free하지 않는다
- Boehm GC가 reachability를 추적하여 자동으로 수집
- Chapter 9에서 설정한 GC infrastructure 활용

### Immutability와 Structural Sharing

**Immutability:**

```mlir
// 리스트 생성
%lst1 = funlang.cons %x, %nil : !funlang.list<i32>

// "수정" 불가능 (새 리스트 생성)
%lst2 = funlang.cons %y, %lst1 : !funlang.list<i32>
// %lst1은 변경되지 않음!
```

**Structural sharing:**

```mlir
%nil = funlang.nil : !funlang.list<i32>
%lst1 = funlang.cons %c1, %nil : !funlang.list<i32>  // [1]
%lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>  // [2, 1]
%lst3 = funlang.cons %c3, %lst1 : !funlang.list<i32>  // [3, 1]

// %lst2와 %lst3는 %lst1을 tail로 공유!
```

**메모리 효율:**

```
Without sharing (mutable arrays):
[2, 1]: 2개 원소 저장
[3, 1]: 2개 원소 저장
Total: 4개 원소

With sharing (immutable lists):
[2, 1]: cell(2) → cell(1) → Nil
[3, 1]: cell(3) ──┘
Total: 3개 cons cells (원소 중복 없음)
```

**장점:**

1. **메모리 효율**: 공통 sublist를 재사용
2. **안전성**: Aliasing bugs 없음 (immutable)
3. **병렬성**: Race conditions 없음
4. **Persistent data structures**: 이전 버전 유지 가능

### Element Type Considerations

리스트는 **parameterized type**이어야 한다:

**타입 안전성:**

```mlir
// 올바른 타입: !funlang.list<i32>
%int_list = funlang.nil : !funlang.list<i32>
%int_cons = funlang.cons %x, %int_list : !funlang.list<i32>
// Type checker verifies: %x must be i32

// 잘못된 타입: !funlang.list (opaque - no element type)
%list = funlang.nil : !funlang.list
%cons = funlang.cons %x, %list : !funlang.list
// Type checker CANNOT verify: %x type unknown
```

**Cons cell storage:**

- Element type은 cons cell에 저장됨 (not in list struct)
- List struct는 tag + pointer만 포함
- Element type은 컴파일 타임 정보 (type safety)

**Type parameter in lowering:**

```
!funlang.list<i32> → !llvm.struct<(i32, ptr)>
!funlang.list<f64> → !llvm.struct<(i32, ptr)>
!funlang.list<!funlang.closure> → !llvm.struct<(i32, ptr)>

// 런타임 표현은 동일! (opaque pointer)
// 컴파일 타임에만 element type 검증
```

### List Representation vs Array Representation

**왜 linked list인가? 배열보다 나은가?**

| Aspect | Linked List | Array |
|--------|-------------|-------|
| Random access | O(n) | O(1) |
| Prepend (cons) | O(1) | O(n) - copy |
| Append | O(n) | O(1) or O(n) |
| Structural sharing | O(1) | Impossible (mutable) |
| Pattern matching | Natural (Nil/Cons) | Complex (length check + index) |
| Memory | Pointer overhead | Contiguous, cache-friendly |

**함수형 언어에서 linked list를 선호하는 이유:**

1. **Immutability**: Sharing이 메모리 효율적
2. **Pattern matching**: Constructor-based decomposition 자연스러움
3. **Recursion**: Recursive structure와 recursive functions 매칭
4. **Prepend**: 대부분의 list operations는 prepend 중심 (cons, map, filter)

**Array가 더 나은 경우:**

- Random access가 주요 operation
- Numeric computing (SIMD, vectorization)
- Cache locality가 중요한 tight loop

**FunLang의 선택:**

- Phase 6는 linked list로 구현 (함수형 언어 교육 목적)
- Phase 7에서 array/vector 추가 가능 (performance-critical code)

### Comparison with Other Implementations

**OCaml list representation:**

```c
// OCaml runtime
typedef uintnat value;

#define Val_int(x) ((value)((x) << 1) + 1)
#define Int_val(x) ((long)(x) >> 1)

// List: []
#define Val_emptylist Val_int(0)

// List: head :: tail
struct list_cell {
    value header;  // GC header
    value head;
    value tail;
};
```

**Haskell list representation (GHC):**

```c
// Haskell runtime
typedef struct {
    StgHeader header;
    StgClosure *head;
    StgClosure *tail;
} StgCons;

// [] is a special constructor (static object)
```

**FunLang's simpler approach:**

- No GC header (Boehm GC handles this internally)
- Tagged union explicit (tag + data)
- Uniform representation (LLVM struct)

### Summary: List Representation Design

**핵심 결정사항:**

1. **Tagged union**: `!llvm.struct<(i32, ptr)>` for Nil/Cons discrimination
2. **Cons cells**: Heap-allocated `{element, tail}` structs via GC_malloc
3. **Immutability**: 리스트는 생성 후 변경 불가
4. **Structural sharing**: 여러 리스트가 tail을 공유 가능
5. **Parameterized type**: `!funlang.list<T>` for type safety

**다음 섹션에서:**

- `!funlang.list<T>` TableGen 정의
- `funlang.nil` operation 구현
- `funlang.cons` operation 구현

---

## FunLang List Type

이제 list를 표현할 **MLIR type**을 정의한다. Chapter 15에서 배운 parameterized type 패턴을 적용한다.

### Parameterized Type의 필요성

**왜 `!funlang.list`가 아니라 `!funlang.list<T>`인가?**

```mlir
// 잘못된 설계: Opaque list type
def FunLang_ListType : FunLang_Type<"List", "list"> {
  // No type parameters!
}

// 사용 예
%list1 = funlang.nil : !funlang.list  // 어떤 타입의 원소?
%list2 = funlang.cons %x, %list1 : !funlang.list  // %x의 타입은?

// 문제점:
// 1. Type checker가 element type을 검증할 수 없음
// 2. funlang.cons의 head 타입이 tail의 element type과 일치하는지 확인 불가
// 3. funlang.match의 cons region에서 head의 타입을 추론할 수 없음
```

**올바른 설계: Parameterized type**

```mlir
def FunLang_ListType : FunLang_Type<"List", "list", [
    TypeParameter<"Type", "elementType">
]> {
  // Type parameter: T
}

// 사용 예
%int_list = funlang.nil : !funlang.list<i32>
%float_list = funlang.nil : !funlang.list<f64>
%closure_list = funlang.nil : !funlang.list<!funlang.closure>

// 장점:
// 1. Type checker가 element type 검증
// 2. funlang.cons %x, %tail에서 %x : T (T는 tail의 element type)
// 3. funlang.match의 ^cons region에서 head : T
```

### TableGen Type Definition

**파일: `mlir/include/mlir/Dialect/FunLang/FunLangOps.td`**

```tablegen
//===----------------------------------------------------------------------===//
// FunLang Types
//===----------------------------------------------------------------------===//

// ClosureType (Chapter 15)
def FunLang_ClosureType : FunLang_Type<"Closure", "closure"> {
  let summary = "FunLang closure type (opaque)";

  let description = [{
    Represents a closure (function + captured environment).

    Syntax: `!funlang.closure`

    Lowering:
    - FunLang dialect: !funlang.closure
    - LLVM dialect: !llvm.ptr

    Internal representation (after lowering):
    ```
    struct {
        ptr fn_ptr;      // function pointer
        T1 capture1;     // captured variable 1
        T2 capture2;     // captured variable 2
        ...
    }
    ```
  }];
}

// ListType (Chapter 18)
def FunLang_ListType : FunLang_Type<"List", "list", [
    TypeParameter<"Type", "elementType">
]> {
  let summary = "FunLang immutable list type";

  let description = [{
    Represents an immutable linked list with type parameter.

    Syntax: `!funlang.list<T>`

    Type parameter:
    - T: Element type (any MLIR type)

    Examples:
    ```
    !funlang.list<i32>          // List of integers
    !funlang.list<f64>          // List of floats
    !funlang.list<!funlang.closure>  // List of closures
    !funlang.list<!funlang.list<i32>>  // List of lists (nested)
    ```

    Lowering:
    - FunLang dialect: !funlang.list<T>
    - LLVM dialect: !llvm.struct<(i32, ptr)>

    Internal representation (after lowering):
    ```
    struct TaggedUnion {
        i32 tag;        // 0 = Nil, 1 = Cons
        ptr data;       // nullptr for Nil, cons cell pointer for Cons
    }

    struct ConsCell {
        T element;      // head element
        TaggedUnion tail;  // tail list
    }
    ```

    Note: Element type T is compile-time information only.
          Runtime representation is uniform (opaque pointer).
  }];

  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
      return Base::get($_ctxt, elementType);
    }]>
  ];
}
```

**핵심 요소:**

1. **Type parameter**: `TypeParameter<"Type", "elementType">`
   - C++ 클래스에서 `Type getElementType() const` 메서드 생성
   - Assembly format에서 `!funlang.list<i32>` 형태로 출력

2. **Assembly format**: `` "`<` $elementType `>`" ``
   - `<T>` syntax for parameterized type
   - TableGen이 parser/printer 자동 생성

3. **Builder**: 편의를 위한 생성자
   - `FunLangListType::get(context, elementType)`

### Generated C++ Interface

TableGen이 생성하는 C++ 코드:

```cpp
// mlir/include/mlir/Dialect/FunLang/FunLangTypes.h

namespace mlir {
namespace funlang {

class FunLangListType : public Type::TypeBase<
    FunLangListType,
    Type,
    detail::FunLangListTypeStorage,   // Storage for type parameters
    TypeTrait::HasTypeParameter> {    // Trait for parameterized types
public:
  using Base::Base;

  /// Create !funlang.list<elementType>
  static FunLangListType get(MLIRContext *context, Type elementType);

  /// Get element type from !funlang.list<T>
  Type getElementType() const;

  /// Parse !funlang.list<T> from assembly
  static Type parse(AsmParser &parser);

  /// Print !funlang.list<T> to assembly
  void print(AsmPrinter &printer) const;

  /// Verify type parameter is valid
  static LogicalResult verify(
      function_ref<InFlightDiagnostic()> emitError,
      Type elementType);
};

} // namespace funlang
} // namespace mlir
```

**Storage implementation (TableGen이 생성):**

```cpp
namespace mlir {
namespace funlang {
namespace detail {

struct FunLangListTypeStorage : public TypeStorage {
  using KeyTy = Type;  // elementType is the key

  FunLangListTypeStorage(Type elementType) : elementType(elementType) {}

  bool operator==(const KeyTy &key) const {
    return elementType == key;
  }

  static FunLangListTypeStorage *construct(
      TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<FunLangListTypeStorage>())
        FunLangListTypeStorage(key);
  }

  Type elementType;
};

} // namespace detail
} // namespace funlang
} // namespace mlir
```

### Type Uniquing

MLIR은 type uniquing을 자동으로 수행한다:

```cpp
// Same element type → same type instance
auto ctx = /* context */;
auto i32Ty = IntegerType::get(ctx, 32);

auto listTy1 = FunLangListType::get(ctx, i32Ty);
auto listTy2 = FunLangListType::get(ctx, i32Ty);

assert(listTy1 == listTy2);  // Same pointer!
```

**장점:**

- Type comparison은 pointer equality (`==`)
- Type hashing 효율적
- Memory 효율적 (각 unique type은 한 번만 저장)

### C API Shim

F#에서 사용하기 위한 C API:

**파일: `mlir/lib/CAPI/Dialect/FunLang.cpp`**

```cpp
//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

/// Create !funlang.list<elementType>
MlirType mlirFunLangListTypeGet(MlirContext ctx, MlirType elementType) {
  return wrap(funlang::FunLangListType::get(
      unwrap(ctx), unwrap(elementType)));
}

/// Check if type is !funlang.list
bool mlirTypeIsAFunLangListType(MlirType ty) {
  return unwrap(ty).isa<funlang::FunLangListType>();
}

/// Get element type from !funlang.list<T>
MlirType mlirFunLangListTypeGetElementType(MlirType ty) {
  auto listTy = unwrap(ty).cast<funlang::FunLangListType>();
  return wrap(listTy.getElementType());
}
```

**헤더 파일: `mlir/include/mlir-c/Dialect/FunLang.h`**

```c
#ifndef MLIR_C_DIALECT_FUNLANG_H
#define MLIR_C_DIALECT_FUNLANG_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

/// Create !funlang.list<elementType> type
MLIR_CAPI_EXPORTED MlirType
mlirFunLangListTypeGet(MlirContext ctx, MlirType elementType);

/// Check if type is !funlang.list
MLIR_CAPI_EXPORTED bool
mlirTypeIsAFunLangListType(MlirType ty);

/// Get element type from !funlang.list<T>
MLIR_CAPI_EXPORTED MlirType
mlirFunLangListTypeGetElementType(MlirType ty);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_FUNLANG_H
```

### F# Bindings

**파일: `FunLang.Compiler/MlirBindings.fs`**

```fsharp
module FunLangBindings =
    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangListTypeGet(MlirContext ctx, MlirType elementType)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern bool mlirTypeIsAFunLangListType(MlirType ty)

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirType mlirFunLangListTypeGetElementType(MlirType ty)
```

**FunLangDialect wrapper:**

```fsharp
type FunLangDialect(ctx: MlirContext) =
    member this.Context = ctx

    //==========================================================================
    // Types
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
```

**OpBuilder extension:**

```fsharp
type OpBuilder with
    /// Create !funlang.list<T> type
    member this.FunLangListType(elementType: MlirType) : MlirType =
        let funlang = FunLangDialect(this.Context)
        funlang.ListType(elementType)
```

### F# Usage Examples

```fsharp
// F# compiler code
let compileListExpr (builder: OpBuilder) =
    // Create type: !funlang.list<i32>
    let i32Type = builder.IntegerType(32)
    let listType = builder.FunLangListType(i32Type)

    // Create empty list
    let nil = builder.CreateNil(listType)

    // Create cons cell
    let head = (* some i32 value *)
    let cons = builder.CreateCons(head, nil)

    cons

// Check if type is list type
let isListType (ty: MlirType) =
    let funlang = FunLangDialect(ctx)
    funlang.IsListType(ty)

// Get element type
let getElementType (listTy: MlirType) =
    let funlang = FunLangDialect(ctx)
    if funlang.IsListType(listTy) then
        Some (funlang.ListElementType(listTy))
    else
        None
```

### Nested List Types

Parameterized type이므로 중첩 가능:

```mlir
// List of lists
!funlang.list<!funlang.list<i32>>

// Example: [[1, 2], [3, 4]]
%inner_nil = funlang.nil : !funlang.list<i32>
%inner1 = funlang.cons %c2, %inner_nil : !funlang.list<i32>
%inner1 = funlang.cons %c1, %inner1 : !funlang.list<i32>  // [1, 2]

%inner2 = funlang.cons %c4, %inner_nil : !funlang.list<i32>
%inner2 = funlang.cons %c3, %inner2 : !funlang.list<i32>  // [3, 4]

%outer_nil = funlang.nil : !funlang.list<!funlang.list<i32>>
%outer = funlang.cons %inner2, %outer_nil : !funlang.list<!funlang.list<i32>>
%outer = funlang.cons %inner1, %outer : !funlang.list<!funlang.list<i32>>
// [[1, 2], [3, 4]]
```

**Lowering:**

```
!funlang.list<!funlang.list<i32>> → !llvm.struct<(i32, ptr)>

// 동일한 표현! Element type은 컴파일 타임 정보만
```

### Type Verification

TableGen이 자동으로 verification 생성하지만, 추가 검증 가능:

```cpp
LogicalResult FunLangListType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    Type elementType) {
  // Element type must be non-null
  if (!elementType)
    return emitError() << "list element type cannot be null";

  // Additional constraints (if needed)
  // e.g., element type must be first-class (no void, etc.)

  return success();
}
```

### Summary: FunLang List Type

**구현 완료:**

- [x] `!funlang.list<T>` parameterized type in TableGen
- [x] C++ interface with `getElementType()` method
- [x] C API shim: `mlirFunLangListTypeGet`, `mlirTypeIsAFunLangListType`, `mlirFunLangListTypeGetElementType`
- [x] F# bindings in `FunLangDialect` class
- [x] OpBuilder extension for convenient usage

**다음 섹션:**

- `funlang.nil` operation으로 empty list 생성
- `funlang.cons` operation으로 cons cell 생성

---

## funlang.nil Operation

Empty list를 생성하는 operation을 구현한다.

### Purpose and Semantics

**funlang.nil의 역할:**

- Empty list (빈 리스트) 생성
- 리스트의 base case (재귀의 종료 조건)
- Runtime allocation 불필요 (constant representation)

**예제:**

```mlir
// Create empty list of integers
%nil = funlang.nil : !funlang.list<i32>

// Create empty list of floats
%nil = funlang.nil : !funlang.list<f64>

// Create empty list of closures
%nil = funlang.nil : !funlang.list<!funlang.closure>
```

**의미:**

```
funlang.nil : !funlang.list<T>

// Equivalent to (after lowering):
{tag: 0, data: null}
```

### TableGen ODS Definition

**파일: `mlir/include/mlir/Dialect/FunLang/FunLangOps.td`**

```tablegen
//===----------------------------------------------------------------------===//
// List Operations
//===----------------------------------------------------------------------===//

def FunLang_NilOp : FunLang_Op<"nil", [Pure]> {
  let summary = "Create empty list";

  let description = [{
    Creates an empty list (Nil constructor).

    Syntax:
    ```
    %nil = funlang.nil : !funlang.list<T>
    ```

    The result type specifies the element type of the list.

    Examples:
    ```
    // Empty list of integers
    %nil_int = funlang.nil : !funlang.list<i32>

    // Empty list of closures
    %nil_closure = funlang.nil : !funlang.list<!funlang.closure>
    ```

    Lowering:
    ```
    %nil = funlang.nil : !funlang.list<i32>

    // Lowers to:
    %tag = arith.constant 0 : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
    %nil = llvm.insertvalue %null, %s1[1] : !llvm.struct<(i32, ptr)>
    ```

    Traits: Pure (no side effects, no memory allocation)
  }];

  let arguments = (ins);

  let results = (outs FunLang_ListType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "Type":$elementType), [{
      auto listType = funlang::FunLangListType::get($_builder.getContext(), elementType);
      $_state.addTypes(listType);
    }]>
  ];
}
```

**핵심 요소:**

1. **Pure trait**: No side effects, 메모리 할당 없음
   - CSE (Common Subexpression Elimination) 가능
   - 같은 element type의 nil은 한 번만 생성 가능

2. **No arguments**: Empty list는 인자 불필요

3. **Result type**: `!funlang.list<T>` (element type 명시 필요)

4. **Assembly format**: `funlang.nil : !funlang.list<i32>`
   - Type suffix로 element type 지정

5. **Builder**: Element type만으로 NilOp 생성 가능

### Generated C++ Interface

TableGen이 생성하는 C++ 코드:

```cpp
// mlir/include/mlir/Dialect/FunLang/FunLangOps.h

namespace mlir {
namespace funlang {

class NilOp : public Op<
    NilOp,
    OpTrait::ZeroOperands,
    OpTrait::OneResult,
    OpTrait::Pure> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "funlang.nil"; }

  /// Get result type (!funlang.list<T>)
  FunLangListType getType() {
    return getResult().getType().cast<FunLangListType>();
  }

  /// Get element type (T from !funlang.list<T>)
  Type getElementType() {
    return getType().getElementType();
  }

  /// Build NilOp with element type
  static void build(
      OpBuilder &builder,
      OperationState &state,
      Type elementType);

  /// Verify operation
  LogicalResult verify();

  /// Parse from assembly
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  /// Print to assembly
  void print(OpAsmPrinter &p);
};

} // namespace funlang
} // namespace mlir
```

### Verification

```cpp
LogicalResult NilOp::verify() {
  // Result must be !funlang.list<T>
  auto resultTy = getResult().getType();
  if (!resultTy.isa<FunLangListType>()) {
    return emitOpError("result must be !funlang.list type");
  }

  // Element type must be valid (checked by FunLangListType::verify)
  return success();
}
```

### C API Shim

**파일: `mlir/lib/CAPI/Dialect/FunLang.cpp`**

```cpp
//===----------------------------------------------------------------------===//
// NilOp
//===----------------------------------------------------------------------===//

MlirOperation mlirFunLangNilOpCreate(
    MlirLocation loc,
    MlirType elementType) {
  mlir::OpBuilder builder(unwrap(loc)->getContext());
  builder.setInsertionPointToStart(/* appropriate block */);

  auto listType = funlang::FunLangListType::get(
      unwrap(loc)->getContext(), unwrap(elementType));

  auto op = builder.create<funlang::NilOp>(
      unwrap(loc), listType);

  return wrap(op.getOperation());
}
```

**헤더 파일: `mlir/include/mlir-c/Dialect/FunLang.h`**

```c
//===----------------------------------------------------------------------===//
// NilOp
//===----------------------------------------------------------------------===//

/// Create funlang.nil operation
/// Returns MlirOperation (not MlirValue - use mlirOperationGetResult)
MLIR_CAPI_EXPORTED MlirOperation
mlirFunLangNilOpCreate(MlirLocation loc, MlirType elementType);
```

### F# Bindings

**파일: `FunLang.Compiler/MlirBindings.fs`**

```fsharp
module FunLangBindings =
    // ... (previous bindings) ...

    //==========================================================================
    // Operations - NilOp
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirFunLangNilOpCreate(MlirLocation loc, MlirType elementType)
```

**FunLangDialect wrapper:**

```fsharp
type FunLangDialect(ctx: MlirContext) =
    // ... (previous members) ...

    //==========================================================================
    // Operation Creation
    //==========================================================================

    /// Create funlang.nil operation
    member this.CreateNilOp(loc: MlirLocation, elementType: MlirType) : MlirOperation =
        FunLangBindings.mlirFunLangNilOpCreate(loc, elementType)

    /// Create funlang.nil and return the result value
    member this.CreateNil(loc: MlirLocation, elementType: MlirType) : MlirValue =
        let op = this.CreateNilOp(loc, elementType)
        MlirHelpers.GetOperationResult(op, 0)
```

**OpBuilder extension:**

```fsharp
type OpBuilder with
    // ... (previous members) ...

    /// Create funlang.nil operation
    member this.CreateNilOp(elementType: MlirType) : MlirOperation =
        let funlang = FunLangDialect(this.Context)
        funlang.CreateNilOp(this.UnknownLoc, elementType)

    /// Create funlang.nil and return result value
    member this.CreateNil(elementType: MlirType) : MlirValue =
        let funlang = FunLangDialect(this.Context)
        funlang.CreateNil(this.UnknownLoc, elementType)
```

### F# Usage Examples

```fsharp
// Example 1: Basic usage
let builder = OpBuilder(ctx)

let i32Type = builder.IntegerType(32)
let nilValue = builder.CreateNil(i32Type)
// %nil = funlang.nil : !funlang.list<i32>

// Example 2: Building list [1, 2, 3] (forward)
let nil = builder.CreateNil(i32Type)
let c1 = builder.CreateConstantInt(1, 32)
let c2 = builder.CreateConstantInt(2, 32)
let c3 = builder.CreateConstantInt(3, 32)

// Build from right to left: 3 → 2 → 1 → nil
let lst1 = builder.CreateCons(c3, nil)    // [3]
let lst2 = builder.CreateCons(c2, lst1)   // [2, 3]
let lst3 = builder.CreateCons(c1, lst2)   // [1, 2, 3]

// Example 3: Empty list of different types
let floatType = builder.FloatType(64)
let nilFloat = builder.CreateNil(floatType)
// %nil = funlang.nil : !funlang.list<f64>

let closureType = builder.FunLangClosureType()
let nilClosure = builder.CreateNil(closureType)
// %nil = funlang.nil : !funlang.list<!funlang.closure>
```

### No Runtime Allocation Needed

**중요한 최적화 기회:**

```mlir
// Multiple nil operations
%nil1 = funlang.nil : !funlang.list<i32>
%nil2 = funlang.nil : !funlang.list<i32>
%nil3 = funlang.nil : !funlang.list<i32>

// Pure trait enables CSE:
// → All replaced with single %nil!

// Lowering (only once):
%tag = arith.constant 0 : i32
%null = llvm.mlir.zero : !llvm.ptr
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%nil = llvm.insertvalue %null, %s1[1] : !llvm.struct<(i32, ptr)>

// No GC_malloc call! (constant struct)
```

**Static empty list (advanced optimization - Phase 7):**

```cpp
// Could use global constant for empty list
static const struct { int tag; void* data; } EMPTY_LIST = {0, NULL};

// All funlang.nil → load from EMPTY_LIST address
```

### Summary: funlang.nil Operation

**구현 완료:**

- [x] TableGen ODS definition with Pure trait
- [x] No arguments, result type is `!funlang.list<T>`
- [x] C API shim: `mlirFunLangNilOpCreate`
- [x] F# bindings: `CreateNilOp`, `CreateNil`
- [x] OpBuilder extension for convenient usage

**특징:**

- Pure operation (CSE 가능)
- No runtime allocation
- Result type으로 element type 지정

**다음 섹션:**

- `funlang.cons` operation으로 cons cell 생성

---

## funlang.cons Operation

Cons cell을 생성하는 operation을 구현한다. 리스트의 핵심 생성자다.

### Purpose and Semantics

**funlang.cons의 역할:**

- Non-empty list 생성 (head :: tail)
- 리스트의 recursive case
- GC를 통한 heap allocation

**예제:**

```mlir
// Prepend element to list
%lst = funlang.cons %head, %tail : !funlang.list<i32>

// Build list [1, 2, 3]
%nil = funlang.nil : !funlang.list<i32>
%c3 = arith.constant 3 : i32
%lst1 = funlang.cons %c3, %nil : !funlang.list<i32>    // [3]
%c2 = arith.constant 2 : i32
%lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>   // [2, 3]
%c1 = arith.constant 1 : i32
%lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>   // [1, 2, 3]
```

**의미:**

```
funlang.cons %head, %tail : !funlang.list<T>

// Equivalent to (after lowering):
cell = GC_malloc(sizeof(T) + sizeof(ptr))
cell->head = %head
cell->tail = %tail
result = {tag: 1, data: cell}
```

### TableGen ODS Definition

**파일: `mlir/include/mlir/Dialect/FunLang/FunLangOps.td`**

```tablegen
def FunLang_ConsOp : FunLang_Op<"cons", []> {
  let summary = "Create cons cell (non-empty list)";

  let description = [{
    Creates a cons cell by prepending an element to a list.

    Syntax:
    ```
    %result = funlang.cons %head, %tail : !funlang.list<T>
    ```

    Arguments:
    - `head`: Element to prepend (type T)
    - `tail`: Existing list (type !funlang.list<T>)

    Result:
    - New list with `head` prepended to `tail` (type !funlang.list<T>)

    Type constraints:
    - `head` type must match element type of `tail` list
    - Result type is same as `tail` type

    Examples:
    ```
    // Create [1]
    %nil = funlang.nil : !funlang.list<i32>
    %c1 = arith.constant 1 : i32
    %lst = funlang.cons %c1, %nil : !funlang.list<i32>

    // Create [1, 2, 3]
    %c3 = arith.constant 3 : i32
    %lst1 = funlang.cons %c3, %nil : !funlang.list<i32>
    %c2 = arith.constant 2 : i32
    %lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>
    %lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>
    ```

    Lowering:
    ```
    %lst = funlang.cons %head, %tail : !funlang.list<i32>

    // Lowers to:
    // 1. Allocate cons cell
    %size = arith.constant 16 : i64  // sizeof(i32) + sizeof(struct)
    %cell = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr

    // 2. Store head
    %head_ptr = llvm.getelementptr %cell[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %head, %head_ptr : i32, !llvm.ptr

    // 3. Store tail
    %tail_ptr = llvm.getelementptr %cell[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %tail, %tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr

    // 4. Build tagged union
    %tag = arith.constant 1 : i32
    %undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
    %lst = llvm.insertvalue %cell, %s1[1] : !llvm.struct<(i32, ptr)>
    ```

    Note: No Pure trait (allocates memory via GC_malloc)
  }];

  let arguments = (ins AnyType:$head, FunLang_ListType:$tail);

  let results = (outs FunLang_ListType:$result);

  let assemblyFormat = "$head `,` $tail attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "Value":$head, "Value":$tail), [{
      auto tailType = tail.getType().cast<funlang::FunLangListType>();
      $_state.addOperands({head, tail});
      $_state.addTypes(tailType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get element type (T from !funlang.list<T>)
    Type getElementType() {
      return getResult().getType().cast<FunLangListType>().getElementType();
    }
  }];
}
```

**핵심 요소:**

1. **No Pure trait**: GC_malloc 호출로 side effect 발생
   - CSE 불가능 (각 cons는 새로운 cell 할당)
   - Dead code elimination 신중하게 (allocation 유지 필요할 수도)

2. **Arguments**: `head` (element), `tail` (list)
   - `head`: AnyType (element type은 tail과 검증)
   - `tail`: FunLang_ListType

3. **Result type**: Same as `tail` type
   - Builder가 자동으로 tail의 타입을 result에 사용

4. **Assembly format**: `funlang.cons %head, %tail : !funlang.list<i32>`

5. **extraClassDeclaration**: `getElementType()` 헬퍼 메서드

### Type Constraints and Verification

```cpp
LogicalResult ConsOp::verify() {
  // Tail must be !funlang.list<T>
  auto tailType = getTail().getType().dyn_cast<FunLangListType>();
  if (!tailType) {
    return emitOpError("tail must be !funlang.list type");
  }

  // Result must be same type as tail
  auto resultType = getResult().getType().dyn_cast<FunLangListType>();
  if (!resultType || resultType != tailType) {
    return emitOpError("result type must match tail type");
  }

  // Head type must match element type of list
  Type headType = getHead().getType();
  Type elemType = tailType.getElementType();
  if (headType != elemType) {
    return emitOpError("head type (")
        << headType << ") must match list element type (" << elemType << ")";
  }

  return success();
}
```

**검증하는 제약조건:**

1. Tail은 `!funlang.list<T>` 타입이어야 함
2. Result 타입은 tail 타입과 동일해야 함
3. Head 타입은 list의 element 타입과 일치해야 함

**예제: Type errors**

```mlir
// Error: head type mismatch
%nil = funlang.nil : !funlang.list<i32>
%f = arith.constant 3.14 : f64
%bad = funlang.cons %f, %nil : !funlang.list<i32>
// Error: head type (f64) must match list element type (i32)

// Error: tail not a list
%x = arith.constant 42 : i32
%bad = funlang.cons %x, %x : !funlang.list<i32>
// Error: tail must be !funlang.list type

// Error: result type mismatch
%nil_int = funlang.nil : !funlang.list<i32>
%x = arith.constant 42 : i32
%bad = funlang.cons %x, %nil_int : !funlang.list<f64>
// Error: result type must match tail type
```

### C API Shim

**파일: `mlir/lib/CAPI/Dialect/FunLang.cpp`**

```cpp
//===----------------------------------------------------------------------===//
// ConsOp
//===----------------------------------------------------------------------===//

MlirOperation mlirFunLangConsOpCreate(
    MlirLocation loc,
    MlirValue head,
    MlirValue tail) {
  mlir::OpBuilder builder(unwrap(loc)->getContext());
  builder.setInsertionPointToStart(/* appropriate block */);

  auto op = builder.create<funlang::ConsOp>(
      unwrap(loc),
      unwrap(head),
      unwrap(tail));

  return wrap(op.getOperation());
}
```

**헤더 파일: `mlir/include/mlir-c/Dialect/FunLang.h`**

```c
//===----------------------------------------------------------------------===//
// ConsOp
//===----------------------------------------------------------------------===//

/// Create funlang.cons operation
/// Arguments:
///   - head: Element to prepend
///   - tail: Existing list
/// Returns MlirOperation (use mlirOperationGetResult to get value)
MLIR_CAPI_EXPORTED MlirOperation
mlirFunLangConsOpCreate(
    MlirLocation loc,
    MlirValue head,
    MlirValue tail);
```

### F# Bindings

**파일: `FunLang.Compiler/MlirBindings.fs`**

```fsharp
module FunLangBindings =
    // ... (previous bindings) ...

    //==========================================================================
    // Operations - ConsOp
    //==========================================================================

    [<DllImport("MLIR-FunLang-CAPI", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirOperation mlirFunLangConsOpCreate(
        MlirLocation loc,
        MlirValue head,
        MlirValue tail)
```

**FunLangDialect wrapper:**

```fsharp
type FunLangDialect(ctx: MlirContext) =
    // ... (previous members) ...

    /// Create funlang.cons operation
    member this.CreateConsOp(loc: MlirLocation, head: MlirValue, tail: MlirValue) : MlirOperation =
        FunLangBindings.mlirFunLangConsOpCreate(loc, head, tail)

    /// Create funlang.cons and return the result value
    member this.CreateCons(loc: MlirLocation, head: MlirValue, tail: MlirValue) : MlirValue =
        let op = this.CreateConsOp(loc, head, tail)
        MlirHelpers.GetOperationResult(op, 0)
```

**OpBuilder extension:**

```fsharp
type OpBuilder with
    // ... (previous members) ...

    /// Create funlang.cons operation
    member this.CreateConsOp(head: MlirValue, tail: MlirValue) : MlirOperation =
        let funlang = FunLangDialect(this.Context)
        funlang.CreateConsOp(this.UnknownLoc, head, tail)

    /// Create funlang.cons and return result value
    member this.CreateCons(head: MlirValue, tail: MlirValue) : MlirValue =
        let funlang = FunLangDialect(this.Context)
        funlang.CreateCons(this.UnknownLoc, head, tail)
```

### F# Usage Examples

```fsharp
// Example 1: Build single-element list [42]
let builder = OpBuilder(ctx)
let i32Type = builder.IntegerType(32)

let nil = builder.CreateNil(i32Type)
let c42 = builder.CreateConstantInt(42, 32)
let lst = builder.CreateCons(c42, nil)
// %lst = funlang.cons %c42, %nil : !funlang.list<i32>

// Example 2: Build list [1, 2, 3]
let nil = builder.CreateNil(i32Type)
let c1 = builder.CreateConstantInt(1, 32)
let c2 = builder.CreateConstantInt(2, 32)
let c3 = builder.CreateConstantInt(3, 32)

// Build from right to left
let lst1 = builder.CreateCons(c3, nil)    // [3]
let lst2 = builder.CreateCons(c2, lst1)   // [2, 3]
let lst3 = builder.CreateCons(c1, lst2)   // [1, 2, 3]

// Example 3: Build list from F# list
let buildList (builder: OpBuilder) (elements: MlirValue list) (elemType: MlirType) =
    let nil = builder.CreateNil(elemType)
    List.foldBack (fun elem acc ->
        builder.CreateCons(elem, acc)
    ) elements nil

let values = [c1; c2; c3]
let lst = buildList builder values i32Type
// funlang.cons %c1, (funlang.cons %c2, (funlang.cons %c3, %nil))

// Example 4: Type inference from tail
let tail = (* existing !funlang.list<i32> *)
let head = builder.CreateConstantInt(99, 32)
let extended = builder.CreateCons(head, tail)
// Result type inferred from tail type
```

### Memory Allocation Details

**Cons cell size calculation:**

```
ConsCell<T> = struct {
    T element;
    TaggedUnion tail;  // struct { i32 tag; ptr data }
}

Size = sizeof(T) + sizeof(i32) + sizeof(ptr)

Examples:
- i32: 4 + 4 + 8 = 16 bytes
- f64: 8 + 4 + 8 = 20 bytes (alignment → 24 bytes)
- !funlang.closure (ptr): 8 + 4 + 8 = 20 bytes (alignment → 24 bytes)
```

**Lowering에서 size 계산:**

```cpp
// ConsOpLowering::matchAndRewrite
Value ConsOpLowering::calculateCellSize(
    OpBuilder &builder, Location loc, Type elementType) {
  auto &dataLayout = getDataLayout();

  // Get element size
  uint64_t elemSize = dataLayout.getTypeSize(elementType);

  // TaggedUnion size: i32 (4 bytes) + ptr (8 bytes) = 12 bytes
  // But alignment: struct<(i32, ptr)> → 16 bytes on 64-bit
  uint64_t tailSize = 16;  // Hardcoded for simplicity

  uint64_t totalSize = elemSize + tailSize;

  // Align to 8 bytes
  totalSize = (totalSize + 7) & ~7;

  return builder.create<arith::ConstantIntOp>(
      loc, totalSize, builder.getI64Type());
}
```

### List Construction Patterns

**Pattern 1: Build from literal**

```fsharp
// F# source: [1; 2; 3]
let lst = [1; 2; 3]

// MLIR output:
%nil = funlang.nil : !funlang.list<i32>
%c3 = arith.constant 3 : i32
%lst1 = funlang.cons %c3, %nil : !funlang.list<i32>
%c2 = arith.constant 2 : i32
%lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>
%c1 = arith.constant 1 : i32
%lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>
```

**Pattern 2: Recursive construction**

```fsharp
// F# source
let rec range n =
    if n <= 0 then []
    else n :: range (n - 1)

// MLIR output (simplified):
func.func @range(%n: i32) -> !funlang.list<i32> {
    %zero = arith.constant 0 : i32
    %cond = arith.cmpi sle, %n, %zero : i32
    %result = scf.if %cond -> !funlang.list<i32> {
        %nil = funlang.nil : !funlang.list<i32>
        scf.yield %nil : !funlang.list<i32>
    } else {
        %one = arith.constant 1 : i32
        %n_minus_1 = arith.subi %n, %one : i32
        %tail = func.call @range(%n_minus_1) : (i32) -> !funlang.list<i32>
        %cons = funlang.cons %n, %tail : !funlang.list<i32>
        scf.yield %cons : !funlang.list<i32>
    }
    func.return %result : !funlang.list<i32>
}
```

**Pattern 3: List transformation (map)**

```fsharp
// F# source
let rec map f lst =
    match lst with
    | [] -> []
    | head :: tail -> f head :: map f tail

// MLIR output (with funlang.match - Chapter 19):
func.func @map(
    %f: !funlang.closure,
    %lst: !funlang.list<i32>
) -> !funlang.list<i32> {
    %result = funlang.match %lst : !funlang.list<i32> -> !funlang.list<i32> {
      ^nil:
        %nil = funlang.nil : !funlang.list<i32>
        funlang.yield %nil : !funlang.list<i32>
      ^cons(%head: i32, %tail: !funlang.list<i32>):
        %new_head = funlang.apply %f(%head) : (i32) -> i32
        %new_tail = func.call @map(%f, %tail)
            : (!funlang.closure, !funlang.list<i32>) -> !funlang.list<i32>
        %new_cons = funlang.cons %new_head, %new_tail : !funlang.list<i32>
        funlang.yield %new_cons : !funlang.list<i32>
    }
    func.return %result : !funlang.list<i32>
}
```

### Summary: funlang.cons Operation

**구현 완료:**

- [x] TableGen ODS definition (no Pure trait)
- [x] Arguments: head (element), tail (list)
- [x] Type verification: head type matches element type
- [x] C API shim: `mlirFunLangConsOpCreate`
- [x] F# bindings: `CreateConsOp`, `CreateCons`
- [x] OpBuilder extension for convenient usage

**특징:**

- GC allocation for cons cells
- Type-safe: head type must match list element type
- Result type inferred from tail type

**다음 Part:**

- TypeConverter for `!funlang.list<T>`
- NilOpLowering pattern
- ConsOpLowering pattern
- Complete lowering pass integration

---

## 튜플 타입과 연산 (Tuple Type and Operations)

리스트와 함께 함수형 프로그래밍에서 필수적인 또 다른 데이터 구조가 있다: **튜플(tuple)**이다. 리스트가 같은 타입의 여러 원소를 가변 개수로 담는다면, 튜플은 서로 다른 타입의 원소들을 고정된 개수로 묶는다.

### 튜플 vs 리스트: 근본적인 차이

**List:**
- 가변 개수 (0개부터 N개까지)
- 동질적 (모든 원소가 같은 타입)
- 런타임에 태그로 Nil/Cons 구분 필요
- 패턴 매칭에서 여러 case 필요

```fsharp
// 리스트: 가변 길이, 같은 타입
let numbers: int list = [1; 2; 3; 4; 5]
let empty: int list = []
let singleton: int list = [42]
```

**Tuple:**
- 고정 개수 (컴파일 타임에 결정)
- 이질적 (원소마다 다른 타입 가능)
- 태그 불필요 (항상 같은 구조)
- 패턴 매칭에서 단일 case (항상 매칭)

```fsharp
// 튜플: 고정 길이, 다른 타입 가능
let pair: int * string = (42, "hello")
let triple: int * float * bool = (1, 3.14, true)
let person: string * int = ("Alice", 30)
```

**메모리 표현의 차이:**

```
List [1, 2, 3] (가변, 태그 필요):
┌─────────┬─────────┐     ┌─────────┬─────────┐     ┌─────────┬─────────┐     ┌─────────┬─────────┐
│ tag=1   │ ptr  ───────► │ head=1  │ tail ────────► │ head=2  │ tail ────────► │ head=3  │ tail=NULL │
│ (Cons)  │         │     │         │         │     │         │         │     │         │         │
└─────────┴─────────┘     └─────────┴─────────┘     └─────────┴─────────┘     └─────────┴─────────┘

Tuple (1, "hello") (고정, 태그 불필요):
┌─────────┬─────────┐
│  int=1  │ ptr ────────► "hello"
│ (slot0) │ (slot1) │
└─────────┴─────────┘
```

### 튜플 타입 설계 (Tuple Type Design)

FunLang에서 튜플 타입의 문법:

```mlir
// 2-tuple (pair)
!funlang.tuple<i32, f64>

// 3-tuple (triple)
!funlang.tuple<i32, string, bool>

// Nested tuple
!funlang.tuple<!funlang.tuple<i32, i32>, f64>

// Tuple of lists
!funlang.tuple<!funlang.list<i32>, !funlang.list<f64>>
```

**타입 시스템에서의 특징:**

1. **Arity가 타입에 인코딩**: `!funlang.tuple<i32>` (1-tuple)과 `!funlang.tuple<i32, i32>` (2-tuple)은 다른 타입
2. **원소 타입 순서가 중요**: `!funlang.tuple<i32, f64>` ≠ `!funlang.tuple<f64, i32>`
3. **Unit type**: 0-tuple `!funlang.tuple<>`은 unit type으로 사용 가능

**LLVM으로의 lowering:**

```mlir
// FunLang tuple type
!funlang.tuple<i32, f64>

// LLVM struct type (no tag needed!)
!llvm.struct<(i32, f64)>
```

리스트와 달리:
- **태그 필요 없음**: 튜플은 항상 같은 구조
- **포인터 indirection 없음**: 값 자체를 struct에 저장 (작은 튜플의 경우)
- **스택 할당 가능**: escape하지 않으면 힙 할당 불필요

### TableGen 정의 (TableGen Definition)

**파일: `mlir/include/Dialect/FunLang/FunLangTypes.td`**

```tablegen
//===----------------------------------------------------------------------===//
// Tuple Type
//===----------------------------------------------------------------------===//

def FunLang_TupleType : FunLang_Type<"Tuple", "tuple"> {
  let summary = "FunLang tuple type";
  let description = [{
    A fixed-size product type with heterogeneous elements.
    Unlike lists, tuples have a known arity at compile time.

    Examples:
    - `!funlang.tuple<i32, f64>` is a pair of integer and float
    - `!funlang.tuple<i32, i32, i32>` is a triple of integers
    - `!funlang.tuple<>` is the unit type (empty tuple)

    Tuples are lowered to LLVM structs directly, without tags,
    because they always have the same structure (no variants).
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::Type", "element types">:$elementTypes
  );

  let assemblyFormat = "`<` $elementTypes `>`";

  let extraClassDeclaration = [{
    /// Get the number of elements in this tuple
    size_t getNumElements() const { return getElementTypes().size(); }

    /// Get the element type at the given index
    mlir::Type getElementType(size_t index) const {
      return getElementTypes()[index];
    }

    /// Check if this is a pair (2-tuple)
    bool isPair() const { return getNumElements() == 2; }

    /// Check if this is a unit type (0-tuple)
    bool isUnit() const { return getNumElements() == 0; }
  }];
}
```

**핵심 요소 분석:**

1. **`ArrayRefParameter`**: 가변 개수의 타입 파라미터
   - `Variadic<Type>`이 아닌 `ArrayRefParameter<"mlir::Type">`
   - TableGen이 자동으로 storage와 accessor 생성

2. **`assemblyFormat`**: `<` 원소타입들 `>`
   - `!funlang.tuple<i32, f64>` 형태로 파싱/프린팅

3. **`extraClassDeclaration`**: 유틸리티 메서드
   - `getNumElements()`, `getElementType(index)` 등

**생성되는 C++ 코드:**

```cpp
// Auto-generated from TableGen
class TupleType : public mlir::Type::TypeBase<TupleType,
                                               mlir::Type,
                                               detail::TupleTypeStorage> {
public:
  using Base::Base;

  static TupleType get(mlir::MLIRContext *context,
                       llvm::ArrayRef<mlir::Type> elementTypes);

  llvm::ArrayRef<mlir::Type> getElementTypes() const;
  size_t getNumElements() const { return getElementTypes().size(); }
  mlir::Type getElementType(size_t index) const {
    return getElementTypes()[index];
  }
  bool isPair() const { return getNumElements() == 2; }
  bool isUnit() const { return getNumElements() == 0; }
};
```

### funlang.make_tuple 연산 (make_tuple Operation)

**파일: `mlir/include/Dialect/FunLang/FunLangOps.td`**

```tablegen
//===----------------------------------------------------------------------===//
// make_tuple Operation
//===----------------------------------------------------------------------===//

def FunLang_MakeTupleOp : FunLang_Op<"make_tuple", [Pure]> {
  let summary = "Create a tuple from values";
  let description = [{
    Constructs a tuple from the given element values.
    The result type must match the types of the input elements.

    Example:
    ```mlir
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 3.14 : f64
    %pair = funlang.make_tuple(%c1, %c2) : !funlang.tuple<i32, f64>
    ```

    The operation is marked Pure because it has no side effects.
    This enables CSE (Common Subexpression Elimination) optimization.
  }];

  let arguments = (ins
    Variadic<AnyType>:$elements
  );

  let results = (outs
    FunLang_TupleType:$result
  );

  let assemblyFormat = [{
    `(` $elements `)` attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$elements), [{
      // Infer result type from element types
      llvm::SmallVector<mlir::Type> elemTypes;
      for (auto elem : elements)
        elemTypes.push_back(elem.getType());

      auto tupleType = TupleType::get($_builder.getContext(), elemTypes);
      build($_builder, $_state, tupleType, elements);
    }]>
  ];

  let hasVerifier = 1;
}
```

**핵심 요소 분석:**

1. **`Variadic<AnyType>`**: 0개 이상의 임의 타입 operands
   - `make_tuple()` (unit), `make_tuple(%a)` (singleton), `make_tuple(%a, %b)` (pair) 모두 가능

2. **`Pure` trait**: 순수 함수
   - 부작용 없음, 같은 입력 → 같은 출력
   - CSE 최적화 가능: 동일한 make_tuple 호출 합치기

3. **Custom builder**: 타입 추론
   - element 타입들로부터 결과 tuple 타입 자동 추론
   - 사용자가 명시적으로 타입을 지정할 필요 없음

4. **Verifier**: 타입 일관성 검증
   - element 개수와 tuple 타입의 arity 일치
   - 각 element 타입과 tuple의 대응 위치 타입 일치

**Verifier 구현:**

```cpp
// FunLangOps.cpp
LogicalResult MakeTupleOp::verify() {
  auto tupleType = getType().cast<TupleType>();
  auto elements = getElements();

  // Check element count matches tuple arity
  if (elements.size() != tupleType.getNumElements()) {
    return emitOpError() << "expected " << tupleType.getNumElements()
                         << " elements but got " << elements.size();
  }

  // Check each element type matches
  for (size_t i = 0; i < elements.size(); ++i) {
    Type expectedType = tupleType.getElementType(i);
    Type actualType = elements[i].getType();
    if (expectedType != actualType) {
      return emitOpError() << "element " << i << " type mismatch: expected "
                           << expectedType << " but got " << actualType;
    }
  }

  return success();
}
```

**사용 예제:**

```mlir
// Empty tuple (unit)
%unit = funlang.make_tuple() : !funlang.tuple<>

// Pair of int and float
%c1 = arith.constant 42 : i32
%c2 = arith.constant 3.14 : f64
%pair = funlang.make_tuple(%c1, %c2) : !funlang.tuple<i32, f64>

// Triple of ints
%a = arith.constant 1 : i32
%b = arith.constant 2 : i32
%c = arith.constant 3 : i32
%triple = funlang.make_tuple(%a, %b, %c) : !funlang.tuple<i32, i32, i32>

// Nested tuple
%inner = funlang.make_tuple(%a, %b) : !funlang.tuple<i32, i32>
%outer = funlang.make_tuple(%inner, %c2) : !funlang.tuple<!funlang.tuple<i32, i32>, f64>

// Tuple containing list
%list = funlang.cons %c1, %nil : !funlang.list<i32>
%mixed = funlang.make_tuple(%list, %c2) : !funlang.tuple<!funlang.list<i32>, f64>
```

### 튜플 로우어링 (Tuple Lowering)

튜플의 lowering은 리스트보다 훨씬 간단하다. 태그 없이 직접 LLVM struct로 변환한다.

**TypeConverter 확장:**

```cpp
// FunLangTypeConverter에 추가
addConversion([](funlang::TupleType type) {
  auto ctx = type.getContext();

  // Convert each element type
  llvm::SmallVector<mlir::Type> llvmTypes;
  for (auto elemType : type.getElementTypes()) {
    // Recursively convert element types
    // (handles nested tuples, lists, etc.)
    auto convertedType = convertType(elemType);
    llvmTypes.push_back(convertedType);
  }

  // Create LLVM struct type
  return LLVM::LLVMStructType::getLiteral(ctx, llvmTypes);
});
```

**변환 예제:**

```mlir
// Before: FunLang types
!funlang.tuple<i32, f64>
!funlang.tuple<i32, i32, i32>
!funlang.tuple<!funlang.list<i32>, f64>

// After: LLVM types
!llvm.struct<(i32, f64)>
!llvm.struct<(i32, i32, i32)>
!llvm.struct<(!llvm.struct<(i32, ptr)>, f64)>  // list becomes tagged union struct
```

**MakeTupleOpLowering 패턴:**

```cpp
class MakeTupleOpLowering : public OpConversionPattern<funlang::MakeTupleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(funlang::MakeTupleOp op,
                                 OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto elements = adaptor.getElements();  // Already converted by TypeConverter

    // Get the converted result type (LLVM struct)
    auto resultType = getTypeConverter()->convertType(op.getType());
    auto structType = resultType.cast<LLVM::LLVMStructType>();

    // Start with undef struct
    Value structVal = rewriter.create<LLVM::UndefOp>(loc, structType);

    // Insert each element at its position
    for (size_t i = 0; i < elements.size(); ++i) {
      structVal = rewriter.create<LLVM::InsertValueOp>(
          loc, structVal, elements[i], i);
    }

    // Replace make_tuple with the constructed struct
    rewriter.replaceOp(op, structVal);
    return success();
  }
};
```

**Lowering 과정 시각화:**

```mlir
// Before lowering
%c1 = arith.constant 42 : i32
%c2 = arith.constant 3.14 : f64
%pair = funlang.make_tuple(%c1, %c2) : !funlang.tuple<i32, f64>

// After lowering
%c1 = arith.constant 42 : i32
%c2 = arith.constant 3.14 : f64
%0 = llvm.mlir.undef : !llvm.struct<(i32, f64)>
%1 = llvm.insertvalue %c1, %0[0] : !llvm.struct<(i32, f64)>
%pair = llvm.insertvalue %c2, %1[1] : !llvm.struct<(i32, f64)>
```

**리스트 vs 튜플 lowering 비교:**

| 구분 | List | Tuple |
|------|------|-------|
| 태그 | 필요 (Nil=0, Cons=1) | 불필요 |
| 힙 할당 | 필요 (GC_malloc) | 불필요 (값 의미론) |
| 간접 참조 | 있음 (ptr → data) | 없음 (직접 저장) |
| Lowering 복잡도 | 높음 | 낮음 |

### C API 및 F# 바인딩 (C API and F# Bindings)

**C API Shim:**

```cpp
// mlir/lib/Dialect/FunLang/CAPI/FunLangCAPI.cpp

//===----------------------------------------------------------------------===//
// Tuple Type
//===----------------------------------------------------------------------===//

extern "C" MlirType funlangTupleTypeGet(MlirContext ctx,
                                        MlirType *elementTypes,
                                        intptr_t numElements) {
  llvm::SmallVector<mlir::Type> types;
  for (intptr_t i = 0; i < numElements; ++i) {
    types.push_back(unwrap(elementTypes[i]));
  }
  return wrap(funlang::TupleType::get(unwrap(ctx), types));
}

extern "C" intptr_t funlangTupleTypeGetNumElements(MlirType type) {
  return unwrap(type).cast<funlang::TupleType>().getNumElements();
}

extern "C" MlirType funlangTupleTypeGetElementType(MlirType type, intptr_t index) {
  return wrap(unwrap(type).cast<funlang::TupleType>().getElementType(index));
}

extern "C" bool funlangTypeIsATupleType(MlirType type) {
  return unwrap(type).isa<funlang::TupleType>();
}

//===----------------------------------------------------------------------===//
// make_tuple Operation
//===----------------------------------------------------------------------===//

extern "C" MlirOperation funlangMakeTupleOpCreate(MlirLocation loc,
                                                   MlirType resultType,
                                                   MlirValue *elements,
                                                   intptr_t numElements,
                                                   MlirBlock block) {
  OpBuilder builder(unwrap(block)->getParent());
  builder.setInsertionPointToEnd(unwrap(block));

  llvm::SmallVector<mlir::Value> values;
  for (intptr_t i = 0; i < numElements; ++i) {
    values.push_back(unwrap(elements[i]));
  }

  auto tupleType = unwrap(resultType).cast<funlang::TupleType>();
  auto op = builder.create<funlang::MakeTupleOp>(
      unwrap(loc), tupleType, values);
  return wrap(op.getOperation());
}
```

**F# Bindings:**

```fsharp
// FunLang.Bindings/FunLangTypes.fs

module FunLangTypes

open System.Runtime.InteropServices

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType funlangTupleTypeGet(MlirContext ctx, MlirType[] elementTypes, nativeint numElements)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint funlangTupleTypeGetNumElements(MlirType type_)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirType funlangTupleTypeGetElementType(MlirType type_, nativeint index)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern bool funlangTypeIsATupleType(MlirType type_)

type MLIRTypeExtensions =
    /// Create a tuple type with the given element types
    static member CreateTupleType(ctx: MlirContext, elementTypes: MlirType list) : MlirType =
        let typesArray = elementTypes |> List.toArray
        funlangTupleTypeGet(ctx, typesArray, nativeint typesArray.Length)

    /// Check if a type is a tuple type
    static member IsTupleType(t: MlirType) : bool =
        funlangTypeIsATupleType(t)

    /// Get the number of elements in a tuple type
    static member GetTupleNumElements(t: MlirType) : int =
        int (funlangTupleTypeGetNumElements(t))

    /// Get an element type from a tuple type
    static member GetTupleElementType(t: MlirType, index: int) : MlirType =
        funlangTupleTypeGetElementType(t, nativeint index)
```

```fsharp
// FunLang.Bindings/FunLangOps.fs

module FunLangOps

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirOperation funlangMakeTupleOpCreate(
    MlirLocation loc,
    MlirType resultType,
    MlirValue[] elements,
    nativeint numElements,
    MlirBlock block)

type OpBuilderExtensions =
    /// Create a make_tuple operation
    member this.CreateMakeTupleOp(loc: MlirLocation, elements: MlirValue list) : MlirOperation =
        // Infer tuple type from elements
        let elementTypes = elements |> List.map (fun e -> e.GetType())
        let tupleType = MLIRTypeExtensions.CreateTupleType(this.Context, elementTypes)
        let elemArray = elements |> List.toArray
        funlangMakeTupleOpCreate(loc, tupleType, elemArray, nativeint elemArray.Length, this.CurrentBlock)

    /// Create a tuple and return its value
    member this.CreateMakeTuple(loc: MlirLocation, elements: MlirValue list) : MlirValue =
        let op = this.CreateMakeTupleOp(loc, elements)
        op.GetResult(0)

    /// Create a pair (2-tuple)
    member this.CreatePair(loc: MlirLocation, first: MlirValue, second: MlirValue) : MlirValue =
        this.CreateMakeTuple(loc, [first; second])
```

**사용 예제:**

```fsharp
// F# code using the bindings
let createPointTuple (builder: OpBuilder) (x: MlirValue) (y: MlirValue) =
    let loc = builder.GetUnknownLoc()

    // Create pair using convenience method
    let point = builder.CreatePair(loc, x, y)

    // Or explicitly with CreateMakeTuple
    let point' = builder.CreateMakeTuple(loc, [x; y])

    point

let createMixedTuple (builder: OpBuilder) (intVal: MlirValue) (floatVal: MlirValue) (listVal: MlirValue) =
    let loc = builder.GetUnknownLoc()

    // 3-tuple with mixed types
    let mixed = builder.CreateMakeTuple(loc, [intVal; floatVal; listVal])

    // Check the type
    let tupleType = mixed.GetType()
    assert (MLIRTypeExtensions.IsTupleType(tupleType))
    assert (MLIRTypeExtensions.GetTupleNumElements(tupleType) = 3)

    mixed
```

### Summary: 튜플 타입과 연산

**구현 완료:**

- [x] `!funlang.tuple<T1, T2, ...>` 타입 정의 (TableGen)
- [x] ArrayRefParameter로 가변 개수 타입 파라미터
- [x] `funlang.make_tuple` 연산 정의
- [x] Pure trait (CSE 최적화 가능)
- [x] TypeConverter에 튜플 → LLVM struct 변환 추가
- [x] MakeTupleOpLowering 패턴
- [x] C API shim 함수
- [x] F# bindings

**튜플의 특징:**

| 특성 | 리스트 | 튜플 |
|------|--------|------|
| Arity | 가변 | 고정 |
| 원소 타입 | 동질적 (T) | 이질적 (T1, T2, ...) |
| 런타임 태그 | 필요 | 불필요 |
| 메모리 할당 | 힙 (GC) | 스택/인라인 가능 |
| 패턴 매칭 case | 다중 (Nil/Cons) | 단일 (항상 매칭) |
| Lowering 대상 | `!llvm.struct<(i32, ptr)>` | `!llvm.struct<(T1, T2, ...)>` |

**다음:**

- Chapter 19에서 튜플 패턴 매칭 구현
- extractvalue로 튜플 원소 추출
- 중첩 패턴 (튜플 + 리스트 조합)

---

## TypeConverter for List Types

Chapter 16에서 우리는 **TypeConverter**를 배웠다. FunLang types를 LLVM types로 변환하는 규칙을 정의한다.

### Chapter 16 복습: TypeConverter란?

**TypeConverter의 역할:**

```cpp
// Type conversion rules
!funlang.closure → !llvm.ptr
!funlang.list<T> → !llvm.struct<(i32, ptr)>
i32 → i32 (identity)
```

**왜 필요한가?**

- Operations를 lowering할 때 operand/result types도 변환해야 함
- Type consistency 유지 필요
- DialectConversion framework가 자동으로 type materialization 수행

### FunLangTypeConverter 확장

Chapter 16에서 closure type 변환만 구현했다. 이제 list type 변환을 추가한다.

**파일: `mlir/lib/Dialect/FunLang/Transforms/FunLangToLLVM.cpp`**

```cpp
class FunLangTypeConverter : public TypeConverter {
public:
  FunLangTypeConverter(MLIRContext *ctx) {
    // Identity conversion for built-in types
    addConversion([](Type type) { return type; });

    // !funlang.closure → !llvm.ptr (Chapter 16)
    addConversion([](funlang::FunLangClosureType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    // !funlang.list<T> → !llvm.struct<(i32, ptr)> (Chapter 18)
    addConversion([](funlang::FunLangListType type) {
      auto ctx = type.getContext();
      auto i32Type = IntegerType::get(ctx, 32);
      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      return LLVM::LLVMStructType::getLiteral(ctx, {i32Type, ptrType});
    });

    // Materialization for unconverted types
    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                  ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      return inputs[0];
    });

    addTargetMaterialization([&](OpBuilder &builder, Type type,
                                  ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
  }
};
```

**핵심 포인트:**

1. **List type conversion**:
   ```cpp
   !funlang.list<T> → !llvm.struct<(i32, ptr)>
   ```
   - Element type `T`는 버려짐 (runtime representation에 불필요)
   - Tagged union: tag (i32) + data (ptr)

2. **Type parameter 무시**:
   ```cpp
   !funlang.list<i32> → !llvm.struct<(i32, ptr)>
   !funlang.list<f64> → !llvm.struct<(i32, ptr)>
   !funlang.list<!funlang.closure> → !llvm.struct<(i32, ptr)>
   // 모두 동일한 LLVM type!
   ```

3. **Opaque pointer**:
   - Cons cell은 `!llvm.ptr`로 표현 (opaque)
   - Element type 정보는 컴파일 타임에만 존재

### Element Type은 어디로?

**질문:** Element type `T`를 버려도 괜찮은가?

**답:** 네, 컴파일 타임에만 필요하기 때문입니다.

**Element type의 용도:**

1. **Type checking** (compile time):
   ```mlir
   %cons = funlang.cons %head, %tail : !funlang.list<i32>
   // Verifier checks: %head must be i32
   ```

2. **Pattern matching** (compile time):
   ```mlir
   %result = funlang.match %list : !funlang.list<i32> -> i32 {
     ^cons(%head: i32, %tail: !funlang.list<i32>):
       // %head type inferred from list element type
   }
   ```

3. **Lowering** (code generation):
   ```cpp
   // ConsOpLowering::matchAndRewrite
   Type elemType = consOp.getElementType();  // Get T from !funlang.list<T>
   uint64_t elemSize = dataLayout.getTypeSize(elemType);  // Calculate cell size
   ```

**Runtime에는 불필요:**

- Runtime에는 tag만 확인 (0=Nil, 1=Cons)
- Cons cell에서 데이터 로드할 때 타입 정보 불필요 (opaque pointer)
- GC가 타입 정보 없이도 메모리 관리 가능

**비유:**

```cpp
// C++ template (compile time)
template<typename T>
struct List {
    int tag;
    void* data;
};

List<int> intList;      // Compile time: T = int
List<double> doubleList;  // Compile time: T = double

// Runtime: sizeof(List<int>) == sizeof(List<double>)
// Runtime에는 T 정보 사라짐 (type erasure)
```

### Recursive List Types

**중첩 리스트:**

```mlir
!funlang.list<!funlang.list<i32>>
```

**TypeConverter가 자동으로 처리:**

```cpp
// Step 1: Convert inner list
!funlang.list<i32> → !llvm.struct<(i32, ptr)>

// Step 2: Convert outer list (element type = inner list)
!funlang.list<!funlang.list<i32>>
  → !funlang.list<!llvm.struct<(i32, ptr)>>  // Inner converted
  → !llvm.struct<(i32, ptr)>                 // Outer converted

// Result: Same as flat list!
```

**이것도 type erasure:**

- Cons cell에는 element가 `!llvm.struct<(i32, ptr)>`로 저장됨
- 하지만 outer list의 표현은 여전히 `!llvm.struct<(i32, ptr)>`

### Type Materialization

**Materialization이란?**

Type conversion 중 intermediate values가 필요할 때 자동으로 생성되는 operations.

**예제:**

```mlir
// Before lowering
func.func @foo(%lst: !funlang.list<i32>) -> i32 {
    // %lst uses: !funlang.list<i32>
}

// After lowering
func.func @foo(%arg: !llvm.struct<(i32, ptr)>) -> i32 {
    // But some operations might still reference old type temporarily
    // Materialization creates cast operations
}
```

**FunLangTypeConverter에서:**

```cpp
// Source materialization: LLVM type → FunLang type (usually no-op)
addSourceMaterialization([&](OpBuilder &builder, Type type,
                              ValueRange inputs, Location loc) -> Value {
  if (inputs.size() != 1)
    return nullptr;
  return inputs[0];  // Identity cast
});

// Target materialization: FunLang type → LLVM type
addTargetMaterialization([&](OpBuilder &builder, Type type,
                              ValueRange inputs, Location loc) -> Value {
  if (inputs.size() != 1)
    return nullptr;
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
      .getResult(0);
});
```

**UnrealizedConversionCastOp:**

- Temporary operation for type conversion
- Should be removed by complete conversion
- If it remains after pass, conversion failed (verification error)

### Complete FunLangTypeConverter

**전체 TypeConverter (Closure + List):**

```cpp
// mlir/lib/Dialect/FunLang/Transforms/FunLangToLLVM.cpp

class FunLangTypeConverter : public TypeConverter {
public:
  FunLangTypeConverter(MLIRContext *ctx, const DataLayout &dataLayout)
      : dataLayout(dataLayout) {
    // Keep identity conversions (i32, f64, etc.)
    addConversion([](Type type) { return type; });

    // Closure type conversion (Phase 5)
    addConversion([](funlang::FunLangClosureType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    // List type conversion (Phase 6)
    addConversion([](funlang::FunLangListType type) {
      auto ctx = type.getContext();
      auto i32Type = IntegerType::get(ctx, 32);
      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      // Tagged union: {i32 tag, ptr data}
      return LLVM::LLVMStructType::getLiteral(ctx, {i32Type, ptrType});
    });

    // Function type conversion
    addConversion([this](FunctionType type) {
      return convertFunctionType(type);
    });

    // Materialization hooks
    addSourceMaterialization(materializeSource);
    addTargetMaterialization(materializeTarget);
    addArgumentMaterialization(materializeSource);
  }

  // Get element type from list type (helper for lowering patterns)
  Type getListElementType(funlang::FunLangListType listType) const {
    return listType.getElementType();
  }

  // Calculate cons cell size for element type
  uint64_t getConsCellSize(Type elementType) const {
    uint64_t elemSize = dataLayout.getTypeSize(elementType);
    uint64_t tailSize = 16;  // sizeof(struct<(i32, ptr)>) with alignment
    uint64_t totalSize = elemSize + tailSize;
    // Align to 8 bytes
    return (totalSize + 7) & ~7;
  }

private:
  const DataLayout &dataLayout;

  FunctionType convertFunctionType(FunctionType type) {
    SmallVector<Type> inputs;
    SmallVector<Type> results;

    if (failed(convertTypes(type.getInputs(), inputs)) ||
        failed(convertTypes(type.getResults(), results)))
      return nullptr;

    return FunctionType::get(type.getContext(), inputs, results);
  }

  static Value materializeSource(OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) {
    if (inputs.size() != 1)
      return nullptr;
    return inputs[0];
  }

  static Value materializeTarget(OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) {
    if (inputs.size() != 1)
      return nullptr;
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  }
};
```

### TypeConverter in Lowering Pass

**Pass에서 TypeConverter 사용:**

```cpp
struct FunLangToLLVMPass : public PassWrapper<FunLangToLLVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    // Get data layout from module
    auto dataLayout = DataLayout(module);

    // Create type converter
    FunLangTypeConverter typeConverter(ctx, dataLayout);

    // Setup conversion target
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect>();
    target.addIllegalDialect<funlang::FunLangDialect>();

    // Populate rewrite patterns
    RewritePatternSet patterns(ctx);
    patterns.add<ClosureOpLowering>(typeConverter, ctx);
    patterns.add<ApplyOpLowering>(typeConverter, ctx);
    patterns.add<NilOpLowering>(typeConverter, ctx);     // New!
    patterns.add<ConsOpLowering>(typeConverter, ctx);    // New!

    // Run conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

**ConversionPattern에서 typeConverter 사용:**

```cpp
class NilOpLowering : public OpConversionPattern<funlang::NilOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::NilOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();

    // Get converted result type: !llvm.struct<(i32, ptr)>
    Type convertedType = typeConverter->convertType(op.getType());

    // Build Nil representation: {0, null}
    // ...
  }
};
```

### Summary: TypeConverter for List Types

**구현 완료:**

- [x] `!funlang.list<T>` → `!llvm.struct<(i32, ptr)>` conversion
- [x] Element type handling (compile-time only)
- [x] Recursive list types (automatic handling)
- [x] Type materialization hooks
- [x] Helper methods for lowering patterns (`getConsCellSize`)

**핵심 통찰:**

- Element type은 컴파일 타임 정보만
- Runtime representation은 모든 list types에 대해 uniform
- Type erasure로 효율적인 메모리 사용

**다음 섹션:**

- NilOpLowering pattern으로 empty list 생성

---

## NilOp Lowering Pattern

이제 `funlang.nil`을 LLVM dialect로 lowering하는 pattern을 작성한다.

### Lowering Strategy

**Before:**

```mlir
%nil = funlang.nil : !funlang.list<i32>
```

**After:**

```mlir
// Build struct {0, null}
%tag = arith.constant 0 : i32
%null = llvm.mlir.zero : !llvm.ptr
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%nil = llvm.insertvalue %null, %s1[1] : !llvm.struct<(i32, ptr)>
```

**핵심 LLVM operations:**

1. **arith.constant**: Create tag value (0 for Nil)
2. **llvm.mlir.zero**: Create null pointer
3. **llvm.mlir.undef**: Create undefined struct (placeholder)
4. **llvm.insertvalue**: Insert values into struct fields

### ConversionPattern Structure

```cpp
class NilOpLowering : public OpConversionPattern<funlang::NilOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::NilOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};
```

**OpConversionPattern vs OpRewritePattern:**

| Aspect | OpConversionPattern | OpRewritePattern |
|--------|---------------------|------------------|
| Framework | DialectConversion | Greedy rewriter |
| Type conversion | Automatic (TypeConverter) | Manual |
| Adaptor | Yes (adaptor.getOperands()) | No (op.getOperands()) |
| Use case | Dialect lowering | Optimization |

**OpAdaptor:**

- Provides **converted operands** (types already converted by TypeConverter)
- Example: `adaptor.getTail()` returns tail with LLVM type, not FunLang type

### Implementation

**파일: `mlir/lib/Dialect/FunLang/Transforms/FunLangToLLVM.cpp`**

```cpp
//===----------------------------------------------------------------------===//
// NilOpLowering
//===----------------------------------------------------------------------===//

class NilOpLowering : public OpConversionPattern<funlang::NilOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::NilOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = op.getContext();

    // Get converted result type: !llvm.struct<(i32, ptr)>
    Type convertedType = typeConverter->convertType(op.getType());
    auto structType = convertedType.cast<LLVM::LLVMStructType>();

    // Step 1: Create tag value (0 for Nil)
    auto i32Type = IntegerType::get(ctx, 32);
    auto tagValue = rewriter.create<arith::ConstantIntOp>(loc, 0, i32Type);

    // Step 2: Create null pointer
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);

    // Step 3: Create undefined struct (placeholder)
    auto undefStruct = rewriter.create<LLVM::UndefOp>(loc, structType);

    // Step 4: Insert tag into struct at index 0
    auto withTag = rewriter.create<LLVM::InsertValueOp>(
        loc, undefStruct, tagValue, ArrayRef<int64_t>{0});

    // Step 5: Insert null pointer into struct at index 1
    auto nilValue = rewriter.create<LLVM::InsertValueOp>(
        loc, withTag, nullPtr, ArrayRef<int64_t>{1});

    // Replace funlang.nil with constructed struct
    rewriter.replaceOp(op, nilValue.getResult());

    return success();
  }
};
```

### Step-by-Step Explanation

**Step 1: Tag value (0)**

```cpp
auto tagValue = rewriter.create<arith::ConstantIntOp>(loc, 0, i32Type);
```

생성되는 MLIR:
```mlir
%tag = arith.constant 0 : i32
```

**Step 2: Null pointer**

```cpp
auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
```

생성되는 MLIR:
```mlir
%null = llvm.mlir.zero : !llvm.ptr
```

**llvm.mlir.zero vs llvm.null:**

- `llvm.mlir.zero`: MLIR의 zero initializer (opaque pointers)
- Old LLVM: `llvm.null` (deprecated with opaque pointers)

**Step 3: Undefined struct**

```cpp
auto undefStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
```

생성되는 MLIR:
```mlir
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
```

**왜 undef부터 시작?**

- LLVM structs는 immutable (SSA form)
- `insertvalue`로 필드를 하나씩 채워 나감
- 초기값은 undefined (나중에 덮어씀)

**Step 4-5: Insert values**

```cpp
auto withTag = rewriter.create<LLVM::InsertValueOp>(
    loc, undefStruct, tagValue, ArrayRef<int64_t>{0});
auto nilValue = rewriter.create<LLVM::InsertValueOp>(
    loc, withTag, nullPtr, ArrayRef<int64_t>{1});
```

생성되는 MLIR:
```mlir
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%nil = llvm.insertvalue %null, %s1[1] : !llvm.struct<(i32, ptr)>
```

**InsertValueOp syntax:**

- `llvm.insertvalue %value, %struct[index]`
- index: struct field index (0 = tag, 1 = data)
- Returns new struct with field updated

**Step 6: Replace operation**

```cpp
rewriter.replaceOp(op, nilValue.getResult());
```

- Remove original `funlang.nil` operation
- Replace all uses with new struct value
- `nilValue.getResult()`: Extract Value from Operation

### No Memory Allocation

**중요한 최적화:**

- NilOp lowering은 **pure computation** (no side effects)
- Stack-only operations (constant, undef, insertvalue)
- **No GC_malloc call** (unlike ConsOp)

**LLVM optimization 기회:**

```mlir
// Multiple nil operations
%nil1 = funlang.nil : !funlang.list<i32>
%nil2 = funlang.nil : !funlang.list<i32>

// After lowering:
%tag = arith.constant 0 : i32
%null = llvm.mlir.zero : !llvm.ptr
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%nil = llvm.insertvalue %null, %s1[1] : !llvm.struct<(i32, ptr)>
// LLVM CSE: %nil1 and %nil2 → same %nil!
```

**Advanced optimization (Phase 7):**

- Global constant for empty list
- All nil operations → load from constant
- Zero runtime cost

### C API Shim (if needed)

NilOpLowering은 C++에서만 사용되므로 C API shim 불필요. 하지만 testing을 위해 제공 가능:

```cpp
// For testing lowering pass from F#
void mlirFunLangRegisterNilOpLowering(MlirRewritePatternSet patterns) {
  auto *ctx = unwrap(patterns)->getContext();
  FunLangTypeConverter typeConverter(ctx, /* dataLayout */);
  unwrap(patterns)->add<NilOpLowering>(typeConverter, ctx);
}
```

### Complete Example

**Input MLIR (FunLang dialect):**

```mlir
func.func @test_nil() -> !funlang.list<i32> {
    %nil = funlang.nil : !funlang.list<i32>
    func.return %nil : !funlang.list<i32>
}
```

**After NilOpLowering (LLVM dialect):**

```mlir
func.func @test_nil() -> !llvm.struct<(i32, ptr)> {
    %c0 = arith.constant 0 : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %1 = llvm.insertvalue %c0, %0[0] : !llvm.struct<(i32, ptr)>
    %nil = llvm.insertvalue %null, %1[1] : !llvm.struct<(i32, ptr)>
    func.return %nil : !llvm.struct<(i32, ptr)>
}
```

**After LLVM optimization:**

```llvm
define { i32, ptr } @test_nil() {
  ; Constant struct {0, null} directly
  ret { i32, ptr } { i32 0, ptr null }
}
```

### Summary: NilOp Lowering Pattern

**구현 완료:**

- [x] OpConversionPattern for funlang.nil
- [x] Tagged union construction: {tag: 0, data: null}
- [x] No memory allocation (pure computation)
- [x] LLVM optimization friendly

**핵심 패턴:**

1. Undefined struct as starting point
2. InsertValueOp for field-by-field construction
3. replaceOp to complete rewriting

**다음 섹션:**

- ConsOpLowering pattern으로 cons cell allocation

---

## ConsOp Lowering Pattern

이제 `funlang.cons`를 LLVM dialect로 lowering한다. NilOp보다 복잡하다 - **memory allocation**이 필요하기 때문이다.

### Lowering Strategy

**Before:**

```mlir
%lst = funlang.cons %head, %tail : !funlang.list<i32>
```

**After:**

```mlir
// 1. Allocate cons cell
%cell_size = arith.constant 16 : i64
%cell_ptr = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr

// 2. Store head element
%head_ptr = llvm.getelementptr %cell_ptr[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %head, %head_ptr : i32, !llvm.ptr

// 3. Store tail list
%tail_ptr = llvm.getelementptr %cell_ptr[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %tail, %tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr

// 4. Build tagged union {1, cell_ptr}
%tag = arith.constant 1 : i32
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%lst = llvm.insertvalue %cell_ptr, %s1[1] : !llvm.struct<(i32, ptr)>
```

**핵심 작업:**

1. **GC_malloc**: Heap에 cons cell 할당
2. **GEP (GetElementPtr)**: Struct field 주소 계산
3. **Store**: Head와 tail을 cell에 저장
4. **InsertValue**: Tagged union 구성

### Memory Layout Recap

**Cons cell structure:**

```
struct ConsCell {
    T element;                    // Offset 0
    TaggedUnion tail;             // Offset sizeof(T)
}

TaggedUnion = struct {
    i32 tag;                      // 4 bytes
    ptr data;                     // 8 bytes
}
```

**예제: `!funlang.list<i32>`**

```
ConsCell<i32> = {
    i32 element;        // 4 bytes at offset 0
    struct {            // 16 bytes at offset 4 (aligned to 8)
        i32 tag;        // 4 bytes
        ptr data;       // 8 bytes
    } tail;
}

Total size: 4 + 16 = 20 bytes → aligned to 24 bytes
```

### Implementation

**파일: `mlir/lib/Dialect/FunLang/Transforms/FunLangToLLVM.cpp`**

```cpp
//===----------------------------------------------------------------------===//
// ConsOpLowering
//===----------------------------------------------------------------------===//

class ConsOpLowering : public OpConversionPattern<funlang::ConsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::ConsOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = op.getContext();

    // Get converted types
    Type convertedResultType = typeConverter->convertType(op.getType());
    auto structType = convertedResultType.cast<LLVM::LLVMStructType>();

    // Get element type (from original FunLang type)
    Type elementType = op.getElementType();

    // Get converted operands (TypeConverter already converted them)
    Value headValue = adaptor.getHead();
    Value tailValue = adaptor.getTail();

    // Step 1: Calculate cons cell size
    auto cellSize = calculateCellSize(rewriter, loc, elementType);

    // Step 2: Allocate cons cell via GC_malloc
    auto cellPtr = allocateConsCell(rewriter, loc, cellSize);

    // Step 3: Store head element
    storeHead(rewriter, loc, cellPtr, headValue, elementType);

    // Step 4: Store tail list
    storeTail(rewriter, loc, cellPtr, tailValue, elementType);

    // Step 5: Build tagged union {1, cellPtr}
    auto consValue = buildTaggedUnion(rewriter, loc, structType, cellPtr);

    // Replace funlang.cons with constructed value
    rewriter.replaceOp(op, consValue);

    return success();
  }

private:
  // Calculate cons cell size: sizeof(element) + sizeof(TaggedUnion)
  Value calculateCellSize(
      OpBuilder &builder, Location loc, Type elementType) const {

    auto *typeConverter = getTypeConverter();
    auto dataLayout = DataLayout::closest(loc.getParentModule());

    // Get element size
    uint64_t elemSize = dataLayout.getTypeSize(elementType);

    // TaggedUnion size: struct<(i32, ptr)> = 4 + 8 = 12, aligned to 16
    uint64_t tailSize = 16;

    uint64_t totalSize = elemSize + tailSize;

    // Align to 8 bytes
    totalSize = (totalSize + 7) & ~7;

    auto i64Type = builder.getI64Type();
    return builder.create<arith::ConstantIntOp>(loc, totalSize, i64Type);
  }

  // Allocate cons cell via GC_malloc
  Value allocateConsCell(
      OpBuilder &builder, Location loc, Value size) const {

    auto ctx = builder.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i64Type = builder.getI64Type();

    // Get or declare GC_malloc
    auto module = loc->getParentOfType<ModuleOp>();
    auto gcMalloc = module.lookupSymbol<LLVM::LLVMFuncOp>("GC_malloc");
    if (!gcMalloc) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {i64Type});
      gcMalloc = builder.create<LLVM::LLVMFuncOp>(
          loc, "GC_malloc", funcType);
    }

    // Call GC_malloc
    auto callOp = builder.create<LLVM::CallOp>(
        loc, gcMalloc, ValueRange{size});

    return callOp.getResult();
  }

  // Store head element at offset 0
  void storeHead(
      OpBuilder &builder, Location loc, Value cellPtr,
      Value headValue, Type elementType) const {

    // GEP to head field: cell[0]
    auto headPtr = builder.create<LLVM::GEPOp>(
        loc, cellPtr.getType(), cellPtr,
        ArrayRef<LLVM::GEPArg>{0},
        elementType);

    // Store head
    builder.create<LLVM::StoreOp>(loc, headValue, headPtr);
  }

  // Store tail list at offset sizeof(element)
  void storeTail(
      OpBuilder &builder, Location loc, Value cellPtr,
      Value tailValue, Type elementType) const {

    auto ctx = builder.getContext();
    auto dataLayout = DataLayout::closest(loc.getParentModule());

    // Calculate tail offset
    uint64_t elemSize = dataLayout.getTypeSize(elementType);
    uint64_t tailOffset = (elemSize + 7) & ~7;  // Align to 8 bytes

    // GEP to tail field: cell + tailOffset bytes
    auto tailPtr = builder.create<LLVM::GEPOp>(
        loc, cellPtr.getType(), cellPtr,
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(tailOffset)},
        builder.getI8Type());

    // Store tail
    builder.create<LLVM::StoreOp>(loc, tailValue, tailPtr);
  }

  // Build tagged union: {tag: 1, data: cellPtr}
  Value buildTaggedUnion(
      OpBuilder &builder, Location loc,
      LLVM::LLVMStructType structType, Value cellPtr) const {

    auto ctx = builder.getContext();
    auto i32Type = builder.getI32Type();

    // Tag = 1 (Cons)
    auto tagValue = builder.create<arith::ConstantIntOp>(loc, 1, i32Type);

    // Start with undefined struct
    auto undefStruct = builder.create<LLVM::UndefOp>(loc, structType);

    // Insert tag
    auto withTag = builder.create<LLVM::InsertValueOp>(
        loc, undefStruct, tagValue, ArrayRef<int64_t>{0});

    // Insert cell pointer
    auto withData = builder.create<LLVM::InsertValueOp>(
        loc, withTag, cellPtr, ArrayRef<int64_t>{1});

    return withData.getResult();
  }
};
```

### Detailed Breakdown

**Step 1: Cell size calculation**

```cpp
uint64_t elemSize = dataLayout.getTypeSize(elementType);
uint64_t tailSize = 16;  // struct<(i32, ptr)> aligned
uint64_t totalSize = elemSize + tailSize;
totalSize = (totalSize + 7) & ~7;  // Align to 8 bytes
```

**Examples:**

```
i32: 4 + 16 = 20 → 24 bytes
f64: 8 + 16 = 24 → 24 bytes
!funlang.closure (ptr): 8 + 16 = 24 → 24 bytes
```

**Step 2: GC_malloc call**

```cpp
auto gcMalloc = module.lookupSymbol<LLVM::LLVMFuncOp>("GC_malloc");
if (!gcMalloc) {
  // Declare if not exists
  auto funcType = LLVM::LLVMFunctionType::get(ptrType, {i64Type});
  gcMalloc = builder.create<LLVM::LLVMFuncOp>(loc, "GC_malloc", funcType);
}
auto callOp = builder.create<LLVM::CallOp>(loc, gcMalloc, ValueRange{size});
```

생성되는 MLIR:
```mlir
llvm.func @GC_malloc(i64) -> !llvm.ptr
%cell_ptr = llvm.call @GC_malloc(%cell_size) : (i64) -> !llvm.ptr
```

**Step 3: Store head**

```cpp
auto headPtr = builder.create<LLVM::GEPOp>(
    loc, cellPtr.getType(), cellPtr,
    ArrayRef<LLVM::GEPArg>{0}, elementType);
builder.create<LLVM::StoreOp>(loc, headValue, headPtr);
```

생성되는 MLIR:
```mlir
%head_ptr = llvm.getelementptr %cell_ptr[0] : (!llvm.ptr) -> !llvm.ptr, i32
llvm.store %head, %head_ptr : i32, !llvm.ptr
```

**GEPOp (GetElementPtr):**

- Opaque pointers 시대의 GEP
- Type hint: `elementType` (i32, f64, etc.)
- Offset: `[0]` (first field)

**Step 4: Store tail**

```cpp
uint64_t tailOffset = (elemSize + 7) & ~7;  // Aligned offset
auto tailPtr = builder.create<LLVM::GEPOp>(
    loc, cellPtr.getType(), cellPtr,
    ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(tailOffset)},
    builder.getI8Type());
builder.create<LLVM::StoreOp>(loc, tailValue, tailPtr);
```

생성되는 MLIR:
```mlir
%tail_ptr = llvm.getelementptr %cell_ptr[8] : (!llvm.ptr) -> !llvm.ptr, i8
llvm.store %tail, %tail_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr
```

**Byte-offset GEP:**

- Type hint: `i8` (byte-addressable)
- Offset: `[8]` (after 4-byte i32, aligned to 8)

**Step 5: Tagged union**

```cpp
auto tagValue = builder.create<arith::ConstantIntOp>(loc, 1, i32Type);
auto undefStruct = builder.create<LLVM::UndefOp>(loc, structType);
auto withTag = builder.create<LLVM::InsertValueOp>(
    loc, undefStruct, tagValue, ArrayRef<int64_t>{0});
auto withData = builder.create<LLVM::InsertValueOp>(
    loc, withTag, cellPtr, ArrayRef<int64_t>{1});
```

생성되는 MLIR:
```mlir
%tag = arith.constant 1 : i32
%undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
%s1 = llvm.insertvalue %tag, %undef[0] : !llvm.struct<(i32, ptr)>
%cons = llvm.insertvalue %cell_ptr, %s1[1] : !llvm.struct<(i32, ptr)>
```

### OpAdaptor Usage

**OpAdaptor가 중요한 이유:**

```cpp
Value headValue = adaptor.getHead();  // Converted type!
Value tailValue = adaptor.getTail();  // Converted type!
```

**Type conversion 자동 처리:**

```mlir
// Before lowering
%cons = funlang.cons %head, %tail : !funlang.list<i32>
// %head: i32
// %tail: !funlang.list<i32>

// During lowering (via OpAdaptor)
// adaptor.getHead(): i32 (unchanged)
// adaptor.getTail(): !llvm.struct<(i32, ptr)> (converted!)
```

**이미 TypeConverter가 처리함:**

- OpAdaptor는 TypeConverter가 변환한 operands 제공
- Pattern 코드는 converted types로 작업
- 수동 type conversion 불필요

### Complete Example

**Input MLIR (FunLang dialect):**

```mlir
func.func @build_list() -> !funlang.list<i32> {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    %nil = funlang.nil : !funlang.list<i32>
    %lst1 = funlang.cons %c3, %nil : !funlang.list<i32>
    %lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>
    %lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>

    func.return %lst3 : !funlang.list<i32>
}
```

**After lowering (LLVM dialect):**

```mlir
func.func @build_list() -> !llvm.struct<(i32, ptr)> {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    // Nil
    %c0_tag = arith.constant 0 : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %nil_undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %nil_1 = llvm.insertvalue %c0_tag, %nil_undef[0] : !llvm.struct<(i32, ptr)>
    %nil = llvm.insertvalue %null, %nil_1[1] : !llvm.struct<(i32, ptr)>

    // Cons %c3, %nil
    %size1 = arith.constant 24 : i64
    %cell1 = llvm.call @GC_malloc(%size1) : (i64) -> !llvm.ptr
    %head1_ptr = llvm.getelementptr %cell1[0] : (!llvm.ptr) -> !llvm.ptr, i32
    llvm.store %c3, %head1_ptr : i32, !llvm.ptr
    %tail1_ptr = llvm.getelementptr %cell1[8] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %nil, %tail1_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr
    %c1_tag = arith.constant 1 : i32
    %lst1_undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %lst1_1 = llvm.insertvalue %c1_tag, %lst1_undef[0] : !llvm.struct<(i32, ptr)>
    %lst1 = llvm.insertvalue %cell1, %lst1_1[1] : !llvm.struct<(i32, ptr)>

    // Cons %c2, %lst1
    %size2 = arith.constant 24 : i64
    %cell2 = llvm.call @GC_malloc(%size2) : (i64) -> !llvm.ptr
    %head2_ptr = llvm.getelementptr %cell2[0] : (!llvm.ptr) -> !llvm.ptr, i32
    llvm.store %c2, %head2_ptr : i32, !llvm.ptr
    %tail2_ptr = llvm.getelementptr %cell2[8] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %lst1, %tail2_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr
    %lst2_1 = llvm.insertvalue %c1_tag, %lst1_undef[0] : !llvm.struct<(i32, ptr)>
    %lst2 = llvm.insertvalue %cell2, %lst2_1[1] : !llvm.struct<(i32, ptr)>

    // Cons %c1, %lst2
    %size3 = arith.constant 24 : i64
    %cell3 = llvm.call @GC_malloc(%size3) : (i64) -> !llvm.ptr
    %head3_ptr = llvm.getelementptr %cell3[0] : (!llvm.ptr) -> !llvm.ptr, i32
    llvm.store %c1, %head3_ptr : i32, !llvm.ptr
    %tail3_ptr = llvm.getelementptr %cell3[8] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %lst2, %tail3_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr
    %lst3_1 = llvm.insertvalue %c1_tag, %lst1_undef[0] : !llvm.struct<(i32, ptr)>
    %lst3 = llvm.insertvalue %cell3, %lst3_1[1] : !llvm.struct<(i32, ptr)>

    func.return %lst3 : !llvm.struct<(i32, ptr)>
}
```

### Summary: ConsOp Lowering Pattern

**구현 완료:**

- [x] OpConversionPattern for funlang.cons
- [x] GC_malloc call for cons cell allocation
- [x] GEP + Store for head and tail
- [x] Tagged union construction with tag=1
- [x] OpAdaptor for converted operands

**핵심 패턴:**

1. Calculate cell size from element type
2. Allocate via GC_malloc
3. Store head and tail with GEP
4. Build tagged union with InsertValueOp

**다음 섹션:**

- Complete lowering pass integration
- Common errors and debugging

---

## Complete Lowering Pass Update

이제 NilOpLowering과 ConsOpLowering을 FunLangToLLVM pass에 등록한다.

### FunLangToLLVM Pass Structure

**파일: `mlir/lib/Dialect/FunLang/Transforms/FunLangToLLVM.cpp`**

```cpp
//===----------------------------------------------------------------------===//
// FunLangToLLVM Pass
//===----------------------------------------------------------------------===//

struct FunLangToLLVMPass
    : public PassWrapper<FunLangToLLVMPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunLangToLLVMPass)

  StringRef getArgument() const final { return "convert-funlang-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert FunLang dialect to LLVM dialect";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    // Get data layout from module
    auto dataLayout = DataLayout::closest(module);

    // Create type converter
    FunLangTypeConverter typeConverter(ctx, dataLayout);

    // Setup conversion target
    ConversionTarget target(*ctx);

    // Legal dialects (after conversion)
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // Illegal dialects (must be converted)
    target.addIllegalDialect<funlang::FunLangDialect>();

    // Function signatures must be converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    // Populate rewrite patterns
    RewritePatternSet patterns(ctx);

    // Phase 5 patterns (Chapter 16)
    patterns.add<ClosureOpLowering>(typeConverter, ctx);
    patterns.add<ApplyOpLowering>(typeConverter, ctx);

    // Phase 6 patterns (Chapter 18)
    patterns.add<NilOpLowering>(typeConverter, ctx);
    patterns.add<ConsOpLowering>(typeConverter, ctx);

    // Function signature conversion
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // Run partial conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// Register pass
void registerFunLangToLLVMPass() {
  PassRegistration<FunLangToLLVMPass>();
}
```

### Pattern Registration Order

**순서가 중요한가?**

일반적으로 **순서 무관**하다. DialectConversion framework가 모든 patterns를 시도한다.

**하지만 성능 최적화를 위해:**

- 자주 매칭되는 patterns를 먼저 등록
- 복잡한 patterns를 나중에 등록 (matching cost 고려)

**FunLang의 경우:**

```cpp
// Frequency: ClosureOp > ApplyOp > ConsOp > NilOp (typical functional code)
patterns.add<ClosureOpLowering>(typeConverter, ctx);    // Most frequent
patterns.add<ApplyOpLowering>(typeConverter, ctx);
patterns.add<ConsOpLowering>(typeConverter, ctx);
patterns.add<NilOpLowering>(typeConverter, ctx);        // Least frequent
```

하지만 **실용적으로는 로직 순서**로 배치:

```cpp
// Logical grouping
// Phase 5 operations
patterns.add<ClosureOpLowering>(typeConverter, ctx);
patterns.add<ApplyOpLowering>(typeConverter, ctx);

// Phase 6 operations
patterns.add<NilOpLowering>(typeConverter, ctx);
patterns.add<ConsOpLowering>(typeConverter, ctx);
```

### Pass Manager Integration

**F# compiler pipeline:**

```fsharp
// FunLang.Compiler/Compiler.fs
let lowerToLLVM (mlirModule: MlirModule) =
    let pm = PassManager(mlirModule.Context)

    // Phase 5-6: FunLang → LLVM
    pm.AddPass("convert-funlang-to-llvm")

    // Standard MLIR lowering
    pm.AddPass("convert-func-to-llvm")
    pm.AddPass("convert-arith-to-llvm")
    pm.AddPass("reconcile-unrealized-casts")

    pm.Run(mlirModule)
```

**Pass order:**

1. **convert-funlang-to-llvm**: FunLang ops → LLVM ops
2. **convert-func-to-llvm**: func.func → llvm.func
3. **convert-arith-to-llvm**: arith ops → llvm ops
4. **reconcile-unrealized-casts**: Remove UnrealizedConversionCastOps

### Testing List Construction

**Test case:**

```fsharp
// F# source
let test_list = [1; 2; 3]
```

**Compiled MLIR (FunLang dialect):**

```mlir
module {
  func.func @test_list() -> !funlang.list<i32> {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    %nil = funlang.nil : !funlang.list<i32>
    %lst1 = funlang.cons %c3, %nil : !funlang.list<i32>
    %lst2 = funlang.cons %c2, %lst1 : !funlang.list<i32>
    %lst3 = funlang.cons %c1, %lst2 : !funlang.list<i32>

    func.return %lst3 : !funlang.list<i32>
  }
}
```

**After lowering:**

```bash
mlir-opt test.mlir \
  --convert-funlang-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts
```

**Result (LLVM dialect):**

```mlir
module {
  llvm.func @GC_malloc(i64) -> !llvm.ptr

  llvm.func @test_list() -> !llvm.struct<(i32, ptr)> {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32

    // Nil
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %nil_undef = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %nil_1 = llvm.insertvalue %c0, %nil_undef[0] : !llvm.struct<(i32, ptr)>
    %nil = llvm.insertvalue %null, %nil_1[1] : !llvm.struct<(i32, ptr)>

    // Cons cells (similar to previous example)
    // ...

    llvm.return %lst3 : !llvm.struct<(i32, ptr)>
  }
}
```

### End-to-End Example

**Complete workflow:**

```fsharp
// 1. F# AST → FunLang MLIR
let ast = parseExpression "[1; 2; 3]"
let mlirModule = compileToFunLang ast

// 2. FunLang MLIR → LLVM MLIR
lowerToLLVM mlirModule

// 3. LLVM MLIR → LLVM IR
let llvmIR = translateToLLVMIR mlirModule

// 4. LLVM IR → Object file
let objFile = compileLLVMIR llvmIR

// 5. Link with runtime
let executable = linkWithRuntime objFile

// 6. Run!
runExecutable executable
```

**Memory diagram at runtime:**

```
Stack:
  %lst3: {1, 0x1000}

Heap (GC-managed):
  0x1000: ConsCell { head: 1, tail: {1, 0x2000} }
  0x2000: ConsCell { head: 2, tail: {1, 0x3000} }
  0x3000: ConsCell { head: 3, tail: {0, null} }
```

### Summary: Complete Lowering Pass

**구현 완료:**

- [x] FunLangToLLVMPass with all patterns
- [x] Pattern registration (Closure, Apply, Nil, Cons)
- [x] Pass manager integration
- [x] End-to-end list construction

**Pass pipeline:**

1. convert-funlang-to-llvm
2. convert-func-to-llvm
3. convert-arith-to-llvm
4. reconcile-unrealized-casts

**다음 섹션:**

- Common errors and debugging strategies

---

## Common Errors

Lowering pass 구현 시 흔히 발생하는 오류와 해결 방법.

### Error 1: Wrong Cons Cell Size

**증상:**

```
Runtime segfault when accessing tail
```

**원인:**

```cpp
// 잘못된 코드
uint64_t totalSize = elemSize + 12;  // struct<(i32, ptr)> = 12 bytes?
// 실제: struct는 alignment 때문에 16 bytes!
```

**해결:**

```cpp
// 올바른 코드
uint64_t tailSize = 16;  // Aligned struct size
uint64_t totalSize = elemSize + tailSize;
totalSize = (totalSize + 7) & ~7;  // Align total to 8 bytes
```

**디버깅:**

```cpp
// Print sizes in lowering pass
llvm::errs() << "Element size: " << elemSize << "\n";
llvm::errs() << "Total cell size: " << totalSize << "\n";
```

### Error 2: Type Mismatch in Store Operations

**증상:**

```
error: 'llvm.store' op operand #0 type 'i32' does not match
  destination pointer element type '!llvm.struct<(i32, ptr)>'
```

**원인:**

```cpp
// 잘못된 GEP - head pointer로 tail을 store
llvm.store %tail, %head_ptr : !llvm.struct<(i32, ptr)>, !llvm.ptr
```

**해결:**

```cpp
// 올바른 GEP offsets
auto headPtr = builder.create<LLVM::GEPOp>(
    loc, cellPtr.getType(), cellPtr, ArrayRef<LLVM::GEPArg>{0}, elementType);
    // ^^^^^^^^ offset 0 for head

auto tailPtr = builder.create<LLVM::GEPOp>(
    loc, cellPtr.getType(), cellPtr,
    ArrayRef<LLVM::GEPArg>{tailOffset}, builder.getI8Type());
    // ^^^^^^^^^^^^^^^^ byte offset for tail
```

### Error 3: Missing TypeConverter Rule

**증상:**

```
error: failed to legalize operation 'funlang.cons'
  operand #1 type '!funlang.list<i32>' is not legal
```

**원인:**

TypeConverter에 list type 변환 규칙 없음.

**해결:**

```cpp
// TypeConverter에 추가
addConversion([](funlang::FunLangListType type) {
  auto ctx = type.getContext();
  auto i32Type = IntegerType::get(ctx, 32);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Type, ptrType});
});
```

### Error 4: GEP Index Confusion

**증상:**

```
Runtime crash: accessing wrong memory offset
```

**원인:**

```cpp
// Element index vs byte offset 혼동
auto tailPtr = builder.create<LLVM::GEPOp>(
    loc, cellPtr.getType(), cellPtr,
    ArrayRef<LLVM::GEPArg>{1},  // Element index 1? No!
    structType);
```

**해결:**

```cpp
// Byte offset 사용
uint64_t tailOffset = (elemSize + 7) & ~7;
auto tailPtr = builder.create<LLVM::GEPOp>(
    loc, cellPtr.getType(), cellPtr,
    ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(tailOffset)},
    builder.getI8Type());  // i8 for byte-addressable
```

**GEP modes:**

- **Type-based**: `GEP ptr, [index]` with element type → element index
- **Byte-based**: `GEP ptr, [offset]` with i8 type → byte offset

### Debugging Strategies

**Strategy 1: Print intermediate MLIR**

```bash
mlir-opt input.mlir \
  --convert-funlang-to-llvm \
  --print-ir-after-all \
  -o output.mlir
```

**Strategy 2: Use mlir-opt with debug flags**

```bash
mlir-opt input.mlir \
  --convert-funlang-to-llvm \
  --debug-only=dialect-conversion \
  --mlir-print-debuginfo
```

**Strategy 3: Add assertions in lowering patterns**

```cpp
LogicalResult matchAndRewrite(...) const override {
  // Check preconditions
  assert(adaptor.getTail().getType().isa<LLVM::LLVMStructType>() &&
         "Tail must be converted to struct type");

  // Pattern logic...
}
```

**Strategy 4: Test incrementally**

```mlir
// Test NilOp alone first
func.func @test_nil() -> !funlang.list<i32> {
    %nil = funlang.nil : !funlang.list<i32>
    func.return %nil : !funlang.list<i32>
}

// Then ConsOp with nil
func.func @test_cons_nil() -> !funlang.list<i32> {
    %nil = funlang.nil : !funlang.list<i32>
    %c1 = arith.constant 1 : i32
    %cons = funlang.cons %c1, %nil : !funlang.list<i32>
    func.return %cons : !funlang.list<i32>
}

// Then multiple cons
// ...
```

### Summary: Common Errors

**주요 실수:**

1. Cons cell size 계산 오류 (alignment 무시)
2. GEP offset 혼동 (element index vs byte offset)
3. TypeConverter 규칙 누락
4. Store type mismatch

**디버깅 도구:**

- `mlir-opt --print-ir-after-all`
- `--debug-only=dialect-conversion`
- Assertions in pattern code
- Incremental testing

**다음 섹션:**

- Chapter 18 summary and Chapter 19 preview

---

## Summary and Chapter 19 Preview

### Chapter 18 복습

**이 장에서 구현한 것:**

1. **List Representation Design**
   - Tagged union: `!llvm.struct<(i32, ptr)>`
   - Cons cells: Heap-allocated `{element, tail}` structs
   - Immutability and structural sharing

2. **FunLang List Type**
   - `!funlang.list<T>` parameterized type
   - TableGen definition with type parameter
   - C API shim and F# bindings

3. **funlang.nil Operation**
   - Empty list constructor
   - Pure trait (no allocation)
   - Lowering: constant struct {0, null}

4. **funlang.cons Operation**
   - Cons cell constructor
   - Type-safe head/tail constraints
   - Lowering: GC_malloc + GEP + store

5. **TypeConverter Extension**
   - `!funlang.list<T>` → `!llvm.struct<(i32, ptr)>`
   - Element type erasure at runtime
   - Integration with FunLangTypeConverter

6. **Lowering Patterns**
   - NilOpLowering: InsertValueOp for struct construction
   - ConsOpLowering: GC_malloc + GEP + store + InsertValueOp
   - Complete pass integration

### List Operations의 의의

**Before Chapter 18:**

```mlir
// 리스트 표현 불가
// 패턴 매칭 불가
```

**After Chapter 18:**

```mlir
// 리스트 생성 가능
%nil = funlang.nil : !funlang.list<i32>
%lst = funlang.cons %head, %tail : !funlang.list<i32>

// Chapter 19에서 패턴 매칭 추가:
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil: ...
  ^cons(%h, %t): ...
}
```

### 성공 기준 달성 확인

- [x] List의 메모리 표현(tagged union)을 이해한다
- [x] `!funlang.list<T>` 타입을 TableGen으로 정의할 수 있다
- [x] `funlang.nil`과 `funlang.cons`의 동작 원리를 안다
- [x] TypeConverter로 FunLang → LLVM 타입 변환을 구현할 수 있다
- [x] Lowering pattern으로 operation을 LLVM dialect로 변환할 수 있다
- [x] Chapter 19에서 `funlang.match` 구현을 시작할 준비가 된다

### Chapter 19 Preview: Match Compilation

**Chapter 19의 목표:**

`funlang.match` operation으로 패턴 매칭을 MLIR로 표현하고, decision tree 알고리즘을 lowering으로 구현한다.

**funlang.match operation (preview):**

```mlir
%sum = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32

  ^cons(%head: i32, %tail: !funlang.list<i32>):
    %sum_tail = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
    %result = arith.addi %head, %sum_tail : i32
    funlang.yield %result : i32
}
```

**Lowering strategy:**

```mlir
// funlang.match lowering → scf.if + tag dispatch

// Extract tag
%tag_ptr = llvm.getelementptr %list[0] : ...
%tag = llvm.load %tag_ptr : ...

// Dispatch
%is_nil = arith.cmpi eq, %tag, %c0 : i32
%result = scf.if %is_nil -> i32 {
  // Nil case
  %zero = arith.constant 0 : i32
  scf.yield %zero : i32
} else {
  // Cons case: extract head and tail
  %data_ptr = llvm.getelementptr %list[1] : ...
  %cell = llvm.load %data_ptr : ...
  %head = llvm.load %head_ptr : ...
  %tail = llvm.load %tail_ptr : ...

  // Execute cons body
  %sum_tail = func.call @sum_list(%tail) : ...
  %result = arith.addi %head, %sum_tail : i32
  scf.yield %result : i32
}
```

**Chapter 19 구조:**

1. **funlang.match Operation**: Region-based pattern matching
2. **MatchOp Lowering**: Decision tree → scf.if/cf.br
3. **Pattern Decomposition**: Tag dispatch + field extraction
4. **Exhaustiveness Checking**: Verification at operation level
5. **End-to-End Examples**: sum_list, length, map, filter

### Phase 6 Progress

**Completed:**

- [x] Chapter 17: Pattern Matching Theory (decision tree algorithm)
- [x] Chapter 18: List Operations (nil, cons, lowering)

**Remaining:**

- [ ] Chapter 19: Match Compilation (funlang.match operation and lowering)
- [ ] Chapter 20: Functional Programs (실전 예제: map, filter, fold)

**Phase 6이 완료되면:**

- 완전한 함수형 언어 (closures + pattern matching + data structures)
- Real-world functional programs 작성 가능
- Phase 7 (optimizations)의 기반 완성

### 마무리

**Chapter 18에서 배운 핵심 개념:**

1. **Parameterized types**: `!funlang.list<T>` for type safety
2. **Tagged unions**: Runtime representation of sum types
3. **GC allocation**: Heap-allocated cons cells
4. **Type erasure**: Element type as compile-time information
5. **ConversionPattern**: OpConversionPattern + TypeConverter + OpAdaptor

**Next chapter: Let's implement pattern matching with funlang.match!**
