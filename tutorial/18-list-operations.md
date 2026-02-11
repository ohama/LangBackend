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
