# Chapter 09: 메모리 관리와 Boehm GC

## 소개

지금까지 FunLang 컴파일러는 모든 값을 SSA 레지스터로 처리했다. 정수, boolean, 심지어 let 바인딩도 메모리 연산 없이 SSA value로만 표현했다.

```mlir
func.func @main() -> i32 {
  %c5 = arith.constant 5 : i32       // SSA value (레지스터)
  %c10 = arith.constant 10 : i32     // SSA value (레지스터)
  %sum = arith.addi %c5, %c10 : i32  // SSA value (레지스터)
  func.return %sum : i32
}
```

이 접근 방식은 단순한 표현식에서는 완벽하게 작동한다. 하지만 앞으로 구현할 기능은 **메모리 할당**이 필요하다:

- **클로저(Closures)**: 외부 스코프의 변수를 캡처하는 함수
- **데이터 구조**: 리스트, 튜플, 문자열 등 동적 크기 데이터
- **함수에서 반환되는 값**: 함수 스코프를 벗어나 생존하는 값

이 장에서는 메모리 관리 전략을 학습한다:
- Stack vs Heap 할당 전략
- MLIR의 `memref` dialect
- **Boehm GC** 통합 (자동 메모리 회수)

**중요한 관점:** Phase 2 프로그램은 아직 메모리 할당이 필요하지 않다. 하지만 Phase 3 (함수와 클로저)에 들어가기 전에 GC 인프라를 미리 준비한다. "필요하기 전에 왜 GC가 필요한지"를 이해하는 것이 목표다.

이 장을 마치면:
- Stack과 heap의 차이를 이해한다
- 어떤 값이 stack에, 어떤 값이 heap에 가는지 안다
- MLIR의 `memref` 연산을 사용할 수 있다
- Boehm GC를 빌드하고 링킹할 수 있다
- 왜 클로저에 GC가 필요한지 이해한다

> **Preview:** Phase 3에서 클로저를 구현할 때, 이 장에서 준비한 GC가 바로 사용된다!

## 메모리 관리 전략

프로그램이 실행될 때 두 종류의 메모리 영역을 사용한다: **Stack**과 **Heap**.

### Stack 할당

**Stack (스택)**은 함수 호출 시 자동으로 관리되는 메모리 영역이다.

**Stack에 저장되는 것:**
- 함수 파라미터
- 지역 변수
- 임시 계산 값
- 함수 반환 주소

**Stack의 특징:**

1. **자동 할당 및 해제**
   ```c
   int foo() {
       int x = 5;    // Stack에 할당
       int y = 10;   // Stack에 할당
       return x + y;
       // 함수 종료 시 x, y 자동 해제
   }
   ```

2. **빠른 할당**
   - Stack pointer만 이동 (포인터 연산 한 번)
   - 별도의 할당자(allocator) 불필요

3. **LIFO (Last-In-First-Out) 구조**
   ```
   foo() 호출:
   ┌──────────────┐
   │ foo의 지역변수│ ← stack top
   ├──────────────┤
   │ main의 지역변수│
   ├──────────────┤
   │    ...       │
   └──────────────┘

   foo() 종료:
   ┌──────────────┐
   │ main의 지역변수│ ← stack top (foo의 프레임 제거됨)
   ├──────────────┤
   │    ...       │
   └──────────────┘
   ```

4. **크기 제한**
   - Stack 크기는 고정 (보통 1-8MB)
   - 너무 많은 지역 변수나 깊은 재귀는 stack overflow 유발

**언제 stack을 사용하는가:**
- 함수 내부에서만 사용되는 값
- 크기가 컴파일 타임에 결정되는 값
- 함수 종료 시 사라져도 되는 값

### Heap 할당

**Heap (힙)**은 명시적으로 할당하고 해제하는 메모리 영역이다.

**Heap에 저장되는 것:**
- 함수 스코프를 벗어나 생존하는 값
- 동적 크기 데이터 (런타임에 크기 결정)
- 여러 함수/클로저가 공유하는 값

**Heap의 특징:**

1. **명시적 할당**
   ```c
   void* ptr = malloc(100);  // Heap에 100바이트 할당
   // ... ptr 사용 ...
   free(ptr);                // 명시적 해제 필요
   ```

2. **느린 할당**
   - 할당자가 적절한 메모리 블록을 찾아야 함
   - Fragmentation (단편화) 관리 필요

3. **유연한 생명주기**
   ```c
   int* create_value() {
       int* p = malloc(sizeof(int));
       *p = 42;
       return p;  // 함수 종료 후에도 값이 살아있다
   }
   ```

4. **크기 제한이 크다**
   - Heap은 시스템 전체 가용 메모리를 사용할 수 있다
   - Stack보다 훨씬 큰 데이터 구조 가능

**언제 heap을 사용하는가:**
- 함수에서 반환되는 값
- 동적 크기 데이터 (리스트 길이를 런타임에 결정)
- 여러 클로저가 공유하는 환경

### FunLang의 메모리 전략

**Phase 2 (현재):**
- 모든 값이 SSA 레지스터
- 정수와 boolean만 존재
- 메모리 할당이 전혀 없다

```mlir
// Phase 2: 모든 것이 SSA value
func.func @main() -> i32 {
  %c5 = arith.constant 5 : i32      // 레지스터
  %c10 = arith.constant 10 : i32    // 레지스터
  %sum = arith.addi %c5, %c10 : i32 // 레지스터
  func.return %sum : i32
}
```

**Phase 3 (클로저):**
- 클로저가 환경을 캡처
- **캡처된 환경은 heap에 할당** (함수를 벗어나 생존)
- GC가 자동으로 회수

```mlir
// Phase 3 예시 (preview):
// let x = 5 in (fun y -> x + y)  // 클로저가 x를 캡처
func.func @main() -> !closure {
  %c5 = arith.constant 5 : i32

  // 클로저 환경을 heap에 할당
  %env_size = arith.constant 8 : i64  // x를 저장할 공간
  %env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

  // x를 환경에 저장
  llvm.store %c5, %env : !llvm.ptr

  // 클로저 생성 (함수 포인터 + 환경 포인터)
  %closure = funlang.create_closure @lambda, %env
  func.return %closure : !closure
}
```

**Phase 6 (데이터 구조):**
- 리스트, 튜플, 문자열
- 모두 heap에 할당
- GC가 관리

### Stack vs Heap 다이어그램

```
함수 호출 스택                      Heap (GC 관리)
┌─────────────────────┐            ┌─────────────────────┐
│ main() 프레임       │            │ 클로저 환경 #1      │
│ - return addr       │     ┌─────>│ - x = 5            │
│ - local: result     │─────┘      │ - y = 10           │
│ - temp: %c5, %c10   │            ├─────────────────────┤
├─────────────────────┤            │ 리스트 노드         │
│ foo() 프레임        │            │ - head = 1         │
│ - return addr       │            │ - tail = ...       │
│ - param: x          │            └─────────────────────┘
│ - local: y          │                     ↑
└─────────────────────┘                     │
   (함수 종료 시 자동 해제)           (GC가 회수)
```

**핵심 차이:**
- **Stack**: 함수 스코프에 묶임, 자동 해제, 빠름
- **Heap**: 스코프 독립, 명시적 할당/해제, 유연함

### 왜 FunLang은 Heap이 필요한가?

**클로저가 핵심 이유다:**

```fsharp
// FunLang 예시
let makeAdder = fun x ->
    fun y -> x + y

let add5 = makeAdder 5   // 클로저: x=5를 캡처
let add10 = makeAdder 10 // 클로저: x=10을 캡처

add5 3    // 결과: 8  (x=5 사용)
add10 3   // 결과: 13 (x=10 사용)
```

**문제:**
- `makeAdder 5`가 반환될 때, `x=5`는 어디에 저장되는가?
- `makeAdder` 함수는 이미 종료되었다 (stack 프레임 해제됨)
- 하지만 `add5`를 호출할 때 `x=5`가 필요하다!

**해답:** `x=5`를 **heap에 할당**한다. 클로저는 heap 포인터를 가진다.

```
makeAdder(5) 실행:
1. Heap에 환경 할당: { x: 5 }
2. 클로저 생성: (function_ptr, env_ptr)
3. makeAdder 종료 (stack 해제)
4. 클로저 반환 (env_ptr는 여전히 유효)

add5(3) 호출:
1. env_ptr에서 x 로드: x = 5
2. y = 3 (파라미터)
3. x + y = 8 반환
```

**GC 없이는?**
- 수동으로 `free(env_ptr)` 호출 필요
- 언제 해제? `add5`가 더 이상 사용되지 않을 때
- 하지만 `add5`가 다른 변수에 할당되었다면?
- **복잡성 폭발!** → Garbage Collection 필요

## MLIR memref Dialect 개요

MLIR은 메모리 연산을 위해 **memref (memory reference)** dialect를 제공한다.

### memref 타입

**memref**는 "메모리 영역에 대한 참조"를 나타낸다:

```mlir
memref<10xi32>           // 10개의 i32 배열
memref<1xi32>            // 단일 i32 (크기 1 배열)
memref<5x5xf32>          // 5×5 float 행렬
memref<*xi32>            // 동적 크기 i32 배열
```

**구성:**
- `memref<shape x type>`: shape은 차원, type은 요소 타입
- `memref<1xi32>`: 하나의 i32를 저장하는 메모리 영역

### Stack 할당: memref.alloca

**Stack에 메모리를 할당**하는 연산:

```mlir
func.func @stack_example() -> i32 {
  // Stack에 i32 하나 할당
  %stack = memref.alloca() : memref<1xi32>

  %c0 = arith.constant 0 : index      // 인덱스 0
  %c42 = arith.constant 42 : i32      // 값 42

  // Stack에 값 저장
  memref.store %c42, %stack[%c0] : memref<1xi32>

  // Stack에서 값 로드
  %loaded = memref.load %stack[%c0] : memref<1xi32>

  func.return %loaded : i32
  // 함수 종료 시 stack 자동 해제
}
```

**동작:**
1. `memref.alloca`: Stack에 공간 할당
2. `memref.store`: 메모리에 값 쓰기
3. `memref.load`: 메모리에서 값 읽기
4. 함수 종료: Stack 자동 해제

**인덱스 타입:**
- `index`: MLIR의 배열 인덱스 전용 타입
- 플랫폼에 따라 i32 또는 i64로 lowering됨

**LLVM IR로 lowering:**

```llvm
define i32 @stack_example() {
  %stack = alloca i32, i32 1         ; Stack 할당
  store i32 42, i32* %stack          ; 저장
  %loaded = load i32, i32* %stack    ; 로드
  ret i32 %loaded
}
```

### Heap 할당: memref.alloc

**Heap에 메모리를 할당**하는 연산:

```mlir
func.func @heap_example() -> memref<10xi32> {
  // Heap에 i32 배열 10개 할당
  %heap = memref.alloc() : memref<10xi32>

  // ... heap 사용 ...

  // 명시적 해제 (수동 메모리 관리)
  // memref.dealloc %heap : memref<10xi32>

  func.return %heap : memref<10xi32>
  // heap은 함수 종료 후에도 유효
}
```

**동작:**
1. `memref.alloc`: Heap에 메모리 할당 (malloc과 유사)
2. 메모리 사용
3. `memref.dealloc`: 명시적 해제 (free와 유사)
   - **주의:** 수동 해제는 에러 유발 (use-after-free, double-free)
   - FunLang은 GC를 사용하므로 dealloc을 호출하지 않는다!

**LLVM IR로 lowering:**

```llvm
define ptr @heap_example() {
  ; malloc 호출
  %size = mul i64 10, 4                    ; 10 * sizeof(i32)
  %heap = call ptr @malloc(i64 %size)

  ; ... heap 사용 ...

  ret ptr %heap
}
```

### memref.load와 memref.store

**메모리 읽기/쓰기:**

```mlir
// 쓰기
memref.store %value, %memref[%index] : memref<10xi32>

// 읽기
%loaded = memref.load %memref[%index] : memref<10xi32>
```

**다차원 배열:**

```mlir
// 5×5 행렬
%matrix = memref.alloc() : memref<5x5xi32>
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c42 = arith.constant 42 : i32

// matrix[1][2] = 42
memref.store %c42, %matrix[%c1, %c2] : memref<5x5xi32>

// value = matrix[1][2]
%value = memref.load %matrix[%c1, %c2] : memref<5x5xi32>
```

### Phase 2에서 memref를 사용하지 않는 이유

**Phase 2 프로그램은 SSA 레지스터만으로 충분하다:**

```mlir
// Phase 2 스타일 (SSA only)
func.func @main() -> i32 {
  %x = arith.constant 5 : i32      // SSA value
  %y = arith.constant 10 : i32     // SSA value
  %sum = arith.addi %x, %y : i32   // SSA value
  func.return %sum : i32
}

// memref 스타일로 작성하면? (불필요하게 복잡)
func.func @main() -> i32 {
  %x_mem = memref.alloca() : memref<1xi32>
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : i32
  memref.store %c5, %x_mem[%c0] : memref<1xi32>

  %y_mem = memref.alloca() : memref<1xi32>
  %c10 = arith.constant 10 : i32
  memref.store %c10, %y_mem[%c0] : memref<1xi32>

  %x = memref.load %x_mem[%c0] : memref<1xi32>
  %y = memref.load %y_mem[%c0] : memref<1xi32>
  %sum = arith.addi %x, %y : i32
  func.return %sum : i32
}
```

첫 번째 버전이 훨씬 간단하다! SSA 레지스터만으로 충분하면 memref를 사용할 필요가 없다.

**memref가 필요한 경우:**
- 값이 함수 스코프를 벗어나야 할 때 (클로저 환경)
- 포인터가 필요할 때 (데이터 구조 간 참조)
- Mutation이 필요할 때 (SSA는 immutable)

### memref 요약

**memref dialect:**
- MLIR의 메모리 연산 추상화
- `memref.alloca`: Stack 할당 (자동 해제)
- `memref.alloc`: Heap 할당 (수동 해제 또는 GC)
- `memref.load/store`: 메모리 읽기/쓰기

**Phase 2 vs Phase 3:**
- Phase 2: SSA 레지스터만 사용 (memref 불필요)
- Phase 3: 클로저 환경 → heap 할당 → memref 필요

**다음 섹션:** 왜 Garbage Collection이 필요한가?

## 왜 Garbage Collection이 필요한가

Heap 메모리는 명시적으로 할당하고 해제해야 한다. 하지만 수동 메모리 관리는 **매우 어렵고 에러가 많다**.

### 수동 메모리 관리의 문제

#### 1. Use-After-Free

**freed 메모리에 접근:**

```c
int* ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);        // 메모리 해제
printf("%d\n", *ptr);  // 에러! freed 메모리 접근
```

**결과:**
- Undefined behavior (프로그램 crash 또는 잘못된 값)
- 보안 취약점 (공격자가 freed 메모리를 재사용)

#### 2. Double-Free

**같은 메모리를 두 번 해제:**

```c
int* ptr = malloc(sizeof(int));
free(ptr);
free(ptr);  // 에러! 이미 freed된 메모리
```

**결과:**
- Heap 메타데이터 손상
- 프로그램 crash

#### 3. Memory Leak

**메모리 해제를 잊음:**

```c
void leak() {
    int* ptr = malloc(sizeof(int));
    *ptr = 42;
    return;  // ptr을 free하지 않음!
}

// leak()을 1000번 호출하면?
for (int i = 0; i < 1000; i++) {
    leak();  // 메모리 누수: 1000 * sizeof(int) 바이트
}
```

**결과:**
- 메모리 사용량 계속 증가
- Out-of-memory 에러

### 클로저가 수동 메모리 관리를 어렵게 만드는 이유

**문제: 언제 클로저 환경을 해제하는가?**

```fsharp
// FunLang 예시
let makeAdder x = fun y -> x + y

let add5 = makeAdder 5   // 클로저 1: env = { x: 5 }
let add10 = makeAdder 10 // 클로저 2: env = { x: 10 }

// Q: env { x: 5 }를 언제 해제하는가?
// A: add5가 더 이상 사용되지 않을 때

// 하지만 이것이 언제인가?
let adders = [add5; add10]  // add5를 리스트에 저장
// 여기서 add5를 해제할 수 있는가? No! 리스트가 참조 중

let result = List.head adders 3  // add5 사용
// 이제 해제? 아직 adders가 add5를 가리킨다

// ... 프로그램 계속 실행 ...
```

**복잡성:**
- `add5`가 언제 "더 이상 사용되지 않는가"를 결정하기 어렵다
- 여러 변수가 같은 클로저를 참조할 수 있다
- 클로저가 다른 클로저를 캡처할 수 있다 (환경이 중첩)

**수동 관리 시도:**

```fsharp
// 명시적 free 추가?
let add5 = makeAdder 5
// ... add5 사용 ...
free(add5)  // 하지만 다른 변수가 add5를 참조하면?

let alias = add5
free(add5)  // alias는 이제 invalid pointer!
```

**불가능한 이유:**
- 참조 추적이 필요 (누가 클로저를 가리키는가?)
- 런타임 추적 메커니즘 필요
- **이미 Garbage Collector를 구현하는 것과 같다!**

### 클로저 생명주기 예시

**복잡한 시나리오:**

```fsharp
let outer x =
    let inner y =
        fun z -> x + y + z  // x와 y를 모두 캡처
    inner

let f = outer 5 10   // f는 클로저, env = { x: 5, y: 10 }

// outer 함수는 종료됨 (stack 해제)
// 하지만 env { x: 5, y: 10 }은 heap에 살아있어야 함

let result = f 3     // x=5, y=10, z=3 → 18

// 언제 env를 해제?
// f가 더 이상 참조되지 않을 때
```

**Garbage Collector의 역할:**
- 런타임에 객체 참조를 추적한다
- 더 이상 참조되지 않는 객체를 찾는다
- 자동으로 메모리를 회수한다

### Garbage Collection의 이점

**1. 안전성**
- Use-after-free: 불가능 (GC가 사용 중인 객체를 해제하지 않음)
- Double-free: 불가능 (GC가 한 번만 해제)
- Memory leak: 최소화 (접근 불가능한 객체는 자동 회수)

**2. 생산성**
- 프로그래머가 메모리 관리를 신경 쓰지 않아도 됨
- 버그가 적다
- 코드가 간결해진다

**3. 클로저 지원**
- 클로저 환경의 생명주기를 자동 관리
- 복잡한 참조 그래프도 처리

**트레이드오프:**
- 성능: GC가 주기적으로 실행됨 (pause time)
- 메모리: GC는 약간의 메모리 오버헤드 존재
- **FunLang의 선택:** 클로저 지원을 위해 GC는 필수

### GC 없이 클로저를 구현한다면?

**대안들:**

1. **Reference Counting**
   - 각 객체의 참조 카운트 추적
   - 카운트가 0이 되면 해제
   - **문제:** 순환 참조 처리 불가
     ```fsharp
     let rec loop x = fun y -> loop y x  // 순환 참조!
     ```

2. **Arena Allocation**
   - 모든 객체를 arena에 할당
   - Arena 전체를 한 번에 해제
   - **문제:** 클로저가 서로 다른 생명주기를 가질 때 비효율

3. **Ownership System (Rust 스타일)**
   - 컴파일 타임에 생명주기 추적
   - 런타임 오버헤드 없음
   - **문제:** FunLang은 타입 추론 언어 (ownership 추가는 언어 복잡성 증가)

**결론:** Garbage Collection이 가장 적합한 선택이다!

### 왜 GC가 필요한가 요약

**문제:**
- 클로저 환경은 heap에 할당해야 한다 (함수 스코프를 벗어남)
- 환경의 생명주기는 복잡하다 (여러 참조, 중첩, 순환)
- 수동 메모리 관리는 에러가 많다 (use-after-free, leak)

**해답: Garbage Collection**
- 런타임에 객체 참조를 추적한다
- 접근 불가능한 객체를 자동으로 회수한다
- 프로그래머가 메모리 관리를 신경 쓰지 않아도 된다

**다음 섹션:** Boehm GC 소개 - FunLang이 사용할 GC!

