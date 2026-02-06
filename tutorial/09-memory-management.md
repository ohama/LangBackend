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

## Boehm GC 소개

FunLang은 **Boehm-Demers-Weiser Garbage Collector** (줄여서 Boehm GC 또는 bdwgc)를 사용한다.

### Boehm GC란?

**Boehm GC**는 C와 C++을 위한 **보수적(conservative) 가비지 컬렉터**다.

**핵심 특징:**

1. **Conservative Collection**
   - "보수적"이란 정확한 타입 정보 없이 동작한다는 의미
   - Stack과 heap을 스캔하여 "포인터처럼 보이는 값"을 찾는다
   - 값이 유효한 heap 주소 범위에 있으면 포인터로 간주한다

2. **Drop-in Replacement for malloc/free**
   ```c
   // 기존 코드
   int* ptr = malloc(sizeof(int) * 10);
   // ... 사용 ...
   free(ptr);

   // Boehm GC 사용
   int* ptr = GC_malloc(sizeof(int) * 10);
   // ... 사용 ...
   // free 불필요! GC가 자동으로 회수
   ```

3. **Battle-Tested**
   - 1988년부터 개발됨 (30년 이상 역사)
   - 많은 프로그래밍 언어 구현에서 사용:
     - GNU Guile (Scheme)
     - Mono (.NET on Linux)
     - W3m (텍스트 브라우저)
   - 안정성이 검증됨

4. **Thread-Safe**
   - 멀티스레드 환경에서 안전
   - 적절한 초기화 필요 (`GC_INIT()`)

### 왜 Boehm GC를 선택했는가?

**장점:**

1. **컴파일러 변경 최소화**
   - Stack map 불필요 (conservative 스캔)
   - Write barrier 불필요
   - GC를 위한 특별한 코드 생성 불필요

2. **간단한 통합**
   - C 라이브러리로 제공
   - `GC_malloc` 호출만으로 사용 가능
   - 기존 C runtime과 함께 링킹

3. **안정성**
   - 오래 사용됨, 버그가 적다
   - 다양한 플랫폼 지원 (Linux, macOS, Windows)

**단점:**

1. **보수적 수집**
   - False positive: 포인터가 아닌 값을 포인터로 오인
   - 결과: 일부 객체가 회수되지 않을 수 있음 (메모리 누수)
   - 실제로는 드물고, 대부분의 프로그램에서 문제없음

2. **Stop-the-world GC**
   - GC 실행 중 프로그램 전체가 일시 중지
   - Latency-critical 애플리케이션에는 부적합
   - FunLang은 교육용이므로 문제없음

### 대안과 비교

**1. Reference Counting**
- **장점:** 즉시 회수, 예측 가능
- **단점:** 순환 참조 처리 불가, 성능 오버헤드 (카운트 업데이트)
- **FunLang:** 클로저는 순환 참조 가능 → 부적합

**2. LLVM Statepoints (Precise GC)**
- **장점:** 정확한 수집 (false positive 없음)
- **단점:** 복잡한 컴파일러 지원 필요 (safepoint 삽입, stack map 생성)
- **FunLang:** 교육용으로는 너무 복잡 → 부적합

**3. Custom Mark-Sweep GC**
- **장점:** 완전한 제어
- **단점:** 구현이 어렵고 버그가 많음
- **FunLang:** Boehm GC가 이미 잘 동작 → 불필요

**결론:** Boehm GC가 FunLang에 가장 적합하다!

### Boehm GC 핵심 함수

**1. GC_INIT()**
```c
GC_INIT();  // 프로그램 시작 시 한 번 호출
```
- GC를 초기화한다
- 반드시 `main()` 시작 부분이나 첫 `GC_malloc` 전에 호출
- Thread-local storage 설정, heap 초기화

**2. GC_malloc(size)**
```c
void* ptr = GC_malloc(100);  // 100바이트 할당
```
- Heap에 메모리 할당
- `malloc`과 동일하게 사용
- GC가 자동으로 회수 (free 불필요)

**3. GC_malloc_atomic(size)**
```c
void* ptr = GC_malloc_atomic(100);  // 포인터 없는 데이터
```
- 포인터를 포함하지 않는 데이터용 할당
- 예: 문자열, 정수 배열
- GC가 스캔하지 않음 (성능 향상)

**4. GC_free(ptr)** (선택 사항)
```c
GC_free(ptr);  // 명시적 해제 (힌트)
```
- GC에게 "이 메모리를 즉시 회수해도 됨"을 알림
- 필수는 아님 (GC가 나중에 자동 회수)
- 성능 최적화용

### Conservative GC 동작 원리

**1. Heap 스캔:**
```
Heap:
┌────────────────┐ 0x1000
│ Object A       │
├────────────────┤ 0x1010
│ Object B       │
├────────────────┤ 0x1020
│ Free space     │
└────────────────┘
```

**2. Stack 스캔:**
```
Stack:
┌────────────────┐
│ var1 = 0x1000  │ ← 포인터처럼 보임 (Object A 가리킴)
├────────────────┤
│ var2 = 42      │ ← 포인터 아님 (heap 범위 밖)
├────────────────┤
│ var3 = 0x1010  │ ← 포인터처럼 보임 (Object B 가리킴)
└────────────────┘
```

**3. Mark Phase:**
- Stack에서 `0x1000`, `0x1010` 발견
- Object A와 Object B를 "live"로 표시

**4. Sweep Phase:**
- Heap 전체를 스캔
- "live" 표시 없는 객체 회수

**False Positive 예시:**
```c
int x = 0x1000;  // 우연히 heap 주소와 같은 정수
// GC는 x를 포인터로 오인할 수 있음
// 결과: 0x1000의 객체가 회수되지 않음 (누수)
```

실제로는 드물고, 대부분의 프로그램에서 문제없음.

## Boehm GC 빌드 및 설치

Boehm GC를 소스에서 빌드하거나 패키지 매니저로 설치할 수 있다.

### 소스에서 빌드

**1. 저장소 클론:**
```bash
# Boehm GC 저장소
git clone https://github.com/ivmai/bdwgc
cd bdwgc

# Atomic operations 라이브러리 (의존성)
git clone https://github.com/ivmai/libatomic_ops
```

**2. libatomic_ops 링크:**
```bash
# bdwgc가 libatomic_ops를 찾을 수 있도록 심볼릭 링크 생성
ln -s $(pwd)/libatomic_ops $(pwd)/libatomic_ops
```

또는:
```bash
cd bdwgc
ln -s ../libatomic_ops libatomic_ops
```

**3. Build 설정:**
```bash
cd bdwgc
autoreconf -vif        # autoconf 파일 생성
automake --add-missing # 누락된 파일 추가
./configure --prefix=$HOME/boehm-gc --enable-threads=posix
```

**configure 옵션:**
- `--prefix=$HOME/boehm-gc`: 설치 경로 (홈 디렉토리)
- `--enable-threads=posix`: 멀티스레드 지원 (POSIX threads)

**4. 빌드 및 설치:**
```bash
make -j$(nproc)        # 병렬 빌드 (CPU 코어 수만큼)
make check             # 테스트 실행 (선택 사항)
make install           # $HOME/boehm-gc에 설치
```

**5. 환경 변수 설정:**
```bash
# 라이브러리 경로 추가
export LD_LIBRARY_PATH=$HOME/boehm-gc/lib:$LD_LIBRARY_PATH

# 헤더 경로 추가
export C_INCLUDE_PATH=$HOME/boehm-gc/include:$C_INCLUDE_PATH

# bashrc에 추가하여 영구 적용
echo 'export LD_LIBRARY_PATH=$HOME/boehm-gc/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export C_INCLUDE_PATH=$HOME/boehm-gc/include:$C_INCLUDE_PATH' >> ~/.bashrc
```

### 패키지 매니저로 설치

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libgc-dev
```

**macOS (Homebrew):**
```bash
brew install bdw-gc
```

**Fedora/RHEL:**
```bash
sudo dnf install gc-devel
```

**Arch Linux:**
```bash
sudo pacman -S gc
```

패키지 매니저로 설치하면 환경 변수 설정이 자동으로 처리된다.

### 설치 확인

**테스트 프로그램 작성:**
```c
// test_gc.c
#include <stdio.h>
#include <gc.h>

int main() {
    GC_INIT();

    void* ptr = GC_malloc(100);
    if (ptr == NULL) {
        printf("GC_malloc failed\n");
        return 1;
    }

    printf("GC_malloc succeeded: %p\n", ptr);
    // GC_free 불필요 - GC가 자동 회수

    return 0;
}
```

**컴파일 및 실행:**
```bash
# 소스 빌드한 경우
gcc test_gc.c -o test_gc -I$HOME/boehm-gc/include -L$HOME/boehm-gc/lib -lgc
./test_gc

# 패키지 매니저로 설치한 경우
gcc test_gc.c -o test_gc -lgc
./test_gc
```

**예상 출력:**
```
GC_malloc succeeded: 0x7f1234567890
```

성공! Boehm GC가 올바르게 설치되었다.

## FunLang Runtime 통합

이제 FunLang 컴파일러가 생성하는 바이너리와 Boehm GC를 연결한다.

### C Runtime 작성

**runtime.c** - FunLang 실행 환경:

```c
// runtime.c - FunLang runtime with Boehm GC
#include <stdio.h>
#include <gc.h>

/**
 * GC 초기화
 * 프로그램 시작 시 한 번 호출
 */
void funlang_init() {
    GC_INIT();
}

/**
 * GC-managed 메모리 할당
 *
 * @param size 할당할 바이트 수
 * @return 할당된 메모리 포인터
 */
void* funlang_alloc(size_t size) {
    return GC_malloc(size);
}

/**
 * Atomic 메모리 할당 (포인터 없는 데이터용)
 *
 * @param size 할당할 바이트 수
 * @return 할당된 메모리 포인터
 */
void* funlang_alloc_atomic(size_t size) {
    return GC_malloc_atomic(size);
}

/**
 * 정수 출력 (Chapter 06에서 구현)
 *
 * @param value 출력할 정수 값
 */
void print_int(int value) {
    printf("%d\n", value);
}

/**
 * MLIR 컴파일된 main 함수
 * F# 컴파일러가 생성한 LLVM IR에서 정의됨
 */
extern int funlang_main();

/**
 * C 프로그램 진입점
 * GC 초기화 후 funlang_main 호출
 */
int main(int argc, char** argv) {
    funlang_init();
    int result = funlang_main();
    return result;
}
```

**Runtime 구조:**

1. **funlang_init()**: GC 초기화
2. **funlang_alloc()**: Heap 할당 (Phase 3+에서 사용)
3. **print_int()**: 정수 출력 (Phase 2에서 이미 사용 중)
4. **main()**: GC 초기화 → funlang_main 호출

### Runtime 컴파일

```bash
# 소스 빌드한 경우
gcc -c runtime.c -o runtime.o -I$HOME/boehm-gc/include

# 패키지 매니저로 설치한 경우
gcc -c runtime.c -o runtime.o
```

**결과:** `runtime.o` 오브젝트 파일 생성

### MLIR에서 GC_malloc 호출

Phase 3에서 클로저 환경을 heap에 할당할 때 사용할 패턴 (미리보기):

**1. GC_malloc 선언 (MLIR):**
```mlir
// External function 선언
llvm.func @GC_malloc(i64) -> !llvm.ptr attributes {
    sym_visibility = "private"
}
```

**2. Heap 할당 호출:**
```mlir
func.func @allocate_closure_env() -> !llvm.ptr {
    // 클로저 환경 크기 (예: 2개의 i64 값)
    %size = arith.constant 16 : i64  // 2 * 8 bytes

    // GC_malloc 호출
    %env = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr

    // env에 캡처된 값 저장
    // (Phase 3에서 구현)

    func.return %env : !llvm.ptr
}
```

**3. F# 코드 생성 패턴:**
```fsharp
// MlirWrapper.fs에 추가할 헬퍼 메서드 (Phase 3)
type OpBuilder(context: Context) =
    // ... 기존 메서드 ...

    /// GC_malloc external function 선언
    member this.DeclareGCMalloc() : MlirOperation =
        let ptrType = this.LLVMPointerType()
        let i64Type = builder.Context.GetIntegerType(64)
        let funcType = MlirNative.mlirFunctionTypeGet(
            ctx.Handle,
            1n, [| i64Type |],
            1n, [| ptrType |]
        )

        let name = MlirHelpers.fromString("GC_malloc")
        let funcOp = MlirNative.mlirLLVMFuncCreate(location, name, funcType)

        // 가시성 속성 설정
        // ...

        funcOp

    /// GC_malloc 호출하여 메모리 할당
    member this.CallGCMalloc(size: MlirValue, location: MlirLocation) : MlirValue =
        let gcMalloc = // ... GC_malloc 함수 참조 ...
        let callOp = MlirNative.mlirLLVMCallCreate(
            location, gcMalloc, 1n, [| size |]
        )
        MlirNative.mlirOperationGetResult(callOp, 0)
```

**Phase 2에서는 사용하지 않지만**, runtime.c에 `funlang_alloc`을 미리 정의하여 Phase 3에서 바로 사용할 수 있다.

## 빌드 파이프라인 업데이트

Boehm GC를 포함한 완전한 빌드 파이프라인:

### 단계별 빌드 과정

**1. FunLang 소스 → LLVM IR:**
```bash
# F# 컴파일러 실행
dotnet run "let x = 5 in if x > 0 then x * 2 else 0"

# 출력: output.ll (LLVM IR 파일)
```

**2. LLVM IR → Object 파일:**
```bash
llc -filetype=obj output.ll -o output.o
```

**3. Runtime 컴파일:**
```bash
# 소스 빌드한 경우
gcc -c runtime.c -o runtime.o -I$HOME/boehm-gc/include

# 패키지 매니저로 설치한 경우
gcc -c runtime.c -o runtime.o
```

**4. 링킹 (Boehm GC 포함):**
```bash
# 소스 빌드한 경우
gcc output.o runtime.o -o program \
    -L$HOME/boehm-gc/lib -lgc \
    -Wl,-rpath,$HOME/boehm-gc/lib

# 패키지 매니저로 설치한 경우
gcc output.o runtime.o -o program -lgc
```

**링커 옵션 설명:**
- `-L$HOME/boehm-gc/lib`: 라이브러리 검색 경로
- `-lgc`: Boehm GC 라이브러리 링크
- `-Wl,-rpath,$HOME/boehm-gc/lib`: 실행 시 라이브러리 경로 (RPATH)

**5. 실행:**
```bash
./program
echo $?   # Exit code 확인
```

### 자동화된 빌드 스크립트

**build.sh:**
```bash
#!/bin/bash
# FunLang 빌드 스크립트

set -e  # 에러 시 중단

FUNLANG_SRC="$1"
OUTPUT="program"

# 1. FunLang → LLVM IR
echo "Compiling FunLang to LLVM IR..."
dotnet run "$FUNLANG_SRC" > output.ll

# 2. LLVM IR → Object
echo "Compiling LLVM IR to object file..."
llc -filetype=obj output.ll -o output.o

# 3. Runtime 컴파일 (필요 시)
if [ ! -f runtime.o ]; then
    echo "Compiling runtime..."
    gcc -c runtime.c -o runtime.o
fi

# 4. 링킹
echo "Linking with Boehm GC..."
if [ -d "$HOME/boehm-gc" ]; then
    # 소스 빌드
    gcc output.o runtime.o -o "$OUTPUT" \
        -L$HOME/boehm-gc/lib -lgc \
        -Wl,-rpath,$HOME/boehm-gc/lib
else
    # 패키지 매니저
    gcc output.o runtime.o -o "$OUTPUT" -lgc
fi

echo "Build complete: $OUTPUT"
```

**사용:**
```bash
chmod +x build.sh
./build.sh "let x = 5 in x + x"
./program
```

### F# 통합

**Compiler.fs에 추가:**
```fsharp
module Compiler =

    /// LLVM IR을 object 파일로 컴파일
    let compileToObject (llvmIR: string) (outputPath: string) =
        // LLVM IR을 파일에 쓰기
        let llPath = Path.ChangeExtension(outputPath, ".ll")
        File.WriteAllText(llPath, llvmIR)

        // llc 호출
        let llcArgs = sprintf "-filetype=obj %s -o %s" llPath outputPath
        let result = Process.Start("llc", llcArgs)
        result.WaitForExit()

        if result.ExitCode <> 0 then
            failwith "llc compilation failed"

    /// Object 파일과 runtime을 링킹
    let linkWithGC (objPath: string) (exePath: string) =
        let runtimePath = "runtime.o"

        // Boehm GC 경로 확인
        let gcPath = Environment.GetEnvironmentVariable("HOME") + "/boehm-gc"
        let hasSourceBuild = Directory.Exists(gcPath)

        let gccArgs =
            if hasSourceBuild then
                sprintf "%s %s -o %s -L%s/lib -lgc -Wl,-rpath,%s/lib"
                    objPath runtimePath exePath gcPath gcPath
            else
                sprintf "%s %s -o %s -lgc"
                    objPath runtimePath exePath

        let result = Process.Start("gcc", gccArgs)
        result.WaitForExit()

        if result.ExitCode <> 0 then
            failwith "gcc linking failed"

    /// 전체 컴파일 파이프라인
    let compileProgram (source: string) (outputExe: string) =
        // 1. Parse
        let ast = Parser.parse source

        // 2. MLIR IR 생성
        let mlirModule = CodeGen.compile ast

        // 3. Lowering
        Lowering.lowerToLLVMDialect mlirModule

        // 4. LLVM IR 변환
        let llvmIR = Lowering.translateToLLVMIR mlirModule

        // 5. Object 컴파일
        let objPath = Path.ChangeExtension(outputExe, ".o")
        compileToObject llvmIR objPath

        // 6. 링킹
        linkWithGC objPath outputExe

        printfn "Compilation successful: %s" outputExe
```

**사용:**
```fsharp
// Program.fs
[<EntryPoint>]
let main argv =
    if argv.Length < 1 then
        printfn "Usage: dotnet run <source> [output]"
        1
    else
        let source = argv.[0]
        let output = if argv.Length > 1 then argv.[1] else "program"

        Compiler.compileProgram source output
        0
```

## Phase 2 vs Phase 3+ 메모리 사용

FunLang의 메모리 사용 패턴은 단계별로 진화한다.

### Phase 2 (현재)

**특징:**
- 모든 값이 SSA 레지스터
- 메모리 할당 없음
- GC 초기화되지만 사용되지 않음

**생성되는 MLIR IR:**
```mlir
module {
  func.func @funlang_main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %sum = arith.addi %c5, %c10 : i32
    func.return %sum : i32
  }
}
```

**GC 호출:** 없음 (`funlang_alloc` 호출 0회)

### Phase 3 (함수와 클로저)

**특징:**
- 클로저가 환경을 캡처
- 환경은 heap에 할당 (`GC_malloc`)
- GC가 죽은 클로저 회수

**예시: 클로저 환경 할당**
```mlir
// let makeAdder x = fun y -> x + y
func.func @makeAdder(%x: i32) -> !llvm.ptr {
    // 클로저 환경 할당 (x를 저장)
    %size = arith.constant 8 : i64
    %env = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr

    // x를 환경에 저장
    %x_i64 = arith.extsi %x : i32 to i64
    llvm.store %x_i64, %env : !llvm.ptr

    // 클로저 생성 (function pointer + env pointer)
    %closure = funlang.make_closure @lambda, %env

    func.return %closure : !llvm.ptr
}

// fun y -> x + y
func.func private @lambda(%env: !llvm.ptr, %y: i32) -> i32 {
    // 환경에서 x 로드
    %x_i64 = llvm.load %env : !llvm.ptr -> i64
    %x = arith.trunci %x_i64 : i64 to i32

    // x + y
    %result = arith.addi %x, %y : i32
    func.return %result : i32
}
```

**GC 호출:** `makeAdder` 호출마다 1회

### Phase 6 (데이터 구조)

**특징:**
- 리스트, 튜플, 문자열 모두 heap 할당
- 재귀적 데이터 구조 (리스트의 tail)
- GC가 복잡한 참조 그래프 처리

**예시: 리스트 cons**
```mlir
// cons(1, cons(2, nil))
func.func @build_list() -> !llvm.ptr {
    // nil
    %nil = llvm.mlir.null : !llvm.ptr

    // cons(2, nil)
    %size = arith.constant 16 : i64  // head + tail
    %cons2 = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr
    %c2 = arith.constant 2 : i64
    llvm.store %c2, %cons2 : !llvm.ptr
    %tail_ptr = llvm.getelementptr %cons2[8] : !llvm.ptr
    llvm.store %nil, %tail_ptr : !llvm.ptr

    // cons(1, cons2)
    %cons1 = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr
    %c1 = arith.constant 1 : i64
    llvm.store %c1, %cons1 : !llvm.ptr
    %tail_ptr1 = llvm.getelementptr %cons1[8] : !llvm.ptr
    llvm.store %cons2, %tail_ptr1 : !llvm.ptr

    func.return %cons1 : !llvm.ptr
}
```

**GC 호출:** cons 노드마다 1회

**메모리 그래프:**
```
%cons1 ─→ [ head: 1 | tail: ─→ %cons2 ─→ [ head: 2 | tail: nil ] ]
```

GC는 `%cons1`이 접근 불가능해지면 전체 체인을 회수한다.

### 메모리 사용 비교

| Phase | 할당 위치 | GC 사용 | 복잡도 |
|-------|----------|---------|--------|
| Phase 2 | SSA 레지스터만 | 초기화만 (호출 0회) | 낮음 |
| Phase 3 | 클로저 환경 → Heap | 클로저 생성 시 | 중간 |
| Phase 6 | 모든 데이터 구조 → Heap | 거의 모든 연산 | 높음 |

**핵심:** Phase 2는 GC 인프라를 준비하지만, 실제 사용은 Phase 3부터다.

## 공통 에러 및 해결

GC 통합 시 자주 발생하는 에러와 해결 방법:

### 에러 1: GC_malloc 호출 시 Segfault

**증상:**
```
Segmentation fault (core dumped)
```

**원인:**
`GC_INIT()`을 호출하지 않고 `GC_malloc`을 사용했다.

**해결:**
`main()` 시작 부분에서 `GC_INIT()` 호출:
```c
int main() {
    GC_INIT();  // 필수!
    // ... 나머지 코드 ...
}
```

**FunLang runtime.c:**
```c
void funlang_init() {
    GC_INIT();
}

int main(int argc, char** argv) {
    funlang_init();  // 첫 번째 호출
    // ...
}
```

### 에러 2: Linker Error - Undefined Reference to GC_malloc

**증상:**
```
undefined reference to `GC_malloc'
collect2: error: ld returned 1 exit status
```

**원인:**
Boehm GC 라이브러리를 링킹하지 않았다.

**해결:**
링킹 시 `-lgc` 옵션 추가:
```bash
gcc output.o runtime.o -o program -lgc
```

또는 라이브러리 경로 지정:
```bash
gcc output.o runtime.o -o program -L$HOME/boehm-gc/lib -lgc
```

### 에러 3: 실행 시 Library Not Found

**증상:**
```
error while loading shared libraries: libgc.so.1: cannot open shared object file
```

**원인:**
실행 시 `libgc.so`를 찾을 수 없다.

**해결:**

**옵션 1: LD_LIBRARY_PATH 설정**
```bash
export LD_LIBRARY_PATH=$HOME/boehm-gc/lib:$LD_LIBRARY_PATH
./program
```

**옵션 2: RPATH 사용 (권장)**
```bash
gcc output.o runtime.o -o program \
    -L$HOME/boehm-gc/lib -lgc \
    -Wl,-rpath,$HOME/boehm-gc/lib
```

RPATH는 바이너리에 라이브러리 경로를 포함시킨다. `LD_LIBRARY_PATH` 설정 불필요.

### 에러 4: GC가 메모리를 회수하지 않음

**증상:**
프로그램 메모리 사용량이 계속 증가한다.

**원인:**
Boehm GC는 보수적(conservative)이므로 일부 객체를 회수하지 못할 수 있다.

**확인 방법:**
```c
#include <gc.h>

int main() {
    GC_INIT();

    for (int i = 0; i < 1000000; i++) {
        void* ptr = GC_malloc(100);
        // ptr을 더 이상 사용하지 않음
    }

    // GC 통계 출력
    GC_gcollect();  // 강제 수집
    printf("Heap size: %zu\n", GC_get_heap_size());
    printf("Free bytes: %zu\n", GC_get_free_bytes());

    return 0;
}
```

**일반적인 경우:**
- Phase 2-3 프로그램에서는 문제없음
- Conservative GC의 false positive는 드물다
- 메모리 누수가 심각하면 정확한(precise) GC 고려

### 에러 5: Multi-threading 관련 Crash

**증상:**
멀티스레드 프로그램에서 random crash.

**원인:**
GC를 멀티스레드 모드로 초기화하지 않았다.

**해결:**

**Phase 2-5:** 싱글스레드만 사용하므로 문제없음.

**Phase 6+ (Future):** 스레드 생성 시 GC-aware 함수 사용:
```c
#include <gc.h>
#include <pthread.h>

void* thread_func(void* arg) {
    GC_pthread_create(...);  // GC-aware thread creation
    // ...
}
```

또는 빌드 시 thread-safe 옵션:
```bash
./configure --enable-threads=posix
```

## 장 요약

이 장에서 메모리 관리의 기초와 Boehm GC 통합을 완료했다.

### 주요 성취

1. **Stack vs Heap 이해**
   - Stack: 함수 스코프, 자동 관리, LIFO
   - Heap: 유연한 생명주기, 명시적 할당/해제

2. **FunLang 메모리 전략**
   - Phase 2: SSA 레지스터만 사용
   - Phase 3+: 클로저 환경 → heap 할당

3. **MLIR memref Dialect**
   - `memref.alloca`: Stack 할당
   - `memref.alloc`: Heap 할당
   - `memref.load/store`: 메모리 읽기/쓰기

4. **GC 필요성 이해**
   - 수동 메모리 관리의 문제: use-after-free, leak, double-free
   - 클로저가 복잡한 생명주기를 가진다
   - GC가 자동으로 회수한다

5. **Boehm GC 통합**
   - Conservative GC: 타입 정보 불필요
   - `GC_INIT()`, `GC_malloc()` 사용
   - 빌드 및 설치 완료

6. **Runtime 작성**
   - `runtime.c`: GC 초기화, 메모리 할당 wrapper
   - `funlang_main()` 호출 전에 `funlang_init()`

7. **빌드 파이프라인**
   - FunLang → LLVM IR → Object → 링킹 (+ Boehm GC)
   - 자동화 스크립트 및 F# 통합

8. **에러 처리**
   - GC_INIT 누락, 링킹 오류, 라이브러리 경로 문제 해결

### 독자가 할 수 있는 것

- Stack과 heap의 차이를 설명할 수 있다 ✓
- 언제 heap 할당이 필요한지 안다 (클로저, 데이터 구조) ✓
- Boehm GC를 빌드하고 설치할 수 있다 ✓
- `runtime.c`를 작성하여 GC를 초기화할 수 있다 ✓
- FunLang 컴파일러 출력을 Boehm GC와 링킹할 수 있다 ✓
- GC 관련 에러를 디버깅할 수 있다 ✓
- 왜 클로저가 GC를 필요로 하는지 이해한다 ✓

### Phase 2 완료!

**Chapter 06:** 산술 표현식 (+, -, *, /, 비교, 부정, print)
**Chapter 07:** Let 바인딩과 SSA 환경 전달
**Chapter 08:** 제어 흐름 (scf.if, block arguments, boolean)
**Chapter 09:** 메모리 관리 (stack/heap 전략, Boehm GC 통합)

**독자가 컴파일할 수 있는 프로그램:**

```fsharp
// 복잡한 예시
let x = 5 in
let y = 10 in
if x > 0 then
    if y < 20 then
        x * y
    else
        x + y
else
    0
```

**생성되는 바이너리:**
- MLIR로 컴파일
- LLVM IR로 lowering
- Native object 생성
- Boehm GC와 링킹
- 실행 가능한 바이너리!

```bash
$ ./program
$ echo $?
50
```

### Phase 3 Preview: 함수와 클로저

다음 Phase에서 다룰 내용:

**함수 정의:**
```fsharp
let add = fun x -> fun y -> x + y
```

**클로저 캡처:**
```fsharp
let makeAdder x = fun y -> x + y
let add5 = makeAdder 5  // x=5를 캡처
```

**메모리 할당:**
- 클로저 환경을 heap에 할당 (`GC_malloc`)
- 함수 포인터 + 환경 포인터 구조
- GC가 죽은 클로저 회수

**MLIR 연산:**
- `llvm.call @GC_malloc`: Heap 할당
- `llvm.store`, `llvm.load`: 환경 읽기/쓰기
- Function 타입과 호출 규약

**이 장에서 준비한 GC 인프라가 바로 사용된다!**

---

**독자는 이제 메모리 관리를 이해하고, Boehm GC를 통합했다. Phase 3로 가자!**

