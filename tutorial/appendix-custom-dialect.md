# Appendix: 커스텀 MLIR Dialect 등록

## 소개

Chapter 01-05에서는 MLIR의 빌트인 dialect를 사용했다:
- `arith`: 산술 연산
- `func`: 함수 정의와 호출
- `scf`: 구조적 제어 흐름 (if/while)
- `llvm`: LLVM IR 타입과 operation

이 dialect들은 강력하지만 범용적이다. FunLang과 같은 도메인별 언어의 경우 언어의 의미를 직접 표현하는 **커스텀 dialect**가 유용하다.

예를 들어 FunLang 클로저를 고려해 본다:

```fsharp
let make_adder x =
    fun y -> x + y
```

빌트인 dialect만 사용하면 클로저를 즉시 struct, 함수 포인터, 환경 캡처로 낮춰야 한다. 하지만 커스텀 dialect를 사용하면 이렇게 표현할 수 있다:

```mlir
%closure = funlang.make_closure @lambda_body, %x : (!funlang.closure)
%result = funlang.apply %closure, %y : (i32)
```

높은 수준에서 의미가 명확하다. 그런 다음 낮추기 pass에서 구현 세부사항 (struct 레이아웃, malloc 호출 등)으로 점진적으로 변환한다.

이 appendix는 다음을 다룬다:
1. 커스텀 dialect를 C++에서 정의하는 방법
2. C API shim으로 F#에 노출하는 방법
3. Phase 5에서 사용할 아키텍처

> **아키텍처 노트:** 커스텀 dialect 등록은 Phase 5의 주제다. 이 appendix는 미리 보기와 기술적 기초를 제공한다.

## C API가 커스텀 Dialect를 등록할 수 없는 이유

MLIR C API (`mlir-c/IR.h`)는 빌트인 dialect를 **로드**하는 함수를 제공한다:

```c
// C API에 있음 - 빌트인 dialect 로드
MlirDialectHandle mlirGetDialectHandle__arith__();
void mlirDialectHandleRegisterDialect(MlirDialectHandle handle, MlirContext ctx);
```

하지만 **새** dialect를 **정의**하는 함수는 없다. 커스텀 dialect 정의는 C++ 코드를 요구한다:

```cpp
// C++만 가능 - 새 dialect 정의
class FunLangDialect : public mlir::Dialect {
public:
  FunLangDialect(mlir::MLIRContext *context);
  static constexpr llvm::StringLiteral getDialectNamespace() {
    return llvm::StringLiteral("funlang");
  }
  // ... operation, type, attribute 정의 ...
};
```

**왜 C API에 없나?**

Dialect 정의는 C++ 클래스 상속, 템플릿, TableGen 생성 코드를 사용한다. 이것들은 C FFI 경계를 넘을 수 없다. C API는 **이미 정의된** dialect의 핸들만 다룰 수 있다.

**해결책:** C++에서 dialect를 정의하고 등록을 위한 **C API shim**을 작성한다. F#은 이 shim을 P/Invoke로 호출한다.

## C++ 래퍼 접근법

아키텍처:

```
┌─────────────────────────────────────────┐
│ F# Compiler (Compiler.fs)              │
│                                         │
│ ctx.LoadCustomDialect("funlang")        │
└────────────────┬────────────────────────┘
                 │ P/Invoke
                 ▼
┌─────────────────────────────────────────┐
│ C API Shim (funlang_dialect.c)         │
│                                         │
│ void funlangRegisterDialect(MlirContext)│
└────────────────┬────────────────────────┘
                 │ Call C++ API
                 ▼
┌─────────────────────────────────────────┐
│ C++ Dialect (FunLangDialect.cpp)       │
│                                         │
│ class FunLangDialect : public Dialect { │
│   // operation, type 정의               │
│ }                                       │
└─────────────────────────────────────────┘
```

C++ dialect을 공유 라이브러리 (`libFunLangDialect.so`)로 컴파일하고 F#이 로드한다.

## 최소 커스텀 Dialect in C++

C++ 파일 `funlang_dialect.cpp`를 만든다:

```cpp
// funlang_dialect.cpp - 최소 FunLang MLIR Dialect
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir-c/IR.h"

namespace mlir {
namespace funlang {

/// FunLang Dialect 정의
class FunLangDialect : public Dialect {
public:
  /// Context에 FunLang dialect 등록
  explicit FunLangDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                mlir::TypeID::get<FunLangDialect>()) {
    // 여기서 operation, type, attribute를 등록할 것
    // Phase 5에서 구현
  }

  /// Dialect 네임스페이스 반환 ("funlang")
  static constexpr llvm::StringLiteral getDialectNamespace() {
    return llvm::StringLiteral("funlang");
  }
};

} // namespace funlang
} // namespace mlir

// C API shim - F#에서 호출 가능
extern "C" {

/// FunLang dialect를 MLIR context에 등록
void funlangRegisterDialect(MlirContext ctx) {
  mlir::MLIRContext *context = unwrap(ctx);
  mlir::DialectRegistry registry;
  registry.insert<mlir::funlang::FunLangDialect>();
  context->appendDialectRegistry(registry);
  context->loadDialect<mlir::funlang::FunLangDialect>();
}

} // extern "C"
```

**Line-by-line 설명:**

1. **`#include "mlir/IR/Dialect.h"`**: MLIR dialect 기본 클래스
2. **`namespace mlir::funlang`**: 네임스페이스 충돌 방지
3. **`class FunLangDialect : public Dialect`**: 커스텀 dialect 정의. `Dialect`는 MLIR 기본 클래스
4. **`explicit FunLangDialect(MLIRContext *context)`**: 생성자. Context에 dialect 등록
5. **`getDialectNamespace()`**: Dialect 이름 반환. MLIR IR에서 `funlang.operation_name`으로 사용됨
6. **`extern "C" { ... }`**: C linkage - name mangling 방지, F# P/Invoke 가능
7. **`void funlangRegisterDialect(MlirContext ctx)`**: C API shim. F#이 호출할 함수
8. **`unwrap(ctx)`**: MLIR C API 유틸리티 - `MlirContext` (불투명 핸들)을 C++ `MLIRContext*`로 변환
9. **`registry.insert<FunLangDialect>()`**: Registry에 dialect 추가
10. **`context->appendDialectRegistry(registry)`**: Context에 registry 추가
11. **`context->loadDialect<FunLangDialect>()`**: Dialect 즉시 로드 (lazy loading 아님)

> **설계 결정:** 이 dialect는 아직 operation이나 type을 정의하지 않는다. Phase 5에서 `funlang.closure`, `funlang.apply` 같은 operation을 추가할 것이다.

## C++ 라이브러리 빌드

`CMakeLists.txt`를 작성한다:

```cmake
# CMakeLists.txt - FunLang Dialect 빌드
cmake_minimum_required(VERSION 3.20)
project(FunLangDialect)

# LLVM/MLIR 찾기
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

# Include 디렉토리
include_directories(${MLIR_INCLUDE_DIRS})

# FunLangDialect 공유 라이브러리
add_library(FunLangDialect SHARED
  funlang_dialect.cpp
)

# MLIR 라이브러리 링크
target_link_libraries(FunLangDialect
  PRIVATE
    MLIRIR
    MLIRDialect
)

# 설치
install(TARGETS FunLangDialect
  LIBRARY DESTINATION lib
)
```

**빌드:**

```bash
# CMake 설정
cmake -S . -B build \
  -DMLIR_DIR=$HOME/mlir-install/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release

# 빌드
cmake --build build

# 설치
cmake --build build --target install
```

이렇게 하면 `libFunLangDialect.so` (Linux), `libFunLangDialect.dylib` (macOS), 또는 `FunLangDialect.dll` (Windows)가 생성된다.

## F#에서 사용

`MlirBindings.fs`에 P/Invoke 선언 추가:

```fsharp
// MlirBindings.fs에 추가

module MlirNative =
    // ... 기존 바인딩 ...

    /// FunLang 커스텀 dialect 등록 (C++ shim 호출)
    [<DllImport("FunLangDialect", CallingConvention = CallingConvention.Cdecl)>]
    extern void funlangRegisterDialect(MlirContext ctx)
```

`MlirWrapper.fs`의 `Context` 클래스에 메서드 추가:

```fsharp
type Context() =
    let mutable handle = MlirNative.mlirContextCreate()
    let mutable disposed = false

    member _.Handle = handle

    /// 빌트인 dialect 로드
    member _.LoadDialect(dialect: string) =
        if disposed then
            raise (ObjectDisposedException("Context"))

        MlirStringRef.WithString dialect (fun nameRef ->
            MlirNative.mlirContextGetOrLoadDialect(handle, nameRef)
            |> ignore)

    /// 커스텀 FunLang dialect 로드
    member _.LoadFunLangDialect() =
        if disposed then
            raise (ObjectDisposedException("Context"))

        MlirNative.funlangRegisterDialect(handle)

    // ... IDisposable 구현 ...
```

**사용:**

```fsharp
use ctx = new Context()
ctx.LoadDialect("arith")
ctx.LoadDialect("func")
ctx.LoadFunLangDialect()  // 커스텀 dialect 로드

// 이제 funlang.* operation 사용 가능 (Phase 5에서 정의)
```

## 커스텀 Operation 추가 (미리 보기)

Phase 5에서 FunLang dialect에 operation을 추가한다. 미리 보기:

**TableGen 정의 (FunLangOps.td):**

```tablegen
// FunLangOps.td - FunLang operation 정의 (TableGen)
include "mlir/IR/OpBase.td"

def FunLang_Dialect : Dialect {
  let name = "funlang";
  let cppNamespace = "::mlir::funlang";
}

class FunLang_Op<string mnemonic, list<Trait> traits = []>
    : Op<FunLang_Dialect, mnemonic, traits>;

// funlang.make_closure operation
def FunLang_MakeClosureOp : FunLang_Op<"make_closure"> {
  let summary = "Create a closure capturing environment";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$captured
  );
  let results = (outs AnyType:$result);
}

// funlang.apply operation
def FunLang_ApplyOp : FunLang_Op<"apply"> {
  let summary = "Apply a closure to arguments";
  let arguments = (ins
    AnyType:$closure,
    Variadic<AnyType>:$args
  );
  let results = (outs AnyType:$result);
}
```

**생성된 C++ 코드:**

TableGen은 위 정의에서 C++ 클래스를 생성한다:

```cpp
// 생성됨: FunLangOps.h.inc
class MakeClosureOp : public Op<MakeClosureOp, /* traits */> {
public:
  static StringRef getOperationName() { return "funlang.make_closure"; }
  // ... getter/setter, verifier ...
};

class ApplyOp : public Op<ApplyOp, /* traits */> {
public:
  static StringRef getOperationName() { return "funlang.apply"; }
  // ... getter/setter, verifier ...
};
```

**Dialect에 등록:**

```cpp
// funlang_dialect.cpp 업데이트
FunLangDialect::FunLangDialect(MLIRContext *context)
    : Dialect(/*...*/) {
  // Operation 등록
  addOperations<
    MakeClosureOp,
    ApplyOp
  >();
}
```

**F#에서 사용:**

```fsharp
// 커스텀 operation 생성 (Phase 5에서 OpBuilder 확장)
let closureOp = builder.CreateMakeClosure("lambda_body", [| xValue |], loc)
let resultOp = builder.CreateApply(closureOp, [| yValue |], loc)
```

**생성된 MLIR IR:**

```mlir
%closure = funlang.make_closure @lambda_body, %x : (!funlang.closure)
%result = funlang.apply %closure, %y : (i32)
```

Phase 5에서 이 operation들을 `scf`, `memref`, `llvm` dialect로 낮추는 pass를 작성할 것이다.

## 커스텀 Dialect를 사용할 때 vs. 빌트인 사용

**커스텀 dialect를 사용해야 하는 경우:**

1. **도메인별 의미**: FunLang 클로저, 패턴 매칭, 리스트 cons는 커스텀 operation으로 더 명확하다
2. **점진적 낮추기**: 높은 수준에서 시작하여 여러 pass를 통해 낮춘다
3. **최적화 기회**: 커스텀 operation의 패턴 매칭 최적화 작성 가능
4. **가독성**: `funlang.make_closure`가 15줄의 `llvm.call`, `memref.alloc`, `memref.store`보다 이해하기 쉽다

**빌트인 dialect를 사용해야 하는 경우:**

1. **단순한 언어**: 산술과 함수만 있으면 `arith` + `func`로 충분하다
2. **빠른 프로토타이핑**: 커스텀 dialect는 C++ 빌드 시스템이 필요하다
3. **MLIR 학습**: 빌트인 dialect로 시작하면 개념을 빠르게 배울 수 있다

**FunLang의 경우:** Phase 1-4는 빌트인 dialect를 사용한다. Phase 5는 클로저와 고급 기능을 위해 커스텀 dialect를 도입한다.

## 요약

이 appendix에서 다음을 배웠다:

1. **C API 제한**: MLIR C API는 커스텀 dialect 정의를 지원하지 않는다 - C++ 필요
2. **C++ 래퍼 패턴**: C++에서 dialect를 정의하고 `extern "C"` shim으로 노출
3. **F# 통합**: P/Invoke로 shim 호출, 빌트인 dialect처럼 사용
4. **TableGen**: Operation 정의를 위한 MLIR의 코드 생성 도구
5. **점진적 낮추기**: 커스텀 operation → 표준 dialect → LLVM

**Phase 5 미리 보기:**
- FunLang dialect 정의 (`funlang.closure`, `funlang.apply`, `funlang.match`)
- TableGen으로 operation 생성
- 낮추기 pass 작성 (pattern rewrite 사용)
- 이전 chapter들을 커스텀 dialect 사용으로 리팩터링

**리소스:**
- [MLIR Dialect 정의 가이드](https://mlir.llvm.org/docs/DefiningDialects/)
- [MLIR TableGen 참조](https://mlir.llvm.org/docs/OpDefinitions/)
- [MLIR C API 문서](https://mlir.llvm.org/docs/CAPI/)

---

**이것으로 Phase 1이 완료되었다!** Chapter 00-05와 이 appendix를 통해 MLIR 기반 컴파일러 구축을 위한 완전한 기초를 갖추었다. Phase 2에서 FunLang의 더 많은 기능을 컴파일하기 시작할 것이다.
