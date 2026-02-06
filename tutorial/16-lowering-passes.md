# Chapter 16: Lowering Passes (Lowering Passes)

## 소개

**Phase 5의 여정이 완성된다.** Chapter 14에서 커스텀 dialect의 **이론**을 다뤘고, Chapter 15에서 FunLang operations를 **정의**했다. 이제 마지막 퍼즐 조각: **lowering**이다.

### Chapter 14-15 복습

**Chapter 14: Custom Dialect Design**
- Progressive lowering 철학 (FunLang → Func/SCF → LLVM)
- TableGen ODS로 operation 정의
- C API shim pattern으로 F# 연결
- FunLang dialect 설계 방향

**Chapter 15: Custom Operations**
- `funlang.closure` operation: 클로저 생성 추상화
- `funlang.apply` operation: 클로저 호출 추상화
- `!funlang.closure` custom type: 타입 안전성
- F# integration: C API → P/Invoke → OpBuilder extensions

**현재 상태:**

```mlir
// Phase 5 FunLang dialect (Chapter 15)
func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    func.return %closure : !funlang.closure
}

func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // 환경에서 n 로드
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}
```

**문제:** `funlang.closure`는 high-level operation이다. LLVM backend는 이걸 이해 못한다. **Lowering pass**가 필요하다.

### Lowering Pass란?

**Lowering pass**는 high-level operation을 low-level operation으로 변환하는 MLIR transformation이다.

**FunLang의 Progressive Lowering:**

```
1. FunLang dialect (Chapter 15)
   funlang.closure, funlang.apply
   ↓
2. Func + SCF + MemRef (중간 추상화)
   func.func, scf.if, memref.alloca
   ↓
3. LLVM dialect (Chapter 12-13 패턴)
   llvm.call, llvm.getelementptr, llvm.store
   ↓
4. LLVM IR (MLIR → LLVM translation)
   call @GC_malloc, getelementptr, store
```

**Chapter 16의 scope:** FunLang dialect → LLVM dialect (Step 1 → 3)

**왜 직접 LLVM dialect로?**

Phase 5에서는 간단한 클로저만 다룬다. 중간 dialect(SCF, MemRef)를 거칠 필요가 없다. **직접 lowering**이 효율적이다.

> **Phase 6 preview:** 패턴 매칭 (`funlang.match`)은 복잡한 제어 흐름을 포함한다. 그때는 SCF dialect를 거쳐서 lowering한다.

### Lowering 목표

**Before lowering (FunLang dialect):**

```mlir
%closure = funlang.closure @lambda, %n : !funlang.closure
%result = funlang.apply %closure(%x) : (i32) -> i32
```

**After lowering (LLVM dialect):**

```mlir
// funlang.closure → GC_malloc + getelementptr + store
%env_size = arith.constant 16 : i64
%env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @lambda : !llvm.ptr
%slot0 = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
%slot1 = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %n, %slot1 : i32, !llvm.ptr

// funlang.apply → getelementptr + load + llvm.call
%fn_ptr_addr = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
%fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
%result = llvm.call %fn_ptr(%env, %x) : (!llvm.ptr, i32) -> i32
```

**Lowering은 Chapter 12-13의 패턴을 재사용한다.** 수동으로 작성하던 코드를, 이제 compiler pass가 자동으로 생성한다.

### Chapter 16 목표

이 장을 마치면:

1. **DialectConversion framework 이해**
   - `ConversionTarget`: 어떤 dialect가 합법적인가?
   - `RewritePatternSet`: 어떻게 변환하는가?
   - `TypeConverter`: 타입은 어떻게 변환하는가?

2. **ConversionPattern 작성 능력**
   - `ClosureOpLowering`: `funlang.closure` → LLVM operations
   - `ApplyOpLowering`: `funlang.apply` → LLVM operations

3. **DRR (Declarative Rewrite Rules) 이해**
   - TableGen 기반 패턴 매칭
   - 최적화 패턴 작성 (empty closure, known closure inlining)

4. **Complete lowering pass 구현**
   - Pass 등록 및 실행
   - C API shim 작성
   - F#에서 pass 호출

5. **End-to-end 이해**
   - FunLang source → LLVM IR → executable
   - 전체 컴파일 파이프라인

**성공 기준:**

```fsharp
// F# source
let makeAdder n = fun x -> x + n
let add5 = makeAdder 5
let result = add5 10   // 15

// Compile and run
let mlir = compileFunLang source
let mlir' = lowerFunLangToLLVM mlir  // <- Chapter 16!
let llvmir = translateToLLVMIR mlir'
let executable = compileAndLink llvmir
runExecutable executable  // Prints: 15
```

**Chapter 16 roadmap:**

1. **DialectConversion Framework** (350+ lines)
2. **ClosureOp Lowering Pattern** (450+ lines)
3. **ApplyOp Lowering Pattern** (350+ lines)
4. **TypeConverter for FunLang Types** (250+ lines)
5. **Declarative Rewrite Rules (DRR)** (300+ lines)
6. **Complete Lowering Pass** (250+ lines)
7. **End-to-End Example** (200+ lines)
8. **Common Errors** (100+ lines)
9. **Summary** (50+ lines)

---

## DialectConversion Framework

MLIR의 **DialectConversion framework**는 dialect 간 변환을 위한 인프라다. 핵심 개념 3가지:

1. **ConversionTarget**: 변환 후 허용되는 operations
2. **RewritePatternSet**: 변환 규칙 집합
3. **TypeConverter**: 타입 변환 규칙

### ConversionTarget: Legal vs Illegal Operations

**ConversionTarget**은 "변환 후 어떤 operations가 남아도 되는가?"를 정의한다.

```cpp
ConversionTarget target(getContext());

// Legal: 이 dialects의 operations는 변환 후에도 OK
target.addLegalDialect<LLVM::LLVMDialect>();
target.addLegalDialect<func::FuncDialect>();
target.addLegalDialect<arith::ArithDialect>();

// Illegal: 이 dialects의 operations는 반드시 변환되어야 함
target.addIllegalDialect<funlang::FunLangDialect>();
```

**의미:**

- **Legal dialect**: 최종 IR에 존재해도 된다
- **Illegal dialect**: 최종 IR에 존재하면 안 된다 (변환 필수)

**예시: FunLangToLLVM pass**

```cpp
ConversionTarget target(getContext());

// Legal: LLVM operations는 OK (최종 목표)
target.addLegalDialect<LLVM::LLVMDialect>();

// Legal: func operations는 OK (func.func, func.return 필요)
target.addLegalDialect<func::FuncDialect>();

// Legal: arith operations는 OK (상수, 산술 연산)
target.addLegalDialect<arith::ArithDialect>();

// Illegal: FunLang operations는 반드시 lowering되어야 함
target.addIllegalDialect<funlang::FunLangDialect>();
```

**변환 후:**

```mlir
// OK - func.func (legal)
func.func @foo() {
    // OK - arith.constant (legal)
    %c = arith.constant 10 : i32

    // OK - llvm.call (legal)
    %ptr = llvm.call @GC_malloc(...) : (...) -> !llvm.ptr

    // ERROR - funlang.closure (illegal!)
    %closure = funlang.closure @bar : !funlang.closure
}
```

`funlang.closure`가 남아있으면 **conversion failure**다.

### addLegalOp vs addIllegalOp: Fine-grained Control

Dialect 전체가 아니라 **특정 operation**만 제어할 수도 있다.

```cpp
// FuncDialect 전체가 아니라 특정 operations만 legal
target.addLegalOp<func::FuncOp, func::ReturnOp>();

// 특정 operation만 illegal
target.addIllegalOp<funlang::ClosureOp, funlang::ApplyOp>();
```

**사용 사례:** Partial lowering (일부만 변환)

```cpp
// SCF dialect 중 일부는 legal (scf.while은 그대로 둠)
target.addLegalDialect<scf::SCFDialect>();
target.addIllegalOp<scf::IfOp>();  // scf.if만 lowering
```

### addDynamicallyLegalOp: Conditional Legality

**Dynamic legality**: 런타임에 판단한다.

```cpp
target.addDynamicallyLegalOp<func::CallOp>(
    [](func::CallOp op) {
        // FunLang 타입을 사용하는 call은 illegal (변환 필요)
        return !llvm::any_of(op.getOperandTypes(), [](Type type) {
            return type.isa<funlang::ClosureType>();
        });
    }
);
```

**의미:** `func.call`이 `!funlang.closure` 타입을 사용하면 illegal (lowering 필요). 그렇지 않으면 legal (그대로 둠).

**사용 사례:** 타입 의존적 변환

```mlir
// Legal (i32 타입만 사용)
%result = func.call @add(%x, %y) : (i32, i32) -> i32

// Illegal (funlang.closure 타입 사용)
%result = func.call @apply(%closure, %x) : (!funlang.closure, i32) -> i32
```

### RewritePatternSet: 변환 규칙 집합

**RewritePatternSet**은 "어떻게 변환하는가?"를 정의한다.

```cpp
RewritePatternSet patterns(&getContext());

// ConversionPattern 추가
patterns.add<ClosureOpLowering>(&getContext());
patterns.add<ApplyOpLowering>(&getContext());

// 여러 patterns를 한 번에 추가
patterns.add<ClosureOpLowering, ApplyOpLowering, MatchOpLowering>(&getContext());
```

**Pattern의 역할:**

- 특정 operation을 매치한다 (`funlang.closure`)
- 새로운 operations로 교체한다 (LLVM operations)

### applyPartialConversion vs applyFullConversion

변환을 실행하는 방법 2가지:

**1. applyPartialConversion: 부분 변환**

```cpp
if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    signalPassFailure();
}
```

- **일부 illegal operations가 남아도 OK** (변환 패턴이 없으면 그냥 둠)
- **사용 사례:** Multi-stage lowering (여러 pass로 나눔)

**2. applyFullConversion: 완전 변환**

```cpp
if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    signalPassFailure();
}
```

- **모든 illegal operations를 변환해야 함** (하나라도 남으면 failure)
- **사용 사례:** Final lowering pass (더 이상 illegal operations 없어야 함)

**FunLangToLLVM pass: Partial conversion 사용**

```cpp
// Partial conversion: 다른 dialect의 operations는 나중에 lowering
if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
}
```

**왜 Partial?**

- `arith` operations는 나중에 별도 pass로 lowering (`--convert-arith-to-llvm`)
- `func` operations도 별도 pass로 lowering (`--convert-func-to-llvm`)
- FunLang operations만 먼저 lowering

### TypeConverter: 타입 변환

**TypeConverter**는 "타입을 어떻게 변환하는가?"를 정의한다.

```cpp
TypeConverter typeConverter;

// FunLang 타입 → LLVM 타입
typeConverter.addConversion([](funlang::ClosureType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
});

typeConverter.addConversion([](funlang::ListType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
});

// 기본 타입은 그대로
typeConverter.addConversion([](Type type) {
    return type;  // i32, i64 etc.
});
```

**변환 예시:**

```mlir
// Before
%closure : !funlang.closure

// After
%closure : !llvm.ptr
```

**TypeConverter의 역할:**

1. **Operation result types 변환**
   ```cpp
   Type resultType = typeConverter.convertType(op.getResult().getType());
   ```

2. **Function signatures 변환**
   ```mlir
   // Before
   func.func @apply(%f: !funlang.closure) -> i32

   // After
   func.func @apply(%f: !llvm.ptr) -> i32
   ```

3. **Block arguments 변환** (region 내부 타입)

**Conversion patterns에서 TypeConverter 사용:**

```cpp
struct ApplyOpLowering : public OpConversionPattern<funlang::ApplyOp> {
  using OpConversionPattern<funlang::ApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::ApplyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // TypeConverter를 통해 result type 변환
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // ...
  }
};
```

**ConversionPattern에 TypeConverter 전달:**

```cpp
RewritePatternSet patterns(&getContext());
patterns.add<ApplyOpLowering>(&getContext(), typeConverter);
//                                           ^^^^^^^^^^^^^^
//                                           TypeConverter 전달
```

### 변환 실패 처리

변환이 실패하면 pass가 실패를 알려야 한다.

```cpp
void runOnOperation() override {
    // ...

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        // 변환 실패 시그널
        signalPassFailure();
        return;
    }
}
```

**실패 원인:**

1. **Illegal operation이 남음**: Pattern이 없거나 매치 실패
2. **타입 변환 실패**: TypeConverter에 규칙 없음
3. **Pattern이 failure 반환**: `matchAndRewrite`에서 `failure()` 리턴

**디버깅:**

```bash
# Verbose mode로 실행
mlir-opt --funlang-to-llvm --debug input.mlir

# 에러 메시지 예시:
# error: failed to legalize operation 'funlang.closure'
# note: see current operation: %0 = "funlang.closure"() ...
```

### DialectConversion 전체 흐름

**1. Target 정의:**

```cpp
ConversionTarget target(getContext());
target.addLegalDialect<LLVM::LLVMDialect>();
target.addIllegalDialect<funlang::FunLangDialect>();
```

**2. TypeConverter 설정:**

```cpp
TypeConverter typeConverter;
typeConverter.addConversion([](funlang::ClosureType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
});
```

**3. Patterns 구성:**

```cpp
RewritePatternSet patterns(&getContext());
patterns.add<ClosureOpLowering, ApplyOpLowering>(&getContext(), typeConverter);
```

**4. 변환 실행:**

```cpp
if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
}
```

**5. 검증:**

변환 후 IR에 illegal operations가 없는지 확인.

```mlir
// 변환 전
%closure = funlang.closure @foo : !funlang.closure

// 변환 후
%env = llvm.call @GC_malloc(...) : (...) -> !llvm.ptr
// ... (LLVM operations only)
```

---

## ClosureOp Lowering Pattern

**`funlang.closure`를 LLVM dialect로 lowering한다.** Chapter 12의 클로저 생성 패턴을 재사용한다.

### Chapter 12 복습: 클로저 생성 패턴

**Closure 구조 (Chapter 12):**

```
Environment layout (heap-allocated):
+--------+----------+----------+-----+
| fn_ptr | var1     | var2     | ... |
+--------+----------+----------+-----+
  slot 0   slot 1     slot 2
  8 bytes  variable   variable
```

**클로저 생성 MLIR (Chapter 12):**

```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // 1. 환경 크기 계산: 8 (fn_ptr) + 8 (n)
    %env_size = arith.constant 16 : i64

    // 2. GC_malloc 호출
    %env = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr

    // 3. 함수 포인터 저장 (slot 0)
    %fn_ptr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %slot0 = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr

    // 4. 캡처된 변수 n 저장 (slot 1)
    %slot1 = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %slot1 : i32, !llvm.ptr

    // 5. 환경 포인터 반환
    func.return %env : !llvm.ptr
}
```

**Lowering 목표:** `funlang.closure`를 위 패턴으로 확장한다.

### funlang.closure Operation (Chapter 15 복습)

**ODS 정의:**

```tablegen
def FunLang_ClosureOp : FunLang_Op<"closure", [Pure]> {
  let summary = "Create a closure";

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$captured
  );

  let results = (outs FunLang_ClosureType:$result);

  let assemblyFormat = "$callee `,` $captured attr-dict `:` type($result)";
}
```

**사용 예시:**

```mlir
// 캡처 변수 없음
%closure = funlang.closure @foo : !funlang.closure

// 캡처 변수 1개
%closure = funlang.closure @bar, %n : !funlang.closure

// 캡처 변수 여러 개
%closure = funlang.closure @baz, %x, %y, %z : !funlang.closure
```

### ClosureOp Lowering 전략

**입력:** `funlang.closure @callee, %captured... : !funlang.closure`

**출력:** LLVM dialect operations

1. **환경 크기 계산**: `8 + (captured 개수 * 8)` bytes
2. **GC_malloc 호출**: 환경 힙 할당
3. **함수 포인터 저장**: `env[0] = @callee`
4. **캡처 변수들 저장**: `env[1] = captured[0]`, `env[2] = captured[1]`, ...
5. **환경 포인터 반환**: `!llvm.ptr`

### ConversionPattern 구조

**OpConversionPattern 템플릿:**

```cpp
struct ClosureOpLowering : public OpConversionPattern<funlang::ClosureOp> {
  using OpConversionPattern<funlang::ClosureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::ClosureOp op,           // 원본 operation
      OpAdaptor adaptor,                // 변환된 operands
      ConversionPatternRewriter &rewriter  // IR 수정 도구
  ) const override {

    // Lowering 로직 구현

    return success();  // 또는 failure()
  }
};
```

**핵심 파라미터:**

1. **`op`**: 원본 `funlang.closure` operation
   - `op.getLoc()`: source location
   - `op.getCalleeAttr()`: 함수 심볼 (`@callee`)
   - `op.getResult()`: result value

2. **`adaptor`**: 변환된 operands
   - `adaptor.getCaptured()`: 캡처된 변수들 (타입 이미 변환됨)

3. **`rewriter`**: IR 빌더
   - `rewriter.create<...>()`: 새 operation 생성
   - `rewriter.replaceOp()`: 원본 operation 교체

### ClosureOpLowering 구현 (Complete)

```cpp
struct ClosureOpLowering : public OpConversionPattern<funlang::ClosureOp> {
  using OpConversionPattern<funlang::ClosureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::ClosureOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // ==============================
    // 1. 환경 크기 계산
    // ==============================
    size_t numCaptured = adaptor.getCaptured().size();

    // fn_ptr (8 bytes) + captured vars (8 bytes each)
    // 단순화: 모든 변수를 8 bytes로 가정 (포인터 크기)
    size_t envSize = 8 + numCaptured * 8;

    auto i64Type = rewriter.getI64Type();
    auto envSizeConst = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(envSize));

    // ==============================
    // 2. GC_malloc 호출
    // ==============================
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto gcMalloc = rewriter.create<LLVM::CallOp>(
        loc, ptrType, "GC_malloc", ValueRange{envSizeConst});
    Value envPtr = gcMalloc.getResult(0);

    // ==============================
    // 3. 함수 포인터 저장 (env[0])
    // ==============================
    auto fnPtrAddr = rewriter.create<LLVM::AddressOfOp>(
        loc, ptrType, op.getCalleeAttr());

    auto slot0 = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, envPtr,
        ArrayRef<LLVM::GEPArg>{0});

    rewriter.create<LLVM::StoreOp>(loc, fnPtrAddr, slot0);

    // ==============================
    // 4. 캡처된 변수들 저장 (env[1..])
    // ==============================
    for (auto [idx, val] : llvm::enumerate(adaptor.getCaptured())) {
      auto slot = rewriter.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, envPtr,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(idx + 1)});

      rewriter.create<LLVM::StoreOp>(loc, val, slot);
    }

    // ==============================
    // 5. 원본 operation 교체
    // ==============================
    rewriter.replaceOp(op, envPtr);
    return success();
  }
};
```

### 코드 상세 설명

#### 1. 환경 크기 계산

```cpp
size_t numCaptured = adaptor.getCaptured().size();
size_t envSize = 8 + numCaptured * 8;
```

- **`adaptor.getCaptured()`**: 캡처된 변수들 (`ValueRange`)
- **환경 레이아웃**: `[fn_ptr(8), var1(8), var2(8), ...]`
- **단순화**: 모든 변수를 8 bytes로 가정 (실제로는 타입별 크기 계산 필요)

**arith.constant 생성:**

```cpp
auto envSizeConst = rewriter.create<arith::ConstantOp>(
    loc, i64Type, rewriter.getI64IntegerAttr(envSize));
```

- **`arith.constant 16 : i64`** 생성 (캡처 변수 1개일 때)
- `GC_malloc`에 전달할 인자

#### 2. GC_malloc 호출

```cpp
auto ptrType = LLVM::LLVMPointerType::get(ctx);
auto gcMalloc = rewriter.create<LLVM::CallOp>(
    loc, ptrType, "GC_malloc", ValueRange{envSizeConst});
Value envPtr = gcMalloc.getResult(0);
```

- **`LLVM::CallOp`**: `llvm.call` operation 생성
- **함수 이름**: `"GC_malloc"` (string, external function)
- **인자**: `ValueRange{envSizeConst}` (환경 크기)
- **반환 타입**: `!llvm.ptr`

**생성된 MLIR:**

```mlir
%0 = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
```

#### 3. 함수 포인터 저장

```cpp
auto fnPtrAddr = rewriter.create<LLVM::AddressOfOp>(
    loc, ptrType, op.getCalleeAttr());
```

- **`LLVM::AddressOfOp`**: `llvm.mlir.addressof` operation
- **심볼**: `op.getCalleeAttr()` (예: `@lambda_adder`)
- **타입**: `!llvm.ptr`

**생성된 MLIR:**

```mlir
%fn_ptr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
```

**GEPOp으로 slot 0 주소 계산:**

```cpp
auto slot0 = rewriter.create<LLVM::GEPOp>(
    loc, ptrType, ptrType, envPtr,
    ArrayRef<LLVM::GEPArg>{0});
```

- **`LLVM::GEPOp`**: `llvm.getelementptr` operation
- **베이스 포인터**: `envPtr`
- **인덱스**: `{0}` (첫 번째 슬롯)
- **타입**: `!llvm.ptr` (opaque pointer, LLVM 15+)

**생성된 MLIR:**

```mlir
%slot0 = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
```

**함수 포인터 저장:**

```cpp
rewriter.create<LLVM::StoreOp>(loc, fnPtrAddr, slot0);
```

**생성된 MLIR:**

```mlir
llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
```

#### 4. 캡처된 변수들 저장

```cpp
for (auto [idx, val] : llvm::enumerate(adaptor.getCaptured())) {
  auto slot = rewriter.create<LLVM::GEPOp>(
      loc, ptrType, ptrType, envPtr,
      ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(idx + 1)});

  rewriter.create<LLVM::StoreOp>(loc, val, slot);
}
```

- **`llvm::enumerate`**: `(index, value)` 쌍으로 순회
- **인덱스**: `idx + 1` (slot 0은 함수 포인터, slot 1부터 변수)
- **각 변수를 GEP + store**

**캡처 변수 2개일 때 생성된 MLIR:**

```mlir
// 첫 번째 변수 (%n)
%slot1 = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %n, %slot1 : i32, !llvm.ptr

// 두 번째 변수 (%m)
%slot2 = llvm.getelementptr %env[2] : (!llvm.ptr) -> !llvm.ptr
llvm.store %m, %slot2 : i32, !llvm.ptr
```

#### 5. 원본 operation 교체

```cpp
rewriter.replaceOp(op, envPtr);
return success();
```

- **`rewriter.replaceOp(op, envPtr)`**: `funlang.closure`를 `envPtr`로 교체
- **SSA value 대체**: `%closure`를 사용하던 곳이 이제 `%envPtr` 사용
- **`return success()`**: 변환 성공

**Before:**

```mlir
%closure = funlang.closure @lambda, %n : !funlang.closure
%result = funlang.apply %closure(%x) : (i32) -> i32
```

**After:**

```mlir
%env_size = arith.constant 16 : i64
%envPtr = llvm.call @GC_malloc(%env_size) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @lambda : !llvm.ptr
%slot0 = llvm.getelementptr %envPtr[0] : (!llvm.ptr) -> !llvm.ptr
llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
%slot1 = llvm.getelementptr %envPtr[1] : (!llvm.ptr) -> !llvm.ptr
llvm.store %n, %slot1 : i32, !llvm.ptr

// %closure가 %envPtr로 교체됨
%result = funlang.apply %envPtr(%x) : (i32) -> i32
```

### OpAdaptor의 역할

**`OpAdaptor`**는 **변환된 operands**를 제공한다.

**왜 필요한가?**

Conversion이 여러 단계로 이뤄질 때, operands의 타입이 이미 변환됐을 수 있다.

**예시:**

```mlir
// Before
%captured = funlang.some_op : !funlang.closure
%closure = funlang.closure @foo, %captured : !funlang.closure

// After first pattern
%captured_lowered = ... : !llvm.ptr  // 이미 lowering됨!
%closure = funlang.closure @foo, %captured_lowered : !funlang.closure
```

`ClosureOpLowering`이 실행될 때:
- `op.getCaptured()[0]`는 원본 타입 (`!funlang.closure`)
- `adaptor.getCaptured()[0]`는 변환된 타입 (`!llvm.ptr`)

**ConversionPattern에서는 항상 `adaptor` 사용:**

```cpp
// 잘못됨!
for (Value val : op.getCaptured()) { ... }  // 원본 타입

// 올바름
for (Value val : adaptor.getCaptured()) { ... }  // 변환된 타입
```

### ConversionPatternRewriter의 역할

**`ConversionPatternRewriter`**는 **IR 수정 인터페이스**다.

**주요 메서드:**

```cpp
// Operation 생성
auto newOp = rewriter.create<SomeOp>(loc, ...);

// Operation 교체
rewriter.replaceOp(oldOp, newValue);

// Operation 삭제
rewriter.eraseOp(op);

// 타입 변환
Type newType = rewriter.getTypeConverter()->convertType(oldType);

// 상수 생성 헬퍼
auto i32Type = rewriter.getI32Type();
auto attr = rewriter.getI32IntegerAttr(42);
```

**왜 일반 `OpBuilder`가 아닌가?**

Conversion framework는 **transactional semantics**를 제공한다:
- 변환 실패 시 모든 변경 롤백
- Operand mapping 자동 처리
- Type conversion tracking

**일반 rewriter 사용 금지:**

```cpp
// 잘못됨!
OpBuilder builder(op.getContext());
builder.setInsertionPoint(op);
builder.create<...>(...);

// 올바름
rewriter.setInsertionPoint(op);
rewriter.create<...>(...);
```

### ClosureOpLowering 테스트

**입력 MLIR:**

```mlir
func.func @test(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @lambda, %n : !funlang.closure
    func.return %closure : !funlang.closure
}
```

**Lowering pass 실행:**

```bash
mlir-opt --funlang-to-llvm test.mlir
```

**출력 MLIR:**

```mlir
func.func @test(%n: i32) -> !llvm.ptr {
    %c16 = arith.constant 16 : i64
    %0 = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
    %1 = llvm.mlir.addressof @lambda : !llvm.ptr
    %2 = llvm.getelementptr %0[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %1, %2 : !llvm.ptr, !llvm.ptr
    %3 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %3 : i32, !llvm.ptr
    func.return %0 : !llvm.ptr
}
```

**검증:**

1. `funlang.closure` 사라짐 ✓
2. `GC_malloc` 호출 있음 ✓
3. 함수 포인터 저장 있음 ✓
4. 캡처 변수 저장 있음 ✓
5. 반환 타입 `!llvm.ptr` ✓

### C API Shim (Preview)

Lowering pass를 F#에서 사용하려면 **C API shim**이 필요하다.

**C++ Pass 등록:**

```cpp
// FunLangPasses.cpp
void registerFunLangToLLVMPass() {
  PassRegistration<FunLangToLLVMPass>(
      "funlang-to-llvm",
      "Lower FunLang dialect to LLVM dialect");
}

// C API shim
extern "C" void mlirFunLangRegisterToLLVMPass() {
  registerFunLangToLLVMPass();
}

extern "C" void mlirFunLangRunToLLVMPass(MlirModule module) {
  ModuleOp moduleOp = unwrap(module);
  PassManager pm(moduleOp.getContext());
  pm.addPass(std::make_unique<FunLangToLLVMPass>());
  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "FunLangToLLVM pass failed\n";
  }
}
```

**F# P/Invoke:**

```fsharp
[<DllImport("funlang-dialect", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirFunLangRegisterToLLVMPass()

[<DllImport("funlang-dialect", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirFunLangRunToLLVMPass(MlirModule module)

// 사용
let lowerToLLVM (module_: MlirModule) =
    mlirFunLangRunToLLVMPass(module_)
```

**전체 pass pipeline 구성:**

```fsharp
let compileToLLVM (mlir: MlirModule) =
    // 1. FunLang → LLVM
    mlirFunLangRunToLLVMPass(mlir)

    // 2. Arith → LLVM
    mlirRunArithToLLVMPass(mlir)

    // 3. Func → LLVM
    mlirRunFuncToLLVMPass(mlir)

    // 4. LLVM dialect → LLVM IR
    mlirTranslateToLLVMIR(mlir)
```

**Section 2와 3 요약:**

- **DialectConversion framework**: Target + Patterns + TypeConverter
- **ClosureOpLowering**: `funlang.closure` → GC_malloc + GEP + store 패턴
- **OpAdaptor**: 변환된 operands 제공
- **ConversionPatternRewriter**: IR 수정 인터페이스
- **C API shim**: F#에서 pass 실행

**다음 Section:** `funlang.apply` lowering pattern 구현

