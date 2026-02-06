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

---

## ApplyOp Lowering Pattern

**`funlang.apply`를 LLVM dialect로 lowering한다.** Chapter 13의 간접 호출 패턴을 재사용한다.

### Chapter 13 복습: 간접 호출 패턴

**Closure application (Chapter 13):**

```mlir
func.func @apply(%f: !llvm.ptr, %x: i32) -> i32 {
    // 1. 환경에서 함수 포인터 로드 (env[0])
    %fn_ptr_addr = llvm.getelementptr %f[0] : (!llvm.ptr) -> !llvm.ptr
    %fn_ptr = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr

    // 2. 간접 호출 (fn_ptr를 통해 호출)
    // 첫 번째 인자: 환경 포인터 (%f)
    // 나머지 인자: 실제 인자 (%x)
    %result = llvm.call %fn_ptr(%f, %x) : (!llvm.ptr, i32) -> i32

    func.return %result : i32
}
```

**핵심 단계:**

1. **함수 포인터 추출**: `env[0]`에서 로드
2. **인자 구성**: `[환경 포인터, 실제 인자들]`
3. **간접 호출**: `llvm.call %fn_ptr(...)`

### funlang.apply Operation (Chapter 15 복습)

**ODS 정의:**

```tablegen
def FunLang_ApplyOp : FunLang_Op<"apply"> {
  let summary = "Apply a closure to arguments";

  let arguments = (ins
    FunLang_ClosureType:$closure,
    Variadic<AnyType>:$args
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $closure `(` $args `)` attr-dict `:` functional-type($args, $result)
  }];
}
```

**사용 예시:**

```mlir
// 인자 1개
%result = funlang.apply %closure(%x) : (i32) -> i32

// 인자 여러 개
%result = funlang.apply %closure(%x, %y) : (i32, i32) -> i32

// 인자 없음 (thunk)
%result = funlang.apply %closure() : () -> i32
```

### ApplyOp Lowering 전략

**입력:** `%result = funlang.apply %closure(%args...) : (...) -> result_type`

**출력:** LLVM dialect operations

1. **함수 포인터 추출**: `env[0]`에서 로드
2. **인자 리스트 구성**: `[closure, args...]`
3. **간접 호출**: `llvm.call %fn_ptr(...)`
4. **결과 반환**: `result_type`로 변환

### ApplyOpLowering 구현 (Complete)

```cpp
struct ApplyOpLowering : public OpConversionPattern<funlang::ApplyOp> {
  using OpConversionPattern<funlang::ApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      funlang::ApplyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // ==============================
    // 1. 함수 포인터 추출 (env[0])
    // ==============================
    Value closure = adaptor.getClosure();

    auto slot0 = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, closure,
        ArrayRef<LLVM::GEPArg>{0});

    auto fnPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, slot0);

    // ==============================
    // 2. 인자 리스트 구성
    // ==============================
    SmallVector<Value> callArgs;

    // 첫 번째 인자: 환경 포인터 (클로저 자체)
    callArgs.push_back(closure);

    // 나머지 인자: 실제 인자들
    callArgs.append(adaptor.getArgs().begin(), adaptor.getArgs().end());

    // ==============================
    // 3. 결과 타입 변환
    // ==============================
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // ==============================
    // 4. 간접 호출
    // ==============================
    auto call = rewriter.create<LLVM::CallOp>(
        loc, resultType, fnPtr, callArgs);

    // ==============================
    // 5. 원본 operation 교체
    // ==============================
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};
```

### 코드 상세 설명

#### 1. 함수 포인터 추출

```cpp
Value closure = adaptor.getClosure();

auto slot0 = rewriter.create<LLVM::GEPOp>(
    loc, ptrType, ptrType, closure,
    ArrayRef<LLVM::GEPArg>{0});

auto fnPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, slot0);
```

- **`adaptor.getClosure()`**: 클로저 포인터 (이미 `!llvm.ptr`로 변환됨)
- **GEP**: `closure[0]` 주소 계산 (함수 포인터 슬롯)
- **Load**: 함수 포인터 로드

**생성된 MLIR:**

```mlir
%slot0 = llvm.getelementptr %closure[0] : (!llvm.ptr) -> !llvm.ptr
%fn_ptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
```

#### 2. 인자 리스트 구성

```cpp
SmallVector<Value> callArgs;
callArgs.push_back(closure);  // 환경 포인터
callArgs.append(adaptor.getArgs().begin(), adaptor.getArgs().end());
```

- **첫 번째 인자**: 클로저 자체 (환경 포인터)
- **나머지 인자**: 실제 application 인자들

**예시:**

```mlir
// funlang.apply %closure(%x, %y)
// callArgs = [%closure, %x, %y]
```

**왜 closure를 첫 번째 인자로?**

Lifted function은 환경 포인터를 첫 번째 파라미터로 받는다:

```mlir
func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // %env에서 캡처된 변수 접근
    ...
}
```

#### 3. 결과 타입 변환

```cpp
Type resultType = getTypeConverter()->convertType(op.getResult().getType());
```

- **`getTypeConverter()`**: Pattern에 연결된 TypeConverter
- **`convertType()`**: FunLang 타입 → LLVM 타입

**변환 예시:**

```cpp
// funlang.closure → !llvm.ptr
!funlang.closure  ->  !llvm.ptr

// 기본 타입은 그대로
i32  ->  i32
i64  ->  i64
```

**왜 필요한가?**

함수가 클로저를 반환할 수 있다:

```mlir
// funlang.apply의 결과가 또 다른 클로저
%closure2 = funlang.apply %closure(%x) : (i32) -> !funlang.closure

// Lowering 후
%closure2 = llvm.call %fn_ptr(%closure, %x) : (!llvm.ptr, i32) -> !llvm.ptr
```

#### 4. 간접 호출

```cpp
auto call = rewriter.create<LLVM::CallOp>(
    loc, resultType, fnPtr, callArgs);
```

- **`LLVM::CallOp`**: `llvm.call` operation
- **Callee**: `fnPtr` (함수 포인터, `Value`)
- **인자**: `callArgs` (환경 + 실제 인자)
- **반환 타입**: `resultType`

**일반 호출 vs 간접 호출:**

```mlir
// 일반 호출 (direct call)
%result = llvm.call @foo(%x) : (i32) -> i32

// 간접 호출 (indirect call)
%result = llvm.call %fn_ptr(%x) : (i32) -> i32
```

**생성된 MLIR:**

```mlir
%result = llvm.call %fn_ptr(%closure, %x) : (!llvm.ptr, i32) -> i32
```

#### 5. 원본 operation 교체

```cpp
rewriter.replaceOp(op, call.getResult(0));
return success();
```

- **`call.getResult(0)`**: `llvm.call`의 반환 값
- **교체**: `funlang.apply` 결과를 `llvm.call` 결과로 대체

### ApplyOpLowering 테스트

**입력 MLIR:**

```mlir
func.func @test(%closure: !funlang.closure, %x: i32) -> i32 {
    %result = funlang.apply %closure(%x) : (i32) -> i32
    func.return %result : i32
}
```

**Lowering pass 실행:**

```bash
mlir-opt --funlang-to-llvm test.mlir
```

**출력 MLIR:**

```mlir
func.func @test(%closure: !llvm.ptr, %x: i32) -> i32 {
    %0 = llvm.getelementptr %closure[0] : (!llvm.ptr) -> !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.call %1(%closure, %x) : (!llvm.ptr, i32) -> i32
    func.return %2 : i32
}
```

**검증:**

1. `funlang.apply` 사라짐 ✓
2. GEP + load로 함수 포인터 추출 ✓
3. 간접 호출 (`llvm.call %fn_ptr`) ✓
4. 인자 리스트 올바름 (`%closure, %x`) ✓

### End-to-End 예시: makeAdder

**Phase 5 FunLang dialect:**

```mlir
func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    func.return %closure : !funlang.closure
}

func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}

func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32

    // makeAdder 5
    %add5 = funlang.closure @lambda_adder, %c5 : !funlang.closure

    // add5 10
    %result = funlang.apply %add5(%c10) : (i32) -> i32

    func.return %result : i32
}
```

**After FunLangToLLVM pass:**

```mlir
func.func @make_adder(%n: i32) -> !llvm.ptr {
    // ClosureOpLowering
    %c16 = arith.constant 16 : i64
    %env = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
    %fn_ptr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %slot0 = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
    %slot1 = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %slot1 : i32, !llvm.ptr
    func.return %env : !llvm.ptr
}

func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
}

func.func @main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32

    // ClosureOpLowering
    %c16 = arith.constant 16 : i64
    %add5 = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
    %fn_ptr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %slot0 = llvm.getelementptr %add5[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
    %slot1 = llvm.getelementptr %add5[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %c5, %slot1 : i32, !llvm.ptr

    // ApplyOpLowering
    %fn_ptr_addr = llvm.getelementptr %add5[0] : (!llvm.ptr) -> !llvm.ptr
    %fn_ptr_loaded = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr_loaded(%add5, %c10) : (!llvm.ptr, i32) -> i32

    func.return %result : i32
}
```

**실행 흐름 추적:**

1. `%c5`, `%c10` 상수 생성
2. **Closure 생성** (ClosureOpLowering):
   - `GC_malloc(16)` → `%add5` (환경 포인터)
   - `%add5[0] = @lambda_adder` (함수 포인터)
   - `%add5[1] = 5` (캡처된 `n`)
3. **Closure 호출** (ApplyOpLowering):
   - `%add5[0]` 로드 → `%fn_ptr_loaded` (함수 포인터)
   - `llvm.call %fn_ptr_loaded(%add5, 10)`
4. **lambda_adder 실행**:
   - `%env[1]` 로드 → `%n = 5`
   - `10 + 5 = 15`
   - 반환: `15`

---

## TypeConverter for FunLang Types

**TypeConverter**는 FunLang 타입을 LLVM 타입으로 변환한다.

### FunLang Custom Types (Chapter 15)

**1. funlang.closure:**

```tablegen
def FunLang_ClosureType : FunLang_Type<"Closure"> {
  let mnemonic = "closure";
  let description = "FunLang closure type (function pointer + environment)";
}
```

**MLIR 표기:** `!funlang.closure`

**2. funlang.list (Phase 6 preview):**

```tablegen
def FunLang_ListType : FunLang_Type<"List"> {
  let mnemonic = "list";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}
```

**MLIR 표기:** `!funlang.list<i32>`, `!funlang.list<!funlang.closure>`

### TypeConverter 구성

```cpp
TypeConverter typeConverter;

// ==============================
// 1. FunLang 타입 변환
// ==============================

// funlang.closure → !llvm.ptr
typeConverter.addConversion([&](funlang::ClosureType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
});

// funlang.list<T> → !llvm.ptr
typeConverter.addConversion([&](funlang::ListType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
});

// ==============================
// 2. 기본 타입은 그대로
// ==============================
typeConverter.addConversion([](Type type) {
    // i32, i64, i1 등은 변환하지 않음
    return type;
});
```

**변환 예시:**

| Before | After |
|--------|-------|
| `!funlang.closure` | `!llvm.ptr` |
| `!funlang.list<i32>` | `!llvm.ptr` |
| `i32` | `i32` |
| `i64` | `i64` |

### Function Signature 변환

**TypeConverter는 자동으로 function signatures를 변환한다.**

**Before:**

```mlir
func.func @apply(%f: !funlang.closure, %x: i32) -> i32 {
    %result = funlang.apply %f(%x) : (i32) -> i32
    func.return %result : i32
}
```

**After:**

```mlir
func.func @apply(%f: !llvm.ptr, %x: i32) -> i32 {
    %0 = llvm.getelementptr %f[0] : (!llvm.ptr) -> !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.call %1(%f, %x) : (!llvm.ptr, i32) -> i32
    func.return %2 : i32
}
```

**변환 지점:**

- **파라미터 타입**: `%f: !funlang.closure` → `%f: !llvm.ptr`
- **반환 타입**: 여기서는 `i32` (변환 없음)
- **Operation result 타입**: `funlang.apply` 결과 타입 변환

### Materialization: 타입 변환 보조

**Materialization**은 타입 변환 중간에 필요한 "접착제" operations을 삽입한다.

**사용 사례:** Conversion이 여러 단계로 나뉠 때, 중간 타입 불일치 해결.

#### Source Materialization

```cpp
typeConverter.addSourceMaterialization(
    [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
      // 원본 타입 (FunLang) → 중간 타입 변환
      if (resultType.isa<funlang::ClosureType>()) {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      }
      return nullptr;
    });
```

#### Target Materialization

```cpp
typeConverter.addTargetMaterialization(
    [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
      // 중간 타입 → 대상 타입 (LLVM) 변환
      if (resultType.isa<LLVM::LLVMPointerType>()) {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      }
      return nullptr;
    });
```

#### unrealized_conversion_cast

**Materialization이 생성하는 operation:**

```mlir
%cast = builtin.unrealized_conversion_cast %input : !funlang.closure to !llvm.ptr
```

**의미:** "이 타입 변환은 아직 완료되지 않았다"

**최종 lowering 후:**

- 모든 `unrealized_conversion_cast`는 사라져야 한다
- 남아있으면 conversion failure

**Phase 5에서는 단순한 변환이므로 materialization 불필요:**

- `funlang.closure` → `!llvm.ptr` (direct mapping)
- 중간 타입 없음

### 타입 변환 체인

**Multi-stage lowering에서 타입 변환 체인:**

```
Phase 5 FunLang dialect:
  !funlang.closure

Phase 5a (optional): High-level abstractions
  !funlang.env_ptr  (환경 포인터 전용 타입)

Phase 5b (final): LLVM dialect
  !llvm.ptr
```

**현재 Phase 5 (단순 버전):**

```
!funlang.closure  →  !llvm.ptr  (direct)
```

**TypeConverter 체인 예시 (multi-stage):**

```cpp
// Stage 1: FunLang → HighLevel
TypeConverter highLevelConverter;
highLevelConverter.addConversion([](funlang::ClosureType type) {
    return funlang::EnvPtrType::get(type.getContext());
});

// Stage 2: HighLevel → LLVM
TypeConverter llvmConverter;
llvmConverter.addConversion([](funlang::EnvPtrType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
});
```

---

## Declarative Rewrite Rules (DRR)

**DRR (Declarative Rewrite Rules)**은 **TableGen 기반 패턴 매칭 시스템**이다. C++ ConversionPattern보다 간단한 변환을 선언적으로 작성할 수 있다.

### DRR이란?

**DRR**은 MLIR의 패턴 매칭 DSL이다:

- **입력**: `.td` 파일에 패턴 작성
- **출력**: C++ 코드 자동 생성 (mlir-tblgen)
- **용도**: 최적화, 정규화, 간단한 lowering

**DRR vs C++ ConversionPattern:**

| Aspect | DRR | C++ ConversionPattern |
|--------|-----|----------------------|
| **문법** | 선언적 (TableGen) | 명령형 (C++) |
| **복잡도** | 간단한 패턴 | 복잡한 로직 가능 |
| **제어 흐름** | 없음 (순수 매칭) | if/for/while 가능 |
| **타입 안전성** | 컴파일 타임 | 런타임 검증 |
| **디버깅** | 어려움 | 쉬움 (breakpoint) |

**언제 DRR을 사용하는가?**

- ✓ 1:1 operation 변환 (A → B)
- ✓ 간단한 패턴 매칭 (조건 1-2개)
- ✓ 최적화 패턴 (constant folding, peephole)

**언제 C++를 사용하는가?**

- ✓ 복잡한 변환 로직 (ClosureOpLowering처럼 여러 ops 생성)
- ✓ 동적 계산 (환경 크기 계산)
- ✓ 제어 흐름 (for loop으로 캡처 변수 처리)

### DRR 문법 기초

**Pat (Pattern) 정의:**

```tablegen
def PatternName : Pat<
  (SourcePattern),   // 매치할 패턴
  (TargetPattern),   // 교체할 패턴
  [(Constraint)]     // 추가 제약 (optional)
>;
```

**예시: 상수 폴딩**

```tablegen
def AddZero : Pat<
  (Arith_AddIOp $x, (Arith_ConstantOp ConstantAttr<I32Attr, "0">)),
  (replaceWithValue $x)
>;
```

**의미:** `x + 0` → `x`

### DRR 예시 1: Empty Closure 최적화

**최적화 목표:**

캡처 변수가 없는 클로저는 함수 포인터만 필요하다. 환경 할당 불필요.

**Before:**

```mlir
// 캡처 없음
%closure = funlang.closure @foo : !funlang.closure

// Lowering 후 (불필요한 GC_malloc!)
%env = llvm.call @GC_malloc(%c8) : (i64) -> !llvm.ptr
%fn_ptr = llvm.mlir.addressof @foo : !llvm.ptr
llvm.store %fn_ptr, %env[0] : !llvm.ptr
```

**After (최적화):**

```mlir
// 함수 포인터만 사용
%fn_ptr = llvm.mlir.addressof @foo : !llvm.ptr

// apply에서 직접 함수 포인터 사용
%result = llvm.call @foo(%null_env, %x) : (!llvm.ptr, i32) -> i32
```

**DRR 패턴:**

```tablegen
def SimplifyEmptyClosure : Pat<
  // Match: funlang.closure with no captured variables
  (FunLang_ClosureOp:$result $callee, (variadic)),

  // Replace: function reference (Phase 6에 FuncRefOp 추가 필요)
  (FunLang_FuncRefOp $callee),

  // Constraint: captured variables must be empty
  [(Constraint<CPred<"$0.empty()">, "$result.getCaptured()">)]
>;
```

**설명:**

- **`(variadic)`**: 가변 인자 (0개 이상)
- **`CPred<"$0.empty()">`**: C++ predicate - 첫 번째 인자가 비어있는가?
- **`FuncRefOp`**: 함수 참조만 담는 operation (Phase 6에서 추가 예정)

### DRR 예시 2: Known Closure Inlining

**최적화 목표:**

클로저 생성 직후 호출하면 인라인 가능.

**Before:**

```mlir
// 클로저 생성 후 즉시 호출
%closure = funlang.closure @lambda, %n : !funlang.closure
%result = funlang.apply %closure(%x) : (i32) -> i32
```

**After (최적화):**

```mlir
// 직접 호출 (환경 할당 불필요)
%result = func.call @lambda(%n, %x) : (i32, i32) -> i32
```

**DRR 패턴:**

```tablegen
def InlineKnownApply : Pat<
  // Match: apply (closure @callee, $captures) ($args)
  (FunLang_ApplyOp
    (FunLang_ClosureOp:$closure $callee, $captures),
    $args),

  // Replace: direct call @callee (concat $captures and $args)
  (Func_CallOp $callee, (ConcatValues $captures, $args))
>;
```

**설명:**

- **`$captures`**: 캡처된 변수들 (variadic)
- **`$args`**: apply 인자들 (variadic)
- **`ConcatValues`**: 두 리스트 합치기 (DRR helper)
- **`Func_CallOp`**: 직접 호출 operation

**제약:**

이 패턴은 **클로저가 escape하지 않을 때만** 안전하다:

```mlir
// OK: 즉시 호출
%result = funlang.apply (funlang.closure @f, %n) (%x)

// NOT OK: 클로저가 반환됨 (인라인 불가!)
func.func @make_adder(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @f, %n : !funlang.closure
    func.return %closure  // Escape!
}
```

**DRR로 escape 검사 불가 → C++ ConversionPattern 필요**

### DRR 예시 3: Constant Propagation

**최적화 목표:**

클로저가 상수만 캡처하면 compile-time에 처리 가능.

**Before:**

```mlir
%c5 = arith.constant 5 : i32
%closure = funlang.closure @lambda, %c5 : !funlang.closure
```

**After (최적화):**

```mlir
// lambda 함수 내부에서 %c5를 직접 사용하도록 인라인
// (복잡한 변환이므로 DRR보다 C++가 적합)
```

**DRR 한계:**

- 함수 본문 수정 필요 (DRR은 local pattern만 매칭)
- Whole-program analysis 필요 (DRR은 single operation 매칭)

**결론:** 이런 최적화는 **C++ pass**로 구현해야 함.

### mlir-tblgen으로 DRR 컴파일

**1. DRR 패턴 작성:**

```tablegen
// FunLangPatterns.td
include "mlir/IR/PatternBase.td"
include "FunLangOps.td"

def SimplifyEmptyClosure : Pat<
  (FunLang_ClosureOp:$result $callee, (variadic)),
  (FunLang_FuncRefOp $callee),
  [(Constraint<CPred<"$0.empty()">, "$result.getCaptured()">)]
>;
```

**2. mlir-tblgen 실행:**

```bash
mlir-tblgen -gen-rewriters FunLangPatterns.td -o FunLangPatterns.cpp.inc
```

**3. 생성된 C++ 코드:**

```cpp
// FunLangPatterns.cpp.inc
struct SimplifyEmptyClosure : public RewritePattern {
  SimplifyEmptyClosure(MLIRContext *context)
      : RewritePattern(ClosureOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto closureOp = cast<ClosureOp>(op);

    // Constraint: captured variables empty
    if (!closureOp.getCaptured().empty())
      return failure();

    // Rewrite: create FuncRefOp
    rewriter.replaceOpWithNewOp<FuncRefOp>(op, closureOp.getCalleeAttr());
    return success();
  }
};
```

**4. Pass에 등록:**

```cpp
void populateFunLangOptimizationPatterns(RewritePatternSet &patterns) {
  patterns.add<SimplifyEmptyClosure>(patterns.getContext());
  // ... other patterns
}
```

### DRR vs C++ ConversionPattern 비교 요약

**ClosureOpLowering을 DRR로 작성하면?**

```tablegen
// 불가능! DRR로는 표현 못함
def LowerClosure : Pat<
  (FunLang_ClosureOp $callee, $captured),
  (??? 어떻게 for loop을 표현?)  // 캡처 변수 개수만큼 GEP + store
>;
```

**왜 불가능?**

- DRR은 **fixed-size patterns**만 매칭
- 가변 개수의 operations 생성 불가 (for loop 없음)
- 동적 계산 불가 (환경 크기 계산)

**결론:**

- **DRR**: 간단한 최적화 패턴 (peephole, constant folding)
- **C++ ConversionPattern**: 복잡한 lowering (ClosureOp, ApplyOp)

---

## Complete Lowering Pass

**FunLangToLLVMPass**는 FunLang dialect를 LLVM dialect로 lowering하는 완전한 pass다.

### Pass 정의

```cpp
// FunLangToLLVMPass.cpp
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "FunLang/FunLangDialect.h"
#include "FunLang/FunLangOps.h"

namespace {

struct FunLangToLLVMPass
    : public PassWrapper<FunLangToLLVMPass, OperationPass<ModuleOp>> {

  // ==============================
  // Pass metadata
  // ==============================
  StringRef getArgument() const final {
    return "funlang-to-llvm";
  }

  StringRef getDescription() const final {
    return "Lower FunLang dialect to LLVM dialect";
  }

  // ==============================
  // Dependent dialects
  // ==============================
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
  }

  // ==============================
  // Pass execution
  // ==============================
  void runOnOperation() override {
    // Get module operation
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // ------------------------------
    // 1. Setup ConversionTarget
    // ------------------------------
    ConversionTarget target(*ctx);

    // Legal: LLVM, func, arith dialects
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    // Illegal: FunLang dialect (must be lowered)
    target.addIllegalDialect<funlang::FunLangDialect>();

    // ------------------------------
    // 2. Setup TypeConverter
    // ------------------------------
    TypeConverter typeConverter;

    // FunLang types → LLVM types
    typeConverter.addConversion([&](funlang::ClosureType type) {
      return LLVM::LLVMPointerType::get(ctx);
    });

    typeConverter.addConversion([&](funlang::ListType type) {
      return LLVM::LLVMPointerType::get(ctx);
    });

    // Default: keep type as-is (i32, i64, etc.)
    typeConverter.addConversion([](Type type) {
      return type;
    });

    // ------------------------------
    // 3. Setup RewritePatternSet
    // ------------------------------
    RewritePatternSet patterns(ctx);

    // Add lowering patterns
    patterns.add<ClosureOpLowering>(ctx, typeConverter);
    patterns.add<ApplyOpLowering>(ctx, typeConverter);

    // ------------------------------
    // 4. Apply conversion
    // ------------------------------
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

// ==============================
// Pass registration
// ==============================
void registerFunLangToLLVMPass() {
  PassRegistration<FunLangToLLVMPass>();
}
```

### Pass 구성 요소 설명

#### 1. PassWrapper 템플릿

```cpp
struct FunLangToLLVMPass
    : public PassWrapper<FunLangToLLVMPass, OperationPass<ModuleOp>> {
```

- **`PassWrapper<Self, Base>`**: CRTP 패턴
- **`OperationPass<ModuleOp>`**: Module 레벨 pass (전체 IR 처리)

**다른 pass 레벨:**

- `OperationPass<func::FuncOp>`: Function 레벨 (함수별 처리)
- `OperationPass<>`: 모든 operation에 대해

#### 2. getDependentDialects

```cpp
void getDependentDialects(DialectRegistry &registry) const override {
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
}
```

**역할:** Pass가 사용할 dialects를 등록한다.

**왜 필요?**

- MLIR은 lazy dialect loading 사용
- Pass가 `LLVM::CallOp`을 생성하려면 `LLVMDialect` 로드 필요
- 명시적 등록으로 dependency 보장

#### 3. runOnOperation

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();
  // ... conversion logic
}
```

**역할:** Pass의 핵심 로직.

**실행 흐름:**

1. Target 설정 (legal/illegal dialects)
2. TypeConverter 설정 (타입 변환 규칙)
3. Patterns 구성 (lowering patterns)
4. Conversion 실행 (applyPartialConversion)
5. 실패 시 signalPassFailure()

### Pass 등록

```cpp
void registerFunLangToLLVMPass() {
  PassRegistration<FunLangToLLVMPass>();
}

// 초기화 함수에서 호출
void registerFunLangPasses() {
  registerFunLangToLLVMPass();
  // ... other passes
}
```

**등록 후 사용:**

```bash
mlir-opt --funlang-to-llvm input.mlir -o output.mlir
```

### C API Shim

**C++ pass를 F#에서 사용하려면 C API가 필요하다.**

```cpp
// FunLangCAPI.cpp
#include "mlir-c/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Pass/PassManager.h"
#include "FunLangPasses.h"

extern "C" {

// Pass 등록
MLIR_CAPI_EXPORTED void mlirFunLangRegisterToLLVMPass() {
  registerFunLangToLLVMPass();
}

// Pass 실행
MLIR_CAPI_EXPORTED void mlirFunLangRunToLLVMPass(MlirModule module) {
  ModuleOp moduleOp = unwrap(module);
  MLIRContext *ctx = moduleOp.getContext();

  PassManager pm(ctx);
  pm.addPass(std::make_unique<FunLangToLLVMPass>());

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "FunLangToLLVMPass failed\n";
  }
}

} // extern "C"
```

**헬퍼 함수:**

```cpp
// Wrap/unwrap helpers (MLIR-C API)
static inline ModuleOp unwrap(MlirModule module) {
  return ::mlir::unwrap(module);
}

static inline MlirModule wrap(ModuleOp module) {
  return ::mlir::wrap(module);
}
```

### F# P/Invoke

```fsharp
// Mlir.FunLang.fs
module Mlir.FunLang

open System.Runtime.InteropServices

// ==============================
// P/Invoke declarations
// ==============================

[<DllImport("funlang-dialect", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirFunLangRegisterToLLVMPass()

[<DllImport("funlang-dialect", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirFunLangRunToLLVMPass(MlirModule module)

// ==============================
// F# wrapper functions
// ==============================

/// Initialize FunLang passes (call once at startup)
let initializePasses () =
    mlirFunLangRegisterToLLVMPass()

/// Lower FunLang dialect to LLVM dialect
let lowerToLLVM (module_: MlirModule) =
    mlirFunLangRunToLLVMPass(module_)
```

### F#에서 Pass 사용

```fsharp
// CompilerPipeline.fs
open Mlir
open Mlir.FunLang

// 초기화 (프로그램 시작 시 1회)
FunLang.initializePasses()

// 컴파일 파이프라인
let compileToExecutable (source: string) =
    // 1. Parse & build AST
    let ast = Parser.parse source

    // 2. Generate FunLang dialect MLIR
    use ctx = Mlir.createContext()
    use module_ = Mlir.createModule(ctx)
    use builder = Mlir.createOpBuilder(ctx)

    // ... code generation (Chapter 15)

    // 3. Lower FunLang → LLVM
    FunLang.lowerToLLVM(module_)

    // 4. Lower other dialects → LLVM
    Mlir.runPass(module_, "convert-arith-to-llvm")
    Mlir.runPass(module_, "convert-func-to-llvm")

    // 5. Translate LLVM dialect → LLVM IR
    let llvmIR = Mlir.translateToLLVMIR(module_)

    // 6. Compile & link
    let objFile = LLVMCompiler.compile(llvmIR)
    let executable = Linker.link([objFile; "runtime.o"], "gc")

    executable
```

---

## End-to-End Example

**makeAdder 함수를 전체 파이프라인으로 추적한다.**

### Source Code

```fsharp
// FunLang source
let makeAdder n =
    fun x -> x + n

let add5 = makeAdder 5
let result = add5 10
```

### Stage 1: AST

```fsharp
type Expr =
    | Let of string * Expr * Expr
    | Lambda of string * Expr
    | App of Expr * Expr
    | BinOp of Operator * Expr * Expr
    | Var of string
    | Const of int

// makeAdder AST
Let ("makeAdder",
     Lambda ("n", Lambda ("x", BinOp (Add, Var "x", Var "n"))),
     Let ("add5",
          App (Var "makeAdder", Const 5),
          Let ("result",
               App (Var "add5", Const 10),
               Var "result")))
```

### Stage 2: FunLang Dialect MLIR (Chapter 15)

```mlir
module {
  // lifted lambda: fun x -> x + n
  func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    // Load captured n from env[1]
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32

    // x + n
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  // makeAdder function
  func.func @makeAdder(%n: i32) -> !funlang.closure {
    %closure = funlang.closure @lambda_adder, %n : !funlang.closure
    func.return %closure : !funlang.closure
  }

  // main function
  func.func @funlang_main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32

    // makeAdder 5
    %add5 = funlang.closure @lambda_adder, %c5 : !funlang.closure

    // add5 10
    %result = funlang.apply %add5(%c10) : (i32) -> i32

    func.return %result : i32
  }
}
```

### Stage 3: After FunLangToLLVM Pass (Chapter 16)

```mlir
module {
  // lambda_adder (unchanged)
  func.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
    %n_slot = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    %n = llvm.load %n_slot : !llvm.ptr -> i32
    %result = arith.addi %x, %n : i32
    func.return %result : i32
  }

  // makeAdder (funlang.closure lowered)
  func.func @makeAdder(%n: i32) -> !llvm.ptr {
    %c16 = arith.constant 16 : i64
    %env = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
    %fn_ptr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %slot0 = llvm.getelementptr %env[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
    %slot1 = llvm.getelementptr %env[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %n, %slot1 : i32, !llvm.ptr
    func.return %env : !llvm.ptr
  }

  // main (both funlang operations lowered)
  func.func @funlang_main() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32

    // ClosureOpLowering
    %c16 = arith.constant 16 : i64
    %add5 = llvm.call @GC_malloc(%c16) : (i64) -> !llvm.ptr
    %fn_ptr = llvm.mlir.addressof @lambda_adder : !llvm.ptr
    %slot0 = llvm.getelementptr %add5[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %fn_ptr, %slot0 : !llvm.ptr, !llvm.ptr
    %slot1 = llvm.getelementptr %add5[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %c5, %slot1 : i32, !llvm.ptr

    // ApplyOpLowering
    %fn_ptr_addr = llvm.getelementptr %add5[0] : (!llvm.ptr) -> !llvm.ptr
    %fn_ptr_loaded = llvm.load %fn_ptr_addr : !llvm.ptr -> !llvm.ptr
    %result = llvm.call %fn_ptr_loaded(%add5, %c10) : (!llvm.ptr, i32) -> i32

    func.return %result : i32
  }
}
```

### Stage 4: After convert-arith-to-llvm

```mlir
// arith.constant → llvm.mlir.constant
%c5 = llvm.mlir.constant(5 : i32) : i32
%c10 = llvm.mlir.constant(10 : i32) : i32
%c16 = llvm.mlir.constant(16 : i64) : i64

// arith.addi → llvm.add
%result = llvm.add %x, %n : i32
```

### Stage 5: After convert-func-to-llvm

```mlir
// func.func → llvm.func
llvm.func @lambda_adder(%env: !llvm.ptr, %x: i32) -> i32 {
  ...
  llvm.return %result : i32
}

// func.call → llvm.call (already indirect, no change)
%result = llvm.call %fn_ptr_loaded(%add5, %c10) : (!llvm.ptr, i32) -> i32
```

### Stage 6: LLVM IR (mlir-translate --mlir-to-llvmir)

```llvm
define i32 @lambda_adder(ptr %env, i32 %x) {
  %n_slot = getelementptr ptr, ptr %env, i32 1
  %n = load i32, ptr %n_slot
  %result = add i32 %x, %n
  ret i32 %result
}

define ptr @makeAdder(i32 %n) {
  %env = call ptr @GC_malloc(i64 16)
  %fn_ptr = ptrtoint ptr @lambda_adder to i64
  %slot0 = getelementptr ptr, ptr %env, i32 0
  store i64 %fn_ptr, ptr %slot0
  %slot1 = getelementptr ptr, ptr %env, i32 1
  store i32 %n, ptr %slot1
  ret ptr %env
}

define i32 @funlang_main() {
  %c5 = 5
  %c10 = 10

  ; Closure creation
  %add5 = call ptr @GC_malloc(i64 16)
  %fn_ptr = ptrtoint ptr @lambda_adder to i64
  %slot0 = getelementptr ptr, ptr %add5, i32 0
  store i64 %fn_ptr, ptr %slot0
  %slot1 = getelementptr ptr, ptr %add5, i32 1
  store i32 %c5, ptr %slot1

  ; Closure application
  %fn_ptr_addr = getelementptr ptr, ptr %add5, i32 0
  %fn_ptr_loaded = load ptr, ptr %fn_ptr_addr
  %result = call i32 %fn_ptr_loaded(ptr %add5, i32 %c10)

  ret i32 %result
}
```

### Stage 7: Native Code (llc → object file → executable)

```bash
# LLVM IR → object file
llc output.ll -o output.o -filetype=obj

# Link with runtime
clang output.o runtime.o -lgc -o program

# Run
./program
# Output: 15
```

### 컴파일 파이프라인 다이어그램

```
┌─────────────────┐
│  FunLang Source │  let makeAdder n = fun x -> x + n
└────────┬────────┘
         │ Parser
         ▼
┌─────────────────┐
│       AST       │  Lambda, App, BinOp nodes
└────────┬────────┘
         │ CodeGen (Chapter 15)
         ▼
┌─────────────────┐
│ FunLang Dialect │  funlang.closure, funlang.apply
│      MLIR       │
└────────┬────────┘
         │ FunLangToLLVM Pass (Chapter 16) ★
         ▼
┌─────────────────┐
│  LLVM Dialect   │  llvm.call, llvm.getelementptr, llvm.store
│      MLIR       │
└────────┬────────┘
         │ convert-arith-to-llvm
         │ convert-func-to-llvm
         ▼
┌─────────────────┐
│  LLVM Dialect   │  All dialects → LLVM dialect
│ (fully lowered) │
└────────┬────────┘
         │ mlir-translate --mlir-to-llvmir
         ▼
┌─────────────────┐
│    LLVM IR      │  %.1 = call ptr @GC_malloc(i64 16)
└────────┬────────┘
         │ llc
         ▼
┌─────────────────┐
│  Object File    │  .o binary
└────────┬────────┘
         │ clang (link)
         ▼
┌─────────────────┐
│   Executable    │  ./program
└─────────────────┘
```

---

## Common Errors

Lowering pass 구현 중 자주 발생하는 에러와 해결 방법.

### Error 1: Illegal Operation Remaining

**증상:**

```
error: failed to legalize operation 'funlang.closure' that was explicitly marked illegal
note: see current operation: %0 = "funlang.closure"() {callee = @foo, ...}
```

**원인:**

- Pattern이 등록되지 않음
- Pattern이 매치 실패 (`matchAndRewrite`에서 `failure()` 리턴)
- Target에 illegal로 설정했지만 pattern 없음

**해결:**

1. **Pattern 등록 확인:**

```cpp
RewritePatternSet patterns(ctx);
patterns.add<ClosureOpLowering>(ctx, typeConverter);  // 추가했는가?
```

2. **Pattern 매치 조건 확인:**

```cpp
LogicalResult matchAndRewrite(...) const override {
  // 디버그 출력
  llvm::errs() << "ClosureOpLowering matched\n";

  // ... lowering logic

  return success();  // failure() 리턴하지 않았는가?
}
```

3. **Target 설정 확인:**

```cpp
target.addIllegalDialect<funlang::FunLangDialect>();  // Illegal로 설정
```

### Error 2: Type Conversion Failure

**증상:**

```
error: type conversion failed for block argument #0
note: see current operation: func.func @foo(%arg0: !funlang.closure) -> i32
```

**원인:**

- TypeConverter에 변환 규칙 없음
- 변환 규칙이 `nullptr` 리턴

**해결:**

1. **TypeConverter에 규칙 추가:**

```cpp
typeConverter.addConversion([&](funlang::ClosureType type) {
    return LLVM::LLVMPointerType::get(ctx);
});
```

2. **변환 실패 체크:**

```cpp
typeConverter.addConversion([&](funlang::ClosureType type) -> std::optional<Type> {
    if (!isConvertible(type))
        return std::nullopt;  // 변환 불가

    return LLVM::LLVMPointerType::get(ctx);
});
```

### Error 3: Wrong Operand Types

**증상:**

```
error: 'llvm.store' op types mismatch between stored value and pointee type
note: stored value: i32, pointee type: !llvm.ptr
```

**원인:**

- Store operation에 타입 불일치
- GEP 결과를 잘못 사용

**해결:**

1. **Store 타입 확인:**

```cpp
// 잘못됨: i32를 !llvm.ptr 슬롯에 저장
rewriter.create<LLVM::StoreOp>(loc, i32Value, ptrSlot);

// 올바름: 타입 일치
rewriter.create<LLVM::StoreOp>(loc, i32Value, i32Slot);
```

2. **GEP 사용 확인:**

```mlir
// 올바른 GEP 패턴
%slot = llvm.getelementptr %ptr[1] : (!llvm.ptr) -> !llvm.ptr
```

### Error 4: Pass Not Registered

**증상:**

```bash
$ mlir-opt --funlang-to-llvm test.mlir
error: unknown command line flag '--funlang-to-llvm'
```

**원인:**

- Pass 등록 함수가 호출되지 않음

**해결:**

1. **Pass 등록 확인:**

```cpp
// 초기화 코드에서 호출
void initializeMLIR() {
  registerFunLangDialect();
  registerFunLangToLLVMPass();  // 등록 함수 호출
}
```

2. **C API shim 확인:**

```cpp
extern "C" void mlirFunLangRegisterToLLVMPass() {
  registerFunLangToLLVMPass();
}
```

3. **F# 초기화 확인:**

```fsharp
// 프로그램 시작 시 호출
FunLang.initializePasses()
```

### Error 5: Segmentation Fault in Pattern

**증상:**

```
Segmentation fault (core dumped)
```

**원인:**

- `rewriter` 대신 일반 builder 사용
- Null pointer dereference
- Use-after-free (op 삭제 후 접근)

**해결:**

1. **항상 `rewriter` 사용:**

```cpp
// 잘못됨!
OpBuilder builder(ctx);
builder.create<...>();

// 올바름
rewriter.create<...>();
```

2. **Op 삭제 후 접근 금지:**

```cpp
// 잘못됨!
rewriter.replaceOp(op, newValue);
auto attr = op.getAttr("foo");  // Use-after-free!

// 올바름
auto attr = op.getAttr("foo");  // 먼저 읽기
rewriter.replaceOp(op, newValue);
```

3. **Null 체크:**

```cpp
Value closure = adaptor.getClosure();
if (!closure) {
  return failure();
}
```

---

## Summary

**Chapter 16에서 배운 것:**

### 1. DialectConversion Framework

- **ConversionTarget**: Legal/illegal operations 정의
- **RewritePatternSet**: 변환 규칙 집합
- **TypeConverter**: 타입 변환 규칙
- **applyPartialConversion**: 부분 변환 실행

### 2. ClosureOp Lowering Pattern

- `funlang.closure` → GC_malloc + GEP + store
- Chapter 12 클로저 생성 패턴 재사용
- OpAdaptor로 변환된 operands 접근
- ConversionPatternRewriter로 IR 수정

### 3. ApplyOp Lowering Pattern

- `funlang.apply` → GEP + load + llvm.call (indirect)
- Chapter 13 간접 호출 패턴 재사용
- 인자 리스트 구성 (환경 포인터 + 실제 인자)
- TypeConverter로 결과 타입 변환

### 4. TypeConverter for FunLang Types

- `!funlang.closure` → `!llvm.ptr`
- `!funlang.list<T>` → `!llvm.ptr`
- Function signatures 자동 변환
- Materialization (optional)

### 5. Declarative Rewrite Rules (DRR)

- TableGen 기반 패턴 매칭
- 간단한 최적화 패턴 (empty closure, known closure inlining)
- DRR vs C++ ConversionPattern 비교
- mlir-tblgen으로 C++ 코드 생성

### 6. Complete Lowering Pass

- FunLangToLLVMPass 구현
- Pass 등록 및 실행
- C API shim for F# integration
- F# wrapper functions

### 7. End-to-End Example

- makeAdder: FunLang source → LLVM IR → executable
- 전체 컴파일 파이프라인 추적
- 각 단계별 IR 확인

### 8. Common Errors

- Illegal operation remaining
- Type conversion failure
- Wrong operand types
- Pass not registered
- Segmentation fault

**Phase 5 완료!**

- **Chapter 14**: Custom dialect design theory
- **Chapter 15**: Custom operations implementation (funlang.closure, funlang.apply)
- **Chapter 16**: Lowering passes (FunLangToLLVM)

**코드 압축 효과:**

| Aspect | Before (Phase 4) | After (Phase 5) |
|--------|-----------------|----------------|
| **Closure creation** | 12 lines | 1 line |
| **Closure application** | 8 lines | 1 line |
| **Compiler code** | ~200 lines | ~100 lines |
| **타입 안전성** | `!llvm.ptr` (opaque) | `!funlang.closure` (typed) |
| **최적화 가능성** | 어려움 | 쉬움 (DRR patterns) |

**Phase 6 Preview: Pattern Matching**

다음 Phase에서는 **패턴 매칭**을 추가한다:

```fsharp
// List operations
let rec length list =
    match list with
    | [] -> 0
    | head :: tail -> 1 + length tail
```

**새로운 operations:**

- `funlang.match`: 패턴 매칭
- `funlang.nil`: 빈 리스트
- `funlang.cons`: 리스트 생성
- `funlang.list_head`, `funlang.list_tail`: 리스트 접근

**새로운 lowering patterns:**

- `funlang.match` → `scf.if` + `llvm.switch` (복잡한 제어 흐름)
- SCF dialect를 거친 multi-stage lowering

**Phase 5와 Phase 6의 차이:**

- **Phase 5**: FunLang → LLVM (direct lowering)
- **Phase 6**: FunLang → SCF → LLVM (multi-stage lowering)

**Chapter 16 완료!** 이제 custom dialect를 설계하고, operations를 정의하고, lowering passes를 구현할 수 있다. FunLang 컴파일러는 high-level 추상화와 low-level 성능을 모두 제공한다.

