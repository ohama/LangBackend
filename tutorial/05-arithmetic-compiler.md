# Chapter 05: 산술 컴파일러 - 첫 번째 네이티브 바이너리

## 소개

지금까지의 여정:

- **Chapter 00**: LLVM/MLIR을 빌드하고 .NET SDK를 설치했다
- **Chapter 01**: MLIR 개념 (dialect, operation, region, block, SSA)을 배웠다
- **Chapter 02**: F#에서 처음으로 MLIR IR을 생성했다
- **Chapter 03**: 완전한 P/Invoke 바인딩 모듈을 구축했다
- **Chapter 04**: 안전하고 관용적인 F# 래퍼 레이어를 만들었다

이제 보상을 받을 시간이다.

이 장에서는 **실제 컴파일러**를 구축한다. 소스 코드를 입력으로 받아 실행 가능한 네이티브 바이너리를 출력하는 컴파일러다. 단순화를 위해 FunLang의 매우 작은 부분집합, 즉 **정수 리터럴**만 다룬다. 이것이 사소해 보일 수 있지만 전체 컴파일 파이프라인을 보여준다:

```
Source code → AST → MLIR IR → Lowering → LLVM IR → Object file → Native binary
```

이 장을 마치면 `42`를 네이티브 실행 파일로 컴파일하고 실행하여 프로그램 종료 코드로 `42`를 볼 수 있다.

> **마일스톤:** 이것은 Phase 1의 정점이다. 이 장 이후에는 실제 코드를 컴파일하고 실행하는 작동하는 컴파일러를 갖게 된다!

## FunLang 부분집합

지금은 단 하나의 구문만 지원한다:

```
program ::= <integer>
```

예시:
- `42`
- `0`
- `1337`

이 프로그램은 정수를 종료 코드로 반환한다. Unix에서는 `$?`로 확인할 수 있다:

```bash
./program
echo $?  # 42 출력
```

단순해 보이지만 이것은 다음을 포함한 **완전한 컴파일 파이프라인**을 요구한다:
1. 소스를 AST로 파싱
2. AST를 MLIR IR로 변환
3. MLIR IR 검증
4. LLVM dialect로 낮추기
5. LLVM IR로 변환
6. 오브젝트 파일 생성
7. 실행 파일로 링크

## 컴파일러 파이프라인 개요

전체 파이프라인을 시각화해 본다:

```
┌─────────────┐
│   42        │  소스 코드 (문자열)
└──────┬──────┘
       │ parse
       ▼
┌─────────────┐
│ IntLiteral  │  타입 있는 AST
│   value=42  │
└──────┬──────┘
       │ translateToMlir
       ▼
┌──────────────────────────────┐
│ func.func @main() -> i32 {   │  MLIR IR (high-level)
│   %c = arith.constant 42     │
│   return %c                  │
│ }                            │
└──────┬───────────────────────┘
       │ mlirPassManagerRun
       │ (convert-to-llvm)
       ▼
┌──────────────────────────────┐
│ llvm.func @main() -> i32 {   │  MLIR IR (LLVM dialect)
│   %c = llvm.mlir.constant 42 │
│   llvm.return %c             │
│ }                            │
└──────┬───────────────────────┘
       │ mlirTranslateModuleToLLVMIR
       ▼
┌──────────────────────────────┐
│ define i32 @main() {         │  LLVM IR
│   ret i32 42                 │
│ }                            │
└──────┬───────────────────────┘
       │ llc -filetype=obj
       ▼
┌─────────────┐
│ program.o   │  오브젝트 파일 (ELF/Mach-O)
└──────┬──────┘
       │ cc -o program
       ▼
┌─────────────┐
│ ./program   │  네이티브 실행 파일
└─────────────┘
```

각 단계를 하나씩 구현해 본다.

## 1단계: AST 정의와 파싱

먼저 FunLang AST의 부분집합을 정의한다. 새 파일 `Ast.fs`를 만든다:

```fsharp
namespace FunLangCompiler

/// FunLang 표현식 AST
type Expr =
    | IntLiteral of int

/// 최상위 프로그램
type Program =
    { expr: Expr }
```

극도로 단순하다. 프로그램은 하나의 표현식이고, 표현식은 정수 리터럴이다.

이제 파서를 작성한다. 실제 프로젝트에서는 LangTutorial의 파서를 재사용할 것이다. 여기서는 단순성을 위해 `int.Parse`를 사용한다:

```fsharp
/// 간단한 파서 - 문자열을 정수로 파싱
module Parser =
    open System

    let parse (source: string) : Program =
        let trimmed = source.Trim()
        match Int32.TryParse(trimmed) with
        | (true, value) ->
            { expr = IntLiteral value }
        | (false, _) ->
            failwithf "Parse error: expected integer, got '%s'" trimmed
```

**테스트:**

```fsharp
let program = Parser.parse "42"
// { expr = IntLiteral 42 }
```

## 2단계: AST를 MLIR로 변환

이제 핵심 컴파일 단계다. AST를 MLIR IR로 변환한다. 목표는 다음 IR을 생성하는 것이다:

```mlir
module {
  func.func @main() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
}
```

새 파일 `CodeGen.fs`를 만든다:

```fsharp
namespace FunLangCompiler

open System
open MlirWrapper
open MlirBindings

/// AST를 MLIR IR로 변환
module CodeGen =

    /// 표현식을 MLIR value로 컴파일
    let rec compileExpr
        (builder: OpBuilder)
        (block: MlirBlock)
        (location: Location)
        (expr: Expr)
        : MlirValue =

        match expr with
        | IntLiteral value ->
            // arith.constant operation 생성
            let i32Type = builder.I32Type()
            let constOp = builder.CreateConstant(value, i32Type, location)

            // block에 operation 추가
            MlirNative.mlirBlockAppendOwnedOperation(block, constOp)

            // 결과 value 반환
            builder.GetResult(constOp, 0)

    /// 프로그램을 MLIR module로 컴파일
    let translateToMlir (program: Program) : Module =
        let ctx = new Context()
        ctx.LoadDialect("arith")
        ctx.LoadDialect("func")

        let loc = Location.Unknown(ctx)
        let mlirMod = new Module(ctx, loc)

        let builder = OpBuilder(ctx)
        let i32Type = builder.I32Type()

        // main 함수 생성: () -> i32
        let funcType = builder.FunctionType([||], [| i32Type |])
        let funcOp = builder.CreateFunction("main", funcType, loc)

        // 함수 body에 entry block 생성
        let bodyRegion = MlirNative.mlirOperationGetRegion(funcOp, 0n)
        let entryBlock = MlirNative.mlirBlockCreate(0n, nativeint 0, nativeint 0)
        MlirNative.mlirRegionAppendOwnedBlock(bodyRegion, entryBlock)

        // 표현식 컴파일 (상수 생성)
        let resultValue = compileExpr builder entryBlock loc program.expr

        // return operation 생성
        let returnOp = builder.CreateReturn([| resultValue |], loc)
        MlirNative.mlirBlockAppendOwnedOperation(entryBlock, returnOp)

        // 함수를 module에 추가
        MlirNative.mlirBlockAppendOwnedOperation(mlirMod.Body, funcOp)

        mlirMod
```

> **설계 결정:** `compileExpr`은 재귀적이다. 현재는 IntLiteral만 처리하지만, 나중 장에서 더 많은 케이스 (BinaryOp, IfThenElse, FunctionCall 등)를 추가할 것이다.

**테스트:**

```fsharp
let program = Parser.parse "42"
let mlirMod = CodeGen.translateToMlir program
printfn "%s" (mlirMod.Print())
```

**출력:**

```mlir
module {
  func.func @main() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}
```

## 3단계: MLIR 검증

MLIR은 강력한 검증 인프라를 제공한다. 모든 operation이 올바른 형식인지 확인한다:
- 모든 block이 terminator (return, branch 등)로 끝나는가?
- SSA dominance 규칙이 존중되는가?
- 타입이 일치하는가?

`CodeGen.fs`에 검증 단계를 추가한다:

```fsharp
    /// MLIR module을 검증. 실패 시 예외 발생.
    let verify (mlirMod: Module) =
        if not (mlirMod.Verify()) then
            eprintfn "MLIR verification failed:"
            eprintfn "%s" (mlirMod.Print())
            failwith "MLIR IR is invalid"
```

**사용:**

```fsharp
let mlirMod = CodeGen.translateToMlir program
CodeGen.verify mlirMod  // 실패 시 예외 발생
```

> **마일스톤:** 이 시점에서 올바른 MLIR IR을 생성할 수 있다. 다음 단계는 LLVM으로 낮추는 것이다.

## 4단계: LLVM Dialect로 낮추기

MLIR IR은 계층적이다. 고수준 dialect (`arith`, `func`)에서 시작하여 LLVM dialect로 점진적으로 낮춘다. 이를 **progressive lowering**이라고 한다 (Chapter 01 참조).

MLIR의 pass manager를 사용하여 변환을 수행한다:

```fsharp
namespace FunLangCompiler

open MlirBindings

/// MLIR lowering pass
module Lowering =

    /// arith와 func dialect를 LLVM dialect로 낮춘다
    let lowerToLLVMDialect (mlirMod: Module) =
        let ctx = mlirMod.Context

        // Pass manager 생성
        let pm = MlirNative.mlirPassManagerCreate(ctx.Handle)

        // convert-func-to-llvm pass 추가
        MlirStringRef.WithString "convert-func-to-llvm" (fun passName ->
            let pass = MlirNative.mlirCreateConversionPass(passName)
            MlirNative.mlirPassManagerAddOwnedPass(pm, pass))

        // convert-arith-to-llvm pass 추가
        MlirStringRef.WithString "convert-arith-to-llvm" (fun passName ->
            let pass = MlirNative.mlirCreateConversionPass(passName)
            MlirNative.mlirPassManagerAddOwnedPass(pm, pass))

        // Pass 실행
        let moduleOp = MlirNative.mlirModuleGetOperation(mlirMod.Handle)
        let success = MlirNative.mlirPassManagerRunOnOp(pm, moduleOp)

        if not success then
            failwith "MLIR lowering failed"

        // Pass manager 정리
        MlirNative.mlirPassManagerDestroy(pm)
```

> **아키텍처 노트:** Pass는 MLIR의 강력한 기능이다. 각 pass는 IR을 변환한다 (최적화, 낮추기, 분석). 여러 pass를 체인으로 연결하여 복잡한 변환을 구성할 수 있다.

**변환 전 (high-level):**

```mlir
func.func @main() -> i32 {
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}
```

**변환 후 (LLVM dialect):**

```mlir
llvm.func @main() -> i32 {
  %c42 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %c42 : i32
}
```

차이를 주목한다:
- `func.func` → `llvm.func`
- `arith.constant` → `llvm.mlir.constant`
- `return` → `llvm.return`

이제 IR이 LLVM IR로 변환할 준비가 되었다.

## 5단계: LLVM IR 변환

MLIR은 LLVM IR로 변환하는 빌트인 변환기를 제공한다. `Lowering.fs`에 추가한다:

```fsharp
    open System.Runtime.InteropServices

    /// MLIR module (LLVM dialect)을 LLVM IR 문자열로 변환
    let translateToLLVMIR (mlirMod: Module) : string =
        let ctx = mlirMod.Context
        let moduleOp = MlirNative.mlirModuleGetOperation(mlirMod.Handle)

        // LLVM context 생성
        let llvmCtx = MlirNative.llvmContextCreate()

        // MLIR을 LLVM IR로 변환
        let llvmModule = MlirNative.mlirTranslateModuleToLLVMIR(
            moduleOp,
            llvmCtx)

        if llvmModule = nativeint 0 then
            failwith "Failed to translate MLIR to LLVM IR"

        // LLVM IR을 문자열로 출력
        let irString = MlirNative.llvmPrintModuleToString(llvmModule)

        // 정리
        MlirNative.llvmDisposeModule(llvmModule)
        MlirNative.llvmContextDispose(llvmCtx)

        Marshal.PtrToStringAnsi(irString)
```

> **구현 참고:** MLIR C API는 LLVM IR로 변환하는 `mlirTranslateModuleToLLVMIR`을 제공한다. 그런 다음 LLVM C API (`llvmPrintModuleToString`)를 사용하여 문자열화한다.

**출력 (LLVM IR):**

```llvm
define i32 @main() {
  ret i32 42
}
```

완벽하다! 이것은 순수한 LLVM IR이다. MLIR 개념이 전혀 없다.

## 6단계: 오브젝트 파일 생성

이제 LLVM IR을 네이티브 머신 코드로 컴파일해야 한다. LLVM의 `llc` 도구를 사용한다:

```fsharp
namespace FunLangCompiler

open System
open System.IO
open System.Diagnostics

/// 네이티브 코드 생성
module NativeCodeGen =

    /// LLVM IR을 오브젝트 파일로 컴파일 (llc 사용)
    let emitObjectFile (llvmIR: string) (outputPath: string) =
        // 임시 .ll 파일에 LLVM IR 쓰기
        let llFile = Path.GetTempFileName() + ".ll"
        File.WriteAllText(llFile, llvmIR)

        try
            // llc 실행: .ll → .o
            let psi = ProcessStartInfo()
            psi.FileName <- "llc"
            psi.Arguments <- sprintf "-filetype=obj -o %s %s" outputPath llFile
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.UseShellExecute <- false

            let proc = Process.Start(psi)
            proc.WaitForExit()

            if proc.ExitCode <> 0 then
                let stderr = proc.StandardError.ReadToEnd()
                failwithf "llc failed:\n%s" stderr

            printfn "Generated object file: %s" outputPath

        finally
            // 임시 파일 정리
            File.Delete(llFile)
```

> **도구 요구사항:** `llc`는 LLVM 도구체인의 일부다. Chapter 00에서 LLVM을 빌드했다면 `$HOME/mlir-install/bin/llc`에 있다. PATH에 있는지 확인한다.

**사용:**

```fsharp
let llvmIR = Lowering.translateToLLVMIR mlirMod
NativeCodeGen.emitObjectFile llvmIR "program.o"
```

이제 `program.o`가 있다 -- ELF 오브젝트 파일 (Linux) 또는 Mach-O (macOS).

## 7단계: 실행 파일로 링크

마지막 단계는 오브젝트 파일을 실행 파일로 링크하는 것이다. 시스템 링커 (`cc` 또는 `clang`)를 사용한다:

```fsharp
    /// 오브젝트 파일을 실행 파일로 링크 (cc 사용)
    let linkExecutable (objectPath: string) (outputPath: string) =
        let psi = ProcessStartInfo()
        psi.FileName <- "cc"  // 또는 "clang"
        psi.Arguments <- sprintf "-o %s %s" outputPath objectPath
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false

        let proc = Process.Start(psi)
        proc.WaitForExit()

        if proc.ExitCode <> 0 then
            let stderr = proc.StandardError.ReadToEnd()
            failwithf "Linking failed:\n%s" stderr

        printfn "Generated executable: %s" outputPath
```

**사용:**

```fsharp
NativeCodeGen.linkExecutable "program.o" "program"
```

완료! `./program` 실행 파일이 생성되었다.

## 완전한 컴파일러 드라이버

모든 것을 `Compiler.fs`에 하나로 모은다:

```fsharp
namespace FunLangCompiler

open System
open System.IO

/// 메인 컴파일러 드라이버
module Compiler =

    /// 소스 파일을 네이티브 실행 파일로 컴파일
    let compile (sourceFile: string) (outputFile: string) =
        printfn "=== FunLang Compiler ==="
        printfn "Source: %s" sourceFile
        printfn "Output: %s" outputFile
        printfn ""

        // 1단계: 파싱
        printfn "[1/7] Parsing..."
        let source = File.ReadAllText(sourceFile)
        let program = Parser.parse source
        printfn "  AST: %A" program

        // 2단계: MLIR로 변환
        printfn "[2/7] Translating to MLIR..."
        let mlirMod = CodeGen.translateToMlir program
        printfn "  MLIR (high-level):"
        printfn "%s" (mlirMod.Print())

        // 3단계: 검증
        printfn "[3/7] Verifying MLIR..."
        CodeGen.verify mlirMod
        printfn "  ✓ Verification passed"

        // 4단계: LLVM dialect로 낮추기
        printfn "[4/7] Lowering to LLVM dialect..."
        Lowering.lowerToLLVMDialect mlirMod
        printfn "  MLIR (LLVM dialect):"
        printfn "%s" (mlirMod.Print())

        // 5단계: LLVM IR로 변환
        printfn "[5/7] Translating to LLVM IR..."
        let llvmIR = Lowering.translateToLLVMIR mlirMod
        printfn "  LLVM IR:"
        printfn "%s" llvmIR

        // 6단계: 오브젝트 파일 생성
        printfn "[6/7] Emitting object file..."
        let objectFile = outputFile + ".o"
        NativeCodeGen.emitObjectFile llvmIR objectFile

        // 7단계: 링크
        printfn "[7/7] Linking executable..."
        NativeCodeGen.linkExecutable objectFile outputFile

        // 정리
        mlirMod.Dispose()

        printfn ""
        printfn "=== Compilation successful ==="
        printfn "Run: ./%s" outputFile
```

## 실행해 보기

테스트 프로그램을 작성한다:

```bash
echo "42" > test.fun
```

컴파일한다:

```bash
dotnet fsi Compiler.fs -- test.fun program
```

**출력:**

```
=== FunLang Compiler ===
Source: test.fun
Output: program

[1/7] Parsing...
  AST: { expr = IntLiteral 42 }
[2/7] Translating to MLIR...
  MLIR (high-level):
module {
  func.func @main() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}
[3/7] Verifying MLIR...
  ✓ Verification passed
[4/7] Lowering to LLVM dialect...
  MLIR (LLVM dialect):
module {
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %0 : i32
  }
}
[5/7] Translating to LLVM IR...
  LLVM IR:
define i32 @main() {
  ret i32 42
}
[6/7] Emitting object file...
Generated object file: program.o
[7/7] Linking executable...
Generated executable: program

=== Compilation successful ===
Run: ./program
```

실행한다:

```bash
./program
echo $?
```

**출력:**

```
42
```

> **마일스톤:** 축하한다! 실제 코드를 컴파일하고 실행했다! 🎉

## 구축한 것

이 장에서 다음을 성취했다:

1. **완전한 컴파일 파이프라인**:
   - 소스 → AST (파싱)
   - AST → MLIR IR (코드 생성)
   - MLIR 검증
   - High-level dialect → LLVM dialect (progressive lowering)
   - MLIR → LLVM IR (변환)
   - LLVM IR → 오브젝트 파일 (`llc`)
   - 오브젝트 파일 → 실행 파일 (링커)

2. **실제 컴파일러**: 단순하지만 이것은 실제 컴파일러다. 텍스트를 받아 네이티브 머신 코드를 생성한다.

3. **확장 가능한 아키텍처**: `compileExpr`은 재귀적이다. 나중 장에서 더 많은 표현식 타입을 추가할 것이다:
   - Chapter 06: 이진 연산 (`+`, `-`, `*`, `/`)
   - Chapter 07: Let 바인딩과 변수
   - Chapter 08: If/then/else
   - Chapter 09: 함수와 재귀
   - Chapter 10+: 클로저, 패턴 매칭, 리스트

## 다음 단계

**Phase 1 완료!** 다음 phase에서는:

- **Phase 2**: 산술 연산자, let 바인딩, if/else
- **Phase 3**: 함수와 재귀
- **Phase 4**: 클로저와 고차 함수
- **Phase 5**: 커스텀 MLIR dialect (Appendix 참조)
- **Phase 6**: 패턴 매칭과 데이터 구조
- **Phase 7**: 최적화와 마무리

**Appendix를 읽는 것을 잊지 마라**: 커스텀 MLIR dialect를 C++에서 정의하고 F#에서 사용하는 방법을 다룬다. 이것은 Phase 5의 기초가 된다.

---

**Phase 1의 정점에 도달했다. 실제 컴파일러를 구축했다!**
