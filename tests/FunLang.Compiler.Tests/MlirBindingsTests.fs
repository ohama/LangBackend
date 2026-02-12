module MlirBindingsTests

open System
open System.Runtime.InteropServices
open Expecto
open FunLang.Compiler

[<Tests>]
let contextTests =
    testList "Context" [
        test "can create and dispose context" {
            use ctx = new Context()
            Expect.isTrue (ctx.Handle.Handle <> nativeint 0) "Context handle should be non-null"
        }

        test "can load standard dialects" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            // If we get here without exception, dialects loaded successfully
        }
    ]

[<Tests>]
let moduleTests =
    testList "Module" [
        test "can create empty module" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let loc = Location.Unknown ctx
            use mlirMod = new Module(ctx, loc)
            Expect.isTrue (mlirMod.Handle.Handle <> nativeint 0) "Module handle should be non-null"
        }

        test "can print empty module" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let loc = Location.Unknown ctx
            use mlirMod = new Module(ctx, loc)
            let ir = mlirMod.Print()
            Expect.stringContains ir "module" "IR should contain 'module'"
        }
    ]

[<Tests>]
let opBuilderTests =
    testList "OpBuilder" [
        test "can get i32 type" {
            use ctx = new Context()
            let builder = OpBuilder(ctx)
            let i32Type = builder.I32Type()
            Expect.isTrue (i32Type.Handle <> nativeint 0) "Type handle should be non-null"
        }

        test "can create integer attribute" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let builder = OpBuilder(ctx)
            let i32Type = builder.I32Type()
            let attr = builder.IntegerAttr(42L, i32Type)
            Expect.isTrue (attr.Handle <> nativeint 0) "Attribute handle should be non-null"
        }

        test "can create function type" {
            use ctx = new Context()
            let builder = OpBuilder(ctx)
            let i32Type = builder.I32Type()
            let funcType = builder.FunctionType([| i32Type |], [| i32Type |])
            Expect.isTrue (funcType.Handle <> nativeint 0) "Function type handle should be non-null"
        }
    ]

[<Tests>]
let integrationTests =
    testList "Integration" [
        test "can create arith.constant operation" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let loc = Location.Unknown ctx
            let builder = OpBuilder(ctx)

            // Create constant: arith.constant 42 : i32
            let i32Type = builder.I32Type()
            let valueAttr = builder.IntegerAttr(42L, i32Type)
            let namedAttr = builder.NamedAttr("value", valueAttr)

            let constOp = builder.CreateOperation(
                "arith.constant",
                loc,
                [| i32Type |],     // result types
                [||],              // operands
                [| namedAttr |],   // attributes
                [||])              // regions

            Expect.isTrue (constOp.Handle <> nativeint 0) "Operation handle should be non-null"

            let result = builder.GetResult(constOp, 0)
            Expect.isTrue (result.Handle <> nativeint 0) "Result value should be non-null"
        }

        test "can build and print simple module with constant" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let loc = Location.Unknown ctx
            use mlirMod = new Module(ctx, loc)
            let builder = OpBuilder(ctx)

            // Create arith.constant
            let i32Type = builder.I32Type()
            let valueAttr = builder.IntegerAttr(42L, i32Type)
            let namedAttr = builder.NamedAttr("value", valueAttr)

            let constOp = builder.CreateOperation(
                "arith.constant",
                loc,
                [| i32Type |],
                [||],
                [| namedAttr |],
                [||])

            // Add to module body
            builder.AppendOperationToBlock(mlirMod.Body, constOp)

            let ir = mlirMod.Print()
            printfn "Generated IR:\n%s" ir

            Expect.stringContains ir "arith.constant" "IR should contain arith.constant"
            Expect.stringContains ir "42" "IR should contain value 42"
        }
    ]

[<Tests>]
let codeGenTests =
    testList "CodeGen" [
        test "compile and run simple number" {
            let result = CodeGen.compileAndRun "42"
            Expect.equal result 42 "42 should return 42"
        }

        test "compile and run addition" {
            let result = CodeGen.compileAndRun "1 + 2"
            Expect.equal result 3 "1 + 2 should return 3"
        }

        test "compile and run complex arithmetic" {
            let result = CodeGen.compileAndRun "(1 + 2) * 3 - 4"
            Expect.equal result 5 "(1 + 2) * 3 - 4 should return 5"
        }

        test "compile and run with negative" {
            let result = CodeGen.compileAndRun "10 - 3"
            Expect.equal result 7 "10 - 3 should return 7"
        }

        test "compile and run division" {
            let result = CodeGen.compileAndRun "10 / 2"
            Expect.equal result 5 "10 / 2 should return 5"
        }
    ]

[<Tests>]
let comparisonTests =
    testList "Comparisons" [
        test "can generate IR for equality comparison" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "1 = 1" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_eq" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
        }

        test "can generate IR for not-equal comparison" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "1 <> 2" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_ne" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
        }

        test "can generate IR for less-than comparison" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "1 < 2" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_lt" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
        }

        test "can generate IR for less-equal comparison" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "1 <= 2" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_le" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
        }

        test "can generate IR for greater-than comparison" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "2 > 1" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_gt" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
        }

        test "can generate IR for greater-equal comparison" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "2 >= 1" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_ge" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
        }
    ]

[<Tests>]
let booleanTests =
    testList "Booleans" [
        test "can generate IR for boolean literal true" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "true" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_true" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.constant" "Should contain arith.constant"
            Expect.stringContains ir "i1" "Should have i1 type"
        }

        test "can generate IR for boolean literal false" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "false" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_false" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.constant" "Should contain arith.constant"
            Expect.stringContains ir "i1" "Should have i1 type"
        }

        test "can generate IR for logical AND" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "true && false" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_and" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.andi" "Should contain arith.andi"
        }

        test "can generate IR for logical OR" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "true || false" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_or" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.ori" "Should contain arith.ori"
        }

        test "can generate IR for combined comparison and logical" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            let expr = CodeGen.parse "(1 < 2) && (3 > 2)" "<test>"
            use mlirMod = CodeGen.compileToFunction ctx "test_combined" expr
            let ir = mlirMod.Print()
            Expect.stringContains ir "arith.cmpi" "Should contain arith.cmpi"
            Expect.stringContains ir "arith.andi" "Should contain arith.andi"
        }
    ]

[<Tests>]
let e2eTests =
    testList "E2E" [
        test "can build function, lower to LLVM, and JIT execute" {
            use ctx = new Context()
            ctx.LoadStandardDialects()
            MlirNative.mlirRegisterAllLLVMTranslations(ctx.Handle)

            let loc = Location.Unknown ctx
            use mlirMod = new Module(ctx, loc)
            let builder = OpBuilder(ctx)

            // Build: func.func @answer() -> i32 { return 42 }
            let i32Type = builder.I32Type()
            let funcType = builder.FunctionType([||], [| i32Type |])

            // Create function body region with entry block
            let region = builder.CreateRegion()
            let entryBlock = builder.CreateBlock([||], loc)
            builder.AppendBlockToRegion(region, entryBlock)

            // arith.constant 42 : i32
            let valueAttr = builder.IntegerAttr(42L, i32Type)
            let constOp = builder.CreateOperation(
                "arith.constant", loc,
                [| i32Type |], [||],
                [| builder.NamedAttr("value", valueAttr) |], [||])
            builder.AppendOperationToBlock(entryBlock, constOp)
            let constResult = builder.GetResult(constOp, 0)

            // func.return %0 : i32
            let returnOp = builder.CreateOperation(
                "func.return", loc,
                [||], [| constResult |], [||], [||])
            builder.AppendOperationToBlock(entryBlock, returnOp)

            // Create func.func @answer with C interface attribute for JIT
            let unitAttr = MlirNative.mlirUnitAttrGet(ctx.Handle)
            let funcOp = builder.CreateOperation(
                "func.func", loc,
                [||], [||],
                [| builder.NamedAttr("sym_name", builder.StringAttr("answer"))
                   builder.NamedAttr("function_type", builder.TypeAttr(funcType))
                   builder.NamedAttr("llvm.emit_c_interface", unitAttr) |],
                [| region |])
            builder.AppendOperationToBlock(mlirMod.Body, funcOp)

            let irBefore = mlirMod.Print()
            printfn "=== Before lowering ===\n%s" irBefore
            Expect.stringContains irBefore "func.func @answer" "Should have func.func"

            // Lower to LLVM - use proper MLIR pipeline syntax
            use pm = new PassManager(ctx)
            pm.AddPipeline("builtin.module(convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)")
            let success = pm.Run(mlirMod)
            Expect.isTrue success "Pass pipeline should succeed"

            let irAfter = mlirMod.Print()
            printfn "=== After lowering ===\n%s" irAfter
            Expect.stringContains irAfter "llvm.func @answer" "Should have llvm.func"

            // JIT execute
            use ee = new ExecutionEngine(mlirMod, 0)
            let mutable result: int32 = 0
            let resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned)
            try
                let resultPtr = resultHandle.AddrOfPinnedObject()
                let argsArray = [| resultPtr |]
                let argsHandle = GCHandle.Alloc(argsArray, GCHandleType.Pinned)
                try
                    let invokeResult = ee.InvokePacked("answer", argsHandle.AddrOfPinnedObject())
                    result <- Marshal.ReadInt32(resultPtr)
                    printfn "=== Execution result: %d ===" result
                    Expect.equal result 42 "Function should return 42"
                finally
                    argsHandle.Free()
            finally
                resultHandle.Free()
        }
    ]
