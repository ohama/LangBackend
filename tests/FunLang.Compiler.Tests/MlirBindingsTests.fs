module MlirBindingsTests

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
