namespace FunLang.Compiler

open System
open FSharp.Text.Lexing
open Ast

/// Code generator: FunLang AST -> MLIR
module CodeGen =

    /// Parse FunLang source code into AST
    let parse (source: string) (filename: string) : Expr =
        let lexbuf = LexBuffer<char>.FromString source
        Lexer.setInitialPos lexbuf filename
        Parser.start Lexer.tokenize lexbuf

    /// Compilation context
    type CompileContext = {
        Context: Context
        Builder: OpBuilder
        Location: Location
        Block: MlirBlock  // Current block to append operations to
    }

    /// Create operation, append to block, return result value
    let private emitOp (ctx: CompileContext) name resultTypes operands attrs regions =
        let op = ctx.Builder.CreateOperation(name, ctx.Location, resultTypes, operands, attrs, regions)
        ctx.Builder.AppendOperationToBlock(ctx.Block, op)
        op

    /// Compile a FunLang expression to MLIR, returns the result value
    let rec compileExpr (ctx: CompileContext) (expr: Expr) : MlirValue =
        let builder = ctx.Builder
        let i32Type = builder.I32Type()

        match expr with
        | Number(n, _) ->
            let valueAttr = builder.IntegerAttr(int64 n, i32Type)
            let op = emitOp ctx "arith.constant" [| i32Type |] [||] [| builder.NamedAttr("value", valueAttr) |] [||]
            builder.GetResult(op, 0)

        | Add(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let op = emitOp ctx "arith.addi" [| i32Type |] [| leftVal; rightVal |] [||] [||]
            builder.GetResult(op, 0)

        | Subtract(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let op = emitOp ctx "arith.subi" [| i32Type |] [| leftVal; rightVal |] [||] [||]
            builder.GetResult(op, 0)

        | Multiply(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let op = emitOp ctx "arith.muli" [| i32Type |] [| leftVal; rightVal |] [||] [||]
            builder.GetResult(op, 0)

        | Divide(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let op = emitOp ctx "arith.divsi" [| i32Type |] [| leftVal; rightVal |] [||] [||]
            builder.GetResult(op, 0)

        | Negate(inner, _) ->
            // -x = 0 - x
            let zeroAttr = builder.IntegerAttr(0L, i32Type)
            let zeroOp = emitOp ctx "arith.constant" [| i32Type |] [||] [| builder.NamedAttr("value", zeroAttr) |] [||]
            let zeroVal = builder.GetResult(zeroOp, 0)
            let innerVal = compileExpr ctx inner
            let op = emitOp ctx "arith.subi" [| i32Type |] [| zeroVal; innerVal |] [||] [||]
            builder.GetResult(op, 0)

        // Comparison operators - compile to arith.cmpi with predicate attribute
        // Predicates: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5
        | Equal(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i64Type = builder.I64Type()
            let predicateAttr = builder.IntegerAttr(0L, i64Type)  // eq = 0
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.cmpi" [| i1Type |]
                        [| leftVal; rightVal |]
                        [| builder.NamedAttr("predicate", predicateAttr) |]
                        [||]
            builder.GetResult(op, 0)

        | NotEqual(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i64Type = builder.I64Type()
            let predicateAttr = builder.IntegerAttr(1L, i64Type)  // ne = 1
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.cmpi" [| i1Type |]
                        [| leftVal; rightVal |]
                        [| builder.NamedAttr("predicate", predicateAttr) |]
                        [||]
            builder.GetResult(op, 0)

        | LessThan(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i64Type = builder.I64Type()
            let predicateAttr = builder.IntegerAttr(2L, i64Type)  // slt = 2
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.cmpi" [| i1Type |]
                        [| leftVal; rightVal |]
                        [| builder.NamedAttr("predicate", predicateAttr) |]
                        [||]
            builder.GetResult(op, 0)

        | LessEqual(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i64Type = builder.I64Type()
            let predicateAttr = builder.IntegerAttr(3L, i64Type)  // sle = 3
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.cmpi" [| i1Type |]
                        [| leftVal; rightVal |]
                        [| builder.NamedAttr("predicate", predicateAttr) |]
                        [||]
            builder.GetResult(op, 0)

        | GreaterThan(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i64Type = builder.I64Type()
            let predicateAttr = builder.IntegerAttr(4L, i64Type)  // sgt = 4
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.cmpi" [| i1Type |]
                        [| leftVal; rightVal |]
                        [| builder.NamedAttr("predicate", predicateAttr) |]
                        [||]
            builder.GetResult(op, 0)

        | GreaterEqual(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i64Type = builder.I64Type()
            let predicateAttr = builder.IntegerAttr(5L, i64Type)  // sge = 5
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.cmpi" [| i1Type |]
                        [| leftVal; rightVal |]
                        [| builder.NamedAttr("predicate", predicateAttr) |]
                        [||]
            builder.GetResult(op, 0)

        // Boolean literals and logical operators
        | Bool(b, _) ->
            let i1Type = builder.I1Type()
            let value = if b then 1L else 0L
            let valueAttr = builder.IntegerAttr(value, i1Type)
            let op = emitOp ctx "arith.constant" [| i1Type |] [||]
                        [| builder.NamedAttr("value", valueAttr) |] [||]
            builder.GetResult(op, 0)

        | And(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.andi" [| i1Type |] [| leftVal; rightVal |] [||] [||]
            builder.GetResult(op, 0)

        | Or(left, right, _) ->
            let leftVal = compileExpr ctx left
            let rightVal = compileExpr ctx right
            let i1Type = builder.I1Type()
            let op = emitOp ctx "arith.ori" [| i1Type |] [| leftVal; rightVal |] [||] [||]
            builder.GetResult(op, 0)

        | _ ->
            failwithf "CodeGen: unsupported expression type"

    /// Compile a FunLang expression into a function that returns i32
    let compileToFunction (ctx: Context) (funcName: string) (expr: Expr) : Module =
        let loc = Location.Unknown ctx
        let mlirMod = new Module(ctx, loc)
        let builder = OpBuilder(ctx)

        let i32Type = builder.I32Type()
        let funcType = builder.FunctionType([||], [| i32Type |])

        // Create function body
        let region = builder.CreateRegion()
        let entryBlock = builder.CreateBlock([||], loc)
        builder.AppendBlockToRegion(region, entryBlock)

        // Compile expression into the entry block
        let compileCtx = {
            Context = ctx
            Builder = builder
            Location = loc
            Block = entryBlock
        }
        let resultVal = compileExpr compileCtx expr

        // Return the result
        let returnOp = builder.CreateOperation(
            "func.return", loc,
            [||], [| resultVal |], [||], [||])
        builder.AppendOperationToBlock(entryBlock, returnOp)

        // Create func.func with C interface for JIT
        let unitAttr = MlirNative.mlirUnitAttrGet(ctx.Handle)
        let funcOp = builder.CreateOperation(
            "func.func", loc,
            [||], [||],
            [| builder.NamedAttr("sym_name", builder.StringAttr(funcName))
               builder.NamedAttr("function_type", builder.TypeAttr(funcType))
               builder.NamedAttr("llvm.emit_c_interface", unitAttr) |],
            [| region |])
        builder.AppendOperationToBlock(mlirMod.Body, funcOp)

        mlirMod

    /// Compile, lower to LLVM, and JIT execute an expression
    let compileAndRun (source: string) : int32 =
        use ctx = new Context()
        ctx.LoadStandardDialects()
        MlirNative.mlirRegisterAllLLVMTranslations(ctx.Handle)

        let expr = parse source "<string>"
        use mlirMod = compileToFunction ctx "main" expr

        // Lower to LLVM
        use pm = new PassManager(ctx)
        pm.AddPipeline("builtin.module(convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)")
        if not (pm.Run(mlirMod)) then
            failwith "Pass pipeline failed"

        // JIT execute
        use ee = new ExecutionEngine(mlirMod, 0)
        let mutable result: int32 = 0
        let resultHandle = System.Runtime.InteropServices.GCHandle.Alloc(result, System.Runtime.InteropServices.GCHandleType.Pinned)
        try
            let resultPtr = resultHandle.AddrOfPinnedObject()
            let argsArray = [| resultPtr |]
            let argsHandle = System.Runtime.InteropServices.GCHandle.Alloc(argsArray, System.Runtime.InteropServices.GCHandleType.Pinned)
            try
                ee.InvokePacked("main", argsHandle.AddrOfPinnedObject()) |> ignore
                System.Runtime.InteropServices.Marshal.ReadInt32(resultPtr)
            finally
                argsHandle.Free()
        finally
            resultHandle.Free()
