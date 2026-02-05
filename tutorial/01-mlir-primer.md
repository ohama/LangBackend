# Chapter 01: MLIR Primer

## Introduction

In the previous chapter, you built LLVM/MLIR from source and set up the .NET SDK. You have all the tools installed. But before we write any F# code that generates MLIR, you need to understand what MLIR is and how it represents programs.

MLIR is not like traditional intermediate representations. It's not just "one IR" — it's a framework for building multiple IRs (called dialects) that can interoperate. This multi-level philosophy is what makes MLIR powerful for compiler development. Instead of forcing you to lower a high-level functional language directly to LLVM IR (which is very low-level), MLIR lets you define intermediate representations that preserve your language's semantics as long as you need them, then progressively lower through stages.

For FunLang, our compilation pipeline will look like this:

```
FunLang Typed AST
    ↓
High-Level MLIR (arith, func, scf dialects)
    ↓
Low-Level MLIR (LLVM dialect)
    ↓
LLVM IR
    ↓
Native Machine Code
```

This chapter gives you the mental model to understand MLIR IR. You'll learn the five core concepts — **dialect**, **operation**, **region**, **block**, and **SSA form** — through concrete examples. By the end, you'll be able to read MLIR textual IR and understand how FunLang programs will map to MLIR structures.

## The MLIR IR Structure

Let's start with a complete MLIR program and dissect every piece of it. Here's a simple function that adds two 32-bit integers:

```mlir
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
```

Let's break this down line by line:

- **`module { ... }`**: Every MLIR program is contained in a module. Think of a module as the top-level container for all your code, similar to a compilation unit in C or an assembly in .NET.

- **`func.func @add(...) -> i32 { ... }`**: This is an operation from the `func` dialect that defines a function named `@add`. The `@` prefix indicates a symbol (function name). The function takes two arguments and returns an `i32` (32-bit integer).

- **`%arg0: i32, %arg1: i32`**: Function parameters. Each parameter is an SSA value (starts with `%`) with a type annotation (`: i32`). These are the function's inputs.

- **`%result = arith.addi %arg0, %arg1 : i32`**: An arithmetic addition operation from the `arith` dialect. It takes two operands (`%arg0` and `%arg1`), adds them, and produces a new SSA value `%result`. The `: i32` suffix specifies the result type.

- **`return %result : i32`**: The function's return operation. It returns the value `%result` to the caller. The `: i32` type annotation ensures type safety.

Every element in MLIR has a purpose and a type. There are no implicit conversions or undefined behaviors. This strictness is what allows MLIR to perform aggressive optimizations and verification.

## Dialects

A **dialect** is a namespace that groups related operations, types, and attributes. Dialects are MLIR's extensibility mechanism — instead of having one giant IR with every possible operation, MLIR lets you define custom dialects for your domain.

### Built-in Dialects We'll Use

For the FunLang compiler, we'll primarily work with these standard dialects:

1. **`arith`** — Arithmetic operations
   - `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi` (signed integer arithmetic)
   - `arith.cmpi` (integer comparison: `<`, `>`, `==`, etc.)
   - `arith.constant` (integer and floating-point constants)

2. **`func`** — Function definitions and calls
   - `func.func` (define a function)
   - `func.call` (call a function)
   - `func.return` (return from a function)

3. **`scf`** — Structured control flow
   - `scf.if` (conditional execution)
   - `scf.for` (counted loops)
   - `scf.while` (conditional loops)

4. **`llvm`** — LLVM dialect (target for lowering)
   - `llvm.func`, `llvm.call`, `llvm.add`, etc.
   - This dialect is a 1:1 mapping to LLVM IR constructs

### Custom Dialects

Later in this tutorial series (Chapter 10-11), you'll define a **FunLang dialect** with operations like:

- `funlang.closure` (create a closure)
- `funlang.apply` (apply a closure to arguments)
- `funlang.match` (pattern matching)

Custom dialects let you preserve high-level semantics during compilation. Instead of immediately translating a FunLang closure into low-level struct allocations and function pointers, you represent it as a high-level `funlang.closure` operation. This makes optimizations easier to write and understand.

### Dialect Naming Convention

Operations are always prefixed with their dialect name, separated by a dot:

```mlir
arith.addi   // "addi" operation from "arith" dialect
func.call    // "call" operation from "func" dialect
llvm.load    // "load" operation from "llvm" dialect
```

This prevents name conflicts. The `arith` dialect's `addi` is distinct from a hypothetical `mydialect.addi`.

## Operations

An **operation** is the fundamental unit of MLIR IR. Everything in MLIR is represented as operations: function definitions, arithmetic instructions, control flow, even types and attributes are attached to operations.

### Anatomy of an Operation

An operation in textual form has this structure:

```mlir
%results = dialect.opname(%operands) {attributes} : (types) -> result_type
```

Let's see each component in the addition example:

```mlir
%result = arith.addi %arg0, %arg1 : i32
```

- **`%result`**: The SSA value produced by this operation. This value can be used by subsequent operations. The `%` prefix distinguishes SSA values from symbols (`@function_name`).

- **`arith.addi`**: The operation name (dialect + opname).

- **`%arg0, %arg1`**: Operands (inputs to the operation). These are SSA values defined earlier (in this case, function arguments).

- **`: i32`**: Type constraint. This operation works on 32-bit integers.

Not all operations produce results. For example, `return` is an operation that terminates the function but doesn't produce a value for later use:

```mlir
return %result : i32
```

### Operations with Multiple Results

Some operations produce multiple values. For example, a division operation that returns both quotient and remainder:

```mlir
%quot, %rem = arith.divrem %dividend, %divisor : i32
```

Now `%quot` and `%rem` are both SSA values available for use.

### Operations with Attributes

Attributes provide compile-time constant metadata. For example, an integer constant:

```mlir
%zero = arith.constant 0 : i32
```

The `0` is an attribute (the constant value), and `i32` is the type. Attributes are not runtime values — they're baked into the IR at compile time.

## Regions and Blocks

MLIR operations can contain **regions**, and regions contain **blocks**. This is how MLIR represents nested scopes and control flow.

### Regions

A **region** is a list of blocks. Function bodies are regions. Control flow operations like `scf.if` have regions for the "then" and "else" branches.

Here's a function (which has one region containing one block):

```mlir
func.func @example() -> i32 {
  %one = arith.constant 1 : i32
  return %one : i32
}
```

The braces `{ ... }` delimit the function's region. Inside the region is a single block with two operations: a constant and a return.

### Blocks

A **block** is a sequence of operations that executes linearly. Every block must end with a **terminator** operation — an operation that transfers control elsewhere (return, branch, etc.). You cannot "fall through" a block.

Blocks become essential when you have control flow. Here's a function with two blocks:

```mlir
func.func @conditional(%cond: i1, %a: i32, %b: i32) -> i32 {
  cf.cond_br %cond, ^then_block, ^else_block

^then_block:
  return %a : i32

^else_block:
  return %b : i32
}
```

Let's dissect this:

- **`cf.cond_br %cond, ^then_block, ^else_block`**: Conditional branch operation (from the `cf` control-flow dialect). If `%cond` is true, jump to `^then_block`; otherwise, jump to `^else_block`. This is the terminator of the entry block.

- **`^then_block:`**: A block label. The `^` prefix indicates a block. Block names are local to the function.

- **`return %a : i32`**: The terminator of `^then_block`. It returns `%a` to the caller.

- **`^else_block:`**: Another block label.

- **`return %b : i32`**: The terminator of `^else_block`. It returns `%b`.

### Block Arguments (The MLIR Way to Handle Phi Nodes)

MLIR uses **block arguments** instead of LLVM's phi nodes. In LLVM IR, you use phi nodes to merge values from multiple predecessors. In MLIR, you pass values as arguments when branching to a block.

Here's an example that merges two values:

```mlir
func.func @merge_example(%cond: i1, %a: i32, %b: i32) -> i32 {
  cf.cond_br %cond, ^merge(%a : i32), ^merge(%b : i32)

^merge(%result: i32):
  return %result : i32
}
```

What's happening:

- **`cf.cond_br %cond, ^merge(%a : i32), ^merge(%b : i32)`**: Branch to `^merge` block, passing `%a` if condition is true, or `%b` if condition is false.

- **`^merge(%result: i32):`**: The `^merge` block declares it expects one argument of type `i32`. Whichever branch is taken, the passed value becomes `%result` inside this block.

This is cleaner than LLVM's phi nodes because the data flow is explicit at the branch point, not reconstructed after the fact.

## SSA Form (Static Single Assignment)

MLIR is in **SSA form**, meaning every value is defined exactly once and never mutates. Once you define `%x`, you cannot reassign it. This property simplifies optimization because you never have to track "which version of this variable am I looking at?"

### SSA in Action

Consider this FunLang code:

```fsharp
let x = 5
let y = x + 3
let z = y * 2
z
```

In MLIR SSA form, each let binding becomes a new SSA value:

```mlir
func.func @example() -> i32 {
  %x = arith.constant 5 : i32
  %three = arith.constant 3 : i32
  %y = arith.addi %x, %three : i32
  %two = arith.constant 2 : i32
  %z = arith.muli %y, %two : i32
  return %z : i32
}
```

Notice:

- Each `let` binding becomes a new SSA value (`%x`, `%y`, `%z`).
- Constants are operations (`arith.constant`) that produce values.
- No value is ever reassigned.

### SSA and Mutability

FunLang is immutable, so SSA maps naturally. But what about imperative code with mutation? Consider:

```c
int x = 1;
x = x + 1;
return x;
```

In SSA, you cannot mutate `x`. Instead, you create a new version:

```mlir
%x0 = arith.constant 1 : i32
%one = arith.constant 1 : i32
%x1 = arith.addi %x0, %one : i32
return %x1 : i32
```

Each "mutation" creates a new SSA value (`%x0`, `%x1`, etc.). This transformation is called **SSA conversion** and is handled automatically by compilers for imperative languages.

Since FunLang is functional, you won't need this — every `let` binding already introduces a new name.

### Key Insight: SSA Enables Optimizations

SSA form makes many compiler optimizations trivial. For example:

- **Dead code elimination:** If an SSA value is defined but never used, delete the operation that defines it.
- **Constant propagation:** If `%x` is defined as `arith.constant 5`, replace all uses of `%x` with `5`.
- **Common subexpression elimination:** If two operations compute the same value, reuse one and delete the other.

All these optimizations rely on the guarantee that values never change after definition.

## Types in MLIR

MLIR is strongly typed. Every SSA value, operation, and function has a type. The type system is extensible (dialects can define custom types), but here are the built-in types you'll use:

### Integer Types

- `i1` — 1-bit integer (boolean)
- `i32` — 32-bit signed integer
- `i64` — 64-bit signed integer
- `i8`, `i16`, `i128`, etc. — arbitrary bit-width integers

### Floating-Point Types

- `f32` — 32-bit IEEE 754 float
- `f64` — 64-bit IEEE 754 double

### Index Type

- `index` — Platform-dependent integer for array indexing (typically 32-bit or 64-bit depending on target architecture)

### Memory Types

- `memref<4xi32>` — A reference to a 4-element array of `i32` values in memory
- `memref<*xf64>` — An unranked (dynamic) memory reference to `f64` values

### Function Types

- `(i32, i32) -> i32` — A function taking two `i32` arguments and returning `i32`

### FunLang Type Mapping

Here's how FunLang types will map to MLIR types:

| FunLang Type | MLIR Type | Notes |
|--------------|-----------|-------|
| `Int` | `i64` | FunLang integers are arbitrary precision in the interpreter, but we'll compile to 64-bit |
| `Bool` | `i1` | True = 1, False = 0 |
| `String` | `!llvm.ptr` (LLVM dialect pointer) | Strings are heap-allocated, null-terminated C strings |
| `Float` | `f64` | Double precision floating-point |
| `List<'a>` | `!llvm.ptr` | Lists are heap-allocated linked structures |
| `Tuple<'a, 'b, ...>` | `!llvm.struct<...>` | Tuples compile to LLVM structs |

The `!` prefix indicates a dialect-defined type (e.g., `!llvm.ptr` is a pointer type from the LLVM dialect).

## Progressive Lowering

The power of MLIR is **progressive lowering**: transforming high-level operations into lower-level operations in multiple stages, rather than one giant leap.

### The FunLang Compilation Pipeline

Here's the pipeline we'll build in this tutorial:

```
Stage 1: AST → High-Level MLIR
    FunLang AST (from type checker)
    ↓
    Translate to MLIR using arith, func, scf dialects
    Example: `let x = 1 + 2` becomes `%x = arith.addi ...`

Stage 2: High-Level MLIR → LLVM Dialect
    Operations like `arith.addi` lower to `llvm.add`
    Structured control flow (`scf.if`) lowers to basic blocks and branches

Stage 3: LLVM Dialect → LLVM IR
    MLIR's LLVM dialect translates to textual LLVM IR

Stage 4: LLVM IR → Native Code
    LLVM backend (llc) compiles to machine code for target platform
```

Each lowering step is a **pass** — a transformation that rewrites IR. MLIR provides infrastructure for defining passes, pattern-based rewrites, and verification after each stage.

### Why Progressive Lowering Matters

Consider compiling FunLang's pattern matching. If we had to lower directly to LLVM IR, we'd immediately expand it into a complex decision tree with basic blocks, phi nodes, and memory loads. But with progressive lowering:

1. **High-level:** Represent pattern matching as a `funlang.match` operation that preserves the structure.
2. **Mid-level:** Lower `funlang.match` to `scf.if` and `scf.while` (structured control flow).
3. **Low-level:** Lower `scf.if` to LLVM basic blocks and branches.

At each stage, you can perform optimizations specific to that level of abstraction. Pattern match optimizations (removing redundant checks) happen at the high level. LLVM-level optimizations (register allocation, instruction scheduling) happen at the low level.

## Putting It All Together

Let's look at a more realistic MLIR example that uses multiple concepts:

```mlir
module {
  func.func @factorial(%n: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %n, %c0 : i64
    cf.cond_br %is_zero, ^base_case, ^recursive_case

  ^base_case:
    return %c1 : i64

  ^recursive_case:
    %n_minus_1 = arith.subi %n, %c1 : i64
    %rec_result = func.call @factorial(%n_minus_1) : (i64) -> i64
    %result = arith.muli %n, %rec_result : i64
    return %result : i64
  }
}
```

Let's trace through this:

1. **`func.func @factorial(%n: i64) -> i64`**: Define a function named `@factorial` that takes one 64-bit integer and returns a 64-bit integer.

2. **`%c0 = arith.constant 0 : i64`**: Create constant `0`.

3. **`%c1 = arith.constant 1 : i64`**: Create constant `1`.

4. **`%is_zero = arith.cmpi eq, %n, %c0 : i64`**: Compare `%n` with `0` for equality. Result is `i1` (boolean).

5. **`cf.cond_br %is_zero, ^base_case, ^recursive_case`**: Branch to `^base_case` if true, else `^recursive_case`.

6. **`^base_case:`**: If n == 0, return 1.

7. **`^recursive_case:`**: If n > 0, compute `n * factorial(n - 1)`:
   - `%n_minus_1 = arith.subi %n, %c1`: Compute `n - 1`.
   - `%rec_result = func.call @factorial(%n_minus_1)`: Recursive call.
   - `%result = arith.muli %n, %rec_result`: Multiply `n` by the recursive result.
   - `return %result`: Return the result.

This example demonstrates:

- **SSA form:** Every value (`%c0`, `%n_minus_1`, etc.) is defined once.
- **Operations:** Constants, comparisons, arithmetic, function calls.
- **Regions and blocks:** The function body is a region with three blocks (entry, `^base_case`, `^recursive_case`).
- **Terminators:** Every block ends with a terminator (`cf.cond_br` or `return`).
- **Dialects:** We use `arith`, `func`, and `cf` dialects.

## What You've Learned

You now understand the five core MLIR concepts:

1. **Dialect:** A namespace for operations, types, and attributes (e.g., `arith`, `func`, `llvm`).
2. **Operation:** The fundamental unit of MLIR IR (e.g., `arith.addi`, `func.call`).
3. **Region:** A list of blocks (e.g., a function body).
4. **Block:** A sequence of operations ending with a terminator (e.g., a basic block in control flow).
5. **SSA form:** Every value defined exactly once and immutable.

You've seen how these concepts work together in concrete examples (arithmetic, control flow, recursion). You understand progressive lowering — the philosophy of transforming IR in stages rather than one giant jump.

## Next Steps

In the next chapter, we'll write our first F# program that creates MLIR IR. You'll use P/Invoke to call MLIR's C API and generate the "Hello, World" of compilers: a program that returns a constant integer.

Continue to **Chapter 02: Hello MLIR from F#** (to be written).

## Further Reading

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) — Official specification of MLIR's textual format, dialects, and semantics.
- [Understanding MLIR IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) — Deep dive into operations, regions, and blocks.
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) — A complete tutorial building a compiler for the "Toy" language using MLIR.
- [Dialects Documentation](https://mlir.llvm.org/docs/Dialects/) — Reference for built-in dialects (arith, func, scf, llvm, etc.).
