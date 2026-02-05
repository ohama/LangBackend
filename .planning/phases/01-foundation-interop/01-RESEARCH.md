# Phase 1: Foundation & Interop - Research

**Researched:** 2026-02-05
**Domain:** MLIR C API Interop, F# P/Invoke Bindings, Tutorial Foundation
**Confidence:** MEDIUM-HIGH

## Summary

Phase 1 establishes the foundation for a tutorial teaching MLIR-based compiler backends from F#. This phase must solve three critical problems: (1) building LLVM/MLIR with C API enabled, (2) creating F# P/Invoke bindings to MLIR C API, and (3) teaching readers enough MLIR concepts to understand the compilation pipeline.

The **critical finding** from this research is that MLIR's C API has **limited support for custom dialect registration**. While the C API covers core IR operations (context, module, types, operations, blocks, regions), custom dialect definition requires either C++ code with manual C wrapper functions or acceptance of using only built-in dialects initially. This confirms the blocker identified in PITFALLS.md.

For a documentation-only tutorial, the recommended approach is:
1. Provide pre-built native libraries (MLIR-C + custom FunLang dialect wrapper)
2. Tutorial Chapter 1 focuses on P/Invoke bindings to standard MLIR dialects
3. Appendix chapter explains custom dialect implementation in C++
4. Early chapters use only `arith`, `func`, and `llvm` dialects before custom dialect

**Primary recommendation:** Build MLIR with `-DMLIR_BUILD_MLIR_C_DYLIB=ON`, create minimal C++ wrapper for custom dialect registration, expose via C API, consume from F# via P/Invoke with SafeHandle wrappers for lifetime management.

## Standard Stack

The established technologies for this domain:

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **LLVM/MLIR** | 19.1.x | Multi-level intermediate representation framework | Industry standard for modern compiler infrastructure; stable C API since LLVM 17; LLVM 19 is latest stable |
| **.NET SDK** | 8.0 or 9.0 | F# runtime and P/Invoke support | .NET 8 is LTS (supported until Nov 2026), .NET 9 current (Nov 2024 release) with improved native interop |
| **F#** | 8.0 or 9.0 | Tutorial implementation language | Matches LangTutorial frontend; functional style suits compiler domain |
| **CMake** | 3.20+ | MLIR build system | Required by LLVM project; industry standard for C++ builds |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Ninja** | 1.11+ | Fast build executor | LLVM build (30-60 min with Ninja vs 2+ hours with Make) |
| **Clang/GCC** | 15+ | C++ compiler for MLIR | Clang recommended for LLVM compatibility, GCC works |
| **Python** | 3.8+ | MLIR build scripts | Required for TableGen, build system |
| **libffi** (optional) | 3.4+ | If using LibraryImport | .NET 7+ P/Invoke source generator (modern alternative to DllImport) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **MLIR C API + P/Invoke** | Direct C++ via C++/CLI | C++/CLI is Windows-only, deprecated on .NET Core; not cross-platform |
| **MLIR C API + P/Invoke** | LLVM-only (skip MLIR) | LLVM IR too low-level; loses structured dialects, no custom ops; harder to teach functional concepts |
| **DllImport** | LibraryImport (.NET 7+) | LibraryImport is modern source generator approach, but DllImport more familiar; both work |
| **Manual P/Invoke wrappers** | Use LLVMSharp bindings | LLVMSharp covers LLVM-C API but NOT MLIR-C API; would still need custom MLIR bindings |

**Build command:**
```bash
# Clone LLVM with MLIR (LLVM 19.1.x stable)
git clone --depth 1 --branch release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project

# Configure MLIR with C API enabled
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install

# Build (30-60 minutes)
cmake --build build --target install

# Result: libMLIR-C.so (Linux), libMLIR-C.dylib (macOS), MLIR-C.dll (Windows)
```

## Architecture Patterns

### Recommended Tutorial Structure

```
tutorial/
├── 00-prerequisites.md          # LLVM build, .NET setup
├── 01-mlir-primer.md           # Concepts: dialect, op, region, block, SSA
├── 02-hello-mlir.md            # First F# program emitting MLIR
├── 03-pinvoke-bindings.md      # F# bindings to MLIR-C API
├── 04-wrapper-layer.md         # Safe F# types over P/Invoke
├── 05-arithmetic-compiler.md   # Compile integer literals to native
└── appendix-custom-dialect.md  # C++ dialect definition (advanced)

code-examples/
├── 01-hello-mlir/
│   ├── HelloMlir.fsx           # Standalone runnable example
│   └── expected-output.mlir
├── 02-pinvoke/
│   ├── MlirBindings.fs         # Low-level P/Invoke
│   └── Tests.fsx
└── 03-arithmetic/
    ├── Compiler.fs
    └── test-programs/
```

### Pattern 1: MLIR C API P/Invoke Binding

**What:** F# extern declarations for MLIR C API functions with proper marshalling.

**When to use:** All MLIR API calls from F#; this is the FFI boundary layer.

**Example:**
```fsharp
// MlirBindings.fs - Low-level P/Invoke layer
module FunLang.Mlir.Bindings

open System
open System.Runtime.InteropServices

// Opaque handle types (C API uses pointers to opaque structs)
[<Struct>]
type MlirContext =
    val mutable ptr: nativeint

[<Struct>]
type MlirModule =
    val mutable ptr: nativeint

[<Struct>]
type MlirType =
    val mutable ptr: nativeint

[<Struct>]
type MlirLocation =
    val mutable ptr: nativeint

// Native library name (platform-specific)
module private NativeLib =
    let name =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then "MLIR-C.dll"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then "libMLIR-C.so"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then "libMLIR-C.dylib"
        else failwith "Unsupported platform"

// P/Invoke declarations
[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirContext mlirContextCreate()

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirContextDestroy(MlirContext ctx)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern bool mlirOperationVerify(MlirOperation op)
```
Source: [MLIR C API Reference](https://mlir.llvm.org/doxygen/mlir-c_2IR_8h.html)

### Pattern 2: Safe F# Wrapper with IDisposable

**What:** Wrap raw MLIR handles in F# types implementing IDisposable for automatic cleanup.

**When to use:** All user-facing MLIR API wrappers; shields tutorial code from unsafe pointers.

**Example:**
```fsharp
// MlirWrapper.fs - Safe F# API
namespace FunLang.Mlir

open System
open FunLang.Mlir.Bindings

type Context() =
    let handle = mlirContextCreate()

    member _.Handle = handle

    interface IDisposable with
        member _.Dispose() =
            mlirContextDestroy(handle)

type Module(context: Context, location: Location) =
    let handle = mlirModuleCreateEmpty(location.Handle)

    member _.Handle = handle

    member _.Verify() =
        let op = mlirModuleGetOperation(handle)
        mlirOperationVerify(op)

    interface IDisposable with
        member _.Dispose() =
            mlirModuleDestroy(handle)

// Usage in tutorial code
let compileProgram () =
    use ctx = new Context()          // Auto-disposed when out of scope
    use location = Location.Unknown(ctx)
    use mlirMod = new Module(ctx, location)

    // Build IR here

    if not (mlirMod.Verify()) then
        failwith "IR verification failed"

    mlirMod.Print()
    // Dispose happens automatically
```
Source: [F# Resource Management](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/resource-management-the-use-keyword), [.NET IDisposable Pattern](https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/dispose-pattern)

### Pattern 3: Compiler Driver Pipeline

**What:** Standard phases for reading source, building MLIR IR, lowering, emitting binary.

**When to use:** Every compiler driver implementation (tutorial final product).

**Example:**
```fsharp
// Compiler.fs - Top-level driver
module FunLang.Compiler

open FunLang.Mlir

let compile (sourceFile: string) (outputFile: string) =
    // Phase 1: Parse source (use LangTutorial frontend)
    let ast = FunLang.Frontend.parse sourceFile

    // Phase 2: Type checking (use LangTutorial type inference)
    let typedAst = FunLang.TypeChecker.infer ast

    // Phase 3: Build MLIR IR
    use ctx = new Context()
    ctx.LoadDialect("arith")
    ctx.LoadDialect("func")

    use mlirMod = AstToMlir.translate ctx typedAst

    // Phase 4: Verify IR
    if not (mlirMod.Verify()) then
        failwithf "MLIR verification failed"

    // Phase 5: Lower to LLVM dialect
    let loweredMod = Lowering.toLLVMDialect mlirMod

    // Phase 6: Translate to LLVM IR
    let llvmIR = Lowering.toLLVMIR loweredMod

    // Phase 7: Emit object file
    llvmIR.EmitObjectFile(outputFile)
```
Source: [MLIR Toy Tutorial Ch2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/), [Compiler Driver Patterns](https://fabiensanglard.net/dc/driver.php)

### Anti-Patterns to Avoid

- **String-based MLIR generation:** Violates MLIR's type safety; cannot use pattern rewrites. Use OpBuilder APIs even if verbose.
- **Skip MLIR verification:** IR bugs only surface at LLVM lowering with cryptic errors. Call `mlirOperationVerify()` after every phase.
- **IntPtr instead of typed handles:** Loses type safety. Use struct wrappers (`MlirContext`, `MlirModule`) for handle types.
- **Manual Dispose calls:** Easy to forget, leads to leaks. Use F# `use` bindings for automatic cleanup.
- **Custom dialect in Chapter 1:** C API doesn't support custom dialect registration easily. Start with built-in dialects (`arith`, `func`).

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MLIR context/module lifecycle | Manual create/destroy tracking | F# IDisposable with `use` bindings | Resource leaks, exception safety, finalizer guarantees |
| Native library loading | Platform-specific DllImport paths | `RuntimeInformation.IsOSPlatform` + conditional string | Cross-platform compatibility (Linux .so, macOS .dylib, Windows .dll) |
| MLIR IR printing | String concatenation | `mlirOperationPrint` with callback | Preserves MLIR syntax, handles escaping, supports custom formats |
| Type marshalling | Manual `Marshal.StringToHGlobalUni` | `MarshalAs(UnmanagedType.LPWStr)` attribute | CLR handles conversion, cleanup; less boilerplate |
| Custom dialect registration | Pure F# P/Invoke to MLIR-C | C++ wrapper with C API shim | MLIR-C API lacks `mlirDialectRegister` for custom dialects; needs C++ |

**Key insight:** F# P/Invoke can call MLIR C API for core IR operations, but custom dialect definition requires C++ code. Tutorial must either (a) provide pre-built C++ dialect library, or (b) defer custom dialect to later phases.

## Common Pitfalls

### Pitfall 1: MLIR C API Gaps for Custom Dialects

**What goes wrong:** Tutorial assumes MLIR C API has functions for custom dialect registration. Developer writes F# code expecting `mlirDialectRegister(ctx, "funlang", ...)` to exist. It doesn't. Custom dialect chapter fails completely.

**Why it happens:** MLIR C API documentation shows dialect loading (`mlirContextGetOrLoadDialect`) but not dialect **definition**. Community discussion confirms C API is limited to built-in dialects; custom dialects need C++ code.

**How to avoid:**
- Phase 1 research must prototype custom dialect registration, not assume it works
- Provide C++ wrapper library (`libFunLangDialect.so`) with C API shim:
  ```cpp
  extern "C" void funlangRegisterDialect(MlirContext ctx) {
      mlir::DialectRegistry registry;
      registry.insert<funlang::FunLangDialect>();
      unwrap(ctx)->appendDialectRegistry(registry);
  }
  ```
- F# code calls: `[<DllImport("FunLangDialect")>] extern void funlangRegisterDialect(MlirContext ctx)`

**Warning signs:**
- No prototype showing custom dialect registration from F# P/Invoke
- Assumption that MLIR C API is feature-complete
- No C++ code in project repository
- Tutorial roadmap shows custom dialect chapter without C++ build system

**Source:** [MLIR Discourse: Dialects and the C API](https://discourse.llvm.org/t/dialects-and-the-c-api/2306) - Community confirms manual C bridging required

### Pitfall 2: P/Invoke Lifetime Management Bugs

**What goes wrong:** F# code creates MlirContext, uses it to create MlirModule, destroys context first. Module handle is now dangling pointer. Next operation crashes with segfault.

**Why it happens:** MLIR C++ uses ownership hierarchies (Module owned by Context). Destroying parent invalidates children. F# GC doesn't track this relationship; manual lifetime management required.

**How to avoid:**
- Document ownership rules in MLIR primer chapter
- F# wrappers enforce lifetime: `Module` stores reference to parent `Context`
  ```fsharp
  type Module(context: Context, location: Location) =
      let handle = mlirModuleCreateEmpty(location.Handle)
      let contextRef = context  // Keep context alive

      interface IDisposable with
          member _.Dispose() =
              mlirModuleDestroy(handle)
              // contextRef ensures context not GC'd before module
  ```
- Use `use` bindings, never manual Dispose unless unavoidable
- Test with AddressSanitizer/Valgrind to catch use-after-free

**Warning signs:**
- F# wrappers don't store parent object references
- Tutorial code shows manual `new` without `use` bindings
- Random segfaults in IR operations
- No mention of object lifetime/ownership in tutorial

**Source:** [.NET Native Interop Best Practices](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices)

### Pitfall 3: Tutorial Code Snippets Not Buildable

**What goes wrong:** Reader copies code from tutorial Chapter 3, gets compile errors: "The namespace or module 'FunLang.Mlir' is not defined." Tutorial loses credibility.

**Why it happens:** Tutorial author writes snippets assuming context (imports, helper functions) that exists in their full project but not in the markdown snippet.

**How to avoid:**
- Every code snippet must be extractable to standalone `.fsx` script
- Include necessary imports at top of each snippet:
  ```fsharp
  // chapter-03-example.fsx
  #r "nuget: System.Runtime.InteropServices"
  open System.Runtime.InteropServices

  // ... rest of code
  ```
- Set up CI job that extracts code blocks, compiles them
- Each chapter's examples directory has runnable `.fsx` files matching tutorial snippets

**Warning signs:**
- Tutorial has code blocks but no `code-examples/` directory
- No CI job compiling tutorial code
- Authors say "assume you have this from previous chapter" without file reference
- Code uses `FunLang.Mlir.Bindings` but never shows how to reference it

**Source:** Industry best practice; observed in "Crafting Interpreters" community feedback

### Pitfall 4: MLIR Verification Skipped Until Debugging

**What goes wrong:** Tutorial builds MLIR IR in Chapter 3 without calling `mlirOperationVerify()`. IR has invalid structure (block missing terminator). LLVM lowering in Chapter 5 crashes with "Block must end with terminator." Reader can't debug.

**Why it happens:** Verification is not automatic; must be explicitly called. Tutorials omit it for brevity, assuming generated IR is correct.

**How to avoid:**
- Chapter 1 (hello-mlir) shows verification with intentional error:
  ```fsharp
  // Create invalid IR (block without terminator)
  let invalidBlock = ...

  if not (mlirOperationVerify(invalidOp)) then
      printfn "Verification failed (expected):"
      mlirOperationDump(invalidOp)
  ```
- Every subsequent chapter includes verification step in compiler driver
- Explain what MLIR verifies: SSA dominance, type consistency, terminator requirements, region structure

**Warning signs:**
- Tutorial code builds MLIR Module but never calls verify
- Debugging tips say "try adding verification" (should be default)
- No examples of verification failure and error messages
- IR construction has no feedback on correctness until LLVM lowering

**Source:** [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) - verification requirements

### Pitfall 5: Cross-Platform Library Loading Not Handled

**What goes wrong:** Tutorial shows `[<DllImport("MLIR-C.dll")>]`. Works on Windows, fails on Linux with "library not found." Tutorial is not cross-platform.

**Why it happens:** DllImport uses Windows naming convention (.dll). Linux needs .so suffix, macOS needs .dylib. F# runtime adds suffixes but tutorial may override.

**How to avoid:**
- Use library name without extension: `[<DllImport("MLIR-C", ...)>]`
- Runtime adds platform-appropriate suffix
- For non-standard paths, use `NativeLibrary.Load`:
  ```fsharp
  open System.Runtime.InteropServices

  let libPath =
      if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then "MLIR-C.dll"
      elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then "libMLIR-C.so"
      else "libMLIR-C.dylib"

  let handle = NativeLibrary.Load(libPath)
  ```
- Document how to add library search paths (LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH on macOS)

**Warning signs:**
- DllImport uses `.dll` extension explicitly
- Tutorial tested only on one platform
- No instructions for setting library search paths
- Windows-specific paths in code examples

**Source:** [.NET Native Library Loading](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/native-library-loading)

## Code Examples

Verified patterns from official sources:

### Creating MLIR Context and Module

```fsharp
// Source: MLIR C API IR.h + F# External Functions
open System
open System.Runtime.InteropServices

module MlirBindings =
    [<Struct>]
    type MlirContext = { ptr: nativeint }

    [<Struct>]
    type MlirModule = { ptr: nativeint }

    [<Struct>]
    type MlirLocation = { ptr: nativeint }

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirContext mlirContextCreate()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirContextDestroy(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirLocation mlirLocationUnknownGet(MlirContext ctx)

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirModule mlirModuleCreateEmpty(MlirLocation loc)

// Usage
let ctx = MlirBindings.mlirContextCreate()
let loc = MlirBindings.mlirLocationUnknownGet(ctx)
let mlirMod = MlirBindings.mlirModuleCreateEmpty(loc)

// Cleanup
MlirBindings.mlirContextDestroy(ctx)
```

### Safe Wrapper with IDisposable

```fsharp
// Source: F# Resource Management docs + .NET IDisposable pattern
type Context() =
    let handle = MlirBindings.mlirContextCreate()
    let mutable disposed = false

    member _.Handle = handle

    interface IDisposable with
        member this.Dispose() =
            this.Dispose(true)
            GC.SuppressFinalize(this)

    member private _.Dispose(disposing) =
        if not disposed then
            if disposing then
                // Dispose managed resources here (none in this case)
                ()
            MlirBindings.mlirContextDestroy(handle)
            disposed <- true

// Usage with automatic cleanup
let example () =
    use ctx = new Context()
    // Use context
    printfn "Context created: %A" ctx.Handle
    // Dispose called automatically when 'use' scope ends
```

### Compiler Driver Skeleton

```fsharp
// Source: MLIR Toy Tutorial + Rust Compiler Dev Guide
module Compiler =

    let compile (sourceFile: string) =
        // 1. Parse (reuse LangTutorial frontend)
        let ast = Parser.parseFile sourceFile

        // 2. Type check (reuse LangTutorial type inference)
        let typedAst = TypeChecker.inferTypes ast

        // 3. Create MLIR context
        use ctx = new Context()

        // 4. Translate AST to MLIR
        use mlirMod = AstToMlir.translate ctx typedAst

        // 5. Verify MLIR
        if not (mlirMod.Verify()) then
            failwith "MLIR verification failed"

        // 6. Lower to LLVM dialect
        let llvmDialect = Lowering.toLLVMDialect mlirMod

        // 7. Translate to LLVM IR
        let llvmIR = Lowering.toLLVMIR llvmDialect

        // 8. Emit native binary
        llvmIR.EmitBinary("output.o")

        printfn "Compilation successful"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LLVM-C API only | MLIR-C API (`-DMLIR_BUILD_MLIR_C_DYLIB=ON`) | LLVM 17.0 (Sep 2023) | Enabled stable C bindings for MLIR; prior versions had C++ only |
| DllImport with manual marshalling | LibraryImport source generator (.NET 7+) | .NET 7 (Nov 2022) | Compile-time P/Invoke generation; better performance, diagnostics |
| Manual char* marshalling | `MarshalAs(UnmanagedType.LPWStr)` with UTF-8 support | .NET 7 | UTF-8 as first-class option (previously only ANSI/Unicode) |
| C++/CLI for .NET/C++ interop | Direct P/Invoke to C APIs | .NET Core 3.0 (2019) | C++/CLI deprecated on .NET Core; P/Invoke cross-platform |

**Deprecated/outdated:**
- **C++/CLI for MLIR bindings:** Windows-only, not supported on .NET Core; use C API + P/Invoke
- **LLVM 16.x and earlier MLIR-C:** Incomplete C API coverage; LLVM 17+ has mature MLIR-C API
- **.NET Framework P/Invoke patterns:** Use .NET 8/9 patterns (SafeHandle, LibraryImport)
- **String-based IR generation:** MLIR design emphasizes structured builders; don't emit strings

## Open Questions

Things that couldn't be fully resolved:

1. **MLIR C API Custom Dialect Registration - Exact Functions Available**
   - What we know: Community confirms C API lacks generic custom dialect registration; requires C++ wrapper
   - What's unclear: Exact C++ API to use for dialect registration in LLVM 19.1.x; TableGen requirements
   - Recommendation: Phase 1 must prototype C++ dialect wrapper before tutorial writing begins

2. **F# SafeHandle vs. IDisposable for MLIR Handles**
   - What we know: SafeHandle preferred for unmanaged resources; provides finalizer guarantees
   - What's unclear: Whether MLIR handles should use SafeHandle (complex) or simple IDisposable (simpler for tutorial)
   - Recommendation: Start with IDisposable + `use` bindings (simpler); mention SafeHandle in advanced section

3. **Native Library Distribution for Tutorial**
   - What we know: MLIR-C is 200+ MB; readers must build LLVM or download pre-built
   - What's unclear: Best practice for distributing pre-built MLIR libraries (GitHub releases? separate repo?)
   - Recommendation: Document build instructions; optionally provide Docker image with pre-built MLIR

4. **Tutorial Chapter Granularity**
   - What we know: Phase 1 covers MLIR build, P/Invoke bindings, hello world compiler
   - What's unclear: Should P/Invoke bindings be one chapter or split into (a) raw P/Invoke, (b) safe wrappers?
   - Recommendation: Two chapters - one for mechanics, one for patterns; more digestible

## Sources

### Primary (HIGH confidence)

- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) - Official C API overview
- [MLIR C API IR.h Reference](https://mlir.llvm.org/doxygen/mlir-c_2IR_8h.html) - Function signatures for context, module, types, operations
- [MLIR Discourse: Dialects and the C API](https://discourse.llvm.org/t/dialects-and-the-c-api/2306) - Community discussion on custom dialect C API limitations
- [F# External Functions Documentation](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/functions/external-functions) - DllImport patterns, marshalling
- [MLIR Toy Tutorial Chapter 2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) - Compiler driver structure, IR emission
- [Understanding MLIR IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) - Concepts: operation, region, block, SSA

### Secondary (MEDIUM confidence)

- [.NET Native Interop Best Practices](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices) - SafeHandle, IDisposable, P/Invoke patterns
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) - Dialect, operation, type system semantics
- [MLIR Getting Started](https://mlir.llvm.org/getting_started/) - Build instructions (verified MLIR_BUILD_MLIR_C_DYLIB flag)
- [Rust Compiler Dev Guide](https://rustc-dev-guide.rust-lang.org/overview.html) - Compiler pipeline phases (parse, IR, lower, codegen)

### Tertiary (LOW confidence)

- WebSearch: "MLIR build MLIR_BUILD_MLIR_C_DYLIB CMake LLVM 19 2026" - Confirms build flag exists in LLVM 19
- WebSearch: "F# P/Invoke DllImport native library patterns 2026" - Modern .NET patterns, LibraryImport
- GitHub Issue [#108253](https://github.com/llvm/llvm-project/issues/108253) - Enabling external dialects as shared libs for C API users (ongoing discussion)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - LLVM 19.x, .NET 8/9, CMake, Ninja are verified current versions
- Architecture (P/Invoke + IDisposable): HIGH - Standard .NET interop patterns, verified in docs
- Pitfalls (C API custom dialect gap): HIGH - Confirmed by LLVM Discourse community discussion
- Tutorial structure: MEDIUM - Based on existing tutorial patterns (Toy, Crafting Interpreters) but not MLIR-specific
- Custom dialect C++ wrapper: MEDIUM - Approach is standard but exact API calls need prototyping

**Research date:** 2026-02-05
**Valid until:** 2026-04-05 (60 days - stable domain; LLVM 20 may release but 19.x remains valid)

**Research scope notes:**
This research focused on Phase 1 (Foundation & Interop). Later phases will need additional research:
- Phase 2: Boehm GC integration with MLIR LLVM dialect
- Phase 3: Closure conversion patterns in MLIR
- Phase 4: Pattern match lowering to decision trees
- Phase 5: Custom dialect TableGen definitions (C++)

**Critical unknowns requiring prototyping before Chapter 1:**
1. Custom dialect registration C++ wrapper - exact API calls for LLVM 19.1.x
2. MLIR-C library loading on all platforms (Windows/Linux/macOS) - test rpath, DLL search paths
3. F# IDisposable wrapper correctness - validate with AddressSanitizer, no leaks
