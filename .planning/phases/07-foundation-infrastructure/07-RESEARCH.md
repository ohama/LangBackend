# Phase 7: Foundation Infrastructure - Research

**Researched:** 2026-02-12
**Domain:** F# P/Invoke bindings to MLIR C API and fluent builder patterns
**Confidence:** HIGH

## Summary

This phase implements the foundational infrastructure for FunLang v2.0 compiler: P/Invoke bindings to MLIR-C API and a fluent F# OpBuilder wrapper. The tutorial (v1.0) already documented the theory in Chapters 2-5; this phase creates the actual working code.

The research identified the standard approach: thin P/Invoke layer with F# struct handles (Chapter 03 pattern), wrapped by IDisposable-implementing classes that manage resource lifetimes and provide fluent APIs (Chapter 04 pattern). The MLIR C API is stable and well-documented, with clear ownership semantics that must be preserved through parent references in F# wrappers.

Key considerations include: cross-platform library loading (`MLIR-C` resolves to platform-specific extensions), string marshalling through `MlirStringRef` (non-owning view requiring caller-managed lifetime), explicit `CallingConvention.Cdecl` specification for portability, and `StructLayout(LayoutKind.Sequential)` for correct memory layout matching C structs.

**Primary recommendation:** Implement exactly the architecture documented in tutorial Chapters 03-04, using IDisposable pattern with parent references for ownership safety and OpBuilder for fluent operation construction.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| .NET 8.0+ SDK | 8.0+ | Runtime and compiler | Current LTS version, supports NativeAOT, native interop features |
| MLIR-C shared library | LLVM 19.x | MLIR C API | Official stable C API for MLIR, platform-independent |
| System.Runtime.InteropServices | Built-in | P/Invoke marshalling | .NET standard for native interop |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| NativeLibrary API | .NET 8.0+ | Custom library loading | For non-standard MLIR install paths |
| FSharp.Core | 8.0+ | F# standard library | Built-in with F# compiler |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| P/Invoke | C++/CLI | P/Invoke is cross-platform; C++/CLI is Windows-only |
| Manual marshalling | SafeHandle | SafeHandle is safer but MLIR handles don't need finalization (context-owned) |
| Fluent API | Functional composition | Fluent is more readable for imperative IR building |

**Installation:**
```bash
# MLIR already built from Phase 1 tutorial
# No additional packages needed - use built-in .NET APIs
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── MlirBindings/          # Thin P/Invoke layer (Chapter 03)
│   └── MlirBindings.fs    # Handle types, DllImport declarations
├── MlirWrapper/           # IDisposable wrappers (Chapter 04)
│   └── MlirWrapper.fs     # Context, Module, Location, OpBuilder
└── tests/
    └── MlirBindings.Tests.fs  # Basic smoke tests
```

### Pattern 1: Handle Type Definition
**What:** F# struct wrapping `nativeint` pointer matching C struct layout
**When to use:** For all MLIR opaque handle types (Context, Module, Operation, etc.)
**Example:**
```fsharp
// Source: Tutorial Chapter 03
[<Struct>]
type MlirContext =
    val Handle: nativeint
    new(handle) = { Handle = handle }
```

### Pattern 2: String Marshalling with MlirStringRef
**What:** Non-owning string view requiring caller-managed lifetime
**When to use:** For all MLIR API calls accepting string parameters
**Example:**
```fsharp
// Source: Tutorial Chapter 03
[<Struct; StructLayout(LayoutKind.Sequential)>]
type MlirStringRef =
    val Data: nativeint      // const char*
    val Length: nativeint    // size_t

    static member FromString(s: string) =
        let bytes = System.Text.Encoding.UTF8.GetBytes(s)
        let ptr = Marshal.AllocHGlobal(bytes.Length)
        Marshal.Copy(bytes, 0, ptr, bytes.Length)
        MlirStringRef(ptr, nativeint bytes.Length)

    member this.Free() =
        if this.Data <> nativeint 0 then
            Marshal.FreeHGlobal(this.Data)

    static member WithString(s: string, f: MlirStringRef -> 'a) =
        let strRef = MlirStringRef.FromString(s)
        try f strRef
        finally strRef.Free()
```

### Pattern 3: P/Invoke Declaration
**What:** Explicit DllImport with Cdecl calling convention
**When to use:** For all MLIR C API function bindings
**Example:**
```fsharp
// Source: Tutorial Chapter 03
module MlirNative =
    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern MlirContext mlirContextCreate()

    [<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirContextDestroy(MlirContext ctx)
```

### Pattern 4: IDisposable Wrapper with Parent Reference
**What:** Wrapper class implementing IDisposable, holding reference to parent to prevent premature GC
**When to use:** For all MLIR objects with ownership hierarchy (Module owns Context, Block owns Operation)
**Example:**
```fsharp
// Source: Tutorial Chapter 04
type Module(context: Context, location: Location) =
    let handle = MlirNative.mlirModuleCreateEmpty(location.Handle)
    let contextRef = context  // Keep parent alive
    let mutable disposed = false

    member _.Handle = handle
    member _.Context = contextRef

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                MlirNative.mlirModuleDestroy(handle)
                disposed <- true
```

### Pattern 5: Fluent OpBuilder API
**What:** Builder class with high-level methods that wrap verbose operation state construction
**When to use:** For common operation creation (constants, functions, arithmetic)
**Example:**
```fsharp
// Source: Tutorial Chapter 04
type OpBuilder(context: Context) =
    member _.CreateConstant(value: int, typ: MlirType, location: Location) =
        let mutable state = MlirNative.mlirOperationStateGet(
            MlirStringRef.FromString("arith.constant"), location.Handle)
        // ... state configuration ...
        MlirNative.mlirOperationCreate(&state)
```

### Anti-Patterns to Avoid
- **Forgetting to specify CallingConvention.Cdecl:** Causes crashes on Windows x86 where default is Stdcall
- **Using mutable struct fields without `mutable` keyword:** F# structs are immutable by default, causing silent bugs
- **Not keeping parent references:** Premature GC of Context while Module still exists causes segfaults
- **Manual Dispose calls without try/finally:** Use F# `use` keyword for automatic cleanup

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Custom library loading | DllImport path resolution | NativeLibrary.SetDllImportResolver | Handles platform differences, search paths, rpath |
| String lifetime management | Manual Marshal.AllocHGlobal/Free | MlirStringRef.WithString helper | Ensures cleanup even on exceptions |
| Resource cleanup | Manual try/finally blocks | F# `use` keyword with IDisposable | Compiler-guaranteed cleanup, fewer bugs |
| Operation state assembly | Direct MlirOperationState manipulation | OpBuilder.CreateXXX methods | Encapsulates complexity, reduces errors |

**Key insight:** MLIR's ownership model is subtle (context owns types, regions own blocks, blocks own operations). Manual management is error-prone; use IDisposable pattern with parent references to enforce ownership at type level.

## Common Pitfalls

### Pitfall 1: Platform-Specific Calling Convention Defaults
**What goes wrong:** DllImport without explicit CallingConvention fails on Windows x86 (expects Stdcall, MLIR uses Cdecl)
**Why it happens:** .NET defaults to Stdcall on Windows x86 for legacy compatibility
**How to avoid:** Always specify `CallingConvention = CallingConvention.Cdecl` in every DllImport
**Warning signs:** AccessViolationException or corrupted stack only on Windows builds

### Pitfall 2: MlirStringRef Lifetime Violations
**What goes wrong:** Passing freed MlirStringRef or F# string that gets GC'd causes access violations
**Why it happens:** MlirStringRef is non-owning view; MLIR doesn't copy the string data
**How to avoid:** Use `MlirStringRef.WithString` helper for automatic lifetime management, or ensure manual Free after MLIR call completes
**Warning signs:** Intermittent crashes, garbage characters in IR output

### Pitfall 3: Premature Parent Object Disposal
**What goes wrong:** Context gets GC'd while Module still references it, causing segfault on next MLIR call
**Why it happens:** F# GC doesn't understand MLIR's C++ ownership semantics
**How to avoid:** Always store parent reference in child wrapper (e.g., Module holds `contextRef: Context`)
**Warning signs:** Crashes when accessing module/operation after scope changes

### Pitfall 4: StructLayout Misalignment
**What goes wrong:** Fields read/written at wrong offsets, causing garbage data or crashes
**Why it happens:** F# default struct layout differs from C Sequential layout
**How to avoid:** Always use `[<StructLayout(LayoutKind.Sequential)>]` for P/Invoke structs
**Warning signs:** Unexpected field values, crashes in MLIR functions expecting specific layout

### Pitfall 5: Forgetting `&` for by-ref Parameters
**What goes wrong:** Value is copied instead of passed by reference, MLIR can't modify output parameters
**Why it happens:** F# requires explicit `&` for by-ref, unlike C# where `ref`/`out` is more common
**How to avoid:** Check MLIR C API headers for pointer parameters (e.g., `MlirType *results`), use `&` in F# call
**Warning signs:** Operations missing results, arrays not populated

## Code Examples

Verified patterns from official sources and tutorial:

### Complete Context Creation with Dialects
```fsharp
// Source: Tutorial Chapter 04
use ctx = new Context()
ctx.LoadDialect("func")
ctx.LoadDialect("arith")
ctx.LoadDialect("scf")
ctx.LoadDialect("llvm")

let loc = Location.Unknown(ctx)
use mlirMod = new Module(ctx, loc)
// ... build IR ...
// Automatic cleanup via 'use'
```

### Fluent Operation Building
```fsharp
// Source: Tutorial Chapter 04
let builder = OpBuilder(ctx)
let i32Type = builder.I32Type()

// Create constant: %c42 = arith.constant 42 : i32
let constOp = builder.CreateConstant(42, i32Type, loc)
let constValue = builder.GetResult(constOp, 0)

// Create function with body
let funcType = builder.FunctionType([||], [| i32Type |])
let funcOp = builder.CreateFunction("main", funcType, loc)

// Create return: return %c42 : i32
let returnOp = builder.CreateReturn([| constValue |], loc)
```

### Safe String Marshalling
```fsharp
// Source: Tutorial Chapter 03
// GOOD: Automatic cleanup
MlirStringRef.WithString "func.func" (fun nameRef ->
    let op = MlirNative.mlirOperationCreate(nameRef)
    // nameRef automatically freed after lambda
    op)

// BAD: Manual management (error-prone)
let nameRef = MlirStringRef.FromString("func.func")
let op = MlirNative.mlirOperationCreate(nameRef)
nameRef.Free()  // Forgot try/finally - leaks on exception
```

### Cross-Platform Library Loading
```fsharp
// Source: Tutorial Chapter 03, Option 2
module LibraryLoader =
    let initialize() =
        NativeLibrary.SetDllImportResolver(
            typeof<MlirContext>.Assembly,
            fun libraryName _ _ ->
                if libraryName = "MLIR-C" then
                    let installPath = Environment.GetEnvironmentVariable("MLIR_INSTALL_PATH")
                    let libPath =
                        if RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
                            Path.Combine(installPath, "lib", "libMLIR-C.so")
                        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
                            Path.Combine(installPath, "lib", "libMLIR-C.dylib")
                        else
                            Path.Combine(installPath, "bin", "MLIR-C.dll")
                    NativeLibrary.Load(libPath)
                else nativeint 0)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual string Marshal.StringToHGlobalAnsi | MlirStringRef with UTF-8 | MLIR C API design | Correct Unicode handling, explicit lifetime |
| No resource management | IDisposable + use keyword | F# best practice | Automatic cleanup, fewer leaks |
| Direct C API calls | Fluent OpBuilder wrapper | Tutorial Chapter 04 | 10x less boilerplate, readable |
| IntPtr for handles | Typed struct wrappers | F# interop pattern | Compile-time type safety |

**Deprecated/outdated:**
- **Ansi string marshalling:** MLIR C API uses UTF-8, not platform-default Ansi encoding
- **Finalizers for MLIR handles:** MLIR objects are context-owned; context destruction handles cleanup, finalizers unnecessary and cause GC pressure

## Open Questions

Things that couldn't be fully resolved:

1. **Custom Dialect Registration via C API**
   - What we know: Tutorial documents C++ dialect registration with TableGen, Chapter 05 theory
   - What's unclear: Whether v2.0 needs custom dialect in Phase 7, or deferred to later phases
   - Recommendation: Start with standard dialects (func, arith, scf, llvm) only in Phase 7; implement custom dialect if Phase 8+ codegen requires it

2. **MLIR Version Compatibility**
   - What we know: Tutorial uses LLVM 19.x, C API marked as "alpha" (unstable)
   - What's unclear: Whether to test against LLVM 20 (if available in 2026), or lock to 19.x
   - Recommendation: Target LLVM 19.x specifically (documented in tutorial); document upgrade path in Phase 11 testing

3. **NativeAOT Compatibility**
   - What we know: .NET 8+ supports NativeAOT, but P/Invoke may have restrictions
   - What's unclear: Whether IDisposable + P/Invoke patterns are NativeAOT-compatible
   - Recommendation: Test with standard .NET runtime first (Phase 7-10); verify NativeAOT in Phase 11 if required for distribution

## Sources

### Primary (HIGH confidence)
- Tutorial Chapter 02 (tutorial/02-hello-mlir.md) - P/Invoke fundamentals, handle types
- Tutorial Chapter 03 (tutorial/03-pinvoke-bindings.md) - Complete bindings module structure
- Tutorial Chapter 04 (tutorial/04-wrapper-layer.md) - IDisposable wrappers, OpBuilder pattern
- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/) - Official MLIR C API design and usage
- [Microsoft Learn: Unmanaged calling conventions](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/calling-conventions) - CallingConvention.Cdecl requirements
- [Microsoft Learn: F# Resource Management](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/resource-management-the-use-keyword) - `use` keyword and IDisposable

### Secondary (MEDIUM confidence)
- [GitHub: llvm/llvm-project MLIR C API](https://github.com/llvm/llvm-project/commit/855ec517a300) - MlirStringRef design commit
- [Microsoft Learn: Customizing structure marshalling](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/customize-struct-marshalling) - StructLayout patterns
- [Melior Rust bindings](https://github.com/mlir-rs/melior) - Alternative language binding for architecture comparison

### Tertiary (LOW confidence)
- [F# MLIR Hello project](https://github.com/speakeztech/fsharp-mlir-hello) - Community example (unverified, may be outdated)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Tutorial chapters 03-04 provide complete reference implementation
- Architecture: HIGH - Patterns directly documented in tutorial with working examples
- Pitfalls: MEDIUM - Derived from tutorial common errors sections and .NET P/Invoke docs; not all tested in practice

**Research date:** 2026-02-12
**Valid until:** 30 days (2026-03-14) - MLIR C API is stable (LLVM 19.x), .NET 8 is LTS, patterns are mature
