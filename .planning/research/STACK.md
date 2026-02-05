# Stack Research

**Domain:** MLIR Compiler Backend from F#/.NET
**Researched:** 2026-02-05
**Confidence:** MEDIUM (training data only, web verification unavailable)

## Executive Summary

Building an MLIR-based compiler backend from F# faces a **critical gap**: there are no mature, maintained F#/MLIR bindings as of early 2025. The standard MLIR stack is C++-native with a C API for language bindings, but .NET/F# bindings are not first-class citizens in the MLIR ecosystem. This research identifies three viable approaches with varying tradeoffs.

**Key Finding:** The F#-to-MLIR interop layer is the highest-risk, highest-effort component. This tutorial will need to pioneer an approach that doesn't have extensive production precedent.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **MLIR** | 19.x (LLVM 19) | IR framework for compiler middle-end | Industry standard for modern compiler infrastructure; structured dialects avoid string-based IR generation; Google/Meta/Apple production use |
| **LLVM** | 19.x | Backend code generation to native | De facto standard for native code generation; MLIR lowers to LLVM IR; mature, well-documented |
| **.NET SDK** | 8.0 or 9.0 | F# runtime and build system | F# 8/9 on .NET 8/9 are current LTS; need modern P/Invoke features; .NET 9 has improved native interop |
| **F#** | 8.0 or 9.0 | Implementation language | Project requirement for consistency with LangTutorial; functional style matches compiler domain |

**Confidence:** HIGH for MLIR/LLVM versions (standard practice), HIGH for .NET/F# versions (verifiable from Microsoft)

### F#-to-MLIR Interop Options (CRITICAL DECISION)

This is the **most important and least certain** part of the stack. Three approaches:

#### Option 1: Raw P/Invoke to MLIR-C API (RECOMMENDED for Tutorial)

| Aspect | Details |
|--------|---------|
| **Approach** | Direct P/Invoke calls to `libMLIR-C.so` / `MLIR-C.dll` |
| **Pros** | No external dependencies; full control; educational value (shows FFI directly); mirrors MLIR's Python bindings architecture |
| **Cons** | Most boilerplate; manual memory management; unsafe code blocks; need to ship native binaries |
| **Effort** | High initial (build C API bindings), low ongoing |
| **Confidence** | HIGH (C API is stable, P/Invoke is proven) |

**Why Recommended:** For a tutorial, showing the raw FFI layer is pedagogically valuable. Readers learn how compilers actually talk to MLIR. The C API is stable and well-documented.

```fsharp
// Example P/Invoke pattern
[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirContext mlirContextCreate()

[<DllImport("MLIR-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirModule mlirModuleCreateEmpty(MlirLocation location)
```

#### Option 2: Use Unofficial .NET Bindings (IF THEY EXIST)

| Aspect | Details |
|--------|---------|
| **Approach** | Search for community packages like `MLIRSharp`, `MLIR.NET`, or similar |
| **Pros** | Less boilerplate if well-maintained; idiomatic .NET APIs |
| **Cons** | **Unknown existence** — no confirmed maintained bindings as of Jan 2025; dependency risk; may lag MLIR versions |
| **Effort** | Low if exists and works, high if needs forking/fixing |
| **Confidence** | LOW (existence unverified without web search) |

**Status:** Requires web search to verify. Likely does NOT exist in mature form.

#### Option 3: C++/CLI Wrapper Layer

| Aspect | Details |
|--------|---------|
| **Approach** | Write C++/CLI bridge that uses MLIR C++ API, expose to F# |
| **Pros** | Access to full C++ MLIR API (richer than C API); type-safe wrappers |
| **Cons** | C++/CLI is Windows-only; adds build complexity; mixed-mode assemblies; deprecated on .NET Core |
| **Effort** | Very high; requires C++/CLI expertise |
| **Confidence** | HIGH (technically feasible) but **NOT RECOMMENDED** |

**Why NOT Recommended:** C++/CLI doesn't work on .NET Core cross-platform. Deal-breaker for modern F#.

### MLIR Native Build Stack

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **CMake** | 3.20+ | MLIR/LLVM build system | Required by LLVM project; industry standard for C++ builds |
| **Ninja** | 1.11+ | Build executor | Faster than Make; recommended by LLVM docs |
| **Clang** | 19.x | C++ compiler for MLIR | LLVM project's native compiler; best compatibility |
| **Python** | 3.8+ | Build scripts, MLIR TableGen | MLIR build system uses Python; TableGen for dialect codegen |

**Build Note:** Must build MLIR with `-DMLIR_BUILD_MLIR_C_DYLIB=ON` to get `libMLIR-C` shared library for P/Invoke.

### F# Project Structure

| Tool | Purpose | Notes |
|------|---------|-------|
| **Paket** or **NuGet** | Dependency management | Paket for reproducible builds; NuGet simpler |
| **FAKE** | Build automation | F# DSL for build scripts; can handle native lib deployment |
| **Expecto** | Testing framework | Functional, composable; good for compiler testing |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **FSharp.Core** | 8.0+ | F# standard library | Always (implicit) |
| **FsCheck** | 2.16+ | Property-based testing | Testing MLIR passes, lowering correctness |
| **Argu** | 6.2+ | CLI argument parsing | If tutorial includes a `funlangc` compiler driver |
| None needed | — | MLIR ops, types, dialects | All via P/Invoke to C API |

**Key Point:** Unlike typical F# projects, this won't use many F# libraries. The core work is FFI to MLIR.

## Installation & Setup

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install cmake ninja-build clang lld python3

# macOS
brew install cmake ninja llvm python3

# Windows
# Install Visual Studio 2022 with C++ tools
# Install CMake, Ninja via chocolatey or manually
choco install cmake ninja
```

### Build MLIR with C API

```bash
# Clone LLVM project (includes MLIR)
git clone --depth 1 --branch release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project

# Configure MLIR build with C API
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install

# Build (takes 30-60 min on modern hardware)
cmake --build build --target install

# Result: libMLIR-C.so (Linux), libMLIR-C.dylib (macOS), MLIR-C.dll (Windows)
# Located in: $HOME/mlir-install/lib/
```

### F# Project Setup

```bash
# Create F# project
dotnet new console -lang F# -n FunLangBackend
cd FunLangBackend

# Add dependencies
dotnet add package FsCheck
dotnet add package Expecto

# Copy MLIR-C native library to output directory
# (Needs FAKE script or post-build MSBuild target)
```

### P/Invoke Wrapper Structure

```fsharp
// MlirBindings.fs - Low-level P/Invoke declarations
namespace FunLang.Mlir.Bindings

open System
open System.Runtime.InteropServices

[<Struct>]
type MlirContext =
    val mutable ptr: nativeint

[<Struct>]
type MlirModule =
    val mutable ptr: nativeint

// ... hundreds more type wrappers

module Native =
    let private lib = "MLIR-C"  // or platform-specific name

    [<DllImport(lib, CallingConvention = CallingConvention.Cdecl)>]
    extern MlirContext mlirContextCreate()

    [<DllImport(lib, CallingConvention = CallingConvention.Cdecl)>]
    extern void mlirContextDestroy(MlirContext ctx)

    // ... hundreds more extern declarations
```

```fsharp
// MlirWrapper.fs - Idiomatic F# API over P/Invoke
namespace FunLang.Mlir

open FunLang.Mlir.Bindings

type Context() =
    let handle = Native.mlirContextCreate()

    member _.Handle = handle

    interface IDisposable with
        member _.Dispose() = Native.mlirContextDestroy(handle)

// ... wrap all C API in F#-friendly types with IDisposable, computation expressions, etc.
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **MLIR C API + P/Invoke** | MLIR C++ API + C++/CLI | Never for .NET Core; only if targeting .NET Framework on Windows |
| **MLIR C API + P/Invoke** | LLVM-only (skip MLIR) | If tutorial focuses on simple imperative languages without custom IR abstractions; MLIR is overkill for basic AST→LLVM |
| **F# Implementation** | C# Implementation | If reader base prefers C# over F#; C# has better tooling for P/Invoke (CsWin32 source generators) |
| **Direct Native Binaries** | .NET NativeAOT | If shipping a standalone compiler binary; not relevant for tutorial |
| **LLVM 19.x** | LLVM 18.x or 20.x | 18.x if compatibility needed; 20.x if very cutting-edge (may not be released yet) |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **String-based MLIR generation** | Brittle, no type safety, defeats MLIR's structured IR design | MLIR Builder APIs via C API |
| **LLVM-C++ API from F#** | No stable FFI boundary; C++ name mangling breaks P/Invoke | MLIR-C API (stable C ABI) |
| **JVM-based MLIR bindings** | Wrong ecosystem; would require JVM interop from .NET | .NET-native P/Invoke |
| **MLIR Toy Dialect example code** | Tutorial example, not production; C++ only, not F# applicable | Custom dialect from scratch |
| **LLVM 17.x or older** | MLIR C API significantly improved in 18.x+; missing features | LLVM/MLIR 18.x minimum, 19.x preferred |
| **Mono runtime** | Obsolete; .NET Core/.NET 5+ is the standard | .NET 8/9 SDK |
| **PCL or .NET Standard libs** | Legacy; use .NET 8+ target | `<TargetFramework>net8.0</TargetFramework>` |

## Stack Patterns by Scenario

### If Targeting Tutorial Readers (RECOMMENDED)

**Pattern: Minimal Dependencies, Maximum Transparency**

- Use raw P/Invoke with extensive comments explaining each FFI call
- Wrap C API in thin F# layer (Context, Module, Type, Operation, Block, Region)
- Keep wrapper code in tutorial chapters (not hidden in library)
- Focus on educational value over API beauty

**Rationale:** Readers learn compiler internals AND FFI patterns.

### If Building a Reusable Library

**Pattern: Idiomatic F# Wrapper**

- Heavy use of computation expressions for Builder pattern
- `IDisposable` for all MLIR handles (auto-cleanup)
- F# unions for MLIR types (e.g., `MlirType = IntType of width | FunctionType of ...`)
- NuGet package with embedded native binaries

**Rationale:** Usable by F# compiler projects beyond this tutorial.

### If Shipping a Standalone Compiler

**Pattern: .NET NativeAOT + Static Linking**

- Use .NET 8+ NativeAOT to compile F# to native binary
- Statically link MLIR-C and LLVM libraries
- Single-file executable, no runtime dependencies

**Rationale:** Professional compiler distribution (e.g., `funlangc` binary).

## Version Compatibility Matrix

| F# SDK | MLIR | Compatible | Notes |
|--------|------|------------|-------|
| .NET 8 | LLVM 18.x | ✅ Yes | Stable combo as of 2024 |
| .NET 8 | LLVM 19.x | ✅ Yes | Recommended (current) |
| .NET 9 | LLVM 19.x | ✅ Yes | Cutting edge as of late 2024 |
| .NET 7 | LLVM 19.x | ⚠️ Works | .NET 7 EOL May 2024, avoid |
| .NET 6 | LLVM 17.x | ⚠️ Works | Both EOL soon, don't use |

**Native Library Path Issues:**

- Linux: `LD_LIBRARY_PATH` or `rpath` in binary
- macOS: `DYLD_LIBRARY_PATH` or `@rpath` in dylib
- Windows: DLL must be in same dir as .exe or in `PATH`

**Solution:** Use MSBuild `<Content>` or FAKE script to copy native libs to output directory.

## Critical Unknown: Existing .NET/MLIR Bindings

**Status:** UNVERIFIED (no web access during research)

**Requires Investigation:**

1. Search NuGet for packages: `MLIR`, `MLIRSharp`, `MLIR.NET`, `LlvmSharp` (latter is LLVM-only, not MLIR)
2. Search GitHub for repos: `fsharp mlir`, `dotnet mlir bindings`, `csharp mlir`
3. Check LLVM Discourse forum for .NET binding discussions

**If bindings exist:**

- Evaluate maturity: last commit date, issue count, MLIR version lag
- Check if they cover MLIR C API fully or partially
- Verify cross-platform support (not just Windows)
- Assess if suitable for tutorial (or too high-level, hiding details)

**If bindings do NOT exist (LIKELY):**

- Tutorial must include "Chapter 0: Building F# Bindings to MLIR-C"
- This becomes a unique contribution to the F#/MLIR ecosystem
- Consider open-sourcing bindings separately as `FSharp.MLIR` library

## Research Gaps & Open Questions

### HIGH Priority (Blocks Stack Decision)

1. **Do mature .NET/MLIR bindings exist?** (Confidence: LOW)
   - **Impact:** Changes tutorial structure entirely
   - **Mitigation:** Web search required; assume NO and plan for P/Invoke approach

2. **What's the exact MLIR-C API coverage in LLVM 19?** (Confidence: MEDIUM)
   - **Impact:** May be missing APIs for custom dialect registration
   - **Mitigation:** Check LLVM 19 release notes; fall back to minimal C++ glue if needed

### MEDIUM Priority (Affects Implementation Effort)

3. **How to distribute native MLIR-C binaries with F# NuGet package?** (Confidence: MEDIUM)
   - **Known:** `runtimes/` folder in NuGet, or separate native packages
   - **Unknown:** Best practice for 100MB+ native libs in tutorial context

4. **Performance of P/Invoke for high-frequency MLIR Builder calls?** (Confidence: MEDIUM)
   - **Known:** P/Invoke has ~10-50ns overhead per call
   - **Unknown:** If this matters for compiler (likely not, I/O dominates)

### LOW Priority (Doesn't Block Tutorial)

5. **MLIR TableGen from F#?** (Confidence: LOW)
   - **Known:** TableGen is Python-based, generates C++
   - **Unknown:** If F# can hook into TableGen for dialect codegen
   - **Likely:** Write dialect definitions in TableGen, use via P/Invoke wrappers

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| **MLIR/LLVM Stack** | HIGH | Standard practice; well-documented; training data current |
| **F# P/Invoke Approach** | HIGH | Proven technique; .NET FFI is mature |
| **.NET Versions** | HIGH | Microsoft's public roadmap |
| **Existence of .NET/MLIR Bindings** | LOW | No web access; likely don't exist but unverified |
| **MLIR-C API Coverage** | MEDIUM | Improved significantly in 18.x, but custom dialects may need C++ |
| **Cross-platform Build** | MEDIUM | MLIR builds on all platforms, but complexity varies |
| **Tutorial Pedagogy** | MEDIUM | Assumes P/Invoke approach is learnable; may be steep |

## Recommendations for Roadmap

### Phase 1: Validate Stack Assumptions

- **Task:** Web search for .NET/MLIR bindings (NuGet, GitHub, LLVM Discourse)
- **Deliverable:** Confirmed binding approach (P/Invoke vs existing library)
- **Risk:** If bindings exist but are immature, may waste time evaluating

### Phase 2: MLIR-C Build Verification

- **Task:** Build LLVM 19.x with MLIR-C on target platforms (Linux, macOS, Windows)
- **Deliverable:** `libMLIR-C.so` / `.dylib` / `.dll` in known location
- **Risk:** Build failures on Windows (common LLVM issue); may need patches

### Phase 3: P/Invoke Prototype

- **Task:** Minimal F# program that creates MlirContext, MlirModule, prints IR
- **Deliverable:** Proof-of-concept that P/Invoke works; measure LOC/API surface
- **Risk:** May discover missing C API coverage, requiring C++ glue

### Phase 4: Dialect Registration

- **Task:** Register custom FunLang dialect via C API
- **Deliverable:** Working custom operation in MLIR IR
- **Risk:** **HIGH** — custom dialects may not be fully exposed in C API; may need minimal C++ shim

### Phase 5: Builder API Wrappers

- **Task:** Wrap MLIR OpBuilder, Block, Region in F#-friendly API
- **Deliverable:** Idiomatic F# API for building MLIR; computation expressions if appropriate
- **Risk:** Boilerplate explosion; may need code generation

### Phase 6: Tutorial Chapter Structure

- **Task:** Decide if bindings are in tutorial or external library
- **Deliverable:** Chapter outline with FFI explanation level
- **Risk:** Too much FFI detail bores readers; too little leaves them unable to reproduce

## Sources

**Note:** This research is based on training data (cut-off January 2025) without web verification due to environment limitations.

- **MLIR Documentation** (llvm.org/mlir) — training data on C API, dialects, passes (Confidence: HIGH)
- **LLVM Release Notes** (llvm.org releases) — MLIR-C API improvements in 17.x-19.x (Confidence: HIGH)
- **.NET P/Invoke Documentation** (learn.microsoft.com) — FFI patterns for F# (Confidence: HIGH)
- **F# Language Design** (fsharp.org) — F# 8/9 features (Confidence: HIGH)
- **Assumed Absence of .NET/MLIR Bindings** — no confirmed packages in training data (Confidence: LOW - needs verification)

**Web verification required for:**
- Current LLVM/MLIR version (may be 20.x by Feb 2026)
- Existence of `MLIRSharp` or similar NuGet packages
- MLIR-C API completeness in latest release
- F# 9 interop improvements (if any)

---

## CRITICAL CALL-OUT for Roadmap

**The F#-to-MLIR binding layer is the project's highest technical risk.**

If no mature bindings exist (likely), the tutorial must:

1. Teach P/Invoke FFI patterns (pedagogical benefit, but steep learning curve)
2. Build ~500-1000 lines of FFI wrapper code before any MLIR compiler work
3. Maintain bindings as MLIR versions update (sustainability risk)

**Alternative:** Write tutorial in C++ (with F# backend as "exercise for reader"). This avoids FFI entirely but breaks the "consistency with LangTutorial" requirement.

**Recommendation:** Proceed with P/Invoke approach, but structure tutorial so Chapter 1 provides pre-built bindings (as a black box), and Chapter 0 (appendix) explains how bindings work. This lets readers start compiling FunLang quickly, then optionally dive into FFI details.

---

*Stack research for: MLIR Compiler Backend from F#*
*Researched: 2026-02-05*
*Confidence: MEDIUM (training data only, web verification unavailable)*
