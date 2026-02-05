# Chapter 00: Prerequisites

## Introduction

Welcome to the LangBackend tutorial series. You're here because you've completed the LangTutorial and built a fully functional FunLang interpreter. You have a working parser, type checker with Hindley-Milner type inference, and a tree-walking evaluator. Now it's time to take FunLang to the next level: compiling it to native machine code.

In this tutorial series, you'll learn how to build an MLIR-based compiler backend that transforms typed FunLang ASTs into executable binaries. MLIR (Multi-Level Intermediate Representation) is a modern compiler framework from the LLVM project that provides the infrastructure we need: structured IR operations, type safety, pluggable dialects, and progressive lowering from high-level semantics to machine code.

This chapter covers the essential prerequisite setup: building LLVM/MLIR from source with the C API enabled, installing the .NET SDK for F# development, and configuring your system so the two can communicate. Without these foundations, the rest of the tutorial cannot proceed.

## System Requirements

Before beginning, ensure your system meets these requirements:

- **Disk space:** ~30 GB (LLVM source + build artifacts + installation)
- **RAM:** 16 GB recommended (8 GB minimum with reduced build parallelism)
- **Build time:** 30-60 minutes on modern hardware (4+ cores, SSD)
- **Supported platforms:**
  - Linux (Ubuntu 22.04+, Fedora 38+, or equivalent)
  - macOS (13 Ventura or later, both Intel and Apple Silicon)
  - Windows (WSL2 with Ubuntu 22.04+ recommended; native MSVC build possible but not covered here)

## Building LLVM/MLIR with C API

MLIR is part of the LLVM project. The MLIR team provides a stable C API that allows non-C++ languages like F# to interact with MLIR infrastructure. This C API is not built by default — you must explicitly enable it during the CMake configuration step.

### Installing Build Dependencies

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  clang \
  lld \
  python3 \
  git
```

#### macOS

First, install Xcode Command Line Tools if you haven't already:

```bash
xcode-select --install
```

Then install CMake and Ninja via Homebrew:

```bash
brew install cmake ninja
```

macOS already includes Clang, so you're ready to build.

#### Windows (WSL2)

We recommend using Windows Subsystem for Linux 2 (WSL2) with Ubuntu 22.04. Follow the [WSL2 installation guide](https://learn.microsoft.com/en-us/windows/wsl/install), then use the Linux (Ubuntu) dependency installation steps above.

> **Note:** Native Windows builds with MSVC are possible but require different CMake configurations and are outside this tutorial's scope. WSL2 provides a consistent Linux environment on Windows.

### Cloning LLVM

Clone the LLVM monorepo at the LLVM 19.x stable release branch. Using `--depth 1` saves disk space and download time by fetching only the latest commit:

```bash
cd $HOME
git clone --depth 1 --branch release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
```

The repository is approximately 2 GB after a shallow clone.

### Configuring the Build

The CMake configuration step is critical. Each flag serves a specific purpose:

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install
```

**Flag explanations:**

- `-S llvm`: Source directory (the `llvm` subdirectory in the repo)
- `-B build`: Build directory (out-of-tree build recommended)
- `-G Ninja`: Use Ninja build system (faster than Make)
- `-DCMAKE_BUILD_TYPE=Release`: Optimized build without debug symbols (significantly smaller and faster)
- `-DLLVM_ENABLE_PROJECTS=mlir`: Build MLIR alongside LLVM (MLIR depends on LLVM)
- **`-DMLIR_BUILD_MLIR_C_DYLIB=ON`**: **Critical flag** — builds the `libMLIR-C` shared library exposing the MLIR C API
- `-DLLVM_TARGETS_TO_BUILD="X86;AArch64"`: Build only x86-64 and ARM64 backends (reduces build time; add other targets if needed)
- `-DCMAKE_INSTALL_PREFIX=$HOME/mlir-install`: Install location (use a writable directory)

The CMake configuration should complete in 1-2 minutes. You'll see output like:

```
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
...
-- Build files have been written to: /home/user/llvm-project/build
```

### Building and Installing

Build MLIR with all available CPU cores (Ninja automatically uses parallelism):

```bash
cmake --build build --target install
```

This step takes 30-60 minutes depending on your hardware. You'll see thousands of compilation lines scroll by. If you run out of memory during the build (system becomes unresponsive), stop the build (Ctrl+C) and restart with reduced parallelism:

```bash
cmake --build build --target install -- -j2
```

The `-j2` flag limits Ninja to 2 parallel compile jobs, reducing peak memory usage at the cost of slower build time.

When the build completes, you should see:

```
[100%] Built target install
```

### Verifying the Installation

Check that the MLIR C API shared library was installed:

```bash
ls -lh $HOME/mlir-install/lib/libMLIR-C*
```

**Expected output:**

- **Linux:** `libMLIR-C.so` and `libMLIR-C.so.19` (symlink to versioned library)
- **macOS:** `libMLIR-C.19.dylib` and `libMLIR-C.dylib` (symlink)
- **Windows (WSL):** Same as Linux

If you see `No such file or directory`, verify that you included `-DMLIR_BUILD_MLIR_C_DYLIB=ON` in your CMake configuration and re-run the build step.

You should also have the `mlir-opt` tool installed:

```bash
$HOME/mlir-install/bin/mlir-opt --version
```

Expected output: `MLIR (http://mlir.llvm.org) version 19.1.x`

## Installing .NET SDK

FunLang's compiler backend will be implemented in F#. You need the .NET SDK to compile and run F# programs.

### Linux (Ubuntu/Debian)

Install the .NET 8.0 SDK (LTS release, supported until November 2026):

```bash
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0
```

The script installs .NET to `$HOME/.dotnet`. Add it to your PATH:

```bash
echo 'export PATH="$HOME/.dotnet:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### macOS

Download and install the .NET 8.0 SDK installer from [https://dotnet.microsoft.com/download/dotnet/8.0](https://dotnet.microsoft.com/download/dotnet/8.0), or use Homebrew:

```bash
brew install --cask dotnet-sdk
```

### Windows (WSL2)

Follow the Linux installation steps above within your WSL2 Ubuntu environment.

### Verifying .NET Installation

Check the .NET version:

```bash
dotnet --version
```

Expected output: `8.0.x`

Verify F# compiler is available:

```bash
dotnet fsi --version
```

Expected output: `Microsoft (R) F# Interactive version 12.8.x.0`

Create a test F# project to ensure everything works:

```bash
dotnet new console -lang F# -o test-fsharp
cd test-fsharp
dotnet run
```

You should see:

```
Hello from F#
```

## Setting Up Library Search Paths

When your F# program calls MLIR C API functions via P/Invoke, the .NET runtime must be able to find the `libMLIR-C` shared library at runtime. The standard approach is to add the MLIR installation library directory to the system's library search path.

### Linux

Add the MLIR library directory to `LD_LIBRARY_PATH`:

```bash
echo 'export LD_LIBRARY_PATH="$HOME/mlir-install/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify the library is discoverable:

```bash
ldconfig -p | grep MLIR
```

You should see entries for `libMLIR-C.so`.

### macOS

Add the MLIR library directory to `DYLD_LIBRARY_PATH`:

```bash
echo 'export DYLD_LIBRARY_PATH="$HOME/mlir-install/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc
```

> **Note:** macOS uses zsh by default since macOS Catalina. If you're using bash, modify `~/.bashrc` instead.

Verify the library exists:

```bash
ls -l $HOME/mlir-install/lib/libMLIR-C.dylib
```

### Windows (WSL2)

Follow the Linux instructions above within WSL2.

### Alternative: Per-Project Configuration

Instead of setting global environment variables, you can specify the library path when running your F# application:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet run
```

This is useful for testing without modifying your shell profile.

## Troubleshooting Common Issues

### Out of Memory During Build

**Symptom:** System becomes unresponsive during MLIR build; swap usage at 100%.

**Solution:** Reduce build parallelism:

```bash
cmake --build build --target install -- -j2
```

For systems with 8 GB RAM, `-j1` may be necessary.

### "MLIR-C library not found" Runtime Error

**Symptom:** F# program fails with `DllNotFoundException: Unable to load shared library 'MLIR-C'`.

**Solution:** Verify library search path is configured:

```bash
# Linux
echo $LD_LIBRARY_PATH
# Should include $HOME/mlir-install/lib

# macOS
echo $DYLD_LIBRARY_PATH
```

Ensure the library file exists:

```bash
ls $HOME/mlir-install/lib/libMLIR-C*
```

If missing, rebuild with `-DMLIR_BUILD_MLIR_C_DYLIB=ON`.

### CMake Version Too Old

**Symptom:** CMake configuration fails with `CMake 3.20 or higher is required`.

**Solution:** Install newer CMake:

```bash
# Linux: Download latest CMake binary
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh
sudo sh cmake-3.28.0-linux-x86_64.sh --prefix=/usr/local --skip-license

# macOS
brew upgrade cmake
```

### Missing Ninja Build System

**Symptom:** CMake configuration fails with `Could not find Ninja`.

**Solution:** Install Ninja (see "Installing Build Dependencies" above), or use Unix Makefiles instead (slower):

```bash
cmake -S llvm -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install

make -C build install -j$(nproc)
```

### Disk Space Exhausted

**Symptom:** Build fails with `No space left on device`.

**Solution:** LLVM build requires ~30 GB. Free up space or build on a different partition. You can delete the `build` directory after installation to reclaim ~20 GB:

```bash
rm -rf $HOME/llvm-project/build
```

## What We Built

At this point, you have:

1. **LLVM/MLIR installed** at `$HOME/mlir-install` with the C API shared library (`libMLIR-C.so`, `libMLIR-C.dylib`, or `MLIR-C.dll`)
2. **.NET 8.0 SDK** installed with F# compiler and runtime
3. **Library search paths configured** so .NET can find MLIR at runtime
4. **Verified build tools** ready for development (`mlir-opt`, `dotnet`)

You're now ready to write F# code that interacts with MLIR. In the next chapter, we'll explore the core MLIR concepts you need to understand before writing any code: dialects, operations, regions, blocks, and SSA form.

## Next Chapter

Continue to [Chapter 01: MLIR Primer](01-mlir-primer.md) to learn the fundamental concepts of MLIR IR.
