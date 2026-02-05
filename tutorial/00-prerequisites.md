# Chapter 00: 사전 준비

## 소개

LangBackend 튜토리얼 시리즈에 오신 것을 환영한다. 여러분은 LangTutorial을 완료하고 완전히 동작하는 FunLang 인터프리터를 구축했기 때문에 이 튜토리얼을 시작하게 되었을 것이다. 이미 파서, Hindley-Milner 타입 추론을 갖춘 타입 체커, 그리고 트리 워킹 평가기를 갖추고 있다. 이제 FunLang을 다음 단계로 끌어올릴 차례다: 네이티브 머신 코드로 컴파일하는 것이다.

이 튜토리얼 시리즈에서는 타입이 지정된 FunLang AST를 실행 가능한 바이너리로 변환하는 MLIR 기반 컴파일러 백엔드를 구축하는 방법을 배운다. MLIR(Multi-Level Intermediate Representation)은 LLVM 프로젝트에서 제공하는 현대적인 컴파일러 프레임워크로, 구조화된 IR 연산, 타입 안전성, 플러그인 가능한 dialect, 그리고 고수준 의미론에서 머신 코드까지의 점진적 lowering 등 필요한 인프라를 제공한다.

이 장에서는 필수 사전 준비 설정을 다룬다: C API를 활성화하여 LLVM/MLIR을 소스에서 빌드하고, F# 개발을 위한 .NET SDK를 설치하며, 두 시스템이 통신할 수 있도록 환경을 구성하는 것이다. 이러한 기초가 없으면 나머지 튜토리얼을 진행할 수 없다.

## 시스템 요구 사항

시작하기 전에 시스템이 다음 요구 사항을 충족하는지 확인한다:

- **디스크 공간:** ~30 GB (LLVM 소스 + 빌드 산출물 + 설치)
- **RAM:** 16 GB 권장 (빌드 병렬 처리를 줄이면 최소 8 GB)
- **빌드 시간:** 최신 하드웨어 기준 30-60분 (4코어 이상, SSD)
- **지원 플랫폼:**
  - Linux (Ubuntu 22.04+, Fedora 38+ 또는 이에 상응하는 배포판)
  - macOS (13 Ventura 이상, Intel 및 Apple Silicon 모두 지원)
  - Windows (Ubuntu 22.04+가 설치된 WSL2 권장; 네이티브 MSVC 빌드도 가능하지만 이 튜토리얼에서는 다루지 않는다)

## C API를 포함한 LLVM/MLIR 빌드

MLIR은 LLVM 프로젝트의 일부이다. MLIR 팀은 F#과 같은 비-C++ 언어가 MLIR 인프라와 상호작용할 수 있도록 안정적인 C API를 제공한다. 이 C API는 기본적으로 빌드되지 않으므로 CMake 구성 단계에서 명시적으로 활성화해야 한다.

### 빌드 의존성 설치

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

먼저 Xcode Command Line Tools가 설치되어 있지 않다면 설치한다:

```bash
xcode-select --install
```

그런 다음 Homebrew를 통해 CMake와 Ninja를 설치한다:

```bash
brew install cmake ninja
```

macOS에는 이미 Clang이 포함되어 있으므로 빌드할 준비가 된 것이다.

#### Windows (WSL2)

Ubuntu 22.04가 설치된 Windows Subsystem for Linux 2 (WSL2)를 사용하는 것을 권장한다. [WSL2 설치 가이드](https://learn.microsoft.com/en-us/windows/wsl/install)를 따른 후, 위의 Linux (Ubuntu) 의존성 설치 단계를 사용한다.

> **참고:** MSVC를 사용한 네이티브 Windows 빌드도 가능하지만 다른 CMake 구성이 필요하며 이 튜토리얼의 범위를 벗어난다. WSL2는 Windows에서 일관된 Linux 환경을 제공한다.

### LLVM 클론

LLVM monorepo를 LLVM 19.x 안정 릴리스 브랜치에서 클론한다. `--depth 1`을 사용하면 최신 커밋만 가져와 디스크 공간과 다운로드 시간을 절약할 수 있다:

```bash
cd $HOME
git clone --depth 1 --branch release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
```

shallow clone 후 저장소 크기는 약 2 GB이다.

### 빌드 구성

CMake 구성 단계는 매우 중요하다. 각 플래그는 특정 목적을 가지고 있다:

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install
```

**플래그 설명:**

- `-S llvm`: 소스 디렉터리 (저장소 내의 `llvm` 하위 디렉터리)
- `-B build`: 빌드 디렉터리 (out-of-tree 빌드 권장)
- `-G Ninja`: Ninja 빌드 시스템 사용 (Make보다 빠름)
- `-DCMAKE_BUILD_TYPE=Release`: 디버그 심볼 없이 최적화된 빌드 (크기가 훨씬 작고 빠름)
- `-DLLVM_ENABLE_PROJECTS=mlir`: LLVM과 함께 MLIR 빌드 (MLIR은 LLVM에 의존)
- **`-DMLIR_BUILD_MLIR_C_DYLIB=ON`**: **핵심 플래그** — MLIR C API를 노출하는 `libMLIR-C` 공유 라이브러리를 빌드한다
- `-DLLVM_TARGETS_TO_BUILD="X86;AArch64"`: x86-64 및 ARM64 백엔드만 빌드 (빌드 시간 단축; 필요시 다른 타겟 추가)
- `-DCMAKE_INSTALL_PREFIX=$HOME/mlir-install`: 설치 위치 (쓰기 가능한 디렉터리 사용)

CMake 구성은 1-2분 내에 완료된다. 다음과 같은 출력이 표시된다:

```
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
...
-- Build files have been written to: /home/user/llvm-project/build
```

### 빌드 및 설치

사용 가능한 모든 CPU 코어를 활용하여 MLIR을 빌드한다 (Ninja는 자동으로 병렬 처리를 사용한다):

```bash
cmake --build build --target install
```

이 단계는 하드웨어에 따라 30-60분이 소요된다. 수천 줄의 컴파일 로그가 스크롤된다. 빌드 중 메모리가 부족해지면 (시스템이 응답하지 않는 경우), 빌드를 중지하고 (Ctrl+C) 병렬 처리를 줄여 다시 시작한다:

```bash
cmake --build build --target install -- -j2
```

`-j2` 플래그는 Ninja의 병렬 컴파일 작업을 2개로 제한하여, 빌드 시간이 느려지는 대신 최대 메모리 사용량을 줄인다.

빌드가 완료되면 다음과 같이 표시된다:

```
[100%] Built target install
```

### 설치 확인

MLIR C API 공유 라이브러리가 설치되었는지 확인한다:

```bash
ls -lh $HOME/mlir-install/lib/libMLIR-C*
```

**예상 출력:**

- **Linux:** `libMLIR-C.so` 및 `libMLIR-C.so.19` (버전이 지정된 라이브러리에 대한 심볼릭 링크)
- **macOS:** `libMLIR-C.19.dylib` 및 `libMLIR-C.dylib` (심볼릭 링크)
- **Windows (WSL):** Linux와 동일

`No such file or directory`가 표시되면 CMake 구성에 `-DMLIR_BUILD_MLIR_C_DYLIB=ON`이 포함되어 있는지 확인하고 빌드 단계를 다시 실행한다.

`mlir-opt` 도구도 설치되어 있어야 한다:

```bash
$HOME/mlir-install/bin/mlir-opt --version
```

예상 출력: `MLIR (http://mlir.llvm.org) version 19.1.x`

## .NET SDK 설치

FunLang의 컴파일러 백엔드는 F#으로 구현된다. F# 프로그램을 컴파일하고 실행하려면 .NET SDK가 필요하다.

### Linux (Ubuntu/Debian)

.NET 8.0 SDK (2026년 11월까지 지원되는 LTS 릴리스)를 설치한다:

```bash
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0
```

스크립트는 .NET을 `$HOME/.dotnet`에 설치한다. PATH에 추가한다:

```bash
echo 'export PATH="$HOME/.dotnet:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### macOS

[https://dotnet.microsoft.com/download/dotnet/8.0](https://dotnet.microsoft.com/download/dotnet/8.0)에서 .NET 8.0 SDK 설치 프로그램을 다운로드하여 설치하거나, Homebrew를 사용한다:

```bash
brew install --cask dotnet-sdk
```

### Windows (WSL2)

WSL2 Ubuntu 환경에서 위의 Linux 설치 단계를 따른다.

### .NET 설치 확인

.NET 버전을 확인한다:

```bash
dotnet --version
```

예상 출력: `8.0.x`

F# 컴파일러가 사용 가능한지 확인한다:

```bash
dotnet fsi --version
```

예상 출력: `Microsoft (R) F# Interactive version 12.8.x.0`

모든 것이 정상적으로 작동하는지 확인하기 위해 테스트 F# 프로젝트를 생성한다:

```bash
dotnet new console -lang F# -o test-fsharp
cd test-fsharp
dotnet run
```

다음과 같이 출력되어야 한다:

```
Hello from F#
```

## 라이브러리 검색 경로 설정

F# 프로그램이 P/Invoke를 통해 MLIR C API 함수를 호출할 때, .NET 런타임은 런타임에 `libMLIR-C` 공유 라이브러리를 찾을 수 있어야 한다. 표준적인 방법은 MLIR 설치 라이브러리 디렉터리를 시스템의 라이브러리 검색 경로에 추가하는 것이다.

### Linux

MLIR 라이브러리 디렉터리를 `LD_LIBRARY_PATH`에 추가한다:

```bash
echo 'export LD_LIBRARY_PATH="$HOME/mlir-install/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

라이브러리가 검색 가능한지 확인한다:

```bash
ldconfig -p | grep MLIR
```

`libMLIR-C.so`에 대한 항목이 표시되어야 한다.

### macOS

MLIR 라이브러리 디렉터리를 `DYLD_LIBRARY_PATH`에 추가한다:

```bash
echo 'export DYLD_LIBRARY_PATH="$HOME/mlir-install/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc
```

> **참고:** macOS Catalina 이후 macOS는 기본적으로 zsh를 사용한다. bash를 사용하고 있다면 `~/.bashrc`를 수정한다.

라이브러리가 존재하는지 확인한다:

```bash
ls -l $HOME/mlir-install/lib/libMLIR-C.dylib
```

### Windows (WSL2)

WSL2에서 위의 Linux 지침을 따른다.

### 대안: 프로젝트별 구성

전역 환경 변수를 설정하는 대신, F# 애플리케이션을 실행할 때 라이브러리 경로를 지정할 수 있다:

```bash
LD_LIBRARY_PATH=$HOME/mlir-install/lib dotnet run
```

이 방법은 셸 프로파일을 수정하지 않고 테스트할 때 유용하다.

## 자주 발생하는 문제 해결

### 빌드 중 메모리 부족

**증상:** MLIR 빌드 중 시스템이 응답하지 않음; 스왑 사용량이 100%.

**해결 방법:** 빌드 병렬 처리를 줄인다:

```bash
cmake --build build --target install -- -j2
```

RAM이 8 GB인 시스템에서는 `-j1`이 필요할 수 있다.

### "MLIR-C library not found" 런타임 오류

**증상:** F# 프로그램이 `DllNotFoundException: Unable to load shared library 'MLIR-C'`로 실패한다.

**해결 방법:** 라이브러리 검색 경로가 구성되어 있는지 확인한다:

```bash
# Linux
echo $LD_LIBRARY_PATH
# $HOME/mlir-install/lib이 포함되어 있어야 합니다

# macOS
echo $DYLD_LIBRARY_PATH
```

라이브러리 파일이 존재하는지 확인한다:

```bash
ls $HOME/mlir-install/lib/libMLIR-C*
```

파일이 없다면 `-DMLIR_BUILD_MLIR_C_DYLIB=ON`으로 다시 빌드한다.

### CMake 버전이 너무 오래됨

**증상:** CMake 구성이 `CMake 3.20 or higher is required`로 실패한다.

**해결 방법:** 최신 CMake를 설치한다:

```bash
# Linux: 최신 CMake 바이너리 다운로드
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh
sudo sh cmake-3.28.0-linux-x86_64.sh --prefix=/usr/local --skip-license

# macOS
brew upgrade cmake
```

### Ninja 빌드 시스템 누락

**증상:** CMake 구성이 `Could not find Ninja`로 실패한다.

**해결 방법:** Ninja를 설치하거나 (위의 "빌드 의존성 설치" 참조), 대신 Unix Makefiles를 사용한다 (더 느림):

```bash
cmake -S llvm -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install

make -C build install -j$(nproc)
```

### 디스크 공간 부족

**증상:** 빌드가 `No space left on device`로 실패한다.

**해결 방법:** LLVM 빌드에는 ~30 GB가 필요하다. 공간을 확보하거나 다른 파티션에서 빌드한다. 설치 후 `build` 디렉터리를 삭제하면 ~20 GB를 회수할 수 있다:

```bash
rm -rf $HOME/llvm-project/build
```

## 이 장에서 완료한 것

이 시점에서 다음 항목이 준비되어 있다:

1. **LLVM/MLIR 설치 완료** — `$HOME/mlir-install`에 C API 공유 라이브러리(`libMLIR-C.so`, `libMLIR-C.dylib`, 또는 `MLIR-C.dll`) 포함
2. **.NET 8.0 SDK** — F# 컴파일러 및 런타임과 함께 설치 완료
3. **라이브러리 검색 경로 구성 완료** — .NET이 런타임에 MLIR을 찾을 수 있도록 설정
4. **빌드 도구 검증 완료** — 개발 준비 완료 (`mlir-opt`, `dotnet`)

이제 MLIR과 상호작용하는 F# 코드를 작성할 준비가 되었다. 다음 장에서는 코드를 작성하기 전에 이해해야 할 핵심 MLIR 개념들을 살펴본다: dialect, operation, region, block, 그리고 SSA 형식이다.

## 다음 장

[Chapter 01: MLIR 입문](01-mlir-primer.md)으로 이동하여 MLIR IR의 기본 개념을 학습한다.
