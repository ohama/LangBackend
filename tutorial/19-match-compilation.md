# Chapter 19: Match Compilation (Match Compilation)

## 소개

**Chapter 17**에서는 패턴 매칭의 **이론적 기반**을 다뤘다:
- Decision tree 알고리즘 (Maranget 2008)
- Pattern matrix와 specialization/defaulting 연산
- Exhaustiveness checking과 unreachable case detection

**Chapter 18**에서는 패턴 매칭이 작동할 **데이터 구조**를 구현했다:
- `!funlang.list<T>` parameterized type
- `funlang.nil`과 `funlang.cons` operations
- TypeConverter로 tagged union 변환
- NilOpLowering과 ConsOpLowering patterns

**Chapter 19**에서는 모든 것을 종합하여 **패턴 매칭 컴파일**을 완성한다. `funlang.match` operation을 정의하고 SCF dialect로 lowering하여 실행 가능한 코드를 생성한다.

### 두 장의 복습: 왜 Match Operation이 필요한가?

Chapter 17에서 우리는 decision tree 알고리즘을 배웠다:

```fsharp
// F# 패턴 매칭 예제
let rec sum_list lst =
    match lst with
    | [] -> 0
    | head :: tail -> head + sum_list tail

sum_list [1; 2; 3]  // 6
```

Decision tree 컴파일 결과:

```
Switch on lst:
  Case Nil -> return 0
  Case Cons(head, tail) -> return head + sum_list tail
```

Chapter 18에서 우리는 리스트 데이터 구조를 구현했다:

```mlir
// Empty list
%empty = funlang.nil : !funlang.list<i32>

// List construction: [1, 2, 3]
%three = arith.constant 3 : i32
%t3 = funlang.nil : !funlang.list<i32>
%l3 = funlang.cons %three, %t3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

%two = arith.constant 2 : i32
%l2 = funlang.cons %two, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

%one = arith.constant 1 : i32
%l1 = funlang.cons %one, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>
```

**이제 이 두 가지를 연결할 방법이 필요하다:**

```mlir
// 목표: sum_list를 MLIR로 표현
func.func @sum_list(%lst: !funlang.list<i32>) -> i32 {
  %result = funlang.match %lst : !funlang.list<i32> -> i32 {
    ^nil:
      %zero = arith.constant 0 : i32
      funlang.yield %zero : i32
    ^cons(%head: i32, %tail: !funlang.list<i32>):
      %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
      %sum = arith.addi %head, %tail_sum : i32
      funlang.yield %sum : i32
  }
  return %result : i32
}
```

### funlang.match: The Most Complex Operation

**왜 `funlang.match`가 가장 복잡한가?**

지금까지 우리가 구현한 FunLang operations:

| Operation | Complexity | Why |
|-----------|-----------|-----|
| `funlang.nil` | Simple | Zero arguments, constant value |
| `funlang.cons` | Moderate | Two operands, GC allocation |
| `funlang.closure` | Moderate | Function ref + captures, GC allocation |
| `funlang.apply` | Moderate | Indirect call, block arguments |
| **`funlang.match`** | **Complex** | **Multiple regions, block arguments, type conversion** |

**`funlang.match`의 복잡성:**

1. **Region-based structure**: 각 case가 별도의 region (not just basic block)
2. **Variable number of cases**: Nil/Cons 2개부터 임의의 pattern 개수까지
3. **Block arguments per case**: Cons case는 `(%head, %tail)` 같은 바인딩 필요
4. **Type conversion in regions**: 각 region 내부의 operations도 lowering 필요
5. **Multi-stage lowering**: FunLang → SCF → CF → LLVM

**Chapter 15 Preview 복습:**

Chapter 15에서 우리는 `funlang.match`를 미리 살짝 봤다:

```tablegen
// Chapter 15의 preview (간략 버전)
def FunLang_MatchOp : FunLang_Op<"match"> {
  let summary = "Pattern matching operation";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let regions = (region VariadicRegion<SizedRegion<1>>:$cases);
}
```

**Chapter 19에서는 완전한 버전을 구현한다:**

- Full TableGen definition with verification
- Custom assembly format (parser/printer)
- C API shim for region-based operation
- F# bindings with builder callback
- Lowering pattern to SCF dialect

### Multi-Stage Lowering: FunLang → SCF → LLVM

**왜 SCF dialect를 거치는가?**

Phase 5에서 우리는 FunLang operations를 직접 LLVM dialect로 lowering했다:

```
funlang.closure → llvm.alloca + llvm.store  (direct lowering)
funlang.apply   → llvm.load + llvm.call     (direct lowering)
```

**하지만 `funlang.match`는 다르다:**

```
funlang.match → scf.index_switch → cf.switch → llvm.switch
              (structured)        (CFG)       (machine)
```

**이유:**

1. **Structured control flow preservation**: SCF는 high-level structure 유지
2. **Optimization opportunities**: SCF level에서 최적화 가능 (dead case elimination, etc.)
3. **Debugging**: SCF IR이 source 구조를 반영하여 디버깅 쉬움
4. **Separation of concerns**: Pattern matching logic과 low-level branching 분리

**SCF Dialect란?**

SCF = **Structured Control Flow**

MLIR의 standard dialect 중 하나로, high-level control flow operations 제공:

```mlir
// scf.if: two-way branching (Chapter 8에서 사용)
%result = scf.if %cond : i1 -> i32 {
  %x = arith.constant 42 : i32
  scf.yield %x : i32
} else {
  %y = arith.constant 0 : i32
  scf.yield %y : i32
}

// scf.index_switch: multi-way branching (Chapter 19에서 사용)
%result = scf.index_switch %tag : index -> i32
case 0 {
  %zero = arith.constant 0 : i32
  scf.yield %zero : i32
}
case 1 {
  %one = arith.constant 1 : i32
  scf.yield %one : i32
}
default {
  %minus = arith.constant -1 : i32
  scf.yield %minus : i32
}
```

**SCF vs CF (Control Flow) dialect:**

| Dialect | Level | Structure | When |
|---------|-------|-----------|------|
| SCF | High-level | Structured (nested regions) | Pattern matching, loops |
| CF | Low-level | Unstructured (goto-like) | After SCF lowering |

**Complete lowering pipeline:**

```
FunLang Dialect
    ↓ (FunLangToSCFPass)
SCF + FunLang (partially lowered)
    ↓ (FunLangToLLVMPass - for nil/cons/closure/apply)
SCF + LLVM
    ↓ (SCFToControlFlowPass)
CF + LLVM
    ↓ (ControlFlowToLLVMPass)
LLVM Dialect only
    ↓ (LLVMToObjectPass)
Machine code
```

### Chapter 19 Goals

**이 장에서 배울 것:**

1. **Match Operation Definition** (Part 1)
   - Region-based operation structure
   - TableGen definition with VariadicRegion
   - Custom verifier for region semantics
   - YieldOp terminator for match results
   - C API shim for region-based operations
   - F# bindings with builder callback pattern

2. **SCF Lowering** (Part 2)
   - SCF dialect overview and `scf.index_switch`
   - MatchOpLowering pattern implementation
   - Region cloning and type conversion
   - Block argument remapping
   - Common errors and debugging strategies

3. **End-to-End Example**
   - length function: complete compilation pipeline
   - Stage-by-stage IR transformation
   - Performance comparison vs naive approach

**Success criteria:**

- ✅ `funlang.match` operation defined and verified
- ✅ Lowering to `scf.index_switch` working
- ✅ Pattern variables bound via block arguments
- ✅ End-to-end compilation of recursive list functions

Let's begin!

---

## Part 1: Match Operation Definition

### Region-Based Operations: The Foundation

**Region이란 무엇인가?**

MLIR에서 **region**은 **basic blocks의 container**다.

```
Region
  ├─ Block 1 (entry block)
  │   ├─ Operation 1
  │   ├─ Operation 2
  │   └─ Terminator
  ├─ Block 2
  │   └─ ...
  └─ Block N
```

**우리가 이미 본 region-based operations:**

Chapter 8에서 `scf.if`:

```mlir
scf.if %cond : i1 -> i32 {
  // "then" region (1 block)
  %x = arith.constant 42 : i32
  scf.yield %x : i32
} else {
  // "else" region (1 block)
  %y = arith.constant 0 : i32
  scf.yield %y : i32
}
```

- `scf.if`는 2개의 regions (then, else)
- 각 region은 exactly 1 block
- 각 block은 `scf.yield` terminator로 끝남

Chapter 10에서 `func.func`:

```mlir
func.func @my_function(%arg: i32) -> i32 {
  // function body region (1 or more blocks)
  %result = arith.addi %arg, %arg : i32
  return %result : i32
}
```

- `func.func`는 1개의 region (body)
- Region은 1개 이상의 blocks (control flow로 여러 block 가능)
- Entry block은 function arguments as block arguments

**왜 basic blocks이 아니라 regions인가?**

**Scenario: match expression with 3 cases**

```fsharp
// F# code
match shape with
| Circle r -> compute_circle_area r
| Rectangle (w, h) -> compute_rectangle_area w h
| Triangle (a, b, c) -> compute_triangle_area a b c
```

**Option 1: Basic blocks (NOT what we do)**

```mlir
// 잘못된 접근: basic blocks only
func.func @match_shape(%shape: !funlang.shape) -> f32 {
  // ... tag extraction ...
  cf.br ^dispatch

^dispatch:
  cf.switch %tag [
    ^circle,
    ^rectangle,
    ^triangle
  ]

^circle:
  // Circle case logic
  cf.br ^exit

^rectangle:
  // Rectangle case logic
  cf.br ^exit

^triangle:
  // Triangle case logic
  cf.br ^exit

^exit(%result: f32):
  return %result : f32
}
```

**문제점:**

1. **All blocks in same scope**: ^circle, ^rectangle, ^triangle은 모두 같은 function body region
2. **No encapsulation**: Case logic이 function CFG에 섞임
3. **Hard to verify**: "각 case가 정확히 1개의 yield를 가지는가?" 검증 어려움
4. **Type conversion complexity**: Lowering pass가 case blocks을 구분하기 어려움

**Option 2: Regions (What we do)**

```mlir
// 올바른 접근: regions
func.func @match_shape(%shape: !funlang.shape) -> f32 {
  %result = funlang.match %shape : !funlang.shape -> f32 {
    ^circle(%r: f32):
      %area = call @compute_circle_area(%r) : (f32) -> f32
      funlang.yield %area : f32
    ^rectangle(%w: f32, %h: f32):
      %area = call @compute_rectangle_area(%w, %h) : (f32, f32) -> f32
      funlang.yield %area : f32
    ^triangle(%a: f32, %b: f32, %c: f32):
      %area = call @compute_triangle_area(%a, %b, %c) : (f32, f32, f32) -> f32
      funlang.yield %area : f32
  }
  return %result : f32
}
```

**장점:**

1. **Encapsulation**: 각 case가 자신만의 region (isolated scope)
2. **Clear structure**: match operation이 모든 cases를 소유
3. **Easy verification**: 각 region은 정확히 1 block, 1 terminator
4. **Lowering-friendly**: Region 단위로 type conversion 수행 가능

**Region vs Block vs Operation:**

```
Operation: funlang.match
  ↓ has
Regions: [case 1 region, case 2 region, ...]
  ↓ each contains
Blocks: [entry block]
  ↓ contains
Operations: [arith.constant, func.call, funlang.yield, ...]
```

### Match Operation Semantics

**`funlang.match`의 의미론:**

```mlir
%result = funlang.match %input : InputType -> ResultType {
  ^case1(...pattern_vars1...):
    // case 1 logic
    funlang.yield %value1 : ResultType
  ^case2(...pattern_vars2...):
    // case 2 logic
    funlang.yield %value2 : ResultType
  ...
}
```

**Execution semantics:**

1. **Input evaluation**: `%input` 값을 runtime에 evaluate
2. **Tag extraction**: Tagged union에서 tag value 추출
3. **Case selection**: Tag에 따라 해당 region 선택
4. **Pattern variable binding**: Region의 block arguments에 values 바인딩
5. **Case execution**: 선택된 region 실행
6. **Result yielding**: Region의 `funlang.yield`가 `%result`에 값 전달

**Example: sum_list**

```mlir
func.func @sum_list(%lst: !funlang.list<i32>) -> i32 {
  %result = funlang.match %lst : !funlang.list<i32> -> i32 {
    ^nil:
      %zero = arith.constant 0 : i32
      funlang.yield %zero : i32
    ^cons(%head: i32, %tail: !funlang.list<i32>):
      %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
      %sum = arith.addi %head, %tail_sum : i32
      funlang.yield %sum : i32
  }
  return %result : i32
}
```

**Runtime execution: sum_list([1, 2])**

1. **Call**: `@sum_list([1, 2])`
2. **Tag extraction**: Tag = 1 (Cons)
3. **Case selection**: ^cons region
4. **Variable binding**: `%head = 1`, `%tail = [2]`
5. **Recursive call**: `@sum_list([2])`
   - Tag = 1 (Cons)
   - `%head = 2`, `%tail = []`
   - Recursive call: `@sum_list([])`
     - Tag = 0 (Nil)
     - Return 0
   - `%sum = 2 + 0 = 2`
   - Return 2
6. **Final sum**: `1 + 2 = 3`
7. **Return**: 3

### Block Arguments for Pattern Variables

**패턴 변수는 어떻게 바인딩되는가?**

Chapter 2에서 우리는 **block arguments**를 배웠다:

```mlir
^entry_block(%arg0: i32, %arg1: i32):
  %sum = arith.addi %arg0, %arg1 : i32
```

Block arguments는 PHI nodes의 structured 대안이다.

**Match operation에서 block arguments 활용:**

```mlir
funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:
    // Nil case: 패턴 변수 없음 → block arguments 없음
    funlang.yield %zero : i32

  ^cons(%head: i32, %tail: !funlang.list<i32>):
    // Cons case: 2개 패턴 변수 → 2개 block arguments
    // %head: i32          → cons cell의 head field
    // %tail: !funlang.list<i32> → cons cell의 tail field
    funlang.yield %sum : i32
}
```

**Lowering이 block arguments를 채우는 방법:**

```mlir
// funlang.match lowering 후 (pseudo-code)
%tag = // extract tag from %lst
scf.index_switch %tag {
  case 0 {  // Nil case
    // No data to extract, no arguments
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
  }
  case 1 {  // Cons case
    // Extract head and tail from cons cell
    %head = // extract field 0 from data pointer
    %tail = // extract field 1 from data pointer
    // Now pass to the ^cons block's body (with arguments bound)
    ^cons(%head, %tail):
      // User code here
  }
}
```

**실제로는 region을 clone하고 IRMapping으로 arguments를 remap한다** (Part 2에서 자세히)

**Block arguments vs Let bindings:**

```mlir
// Option 1: Block arguments (what we do)
^cons(%head: i32, %tail: !funlang.list<i32>):
  %sum = arith.addi %head, ... : i32

// Option 2: Let-style extraction (what we DON'T do)
^cons:
  %head = funlang.extract_head %lst : !funlang.list<i32> -> i32
  %tail = funlang.extract_tail %lst : !funlang.list<i32> -> !funlang.list<i32>
  %sum = arith.addi %head, ... : i32
```

**Block arguments가 더 나은 이유:**

1. **Declarative**: Pattern structure가 arguments에 직접 반영
2. **SSA-friendly**: Block entry에서 values가 이미 available
3. **No redundant ops**: extract operations 불필요
4. **Verification**: Argument types로 pattern structure 검증 가능

### TableGen Definition: MatchOp

이제 `funlang.match` operation의 TableGen 정의를 작성한다.

**File: `FunLang/FunLangOps.td`** (conceptual, 실제로는 C++ codebase)

```tablegen
def FunLang_MatchOp : FunLang_Op<"match", [
    RecursiveSideEffect,
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Pattern matching operation";
  let description = [{
    The `funlang.match` operation performs pattern matching on a value.
    Each case is represented as a separate region with exactly one block.

    The entry block of each region may have arguments corresponding to
    pattern variables. For example, a Cons case has two arguments:
    the head element and the tail list.

    Each region must terminate with a `funlang.yield` operation that
    returns a value of the result type.

    Example:
    ```mlir
    %result = funlang.match %lst : !funlang.list<i32> -> i32 {
      ^nil:
        %zero = arith.constant 0 : i32
        funlang.yield %zero : i32
      ^cons(%head: i32, %tail: !funlang.list<i32>):
        %sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
        %result = arith.addi %head, %sum : i32
        funlang.yield %result : i32
    }
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let regions = (region VariadicRegion<SizedRegion<1>>:$cases);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
```

**핵심 요소 설명:**

**1. Traits: RecursiveSideEffect**

```tablegen
RecursiveSideEffect
```

**의미:** 이 operation의 side effects는 내부 regions의 operations에 의존한다.

**왜 필요한가?**

MLIR optimizer는 side effects를 분석하여 dead code elimination, common subexpression elimination 등을 수행한다.

- `funlang.nil`은 `Pure` trait → no side effects
- `funlang.cons`는 side effects 있음 (GC allocation)

**Match operation은?**

```mlir
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:
    %x = funlang.nil : !funlang.list<i32>  // Pure
    funlang.yield %zero : i32
  ^cons(%h, %t):
    %y = funlang.cons %h, %t : ...  // Side effect!
    funlang.yield %sum : i32
}
```

- Nil case: no side effects
- Cons case: side effect (funlang.cons)

**RecursiveSideEffect trait는 MLIR에게 말한다:**

"이 operation의 side effects는 내부 regions을 재귀적으로 분석해서 결정해라"

**없으면 어떻게 되나?**

- Conservative assumption: match는 항상 side effects 있음
- Optimizer가 legitimate optimizations를 못함
- 예: dead match elimination 불가

**2. Traits: SingleBlockImplicitTerminator**

```tablegen
SingleBlockImplicitTerminator<"YieldOp">
```

**의미:** 각 region은 정확히 1개의 block을 가지며, 그 block은 `YieldOp`로 끝나야 한다.

**검증 자동화:**

이 trait가 있으면 MLIR이 자동으로 검증:

1. 각 region이 정확히 1 block인가?
2. 그 block이 `funlang.yield`로 끝나는가?

**없으면 어떻게 되나?**

Custom verifier에서 수동 검증 필요:

```cpp
// Without the trait (manual verification)
LogicalResult MatchOp::verify() {
  for (Region& region : getCases()) {
    if (!region.hasOneBlock()) {
      return emitError("each case must have exactly one block");
    }
    Block& block = region.front();
    if (!isa<YieldOp>(block.getTerminator())) {
      return emitError("each case must terminate with funlang.yield");
    }
  }
  return success();
}
```

Trait가 이 boilerplate를 제거한다!

**3. Regions: VariadicRegion<SizedRegion<1>>**

```tablegen
let regions = (region VariadicRegion<SizedRegion<1>>:$cases);
```

**분해:**

- `VariadicRegion`: 가변 개수의 regions (Nil/Cons = 2개, 더 많은 patterns = N개)
- `SizedRegion<1>`: 각 region은 정확히 1개의 block
- `:$cases`: C++ accessor name → `getCases()` method

**대안들과 비교:**

| Declaration | Meaning |
|-------------|---------|
| `region AnyRegion:$body` | Exactly 1 region, any number of blocks |
| `region SizedRegion<1>:$body` | Exactly 1 region, exactly 1 block |
| `region VariadicRegion<AnyRegion>:$cases` | N regions, each with any blocks |
| `region VariadicRegion<SizedRegion<1>>:$cases` | N regions, each with 1 block ✅ |

**scf.if와 비교:**

```tablegen
// scf.if (exactly 2 regions)
def SCF_IfOp : ... {
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);
}

// funlang.match (variable number of regions)
def FunLang_MatchOp : ... {
  let regions = (region VariadicRegion<SizedRegion<1>>:$cases);
}
```

**4. Custom Assembly Format**

```tablegen
let hasCustomAssemblyFormat = 1;
```

**이유:** Generic format은 readable하지 않다.

**Generic format (자동 생성):**

```mlir
%result = "funlang.match"(%lst) ({
  ^bb0:
    %zero = arith.constant 0 : i32
    "funlang.yield"(%zero) : (i32) -> ()
}, {
  ^bb0(%head: i32, %tail: !funlang.list<i32>):
    %sum = arith.addi %head, %tail_sum : i32
    "funlang.yield"(%sum) : (i32) -> ()
}) : (!funlang.list<i32>) -> i32
```

**Custom format (우리가 작성):**

```mlir
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32
  ^cons(%head: i32, %tail: !funlang.list<i32>):
    %sum = arith.addi %head, %tail_sum : i32
    funlang.yield %sum : i32
}
```

**Custom parser/printer 구현 필요 (C++ code):**

```cpp
// File: FunLangOps.cpp

void MatchOp::print(OpAsmPrinter& p) {
  p << " " << getInput() << " : " << getInput().getType()
    << " -> " << getResult().getType() << " ";

  p.printRegion(getCases(), /*printEntryBlockArgs=*/true);
}

ParseResult MatchOp::parse(OpAsmParser& parser, OperationState& result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType, resultType;
  Region* casesRegion = result.addRegion();

  if (parser.parseOperand(input) ||
      parser.parseColon() ||
      parser.parseType(inputType) ||
      parser.parseArrow() ||
      parser.parseType(resultType) ||
      parser.parseRegion(*casesRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  result.addTypes(resultType);
  return success();
}
```

*실제 구현은 더 복잡하지만, F# tutorial에서는 C API로 추상화됨*

**5. Custom Verifier**

```tablegen
let hasVerifier = 1;
```

**검증할 내용:**

1. ✅ Region count > 0
2. ✅ 각 region의 block arguments types 검증
3. ✅ 각 region의 yield type이 result type과 일치
4. ✅ Input type이 matchable type (현재는 !funlang.list<T>)

**C++ verifier implementation:**

```cpp
// File: FunLangOps.cpp

LogicalResult MatchOp::verify() {
  // Check: at least one case
  if (getCases().empty()) {
    return emitError("match must have at least one case");
  }

  Type resultType = getResult().getType();

  // Check each case region
  for (Region& region : getCases()) {
    if (region.empty())
      return emitError("case region cannot be empty");

    Block& block = region.front();

    // Verify terminator (already checked by SingleBlockImplicitTerminator)
    auto yieldOp = dyn_cast<YieldOp>(block.getTerminator());
    if (!yieldOp)
      return emitError("case must terminate with funlang.yield");

    // Verify yield type matches result type
    if (yieldOp.getValue().getType() != resultType) {
      return emitError("yield type ")
             << yieldOp.getValue().getType()
             << " does not match result type " << resultType;
    }
  }

  return success();
}
```

**실전 예제:**

```mlir
// ERROR: No cases
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
}
// Error: match must have at least one case

// ERROR: Type mismatch
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:
    %x = arith.constant 3.14 : f32  // Wrong type!
    funlang.yield %x : f32
}
// Error: yield type f32 does not match result type i32
```

### YieldOp: Match Result Terminator

**각 match case는 `funlang.yield`로 끝나야 한다.**

**TableGen definition:**

```tablegen
def FunLang_YieldOp : FunLang_Op<"yield", [
    Terminator,
    HasParent<"MatchOp">
  ]> {
  let summary = "Yield a value from a match case";
  let description = [{
    The `funlang.yield` operation terminates a match case region and
    returns a value to the parent `funlang.match` operation.

    Example:
    ```mlir
    funlang.match %lst : !funlang.list<i32> -> i32 {
      ^nil:
        %zero = arith.constant 0 : i32
        funlang.yield %zero : i32  // Yield from nil case
      ^cons(%h, %t):
        %sum = arith.addi %h, ... : i32
        funlang.yield %sum : i32   // Yield from cons case
    }
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}
```

**핵심 요소:**

**1. Trait: Terminator**

```tablegen
Terminator
```

**의미:** 이 operation은 basic block을 종료한다.

**Block의 terminator 규칙:**

- 모든 block은 정확히 1개의 terminator로 끝나야 함
- Terminator는 block의 마지막 operation이어야 함
- Terminator 예: `func.return`, `cf.br`, `scf.yield`, `funlang.yield`

**2. Trait: HasParent<"MatchOp">**

```tablegen
HasParent<"MatchOp">
```

**의미:** 이 operation은 `MatchOp`의 region 내에서만 사용 가능.

**검증 자동화:**

```mlir
// OK: inside funlang.match
funlang.match %lst {
  ^nil:
    funlang.yield %zero : i32  // ✅
}

// ERROR: outside match
func.func @wrong() -> i32 {
  %x = arith.constant 42 : i32
  funlang.yield %x : i32  // ❌ Error: funlang.yield must be inside MatchOp
}
```

**3. Assembly Format**

```tablegen
let assemblyFormat = "$value attr-dict `:` type($value)";
```

**생성되는 format:**

```mlir
funlang.yield %sum : i32
```

**Generic format과 비교:**

```mlir
// Generic (verbose)
"funlang.yield"(%sum) : (i32) -> ()

// Custom (readable)
funlang.yield %sum : i32
```

### scf.yield와 비교

**MLIR에는 여러 yield operations이 있다:**

| Operation | Parent | Purpose |
|-----------|--------|---------|
| `scf.yield` | `scf.if`, `scf.for`, `scf.while` | SCF control flow |
| `funlang.yield` | `funlang.match` | FunLang pattern matching |
| `affine.yield` | `affine.for`, `affine.if` | Affine loops |

**왜 `scf.yield`를 재사용하지 않는가?**

**Option 1: 재사용 (하지 않음)**

```mlir
funlang.match %lst {
  ^nil:
    scf.yield %zero : i32  // Reuse scf.yield?
}
```

**문제:**

1. **Trait conflict**: `scf.yield`는 `HasParent<"IfOp", "ForOp", ...>`
   - `MatchOp`이 parent list에 없으면 verifier 실패
   - SCF dialect 수정 필요 (bad coupling)

2. **Semantic confusion**: `scf.yield`는 SCF dialect semantics
   - Lowering pass에서 `scf.yield` 처리 시 match context 고려해야 함
   - Separation of concerns 위반

**Option 2: 전용 operation (우리가 하는 것)**

```mlir
funlang.match %lst {
  ^nil:
    funlang.yield %zero : i32  // FunLang-specific yield
}
```

**장점:**

1. **Clear ownership**: FunLang dialect이 자신의 terminators 소유
2. **Lowering flexibility**: `funlang.yield` → `scf.yield` 변환을 명시적으로 제어
3. **Future extensions**: 나중에 `funlang.yield`에 attributes 추가 가능

### C API and F# Integration

**Region-based operations는 C API 설계가 복잡하다.**

**문제: Regions를 어떻게 F#에서 구축하는가?**

**Simple operations (Chapter 15):**

```fsharp
// funlang.cons: no regions, straightforward
let cons =
    FunLangOps.CreateConsOp(builder, head, tail, listType)
```

**Region-based operations (Chapter 19):**

```fsharp
// funlang.match: multiple regions, complex
let matchOp = FunLangOps.CreateMatchOp(builder, input, resultType, [
    // How to build regions here???
    nilRegion;
    consRegion
])
```

**Challenge:**

- Region 구축은 F# side에서 일어나야 함 (pattern cases logic)
- 하지만 MLIR C++ API를 직접 호출할 수 없음 (C API만 가능)
- Builder callback pattern 필요!

### C API Shim: Builder Callback Pattern

**Pattern: C API가 F# callback을 받아서 region을 채움**

**C API shim (C wrapper):**

```c
// File: FunLang-C/FunLangOps.h

typedef void (*FunLangMatchCaseBuilder)(
    MlirOpBuilder builder,
    MlirBlock block,
    void* userData
);

typedef struct {
    FunLangMatchCaseBuilder builder;
    void* userData;
} FunLangMatchCase;

MLIR_CAPI_EXPORTED MlirOperation funlangMatchOpCreate(
    MlirOpBuilder builder,
    MlirLocation loc,
    MlirValue input,
    MlirType resultType,
    FunLangMatchCase* cases,
    intptr_t numCases
);
```

**Implementation (C++):**

```cpp
// File: FunLang-C/FunLangOps.cpp

MlirOperation funlangMatchOpCreate(
    MlirOpBuilder builder,
    MlirLocation loc,
    MlirValue input,
    MlirType resultType,
    FunLangMatchCase* cases,
    intptr_t numCases
) {
  OpBuilder& cppBuilder = unwrap(builder);
  Location cppLoc = unwrap(loc);
  Value cppInput = unwrap(input);
  Type cppResultType = unwrap(resultType);

  // Create match operation
  auto matchOp = cppBuilder.create<MatchOp>(
      cppLoc, cppResultType, cppInput, numCases);

  // Build each case region
  for (intptr_t i = 0; i < numCases; ++i) {
    Region& region = matchOp.getCases()[i];
    Block* block = cppBuilder.createBlock(&region);

    // Invoke F# callback to populate the block
    MlirBlock wrappedBlock = wrap(block);
    cases[i].builder(builder, wrappedBlock, cases[i].userData);
  }

  return wrap(matchOp.getOperation());
}
```

**핵심 아이디어:**

1. C API가 empty regions를 가진 `MatchOp` 생성
2. 각 region에 대해 F# callback 호출
3. F# callback이 region의 block을 채움 (operations + yield)

### F# Bindings

**Low-level binding:**

```fsharp
// File: FunLang.Interop/FunLangOps.fs

type MatchCaseBuilder =
    MlirOpBuilder -> MlirBlock -> nativeint -> unit

[<Struct>]
type MatchCase =
    val Builder: MatchCaseBuilder
    val UserData: nativeint

    new(builder, userData) =
        { Builder = builder; UserData = userData }

[<DllImport("FunLang-C", CallingConvention = CallingConvention.Cdecl)>]
extern MlirOperation funlangMatchOpCreate(
    MlirOpBuilder builder,
    MlirLocation loc,
    MlirValue input,
    MlirType resultType,
    MatchCase[] cases,
    nativeint numCases
)
```

**High-level wrapper:**

```fsharp
// File: FunLang.Compiler/OpBuilder.fs

type OpBuilder with
    member this.CreateMatchOp(
        input: MlirValue,
        resultType: MlirType,
        buildCases: (OpBuilder -> Block -> unit) list
    ) : MlirOperation =

        // Convert F# functions to C callbacks
        let cases =
            buildCases
            |> List.map (fun buildCase ->
                let callback builder block userData =
                    let opBuilder = new OpBuilder(builder)
                    let mlirBlock = new Block(block)
                    buildCase opBuilder mlirBlock

                MatchCase(callback, 0n)
            )
            |> List.toArray

        let numCases = nativeint cases.Length
        let loc = this.UnknownLoc()

        funlangMatchOpCreate(
            this.Handle,
            loc,
            input,
            resultType,
            cases,
            numCases
        )
```

**사용 예제 (F# compiler code):**

```fsharp
// File: FunLang.Compiler/Codegen.fs

let compileMatch (builder: OpBuilder) (scrutinee: MlirValue) (cases: MatchCase list) =
    let resultType = // infer from cases

    let buildCases =
        cases |> List.map (fun case ->
            fun (builder: OpBuilder) (block: Block) ->
                match case with
                | NilCase expr ->
                    // Build nil case body
                    let value = compileExpr builder env expr
                    builder.CreateYieldOp(value) |> ignore

                | ConsCase (headVar, tailVar, expr) ->
                    // Add block arguments for head and tail
                    let headType = builder.GetIntegerType(32)
                    let tailType = builder.GetFunLangListType(headType)
                    block.AddArgument(headType) |> ignore
                    block.AddArgument(tailType) |> ignore

                    // Build cons case body with extended environment
                    let env' =
                        env
                        |> Map.add headVar (block.GetArgument(0))
                        |> Map.add tailVar (block.GetArgument(1))

                    let value = compileExpr builder env' expr
                    builder.CreateYieldOp(value) |> ignore
        )

    builder.CreateMatchOp(scrutinee, resultType, buildCases)
```

**Generated MLIR:**

```mlir
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32
  ^cons(%head: i32, %tail: !funlang.list<i32>):
    %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
    %sum = arith.addi %head, %tail_sum : i32
    funlang.yield %sum : i32
}
```

**Builder callback pattern의 장점:**

1. **Flexibility**: F# code가 region 내용을 완전히 제어
2. **Type safety**: F# compiler가 callback signature 검증
3. **Composability**: Nested match expressions 지원 (callback 안에서 또 match 생성)

### Block Arguments in Builder Callback

**위 코드에서 중요한 부분:**

```fsharp
| ConsCase (headVar, tailVar, expr) ->
    // Add block arguments for pattern variables
    block.AddArgument(headType) |> ignore
    block.AddArgument(tailType) |> ignore

    // Use block arguments in environment
    let env' =
        env
        |> Map.add headVar (block.GetArgument(0))
        |> Map.add tailVar (block.GetArgument(1))
```

**F# callback이 하는 일:**

1. **Pattern structure 분석**: ConsCase는 2개 변수 (head, tail)
2. **Block arguments 추가**: Cons case block에 2개 arguments
3. **Environment extension**: Pattern variables를 block arguments로 바인딩
4. **Body compilation**: Extended environment로 case expression 컴파일

**Lowering pass의 책임:**

Lowering pass는 이 block arguments를 실제 데이터로 채운다:

```cpp
// MatchOpLowering (Part 2에서 자세히)
// 1. Extract head and tail from cons cell
Value head = extractHead(builder, consCellPtr);
Value tail = extractTail(builder, consCellPtr);

// 2. Clone cons region
IRMapping mapper;
mapper.map(consBlock->getArgument(0), head);  // Map %head
mapper.map(consBlock->getArgument(1), tail);  // Map %tail

// 3. Clone operations with mapped values
for (Operation& op : consBlock->getOperations()) {
    builder.clone(op, mapper);
}
```

**결과: Block arguments가 실제 values로 대체됨**

---

## 중간 정리: Part 1 완료

**Part 1에서 다룬 내용:**

✅ **Region-based operation structure**
- Regions vs basic blocks
- Encapsulation과 verification 장점

✅ **Match operation semantics**
- Runtime execution model
- Tag extraction → case selection → variable binding → yield

✅ **TableGen definition**
- Traits: RecursiveSideEffect, SingleBlockImplicitTerminator
- VariadicRegion<SizedRegion<1>> for variable cases
- Custom assembly format과 verifier

✅ **YieldOp terminator**
- Terminator trait
- HasParent<"MatchOp"> constraint
- Comparison with scf.yield

✅ **C API and F# integration**
- Builder callback pattern
- High-level wrapper for match construction
- Block arguments for pattern variables

**다음 Part 2에서:**
- SCF dialect 상세 설명
- MatchOpLowering pattern 완전 구현
- Region cloning과 IRMapping
- 전체 pipeline 예제 (sum_list)
- Common errors와 debugging

---

## Part 2: SCF Lowering and Pipeline

### SCF Dialect: Structured Control Flow

**SCF = Structured Control Flow**

Chapter 8에서 우리는 `scf.if`를 사용했다:

```mlir
%result = scf.if %cond : i1 -> i32 {
  %then_val = arith.constant 42 : i32
  scf.yield %then_val : i32
} else {
  %else_val = arith.constant 0 : i32
  scf.yield %else_val : i32
}
```

**SCF dialect의 핵심 operations:**

| Operation | Purpose | Regions |
|-----------|---------|---------|
| `scf.if` | Two-way branch | 2 (then, else) |
| `scf.index_switch` | Multi-way branch | N (cases) + default |
| `scf.for` | Counted loop | 1 (body) |
| `scf.while` | Conditional loop | 2 (before, after) |

**Chapter 19에서는 `scf.index_switch`를 사용한다.**

### scf.index_switch: Multi-Way Branching

**Syntax:**

```mlir
%result = scf.index_switch %selector : index -> ResultType
case 0 {
  // Case 0 operations
  scf.yield %value0 : ResultType
}
case 1 {
  // Case 1 operations
  scf.yield %value1 : ResultType
}
default {
  // Default case (optional)
  scf.yield %default_val : ResultType
}
```

**Semantics:**

1. **Selector evaluation**: `%selector` 값을 runtime에 evaluate (index type)
2. **Case selection**: Selector 값에 해당하는 case region 선택
3. **Fallback**: 해당하는 case가 없으면 default region (있다면)
4. **Result yielding**: 선택된 region의 `scf.yield`가 결과 전달

**Example: Tag dispatch for list**

```mlir
// %lst: !funlang.list<i32>
// Tag extraction
%struct = // convert %lst to !llvm.struct<(i32, ptr)>
%tag = llvm.extractvalue %struct[0] : !llvm.struct<(i32, ptr)>
%tag_index = arith.index_cast %tag : i32 to index

// Dispatch on tag
%result = scf.index_switch %tag_index : index -> i32
case 0 {  // Nil case (tag = 0)
  %zero = arith.constant 0 : i32
  scf.yield %zero : i32
}
case 1 {  // Cons case (tag = 1)
  %ptr = llvm.extractvalue %struct[1] : !llvm.struct<(i32, ptr)>
  %head = llvm.load %ptr : !llvm.ptr -> i32
  // ... compute with head ...
  scf.yield %sum : i32
}
default {
  // Unreachable for {Nil, Cons} (complete constructor set)
  %minus = arith.constant -1 : i32
  scf.yield %minus : i32
}
```

### Why SCF Before LLVM?

**Option 1: Direct lowering funlang.match → LLVM (what we DON'T do)**

```mlir
// Directly to LLVM dialect
%tag = llvm.extractvalue ...
llvm.switch %tag [
  0: ^nil_block,
  1: ^cons_block
]

^nil_block:
  // ... operations ...
  llvm.br ^merge_block(%zero)

^cons_block:
  // ... operations ...
  llvm.br ^merge_block(%sum)

^merge_block(%result: i32):
  llvm.return %result
```

**문제점:**

1. **Lost structure**: CFG는 원래 match의 case structure를 상실
2. **Harder optimization**: Which blocks belong to which case? 불명확
3. **Debugging**: LLVM IR에서 source pattern matching 추적 어려움
4. **Lowering complexity**: funlang.match → LLVM을 한 번에 구현해야 함

**Option 2: Progressive lowering funlang.match → SCF → CF → LLVM (what we do)**

```mlir
// Stage 1: FunLang
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil: funlang.yield %zero : i32
  ^cons(%h, %t): funlang.yield %sum : i32
}

// Stage 2: SCF (structured, high-level)
%tag_index = // extract tag and cast to index
%result = scf.index_switch %tag_index : index -> i32
case 0 { scf.yield %zero : i32 }
case 1 { scf.yield %sum : i32 }

// Stage 3: CF (goto-style, low-level)
cf.switch %tag_index [
  0: ^block_0,
  1: ^block_1
]
^block_0: cf.br ^merge(%zero)
^block_1: cf.br ^merge(%sum)
^merge(%result: i32): ...

// Stage 4: LLVM (machine-level)
llvm.switch %tag_i8 [
  0: ^llvm_0,
  1: ^llvm_1
]
// ... LLVM blocks ...
```

**장점:**

1. **Separation of concerns**: 각 lowering pass는 하나의 변환만 책임
2. **Optimization hooks**: SCF level에서 pattern-specific optimizations
3. **Incremental verification**: 각 stage마다 IR 검증 가능
4. **Easier debugging**: 문제 발생 시 어느 stage에서 일어났는지 명확

**Comparison: SCF vs CF**

| Aspect | SCF | CF |
|--------|-----|-----|
| Structure | Nested regions | Flat blocks |
| Control flow | Implicit (yield returns) | Explicit (br/switch) |
| Source mapping | Preserves match structure | Lost |
| Optimization | High-level (dead case elimination) | Low-level (block merging) |
| Readability | High (similar to source) | Low (machine-like) |

**Example: Dead case elimination at SCF level**

```mlir
// Input: match on statically-known value
%nil = funlang.nil : !funlang.list<i32>
%result = funlang.match %nil : !funlang.list<i32> -> i32 {
  ^nil: funlang.yield %zero : i32
  ^cons(%h, %t): funlang.yield %sum : i32  // Dead!
}

// After lowering to SCF
%tag_index = arith.constant 0 : index  // Statically known!
%result = scf.index_switch %tag_index : index -> i32
case 0 { scf.yield %zero : i32 }
case 1 { scf.yield %sum : i32 }  // Dead case!

// SCF optimizer can eliminate case 1
%result = scf.index_switch %tag_index : index -> i32
case 0 { scf.yield %zero : i32 }
// case 1 removed

// Further optimization: constant folding
%result = %zero  // Direct replacement!
```

이런 최적화는 CF level에서는 훨씬 어렵다.

### MatchOp Lowering Strategy

**Goal: `funlang.match` → `scf.index_switch` 변환**

**Input (FunLang):**

```mlir
%result = funlang.match %lst : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32
  ^cons(%head: i32, %tail: !funlang.list<i32>):
    %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
    %sum = arith.addi %head, %tail_sum : i32
    funlang.yield %sum : i32
}
```

**Output (SCF + LLVM):**

```mlir
// 1. Convert list type to struct
%struct = builtin.unrealized_conversion_cast %lst
    : !funlang.list<i32> to !llvm.struct<(i32, ptr)>

// 2. Extract tag
%tag_i32 = llvm.extractvalue %struct[0] : !llvm.struct<(i32, ptr)>
%tag_index = arith.index_cast %tag_i32 : i32 to index

// 3. Extract data pointer (for cons case)
%data_ptr = llvm.extractvalue %struct[1] : !llvm.struct<(i32, ptr)>

// 4. Multi-way switch
%result = scf.index_switch %tag_index : index -> i32
case 0 {
  // Nil case: no data to extract
  %zero = arith.constant 0 : i32
  scf.yield %zero : i32
}
case 1 {
  // Cons case: extract head and tail
  %head_ptr = %data_ptr  // Points to [head, tail] array
  %head = llvm.load %head_ptr : !llvm.ptr -> i32

  %tail_ptr = llvm.getelementptr %data_ptr[1] : (!llvm.ptr) -> !llvm.ptr
  %tail_struct_ptr = llvm.load %tail_ptr : !llvm.ptr -> !llvm.ptr
  %tail_struct = llvm.load %tail_struct_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>
  %tail = builtin.unrealized_conversion_cast %tail_struct
      : !llvm.struct<(i32, ptr)> to !funlang.list<i32>

  // Cons case body (converted)
  %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
  %sum = arith.addi %head, %tail_sum : i32
  scf.yield %sum : i32
}
default {
  // Unreachable for {Nil, Cons}
  %minus = arith.constant -1 : i32
  scf.yield %minus : i32
}
```

**Lowering steps:**

1. **Type conversion**: `!funlang.list<T>` → `!llvm.struct<(i32, ptr)>`
2. **Tag extraction**: `llvm.extractvalue` to get tag field
3. **Index casting**: `arith.index_cast` for `scf.index_switch` selector
4. **Case region cloning**: 각 funlang.match case를 scf.index_switch case로 복사
5. **Block argument mapping**: Pattern variables를 extracted values로 대체
6. **Terminator conversion**: `funlang.yield` → `scf.yield`

### Tag Value Mapping

**Chapter 18 recap: List representation**

```cpp
// NilOpLowering
Value tag = builder.create<arith::ConstantIntOp>(loc, 0, builder.getI32Type());

// ConsOpLowering
Value tag = builder.create<arith::ConstantIntOp>(loc, 1, builder.getI32Type());
```

**Tag mapping:**

| Constructor | Tag Value |
|-------------|-----------|
| Nil | 0 |
| Cons | 1 |

**MatchOpLowering은 이 mapping을 알아야 한다:**

```cpp
// In MatchOpLowering::matchAndRewrite
// Case 0 → Nil pattern
// Case 1 → Cons pattern
for (auto [index, region] : llvm::enumerate(matchOp.getCases())) {
  // index = 0 → Nil
  // index = 1 → Cons
  builder.create<scf::IndexSwitchCaseOp>(loc, index);
  // ... clone region ...
}
```

**Future extension: 임의의 ADT**

지금은 hardcoded mapping (Nil=0, Cons=1)이지만, 나중에는:

```tablegen
// Extensible ADT definition
def Shape : FunLang_ADT<"shape"> {
  let constructors = [
    Constructor<"circle", [F32]>,           // tag = 0
    Constructor<"rectangle", [F32, F32]>,   // tag = 1
    Constructor<"triangle", [F32, F32, F32]>  // tag = 2
  ];
}
```

Compiler가 자동으로 tag 할당.

### Pattern Variable Binding

**Cons case의 challenge: block arguments를 어떻게 채우는가?**

**Source (FunLang):**

```mlir
^cons(%head: i32, %tail: !funlang.list<i32>):
  // %head와 %tail이 어디서 오는가?
  funlang.yield %sum : i32
```

**Lowering 후 (SCF):**

```mlir
case 1 {
  // 여기서 %head와 %tail을 extract해야 함
  %head = llvm.load %data_ptr : !llvm.ptr -> i32
  %tail = // ... complex extraction ...

  // 이제 body를 clone하면서 block arguments를 이 values로 map
  // (IRMapping 사용)
}
```

**IRMapping: SSA Value Remapping**

MLIR의 `IRMapping` class는 "old value → new value" mapping을 저장한다.

```cpp
IRMapping mapper;
mapper.map(oldValue1, newValue1);
mapper.map(oldValue2, newValue2);

// Clone operation with mapped values
Operation* newOp = builder.clone(*oldOp, mapper);
// oldOp의 operands가 oldValue1, oldValue2였다면
// newOp의 operands는 newValue1, newValue2로 대체됨
```

**MatchOpLowering에서 IRMapping 사용:**

```cpp
// Cons case region
Region& consRegion = matchOp.getCases()[1];
Block* consBlock = &consRegion.front();

// consBlock의 block arguments:
// consBlock->getArgument(0) = %head (i32)
// consBlock->getArgument(1) = %tail (!funlang.list<i32>)

// Extract actual values
Value actualHead = extractHead(builder, dataPtrConverted);
Value actualTail = extractTail(builder, dataPtrConverted, typeConverter);

// Map block arguments to extracted values
IRMapping mapper;
mapper.map(consBlock->getArgument(0), actualHead);
mapper.map(consBlock->getArgument(1), actualTail);

// Clone operations in consBlock with mapping
for (Operation& op : consBlock->getOperations()) {
  if (isa<YieldOp>(op)) {
    // Convert funlang.yield → scf.yield
    builder.create<scf::YieldOp>(op.getLoc(),
                                  mapper.lookupOrDefault(op.getOperand(0)));
  } else {
    // Clone other operations with mapped operands
    builder.clone(op, mapper);
  }
}
```

**Result: Block arguments가 사라지고 extracted values로 대체됨**

```mlir
// Before (funlang.match case)
^cons(%head: i32, %tail: !funlang.list<i32>):
  %sum = arith.addi %head, %tail_sum : i32
  funlang.yield %sum : i32

// After (scf.index_switch case)
case 1 {
  %head = llvm.load ...  // Extracted value
  %tail = ...            // Extracted value
  %sum = arith.addi %head, %tail_sum : i32  // %head mapped
  scf.yield %sum : i32
}
```

### MatchOpLowering Pattern: Complete Implementation

**이제 전체 lowering pattern을 구현한다.**

**File: `FunLang/Transforms/FunLangToSCF.cpp`** (conceptual C++ code)

```cpp
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVM/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "FunLang/IR/FunLangOps.h"
#include "FunLang/Transforms/TypeConverter.h"

using namespace mlir;
using namespace mlir::funlang;

namespace {

// Helper: Extract head from cons cell
// Input: %data_ptr points to [head, tail] array
// Output: %head value
Value extractHead(OpBuilder& builder, Location loc,
                  Value dataPtrConverted, Type headType) {
  // %data_ptr already points to cons cell array
  // Load first element (head)
  Value head = builder.create<LLVM::LoadOp>(loc, headType, dataPtrConverted);
  return head;
}

// Helper: Extract tail from cons cell
// Input: %data_ptr points to [head, tail] array
// Output: %tail value (converted back to !funlang.list<T>)
Value extractTail(OpBuilder& builder, Location loc,
                  Value dataPtrConverted,
                  FunLangTypeConverter* typeConverter,
                  Type tailFunLangType) {
  // GEP to second element (tail)
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, builder.getI32Type());
  Value tailPtr = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()),
      dataPtrConverted, ValueRange{one});

  // Load tail pointer
  Value tailStructPtr = builder.create<LLVM::LoadOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), tailPtr);

  // Load tail struct
  Type tailStructType = typeConverter->convertType(tailFunLangType);
  Value tailStruct = builder.create<LLVM::LoadOp>(
      loc, tailStructType, tailStructPtr);

  // Convert back to FunLang type (for remaining funlang operations in body)
  Value tail = builder.create<UnrealizedConversionCastOp>(
      loc, tailFunLangType, tailStruct).getResult(0);

  return tail;
}

// Helper: Convert funlang.yield to scf.yield in region
void convertYieldOps(Region& region, OpBuilder& builder, IRMapping& mapper) {
  for (Block& block : region) {
    for (Operation& op : llvm::make_early_inc_range(block)) {
      if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
        builder.setInsertionPoint(yieldOp);
        Value yieldValue = mapper.lookupOrDefault(yieldOp.getValue());
        builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValue);
        yieldOp.erase();
      }
    }
  }
}

// Main lowering pattern
class MatchOpLowering : public OpConversionPattern<MatchOp> {
public:
  using OpConversionPattern<MatchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MatchOp matchOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {

    Location loc = matchOp.getLoc();
    Value input = adaptor.getInput();
    Type resultType = matchOp.getResult().getType();

    auto* typeConverter = getTypeConverter<FunLangTypeConverter>();

    // 1. Convert input to LLVM struct
    // input: !funlang.list<T> → !llvm.struct<(i32, ptr)>
    Type structType = typeConverter->convertType(input.getType());
    Value structVal = rewriter.create<UnrealizedConversionCastOp>(
        loc, structType, input).getResult(0);

    // 2. Extract tag field
    Value tag = rewriter.create<LLVM::ExtractValueOp>(loc, structVal, 0);

    // 3. Cast tag to index (for scf.index_switch)
    Value tagIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), tag);

    // 4. Extract data pointer (needed for cons case)
    Value dataPtr = rewriter.create<LLVM::ExtractValueOp>(loc, structVal, 1);

    // 5. Create scf.index_switch
    auto indexSwitchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultType, tagIndex, matchOp.getCases().size());

    // 6. Process each case region
    for (auto [caseIndex, caseRegion] :
         llvm::enumerate(matchOp.getCases())) {

      Block* originalBlock = &caseRegion.front();
      Region& switchCaseRegion = indexSwitchOp.getCaseRegions()[caseIndex];
      Block* caseBlock = rewriter.createBlock(&switchCaseRegion);

      rewriter.setInsertionPointToStart(caseBlock);

      IRMapping mapper;

      // Handle block arguments (pattern variables)
      if (caseIndex == 1) {  // Cons case
        // originalBlock has 2 arguments: %head, %tail

        // Extract head
        Type headFunLangType = originalBlock->getArgument(0).getType();
        Type headLLVMType = typeConverter->convertType(headFunLangType);
        Value head = extractHead(rewriter, loc, dataPtr, headLLVMType);

        // Convert head to FunLang type if needed
        Value headFunLang = head;
        if (headFunLangType != headLLVMType) {
          headFunLang = rewriter.create<UnrealizedConversionCastOp>(
              loc, headFunLangType, head).getResult(0);
        }

        // Extract tail
        Type tailFunLangType = originalBlock->getArgument(1).getType();
        Value tail = extractTail(rewriter, loc, dataPtr,
                                  typeConverter, tailFunLangType);

        // Map block arguments to extracted values
        mapper.map(originalBlock->getArgument(0), headFunLang);
        mapper.map(originalBlock->getArgument(1), tail);
      }
      // Nil case (caseIndex == 0): no block arguments, no extraction

      // Clone operations from original region
      for (Operation& op : originalBlock->getOperations()) {
        if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
          // Convert funlang.yield → scf.yield
          Value yieldValue = mapper.lookupOrDefault(yieldOp.getValue());
          rewriter.create<scf::YieldOp>(loc, yieldValue);
        } else {
          // Clone operation with mapped operands
          rewriter.clone(op, mapper);
        }
      }
    }

    // 7. Add default region (unreachable for complete constructor sets)
    {
      Region& defaultRegion = indexSwitchOp.getDefaultRegion();
      Block* defaultBlock = rewriter.createBlock(&defaultRegion);
      rewriter.setInsertionPointToStart(defaultBlock);

      // Emit error value (this should never execute)
      Value errorVal;
      if (resultType.isIntOrIndex()) {
        errorVal = rewriter.create<arith::ConstantIntOp>(loc, -1, resultType);
      } else {
        // For other types, emit unreachable or null
        errorVal = rewriter.create<LLVM::ZeroOp>(loc, resultType);
      }

      rewriter.create<scf::YieldOp>(loc, errorVal);
    }

    // 8. Replace match operation with index_switch result
    rewriter.replaceOp(matchOp, indexSwitchOp.getResult(0));

    return success();
  }
};

} // namespace

// Pass definition
struct FunLangToSCFPass
    : public PassWrapper<FunLangToSCFPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect,
                    scf::SCFDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto* context = &getContext();

    FunLangTypeConverter typeConverter;
    ConversionTarget target(*context);

    // Mark funlang.match as illegal (must be lowered)
    target.addIllegalOp<MatchOp>();

    // Mark SCF operations as legal
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // Keep other FunLang ops legal (lowered in FunLangToLLVM pass)
    target.addLegalOp<NilOp, ConsOp, ClosureOp, ApplyOp>();

    RewritePatternSet patterns(context);
    patterns.add<MatchOpLowering>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createFunLangToSCFPass() {
  return std::make_unique<FunLangToSCFPass>();
}
```

**핵심 로직 분석:**

**1. Type conversion (lines ~95-100)**

```cpp
Type structType = typeConverter->convertType(input.getType());
Value structVal = rewriter.create<UnrealizedConversionCastOp>(
    loc, structType, input).getResult(0);
```

`!funlang.list<i32>` → `!llvm.struct<(i32, ptr)>` 변환.

`UnrealizedConversionCastOp`는 type conversion의 placeholder다. 나중에 다른 pass가 이를 실제 operations로 대체하거나 제거한다.

**2. Tag extraction (lines ~103-108)**

```cpp
Value tag = rewriter.create<LLVM::ExtractValueOp>(loc, structVal, 0);
Value tagIndex = rewriter.create<arith::IndexCastOp>(
    loc, rewriter.getIndexType(), tag);
```

Struct의 첫 번째 field (tag)를 추출하고 index type으로 cast.

**3. scf.index_switch creation (lines ~113-115)**

```cpp
auto indexSwitchOp = rewriter.create<scf::IndexSwitchOp>(
    loc, resultType, tagIndex, matchOp.getCases().size());
```

N개의 cases를 가진 index_switch 생성.

**4. Region cloning (lines ~118-160)**

각 case region을 iterate하며:

- **Nil case (caseIndex == 0)**: Block arguments 없음, 그냥 clone
- **Cons case (caseIndex == 1)**: Block arguments 있음, extract + map

**5. IRMapping for block arguments (lines ~130-148)**

```cpp
mapper.map(originalBlock->getArgument(0), headFunLang);
mapper.map(originalBlock->getArgument(1), tail);
```

Original block의 arguments를 extracted values로 mapping.

**6. Operation cloning (lines ~152-159)**

```cpp
for (Operation& op : originalBlock->getOperations()) {
  if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
    Value yieldValue = mapper.lookupOrDefault(yieldOp.getValue());
    rewriter.create<scf::YieldOp>(loc, yieldValue);
  } else {
    rewriter.clone(op, mapper);
  }
}
```

- `funlang.yield` → `scf.yield` 변환
- 다른 operations는 mapper와 함께 clone

**7. Default region (lines ~163-176)**

Unreachable case를 위한 default region 생성.

### Complete Pipeline: Pass Registration

**전체 lowering pipeline:**

```
FunLang Dialect (with match, nil, cons, closure, apply)
    ↓
[FunLangToSCFPass]
    ↓
FunLang (without match) + SCF
    ↓
[FunLangToLLVMPass] (lowers nil, cons, closure, apply)
    ↓
LLVM + SCF
    ↓
[SCFToControlFlowPass]
    ↓
LLVM + CF
    ↓
[ConvertControlFlowToLLVMPass]
    ↓
LLVM Dialect only
    ↓
[LLVMToObjectPass]
    ↓
Object file
```

**Pass manager setup (F# code):**

```fsharp
// File: FunLang.Compiler/Pipeline.fs

let lowerToLLVM (module_: Module) =
    let pm = PassManager.Create(module_.Context)

    // 1. FunLang → SCF (lower match operation)
    pm.AddPass(FunLangPasses.CreateFunLangToSCFPass())

    // 2. FunLang → LLVM (lower nil, cons, closure, apply)
    pm.AddPass(FunLangPasses.CreateFunLangToLLVMPass())

    // 3. SCF → CF
    pm.AddPass(SCFPasses.CreateSCFToControlFlowPass())

    // 4. CF → LLVM
    pm.AddPass(ConversionPasses.CreateConvertControlFlowToLLVMPass())

    // 5. Func → LLVM
    pm.AddPass(ConversionPasses.CreateConvertFuncToLLVMPass())

    // 6. Arith → LLVM
    pm.AddPass(ConversionPasses.CreateConvertArithToLLVMPass())

    pm.Run(module_) |> ignore
```

**Pass dependencies:**

- `FunLangToSCFPass` must run **before** `FunLangToLLVMPass`
  - Reason: MatchOp의 regions에 다른 FunLang ops (nil, cons, etc.) 포함
  - SCF로 변환 후 남은 FunLang ops를 LLVM으로 변환

- `SCFToControlFlowPass` must run **after** all FunLang lowering
  - Reason: SCF ops는 다른 dialects가 모두 LLVM으로 변환된 후 lower

- `ConvertFuncToLLVMPass` must run **after** SCF/CF conversion
  - Reason: Function signatures에 FunLang types가 남아있으면 안 됨

### End-to-End Example: sum_list Function

**F# source code:**

```fsharp
// FunLang source
let rec sum_list lst =
    match lst with
    | [] -> 0
    | head :: tail -> head + sum_list tail

let main () =
    let my_list = [1; 2; 3]
    sum_list my_list
```

**Stage 1: FunLang Dialect (after F# compiler)**

```mlir
module {
  func.func @sum_list(%lst: !funlang.list<i32>) -> i32 {
    %result = funlang.match %lst : !funlang.list<i32> -> i32 {
      ^nil:
        %zero = arith.constant 0 : i32
        funlang.yield %zero : i32
      ^cons(%head: i32, %tail: !funlang.list<i32>):
        %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32
        %sum = arith.addi %head, %tail_sum : i32
        funlang.yield %sum : i32
    }
    return %result : i32
  }

  func.func @main() -> i32 {
    // Build list [1, 2, 3]
    %nil = funlang.nil : !funlang.list<i32>

    %c3 = arith.constant 3 : i32
    %l3 = funlang.cons %c3, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>

    %c2 = arith.constant 2 : i32
    %l2 = funlang.cons %c2, %l3 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

    %c1 = arith.constant 1 : i32
    %l1 = funlang.cons %c1, %l2 : (i32, !funlang.list<i32>) -> !funlang.list<i32>

    // Call sum_list
    %sum = func.call @sum_list(%l1) : (!funlang.list<i32>) -> i32
    return %sum : i32
  }
}
```

**Stage 2: After FunLangToSCFPass**

```mlir
module {
  func.func @sum_list(%lst: !funlang.list<i32>) -> i32 {
    // Type conversion
    %struct = builtin.unrealized_conversion_cast %lst
        : !funlang.list<i32> to !llvm.struct<(i32, ptr)>

    // Tag extraction
    %tag = llvm.extractvalue %struct[0] : !llvm.struct<(i32, ptr)>
    %tag_index = arith.index_cast %tag : i32 to index

    // Data pointer
    %data_ptr = llvm.extractvalue %struct[1] : !llvm.struct<(i32, ptr)>

    // Index switch
    %result = scf.index_switch %tag_index : index -> i32
    case 0 {
      %zero = arith.constant 0 : i32
      scf.yield %zero : i32
    }
    case 1 {
      // Extract head
      %head = llvm.load %data_ptr : !llvm.ptr -> i32

      // Extract tail
      %one = arith.constant 1 : i32
      %tail_ptr = llvm.getelementptr %data_ptr[%one] : (!llvm.ptr, i32) -> !llvm.ptr
      %tail_struct_ptr = llvm.load %tail_ptr : !llvm.ptr -> !llvm.ptr
      %tail_struct = llvm.load %tail_struct_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>
      %tail = builtin.unrealized_conversion_cast %tail_struct
          : !llvm.struct<(i32, ptr)> to !funlang.list<i32>

      // Recursive call
      %tail_sum = func.call @sum_list(%tail) : (!funlang.list<i32>) -> i32

      // Sum
      %sum = arith.addi %head, %tail_sum : i32
      scf.yield %sum : i32
    }
    default {
      %error = arith.constant -1 : i32
      scf.yield %error : i32
    }

    return %result : i32
  }

  func.func @main() -> i32 {
    // Still has funlang.nil and funlang.cons (not lowered yet)
    %nil = funlang.nil : !funlang.list<i32>
    %c3 = arith.constant 3 : i32
    %l3 = funlang.cons %c3, %nil : (i32, !funlang.list<i32>) -> !funlang.list<i32>
    // ...
    %sum = func.call @sum_list(%l1) : (!funlang.list<i32>) -> i32
    return %sum : i32
  }
}
```

**Stage 3: After FunLangToLLVMPass**

```mlir
module {
  func.func @sum_list(%lst: !llvm.struct<(i32, ptr)>) -> i32 {
    // ... same as Stage 2 but types converted ...
    %tag = llvm.extractvalue %lst[0] : !llvm.struct<(i32, ptr)>
    %tag_index = arith.index_cast %tag : i32 to index
    %data_ptr = llvm.extractvalue %lst[1] : !llvm.struct<(i32, ptr)>

    %result = scf.index_switch %tag_index : index -> i32
    case 0 { /* ... */ }
    case 1 { /* ... */ }
    default { /* ... */ }

    return %result : i32
  }

  func.func @main() -> i32 {
    // funlang.nil and funlang.cons lowered to LLVM
    %c0 = arith.constant 0 : i32
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_nil = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %s1_nil = llvm.insertvalue %c0, %undef_nil[0] : !llvm.struct<(i32, ptr)>
    %nil = llvm.insertvalue %null, %s1_nil[1] : !llvm.struct<(i32, ptr)>

    %c3 = arith.constant 3 : i32
    %c1_tag = arith.constant 1 : i32
    %size = arith.constant 16 : i64  // sizeof(cons cell)
    %ptr = llvm.call @GC_malloc(%size) : (i64) -> !llvm.ptr
    llvm.store %c3, %ptr : i32, !llvm.ptr
    %tail_ptr = llvm.getelementptr %ptr[1] : (!llvm.ptr) -> !llvm.ptr
    // ... store tail ...
    %undef_cons = llvm.mlir.undef : !llvm.struct<(i32, ptr)>
    %s1_cons = llvm.insertvalue %c1_tag, %undef_cons[0] : !llvm.struct<(i32, ptr)>
    %l3 = llvm.insertvalue %ptr, %s1_cons[1] : !llvm.struct<(i32, ptr)>

    // ...
    %sum = func.call @sum_list(%l1) : (!llvm.struct<(i32, ptr)>) -> i32
    return %sum : i32
  }
}
```

**Stage 4: After SCFToControlFlowPass**

```mlir
module {
  func.func @sum_list(%lst: !llvm.struct<(i32, ptr)>) -> i32 {
    %tag = llvm.extractvalue %lst[0] : !llvm.struct<(i32, ptr)>
    %tag_index = arith.index_cast %tag : i32 to index
    %data_ptr = llvm.extractvalue %lst[1] : !llvm.struct<(i32, ptr)>

    // scf.index_switch → cf.switch
    cf.switch %tag_index : index, [
      default: ^default,
      0: ^case_0,
      1: ^case_1
    ]

  ^case_0:
    %zero = arith.constant 0 : i32
    cf.br ^merge(%zero : i32)

  ^case_1:
    %head = llvm.load %data_ptr : !llvm.ptr -> i32
    // ... extract tail ...
    %tail_sum = func.call @sum_list(%tail) : (!llvm.struct<(i32, ptr)>) -> i32
    %sum = arith.addi %head, %tail_sum : i32
    cf.br ^merge(%sum : i32)

  ^default:
    %error = arith.constant -1 : i32
    cf.br ^merge(%error : i32)

  ^merge(%result: i32):
    return %result : i32
  }

  func.func @main() -> i32 {
    // ... LLVM code for list construction ...
    %sum = func.call @sum_list(%l1) : (!llvm.struct<(i32, ptr)>) -> i32
    return %sum : i32
  }
}
```

**Stage 5: After ConvertControlFlowToLLVMPass + ConvertFuncToLLVMPass**

```mlir
llvm.func @sum_list(%arg0: !llvm.struct<(i32, ptr)>) -> i32 {
  %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, ptr)>
  %1 = llvm.sext %0 : i32 to i64  // index cast
  %2 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, ptr)>

  llvm.switch %1 : i64, ^default [
    0: ^case_0,
    1: ^case_1
  ]

^case_0:
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^merge(%c0 : i32)

^case_1:
  %head = llvm.load %2 : !llvm.ptr -> i32
  // ... tail extraction ...
  %tail_sum = llvm.call @sum_list(%tail) : (!llvm.struct<(i32, ptr)>) -> i32
  %sum = llvm.add %head, %tail_sum : i32
  llvm.br ^merge(%sum : i32)

^default:
  %error = llvm.mlir.constant(-1 : i32) : i32
  llvm.br ^merge(%error : i32)

^merge(%result: i32):
  llvm.return %result : i32
}

llvm.func @main() -> i32 {
  // ... LLVM code ...
  %sum = llvm.call @sum_list(%l1) : (!llvm.struct<(i32, ptr)>) -> i32
  llvm.return %sum : i32
}
```

**Stage 6: Native code (after llc + linking)**

```bash
$ ./funlang_program
6
```

**Pipeline verification at each stage:**

```bash
# After each pass, verify IR
$ mlir-opt --funlang-to-scf --verify-diagnostics input.mlir
$ mlir-opt --funlang-to-llvm --verify-diagnostics input.mlir
$ mlir-opt --convert-scf-to-cf --verify-diagnostics input.mlir
```

### Common Errors and Debugging

**Error 1: Block argument count mismatch**

**Symptom:**

```
error: 'scf.yield' op result type mismatch
```

**Cause:**

Cons case region의 block arguments 개수가 틀림.

```cpp
// Wrong: forgot to map tail argument
mapper.map(originalBlock->getArgument(0), headFunLang);
// Missing: mapper.map(originalBlock->getArgument(1), tail);
```

**Fix:**

모든 block arguments를 map해야 함.

```cpp
mapper.map(originalBlock->getArgument(0), headFunLang);
mapper.map(originalBlock->getArgument(1), tail);  // ✅
```

**Error 2: Type mismatch after region cloning**

**Symptom:**

```
error: 'func.call' op operand type mismatch: expected '!llvm.struct<...>', got '!funlang.list<...>'
```

**Cause:**

Region 내부의 operations가 아직 type conversion 안 됨.

**Why:**

`FunLangToSCFPass`는 partial conversion이다. Match operation만 lower하고 나머지 FunLang ops는 그대로 둔다.

**Fix:**

Region cloning 후 남은 FunLang operations는 다음 pass (`FunLangToLLVMPass`)에서 처리됨.

Temporary workaround: `UnrealizedConversionCastOp` 사용.

```cpp
Value tail = extractTail(...);  // Returns LLVM struct
// Cast back to FunLang type for func.call
Value tailFunLang = rewriter.create<UnrealizedConversionCastOp>(
    loc, tailFunLangType, tail).getResult(0);
```

**Error 3: Missing scf.yield in converted regions**

**Symptom:**

```
error: block must terminate with scf.yield
```

**Cause:**

`funlang.yield`를 `scf.yield`로 변환하는 걸 까먹음.

```cpp
// Wrong: just clone YieldOp as-is
for (Operation& op : originalBlock->getOperations()) {
  rewriter.clone(op, mapper);  // funlang.yield gets cloned!
}
```

**Fix:**

YieldOp를 특별히 처리해서 변환.

```cpp
for (Operation& op : originalBlock->getOperations()) {
  if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
    Value yieldValue = mapper.lookupOrDefault(yieldOp.getValue());
    rewriter.create<scf::YieldOp>(loc, yieldValue);  // ✅ Convert
  } else {
    rewriter.clone(op, mapper);
  }
}
```

**Error 4: Wrong tag values (0 vs 1 confusion)**

**Symptom:**

런타임에 엉뚱한 case가 실행됨. 예: Nil list인데 Cons case 실행.

**Cause:**

Tag mapping이 틀림.

```cpp
// Wrong: reversed mapping
// case 0 → Cons (wrong!)
// case 1 → Nil (wrong!)
```

**Fix:**

Chapter 18의 tag values와 일치시켜야 함:

```cpp
// Correct mapping
// case 0 → Nil  (tag = 0)
// case 1 → Cons (tag = 1)
for (auto [caseIndex, caseRegion] : llvm::enumerate(matchOp.getCases())) {
  // caseIndex = 0 → Nil region (first in match)
  // caseIndex = 1 → Cons region (second in match)
}
```

**F# compiler는 pattern 순서를 보장해야 함:**

```fsharp
// F# compiler must emit cases in this order:
// Case 0: Nil
// Case 1: Cons
match lst with
| [] -> ...        // Must be first case
| head :: tail -> ... // Must be second case
```

**Error 5: Incorrect data extraction from cons cell**

**Symptom:**

런타임 segfault or garbage values.

**Cause:**

GEP indices 틀림.

```cpp
// Wrong: GEP from struct pointer
Value tailPtr = builder.create<LLVM::GEPOp>(
    loc, ptrType, structVal, ValueRange{one});  // ❌ structVal is value not pointer
```

**Fix:**

Data pointer는 이미 cons cell array를 가리킴.

```cpp
// Correct: dataPtr already points to [head, tail] array
Value headPtr = dataPtr;  // Points to head
Value head = builder.create<LLVM::LoadOp>(loc, headType, headPtr);

Value one = builder.create<arith::ConstantIntOp>(loc, 1, i32Type);
Value tailPtr = builder.create<LLVM::GEPOp>(
    loc, ptrType, dataPtr, ValueRange{one});  // ✅ GEP from array pointer
```

**Debugging strategies:**

1. **Print IR after each pass:**

```bash
$ mlir-opt --funlang-to-scf --print-ir-after-all input.mlir
```

2. **Use verifier:**

```bash
$ mlir-opt --funlang-to-scf --verify-diagnostics input.mlir
```

3. **Dump operations in lowering code:**

```cpp
matchOp.dump();  // Before lowering
indexSwitchOp.dump();  // After lowering
```

4. **Check IRMapping:**

```cpp
for (auto [caseIndex, region] : enumerate(matchOp.getCases())) {
  Block* block = &region.front();
  llvm::errs() << "Case " << caseIndex << ":\n";
  for (BlockArgument arg : block->getArguments()) {
    llvm::errs() << "  Arg: " << arg << " → "
                 << mapper.lookupOrDefault(arg) << "\n";
  }
}
```

---

## 리터럴 패턴 로우어링

지금까지 constructor patterns (Nil, Cons)를 위한 lowering을 설명했다. 이제 **리터럴 패턴**을 위한 lowering 전략을 다룬다.

### Constructor vs Literal Dispatch

**Constructor 패턴과 리터럴 패턴의 핵심 차이:**

| 특성 | Constructor | Literal |
|------|-------------|---------|
| 값의 범위 | 유한 (finite) | 무한 (infinite) |
| 테스트 | Tag extraction | Value comparison |
| MLIR dispatch | `scf.index_switch` | `arith.cmpi` + `scf.if` |
| Complexity | O(1) | O(n) sequential |

**Constructor patterns use `scf.index_switch`:**

Tag는 유한한 범위 (예: 0 = Nil, 1 = Cons)이므로 jump table이 가능하다.

```mlir
// Constructor dispatch: O(1)
%tag = llvm.extractvalue %list[0] : !llvm.struct<(i32, ptr)>
%tag_index = arith.index_cast %tag : i32 to index

%result = scf.index_switch %tag_index : index -> i32
case 0 { /* Nil case */ scf.yield ... }
case 1 { /* Cons case */ scf.yield ... }
default { scf.yield %unreachable : i32 }
```

**Literal patterns use `arith.cmpi` + `scf.if` chain:**

정수 리터럴은 무한하므로 순차적 비교가 필요하다.

```mlir
// Literal dispatch: O(n)
%is_zero = arith.cmpi eq, %x, %c0 : i32
%result = scf.if %is_zero -> i32 {
    scf.yield %zero_result : i32
} else {
    %is_one = arith.cmpi eq, %x, %c1 : i32
    %inner = scf.if %is_one -> i32 {
        scf.yield %one_result : i32
    } else {
        scf.yield %default_result : i32
    }
    scf.yield %inner : i32
}
```

### 리터럴 패턴 Lowering 구현

**리터럴 매칭을 위한 C++ lowering pattern:**

```cpp
// LiteralMatchLowering.cpp

class LiteralMatchOpLowering : public OpConversionPattern<LiteralMatchOp> {
public:
  using OpConversionPattern<LiteralMatchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LiteralMatchOp matchOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {

    Location loc = matchOp.getLoc();
    Value scrutinee = adaptor.getScrutinee();
    Type resultType = matchOp.getResult().getType();

    // Collect all cases: (literal_value, region)
    auto cases = matchOp.getCases();
    Region* defaultRegion = matchOp.getDefaultRegion();

    // Build nested scf.if chain from bottom up
    Value result = buildIfChain(rewriter, loc, scrutinee,
                                 cases, defaultRegion, resultType);

    rewriter.replaceOp(matchOp, result);
    return success();
  }

private:
  Value buildIfChain(
      ConversionPatternRewriter& rewriter,
      Location loc,
      Value scrutinee,
      ArrayRef<std::pair<int64_t, Region*>> cases,
      Region* defaultRegion,
      Type resultType) const {

    // Base case: no more cases, use default
    if (cases.empty()) {
      return cloneRegionAndGetResult(rewriter, loc, defaultRegion, resultType);
    }

    // Current case
    auto [literalValue, caseRegion] = cases.front();
    auto remainingCases = cases.drop_front();

    // Create comparison: scrutinee == literal
    Value literalConst = rewriter.create<arith::ConstantIntOp>(
        loc, literalValue, scrutinee.getType());
    Value isMatch = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, scrutinee, literalConst);

    // Create scf.if
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, resultType, isMatch,
        /*thenBuilder=*/[&](OpBuilder& b, Location loc) {
          Value result = cloneRegionAndGetResult(b, loc, caseRegion, resultType);
          b.create<scf::YieldOp>(loc, result);
        },
        /*elseBuilder=*/[&](OpBuilder& b, Location loc) {
          Value result = buildIfChain(
              rewriter, loc, scrutinee, remainingCases, defaultRegion, resultType);
          b.create<scf::YieldOp>(loc, result);
        });

    return ifOp.getResult(0);
  }

  Value cloneRegionAndGetResult(
      OpBuilder& builder,
      Location loc,
      Region* region,
      Type resultType) const {
    // Clone operations from region
    IRMapping mapper;
    for (Operation& op : region->front()) {
      if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
        return mapper.lookupOrDefault(yieldOp.getValue());
      } else {
        builder.clone(op, mapper);
      }
    }
    llvm_unreachable("Region must end with yield");
  }
};
```

**생성된 IR 예제:**

```fsharp
// FunLang source
match x with
| 0 -> "zero"
| 1 -> "one"
| _ -> "other"
```

```mlir
// After lowering: nested scf.if chain
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32

%is_zero = arith.cmpi eq, %x, %c0 : i32
%result = scf.if %is_zero -> !llvm.ptr<i8> {
    scf.yield %zero_str : !llvm.ptr<i8>
} else {
    %is_one = arith.cmpi eq, %x, %c1 : i32
    %inner = scf.if %is_one -> !llvm.ptr<i8> {
        scf.yield %one_str : !llvm.ptr<i8>
    } else {
        // Default case: no comparison
        scf.yield %other_str : !llvm.ptr<i8>
    }
    scf.yield %inner : !llvm.ptr<i8>
}
```

### 최적화 기회 (Optimization Opportunities)

**1. Dense Range Detection**

리터럴이 0, 1, 2, ... 연속일 때 `scf.index_switch`로 변환 가능:

```mlir
// Before: sequential comparisons
%is_0 = arith.cmpi eq, %x, %c0
scf.if %is_0 { ... } else {
    %is_1 = arith.cmpi eq, %x, %c1
    scf.if %is_1 { ... } else {
        %is_2 = arith.cmpi eq, %x, %c2
        // ...
    }
}

// After: range check + index_switch
%in_range = arith.cmpi ult, %x, %c3 : i32
scf.if %in_range {
    %idx = arith.index_cast %x : i32 to index
    scf.index_switch %idx : index -> i32
    case 0 { /* case 0 */ }
    case 1 { /* case 1 */ }
    case 2 { /* case 2 */ }
} else {
    // default
}
```

**Dense range detection algorithm:**

```cpp
bool isDenseRange(ArrayRef<int64_t> literals) {
  if (literals.empty()) return false;

  // Sort literals
  SmallVector<int64_t> sorted(literals.begin(), literals.end());
  llvm::sort(sorted);

  // Check if consecutive
  for (size_t i = 1; i < sorted.size(); ++i) {
    if (sorted[i] != sorted[i-1] + 1)
      return false;
  }

  // Starts from 0 or 1 (common case)
  return sorted[0] == 0 || sorted[0] == 1;
}
```

**2. Sparse Set Optimization**

리터럴이 sparse할 때 (예: 0, 10, 100) binary search 가능:

```mlir
// O(log n) with binary search
%mid = arith.constant 10 : i32
%less_than_mid = arith.cmpi slt, %x, %mid : i32
scf.if %less_than_mid {
    // Check 0
    %is_0 = arith.cmpi eq, %x, %c0
    scf.if %is_0 { ... } else { /* default */ }
} else {
    // Check 10, 100
    %is_10 = arith.cmpi eq, %x, %c10
    scf.if %is_10 { ... } else {
        %is_100 = arith.cmpi eq, %x, %c100
        scf.if %is_100 { ... } else { /* default */ }
    }
}
```

이 최적화는 MLIR transformation pass로 구현 가능 (Phase 7).

**3. LLVM Backend Optimization**

SCF → CF → LLVM pipeline 후 LLVM backend가 추가 최적화:

```llvm
; LLVM will recognize this pattern
%cmp0 = icmp eq i32 %x, 0
br i1 %cmp0, label %case0, label %check1
check1:
%cmp1 = icmp eq i32 %x, 1
br i1 %cmp1, label %case1, label %default

; And optimize to switch:
switch i32 %x, label %default [
    i32 0, label %case0
    i32 1, label %case1
]
```

**LLVM switch lowering:**

- Dense: jump table (O(1))
- Sparse: binary search tree (O(log n))
- Very sparse: linear search (O(n))

### Mixed Patterns: Constructor + Literal

**실제 코드는 constructor와 literal을 섞어 쓴다:**

```fsharp
match (list, n) with
| (Nil, _) -> 0
| (Cons(x, _), 0) -> x
| (Cons(x, xs), n) -> x + process xs (n - 1)
```

**Lowering 전략:**

1. **First column (list)**: Constructor pattern → `scf.index_switch`
2. **Second column (n)**: Literal pattern → `arith.cmpi` + `scf.if`

```mlir
// Step 1: Constructor dispatch on list
%list_tag = llvm.extractvalue %list[0] : !llvm.struct<(i32, ptr)>
%tag_index = arith.index_cast %list_tag : i32 to index

%result = scf.index_switch %tag_index : index -> i32
case 0 {
    // Nil case: wildcard on n (no test)
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
}
case 1 {
    // Cons case: extract data
    %data = llvm.extractvalue %list[1] : !llvm.struct<(i32, ptr)>
    %x = llvm.load %data : !llvm.ptr -> i32

    // Step 2: Literal dispatch on n
    %is_zero = arith.cmpi eq, %n, %c0 : i32
    %inner = scf.if %is_zero -> i32 {
        // Case: Cons(x, _), 0 → x
        scf.yield %x : i32
    } else {
        // Case: Cons(x, xs), n → x + process xs (n-1)
        %tail_ptr = llvm.getelementptr %data[1] : (!llvm.ptr) -> !llvm.ptr
        %xs = llvm.load %tail_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>
        %n_minus_1 = arith.subi %n, %c1 : i32
        %rest = func.call @process(%xs, %n_minus_1) : (...) -> i32
        %sum = arith.addi %x, %rest : i32
        scf.yield %sum : i32
    }
    scf.yield %inner : i32
}
```

**핵심 원칙:**

- Constructor column: `scf.index_switch`로 O(1) dispatch
- Literal column: `scf.if` chain으로 O(n) dispatch
- Wildcard: test 없음 (fallthrough 또는 skip)

### 와일드카드 Default Case 처리

**Wildcard (`_`) pattern의 lowering:**

Wildcard는 **어떤 테스트도 생성하지 않는다.**

```fsharp
match x with
| 0 -> "zero"
| 1 -> "one"
| _ -> "other"  // Wildcard: no test
```

```mlir
%is_zero = arith.cmpi eq, %x, %c0 : i32
scf.if %is_zero {
    scf.yield %zero_str : !llvm.ptr<i8>
} else {
    %is_one = arith.cmpi eq, %x, %c1 : i32
    scf.if %is_one {
        scf.yield %one_str : !llvm.ptr<i8>
    } else {
        // _ case: NO comparison, just yield
        scf.yield %other_str : !llvm.ptr<i8>
    }
}
```

**Wildcard optimization in subpatterns:**

```fsharp
match list with
| Cons(_, tail) -> length tail + 1  // Don't extract head
| Nil -> 0
```

```mlir
case 1 {  // Cons
    // Wildcard _: Skip head extraction
    // %head = llvm.load %data  -- NOT generated!

    %tail_ptr = llvm.getelementptr %data[1] : (!llvm.ptr) -> !llvm.ptr
    %tail = llvm.load %tail_ptr : !llvm.ptr -> !llvm.struct<(i32, ptr)>
    // ...
}
```

**Wildcard 최적화의 효과:**

1. **메모리 접근 감소**: 불필요한 load 제거
2. **레지스터 절약**: unused 값을 저장 안 함
3. **DCE 촉진**: Dead code elimination이 더 쉬워짐

### Type Dispatch Pattern

**타입 기반 dispatch (future extension):**

일부 언어는 runtime type으로 dispatch한다:

```fsharp
// Hypothetical type dispatch
match value with
| :? int as n -> n + 1
| :? string as s -> String.length s
| _ -> 0
```

이는 다음으로 lowering 가능:

```mlir
// Type tag dispatch (similar to ADT constructor)
%type_tag = llvm.extractvalue %boxed_value[0] : !llvm.struct<(i32, ptr)>
%tag_index = arith.index_cast %type_tag : i32 to index

scf.index_switch %tag_index : index -> i32
case 0 { /* int case */ }
case 1 { /* string case */ }
default { scf.yield %zero : i32 }
```

현재 FunLang은 ADT constructor만 지원하지만, 동일한 패턴이 적용된다.

---

## Summary and Chapter 20 Preview

### Chapter 19 Recap

**이 장에서 배운 것:**

✅ **Part 1: Match Operation Definition**

1. **Region-based operations**
   - Regions vs basic blocks: encapsulation, verification 장점
   - `funlang.match`는 multiple regions (variadic, each with 1 block)

2. **Match operation semantics**
   - Runtime execution: tag extraction → case selection → variable binding → yield
   - Block arguments for pattern variables

3. **TableGen definition**
   - Traits: `RecursiveSideEffect`, `SingleBlockImplicitTerminator<"YieldOp">`
   - `VariadicRegion<SizedRegion<1>>` for flexible case count
   - Custom assembly format, verifier

4. **YieldOp terminator**
   - Terminator trait, HasParent<"MatchOp"> constraint
   - Dedicated operation (not reusing scf.yield)

5. **C API and F# integration**
   - Builder callback pattern for region construction
   - High-level wrapper: `CreateMatchOp(scrutinee, resultType, buildCases)`
   - Block arguments added in F# callback, mapped in lowering pass

✅ **Part 2: SCF Lowering and Pipeline**

1. **SCF dialect overview**
   - Structured control flow (regions, not goto)
   - `scf.index_switch` for multi-way branching
   - Why SCF before LLVM: structure preservation, optimization, debugging

2. **MatchOpLowering pattern**
   - Tag extraction and index casting
   - Data extraction for pattern variables
   - IRMapping for block argument remapping
   - Region cloning with mapped values
   - `funlang.yield` → `scf.yield` conversion

3. **Complete pipeline**
   - FunLangToSCFPass → FunLangToLLVMPass → SCFToControlFlowPass → ...
   - Pass dependencies and ordering
   - End-to-end example: sum_list

4. **Common errors**
   - Block argument count mismatch
   - Type mismatch in regions
   - Missing scf.yield
   - Wrong tag values
   - Incorrect data extraction

### Pattern Matching Pipeline: Complete

**Phase 6 journey:**

```
Chapter 17: Theory
  ↓
  Decision tree algorithm
  Pattern matrix, specialization/defaulting
  Exhaustiveness checking

Chapter 18: Data Structures
  ↓
  !funlang.list<T> type
  funlang.nil, funlang.cons operations
  TypeConverter, lowering patterns

Chapter 19: Match Compilation (현재)
  ↓
  funlang.match operation
  Region-based structure
  MatchOpLowering to scf.index_switch
  Complete pipeline

Chapter 20: Functional Programs (next)
  ↓
  Realistic examples: map, filter, fold
  Performance analysis
  Debugging functional code
```

**지금까지의 성과:**

| Feature | Chapters | Operations | Status |
|---------|----------|------------|--------|
| Arithmetic | 5-6 | arith.* | ✅ Phase 2 |
| Let bindings | 7 | SSA values | ✅ Phase 2 |
| Control flow | 8 | scf.if | ✅ Phase 2 |
| Functions | 10 | func.func, func.call | ✅ Phase 3 |
| Recursion | 11 | func.call @self | ✅ Phase 3 |
| Closures | 12 | funlang.closure | ✅ Phase 5 |
| Higher-order | 13 | funlang.apply | ✅ Phase 5 |
| Custom dialect | 14-16 | Lowering passes | ✅ Phase 5 |
| Pattern matching | 17-19 | funlang.match | ✅ Phase 6 (현재) |
| Data structures | 17-19 | funlang.nil, funlang.cons | ✅ Phase 6 (현재) |

**다음: Realistic functional programs**

### Chapter 20 Preview: Functional Programs

**Chapter 20에서 할 것:**

1. **Classic list functions**
   - `length`, `map`, `filter`, `fold_left`, `fold_right`
   - Pattern matching + recursion 결합

2. **Composed functions**
   - `sum = fold_left (+) 0`
   - `product = fold_left (*) 1`
   - Higher-order functions로 추상화

3. **Performance analysis**
   - Tail recursion vs non-tail recursion
   - Closure allocation overhead
   - GC pressure measurement

4. **Debugging techniques**
   - IR dumping at each stage
   - printf debugging in functional code
   - Stack trace interpretation

5. **Complete FunLang compiler**
   - All features integrated
   - End-to-end compilation
   - Real-world program examples

**Chapter 20 목표:**

지금까지 배운 모든 기능을 종합하여 **실용적인 함수형 프로그램**을 작성하고 컴파일한다.

**Example program (Chapter 20):**

```fsharp
// Functional list library
let rec map f lst =
    match lst with
    | [] -> []
    | head :: tail -> f head :: map f tail

let rec filter pred lst =
    match lst with
    | [] -> []
    | head :: tail ->
        if pred head then
            head :: filter pred tail
        else
            filter pred tail

let rec fold_left f acc lst =
    match lst with
    | [] -> acc
    | head :: tail -> fold_left f (f acc head) tail

// Usage
let double x = x * 2
let is_even x = x % 2 = 0

let main () =
    let numbers = [1; 2; 3; 4; 5; 6]
    let doubled = map double numbers         // [2; 4; 6; 8; 10; 12]
    let evens = filter is_even doubled       // [2; 4; 6; 8; 10; 12]
    let sum = fold_left (+) 0 evens          // 42
    sum
```

**Generated MLIR (high-level view):**

```mlir
module {
  func.func @map(%f: !funlang.closure, %lst: !funlang.list<i32>)
      -> !funlang.list<i32> {
    %result = funlang.match %lst {
      ^nil: ...
      ^cons(%h, %t): ...
    }
    return %result
  }

  func.func @filter(%pred: !funlang.closure, %lst: !funlang.list<i32>)
      -> !funlang.list<i32> { ... }

  func.func @fold_left(%f: !funlang.closure, %acc: i32, %lst: !funlang.list<i32>)
      -> i32 { ... }

  func.func @main() -> i32 {
    // Build list [1; 2; 3; 4; 5; 6]
    %numbers = ...

    // map double numbers
    %double = funlang.closure @double, () : !funlang.closure
    %doubled = func.call @map(%double, %numbers) : ...

    // filter is_even doubled
    %is_even = funlang.closure @is_even, () : !funlang.closure
    %evens = func.call @filter(%is_even, %doubled) : ...

    // fold_left (+) 0 evens
    %plus = funlang.closure @plus, () : !funlang.closure
    %zero = arith.constant 0 : i32
    %sum = func.call @fold_left(%plus, %zero, %evens) : ...

    return %sum : i32
  }
}
```

**Chapter 20 will show:**

- Complete compilation to native code
- Performance benchmarks
- Comparison with imperative equivalents
- Debugging workflow for functional programs

---

## 튜플 패턴 매칭 (Tuple Pattern Matching)

Chapter 18에서 우리는 `!funlang.tuple<T1, T2, ...>` 타입과 `funlang.make_tuple` 연산을 구현했다. 이제 튜플에 대한 패턴 매칭을 구현하자.

### 튜플 패턴의 특성 (Tuple Pattern Characteristics)

**튜플 패턴은 리스트 패턴과 근본적으로 다르다:**

| 특성 | 리스트 패턴 | 튜플 패턴 |
|------|-------------|-----------|
| 태그 검사 | 필요 (Nil/Cons 구분) | 불필요 |
| 패턴 case 수 | 최소 2개 (Nil, Cons) | 항상 1개 |
| 매칭 실패 가능성 | 있음 | 없음 (항상 매칭) |
| Lowering 대상 | scf.index_switch | 직접 extractvalue |
| 제어 흐름 | 조건부 분기 | 선형 |

**핵심 통찰:**

튜플 패턴은 본질적으로 **구조 분해(destructuring)**다. 항상 매칭이 성공하므로 조건 분기가 필요 없다.

```fsharp
// 리스트: 두 가지 가능성, 조건 분기 필요
match list with
| [] -> expr1       // Nil case
| x :: xs -> expr2  // Cons case

// 튜플: 한 가지 가능성, 조건 분기 불필요
match pair with
| (x, y) -> x + y   // 항상 이 case로
```

### funlang.match 튜플 지원 (Tuple Support in funlang.match)

**튜플 패턴 매칭의 MLIR 표현:**

```mlir
// 소스 코드: let (x, y) = pair in x + y
%pair = funlang.make_tuple(%a, %b) : !funlang.tuple<i32, i32>

%sum = funlang.match %pair : !funlang.tuple<i32, i32> -> i32 {
  ^case(%x: i32, %y: i32):
    %result = arith.addi %x, %y : i32
    funlang.yield %result : i32
}
```

**리스트 패턴과 비교:**

```mlir
// 리스트 패턴: 두 case
%result = funlang.match %list : !funlang.list<i32> -> i32 {
  ^nil:
    %zero = arith.constant 0 : i32
    funlang.yield %zero : i32
  ^cons(%head: i32, %tail: !funlang.list<i32>):
    // ...
    funlang.yield %sum : i32
}

// 튜플 패턴: 한 case만
%result = funlang.match %tuple : !funlang.tuple<i32, i32> -> i32 {
  ^case(%x: i32, %y: i32):
    // 항상 이 case 실행
    funlang.yield %result : i32
}
```

**튜플 패턴의 핵심:**

1. **단일 case**: 분기 불필요
2. **block arguments = 구조 분해된 원소**: `(%x, %y)` → `^case(%x: i32, %y: i32)`
3. **원소 개수 = block argument 개수**: 타입의 arity와 일치

### 튜플 로우어링 구현 (Tuple Lowering Implementation)

**핵심 차이점:**

리스트 패턴 lowering:
1. 태그 추출 (extractvalue [0])
2. scf.index_switch로 분기
3. 각 case에서 데이터 추출

튜플 패턴 lowering:
1. 각 원소 추출 (extractvalue [i])
2. 원래 block의 operation들을 inline
3. 분기 없음!

**TupleMatchLowering 패턴:**

```cpp
// 튜플 패턴 매칭의 lowering은 특별 처리가 필요
// MatchOpLowering 내부에서 튜플 타입 감지 시

LogicalResult matchTuplePattern(MatchOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value input = adaptor.getInput();

  // 튜플은 단일 case만 가짐
  assert(op.getCases().size() == 1 && "Tuple match must have exactly one case");

  Region& caseRegion = op.getCases().front();
  Block& caseBlock = caseRegion.front();

  // 구조체에서 각 원소 추출
  auto structType = input.getType().cast<LLVM::LLVMStructType>();
  IRMapping mapper;

  for (size_t i = 0; i < caseBlock.getNumArguments(); ++i) {
    Value extracted = rewriter.create<LLVM::ExtractValueOp>(
        loc, structType.getBody()[i], input, i);
    mapper.map(caseBlock.getArgument(i), extracted);
  }

  // 원래 block의 operations를 현재 위치에 inline
  for (Operation& op : caseBlock.getOperations()) {
    if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
      // yield의 값으로 match 결과를 대체
      Value yieldValue = mapper.lookupOrDefault(yieldOp.getValue());
      rewriter.replaceOp(op, yieldValue);
    } else {
      rewriter.clone(op, mapper);
    }
  }

  return success();
}
```

**Lowering 결과 비교:**

```mlir
// Before lowering (FunLang)
%sum = funlang.match %pair : !funlang.tuple<i32, i32> -> i32 {
  ^case(%x: i32, %y: i32):
    %result = arith.addi %x, %y : i32
    funlang.yield %result : i32
}

// After lowering (LLVM dialect) - 분기 없음!
%x = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i32)>
%y = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i32)>
%sum = arith.addi %x, %y : i32
```

**리스트 패턴 lowering과 비교:**

```mlir
// 리스트: scf.index_switch 필요
%tag = llvm.extractvalue %list[0] : !llvm.struct<(i32, ptr)>
%tagIndex = arith.index_cast %tag : i32 to index
%result = scf.index_switch %tagIndex : index -> i32
case 0 {  // Nil
  %zero = arith.constant 0 : i32
  scf.yield %zero : i32
}
case 1 {  // Cons
  %ptr = llvm.extractvalue %list[1] : !llvm.struct<(i32, ptr)>
  %head = llvm.load %ptr : !llvm.ptr -> i32
  // ...
  scf.yield %sum : i32
}
default {
  llvm.unreachable
}

// 튜플: extractvalue만으로 충분
%x = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i32)>
%y = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i32)>
%result = arith.addi %x, %y : i32
```

### 중첩 패턴 (Nested Patterns)

**튜플 안에 리스트가 있는 경우:**

```fsharp
// 두 리스트를 튜플로 묶어서 동시에 패턴 매칭
let rec zip xs ys =
  match (xs, ys) with
  | ([], _) -> []
  | (_, []) -> []
  | (x :: xs', y :: ys') -> (x, y) :: zip xs' ys'
```

**MLIR 표현:**

```mlir
// 1단계: 튜플 구조 분해
%tuple = funlang.make_tuple(%xs, %ys) : !funlang.tuple<!funlang.list<i32>, !funlang.list<i32>>

// 2단계: 튜플에서 두 리스트 추출
%xs_extracted = ... extractvalue [0] ...
%ys_extracted = ... extractvalue [1] ...

// 3단계: 첫 번째 리스트에 대해 패턴 매칭
%result = funlang.match %xs_extracted : !funlang.list<i32> -> ... {
  ^nil:
    // 빈 리스트 반환
  ^cons(%x: i32, %xs_tail: !funlang.list<i32>):
    // 4단계: 두 번째 리스트에 대해 중첩 패턴 매칭
    %inner = funlang.match %ys_extracted : !funlang.list<i32> -> ... {
      ^nil:
        // 빈 리스트 반환
      ^cons(%y: i32, %ys_tail: !funlang.list<i32>):
        // (x, y) :: zip xs_tail ys_tail
        %pair = funlang.make_tuple(%x, %y) : !funlang.tuple<i32, i32>
        %rest = func.call @zip(%xs_tail, %ys_tail) : ...
        %result = funlang.cons %pair, %rest : ...
        funlang.yield %result
    }
    funlang.yield %inner
}
```

**중첩 패턴 lowering 전략:**

1. **외부에서 내부로**: 가장 바깥 패턴부터 lowering
2. **튜플 먼저**: 튜플 분해는 조건 없이 extractvalue
3. **리스트는 분기**: 각 리스트 패턴은 scf.index_switch 필요
4. **깊이 우선**: 내부 패턴이 완전히 lowering된 후 외부로

### MatchOpLowering 확장: 튜플 지원

**기존 MatchOpLowering에 튜플 분기 추가:**

```cpp
class MatchOpLowering : public OpConversionPattern<MatchOp> {
public:
  LogicalResult matchAndRewrite(MatchOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Type inputType = op.getInput().getType();

    // 튜플인지 확인
    if (auto tupleType = inputType.dyn_cast<funlang::TupleType>()) {
      return matchTuplePattern(op, adaptor, rewriter, tupleType);
    }

    // 리스트인 경우 기존 로직 사용
    if (auto listType = inputType.dyn_cast<funlang::ListType>()) {
      return matchListPattern(op, adaptor, rewriter, listType);
    }

    return op.emitError("unsupported match input type");
  }

private:
  LogicalResult matchTuplePattern(MatchOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   funlang::TupleType tupleType) const {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // 튜플은 단일 case만 허용
    if (op.getCases().size() != 1) {
      return op.emitError("tuple match must have exactly one case");
    }

    Region& caseRegion = op.getCases().front();
    Block& caseBlock = caseRegion.front();

    // block argument 개수 검증
    if (caseBlock.getNumArguments() != tupleType.getNumElements()) {
      return op.emitError() << "tuple arity mismatch: type has "
                            << tupleType.getNumElements() << " elements but pattern has "
                            << caseBlock.getNumArguments();
    }

    // 각 원소 추출 및 매핑
    auto structType = getTypeConverter()->convertType(tupleType);
    IRMapping mapper;

    for (size_t i = 0; i < caseBlock.getNumArguments(); ++i) {
      auto elemType = structType.cast<LLVM::LLVMStructType>().getBody()[i];
      Value extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, elemType, input, i);
      mapper.map(caseBlock.getArgument(i), extracted);
    }

    // 현재 위치에 operations inline
    Value resultValue;
    for (Operation& caseOp : caseBlock.getOperations()) {
      if (auto yieldOp = dyn_cast<YieldOp>(&caseOp)) {
        resultValue = mapper.lookupOrDefault(yieldOp.getValue());
      } else {
        rewriter.clone(caseOp, mapper);
      }
    }

    rewriter.replaceOp(op, resultValue);
    return success();
  }

  LogicalResult matchListPattern(MatchOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  funlang::ListType listType) const {
    // 기존 리스트 패턴 매칭 로직...
    // (scf.index_switch 사용)
  }
};
```

### 리스트 vs 튜플 패턴 매칭 종합 비교

| 구분 | 리스트 패턴 | 튜플 패턴 |
|------|-------------|-----------|
| **타입** | `!funlang.list<T>` | `!funlang.tuple<T1, T2, ...>` |
| **case 수** | 2개 이상 (Nil, Cons, ...) | 정확히 1개 |
| **태그 검사** | 필요 (extractvalue [0]) | 불필요 |
| **분기** | scf.index_switch | 없음 |
| **데이터 추출** | ptr load 필요 | extractvalue만 |
| **패턴 변수 바인딩** | 조건부 (case 안에서) | 무조건 |
| **default case** | 있음 (unreachable) | 없음 |
| **lowering 복잡도** | 높음 | 낮음 |
| **최종 코드 크기** | 크다 (분기 포함) | 작다 (선형) |

**생성되는 코드 비교:**

```mlir
// 리스트 패턴 매칭 결과 (복잡)
%tag = llvm.extractvalue %list[0] : !llvm.struct<(i32, ptr)>
%idx = arith.index_cast %tag : i32 to index
%result = scf.index_switch %idx : index -> i32
case 0 {  // ~10 lines
  ...
  scf.yield %val0 : i32
}
case 1 {  // ~15 lines
  ...
  scf.yield %val1 : i32
}
default {  // ~3 lines
  llvm.unreachable
}

// 튜플 패턴 매칭 결과 (단순)
%x = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i32)>
%y = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i32)>
// ... operations inline ...
```

### 튜플 패턴의 최적화 기회

**튜플 패턴 매칭은 이미 최적화되어 있다:**

1. **분기 제거**: 조건문 없이 바로 연산
2. **Inlining 자동**: 별도 함수 호출 없음
3. **레지스터 친화적**: 작은 튜플은 레지스터에 유지

**추가 최적화 가능:**

```mlir
// Before: 사용하지 않는 원소도 추출
%pair = funlang.make_tuple(%a, %b) : !funlang.tuple<i32, i32>
%result = funlang.match %pair {
  ^case(%x: i32, %y: i32):  // %y 사용 안 함
    funlang.yield %x : i32
}

// After: Dead code elimination으로 %y 추출 제거
%x = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i32)>
// %y 추출 생략됨
%result = %x
```

### 와일드카드 패턴 (Wildcard Pattern)

**사용하지 않는 원소는 와일드카드로:**

```fsharp
// 첫 번째 원소만 사용
let fst pair = match pair with (x, _) -> x

// 두 번째 원소만 사용
let snd pair = match pair with (_, y) -> y
```

**MLIR에서 와일드카드:**

```mlir
// 와일드카드는 block argument가 없음
%fst = funlang.match %pair : !funlang.tuple<i32, i32> -> i32 {
  ^case(%x: i32):  // %y 자리에 block argument 없음 (또는 unused)
    funlang.yield %x : i32
}
```

**Lowering 시 최적화:**

```cpp
// 와일드카드 패턴 처리
for (size_t i = 0; i < tupleType.getNumElements(); ++i) {
  BlockArgument arg = caseBlock.getArgument(i);
  if (!arg.use_empty()) {  // 사용되는 경우에만 추출
    Value extracted = rewriter.create<LLVM::ExtractValueOp>(...);
    mapper.map(arg, extracted);
  }
  // 와일드카드 (미사용)면 extractvalue 생략
}
```

### Summary: 튜플 패턴 매칭

**구현 완료:**

- [x] 튜플 패턴의 특성 이해 (단일 case, 분기 없음)
- [x] funlang.match에서 튜플 타입 지원
- [x] MatchOpLowering에서 튜플/리스트 분기 처리
- [x] extractvalue 체인으로 원소 추출
- [x] 분기 없이 inline lowering
- [x] 중첩 패턴 (튜플 + 리스트) 처리 전략
- [x] 와일드카드 패턴과 dead code elimination

**튜플 패턴 vs 리스트 패턴:**

| 측면 | 리스트 | 튜플 |
|------|--------|------|
| 패턴 case | 다중 | 단일 |
| 제어 흐름 | scf.index_switch | 선형 |
| 태그 검사 | 필요 | 불필요 |
| Lowering | 복잡 | 단순 |
| 생성 코드 | 분기 포함 | extractvalue만 |

**다음:**

- Chapter 20에서 튜플을 활용한 실제 프로그램 (zip, fst/snd)
- 중첩 튜플과 포인트 예제

---

## Conclusion

**Chapter 19 완료!**

우리는 `funlang.match` operation을 정의하고 SCF dialect로 lowering하여 **패턴 매칭 컴파일 파이프라인**을 완성했다.

**핵심 개념:**

1. **Region-based operations**: Encapsulation과 verification을 위한 구조
2. **Multi-stage lowering**: FunLang → SCF → CF → LLVM (progressive refinement)
3. **IRMapping**: Block arguments를 실제 values로 remapping
4. **Builder callback pattern**: F#에서 regions를 구축하는 방법
5. **튜플 패턴 매칭**: 분기 없는 extractvalue 기반 lowering

**Phase 6 진행 상황:**

- ✅ Chapter 17: Pattern matching theory (Decision tree algorithm)
- ✅ Chapter 18: List operations (funlang.nil, funlang.cons, funlang.tuple, funlang.make_tuple)
- ✅ Chapter 19: Match compilation (funlang.match, 리스트/튜플 패턴 lowering)
- ⏭️ Chapter 20: Functional programs (map, filter, fold, zip - realistic examples)

**다음 장에서 만나요!**

