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
