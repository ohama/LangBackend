# LangBackend Feature Tests
# Uses fslit (https://github.com/ohama/fslit) for file-based testing

FSLIT := fslit
MLIR_OPT := mlir-opt

# Test directories by phase
PHASES := phase2-core phase3-functions phase4-closures phase5-dialect phase6-patterns

.PHONY: all $(PHASES) mlir e2e compiler clean help check

# Run v2.0 compiler tests (FunLang.Compiler.Cli)
compiler:
	@echo "=== Compiler E2E Tests (Phase 8+) ==="
	@$(FSLIT) compiler/

# Run all tests (interpreter + MLIR IR + E2E)
all: $(PHASES)
	@echo ""
	@echo "=== Interpreter tests completed ==="
	@echo ""
	@$(MAKE) -C mlir check
	@echo ""
	@$(MAKE) -C e2e
	@echo ""
	@echo "=== All tests completed ==="

# Run MLIR IR tests only
mlir:
	@$(MAKE) -C mlir check

# Run E2E compilation tests only
e2e:
	@$(MAKE) -C e2e

# Run tests by phase
phase2-core:
	@echo "=== Phase 2: Core Language (Arithmetic, Let, Control Flow) ==="
	@$(FSLIT) phase2-core/

phase3-functions:
	@echo "=== Phase 3: Functions & Recursion ==="
	@$(FSLIT) phase3-functions/

phase4-closures:
	@echo "=== Phase 4: Closures & Higher-Order Functions ==="
	@$(FSLIT) phase4-closures/

phase5-dialect:
	@echo "=== Phase 5: Custom Dialect Operations ==="
	@$(FSLIT) phase5-dialect/

phase6-patterns:
	@echo "=== Phase 6: Pattern Matching & Data Structures ==="
	@$(FSLIT) phase6-patterns/

# Build FunLang first, then run tests
check:
	@echo "Building FunLang..."
	@dotnet build ../LangTutorial/FunLang/FunLang.fsproj -v q
	@$(MAKE) all

# Show test counts
count:
	@echo "Interpreter tests (fslit):"
	@echo "  phase2-core:      $$(ls phase2-core/*.flt 2>/dev/null | wc -l) tests"
	@echo "  phase3-functions: $$(ls phase3-functions/*.flt 2>/dev/null | wc -l) tests"
	@echo "  phase4-closures:  $$(ls phase4-closures/*.flt 2>/dev/null | wc -l) tests"
	@echo "  phase5-dialect:   $$(ls phase5-dialect/*.flt 2>/dev/null | wc -l) tests"
	@echo "  phase6-patterns:  $$(ls phase6-patterns/*.flt 2>/dev/null | wc -l) tests"
	@echo "  ----------------------------------------"
	@echo "  Subtotal:         $$(find . -name '*.flt' | wc -l) tests"
	@echo ""
	@echo "MLIR IR tests (mlir-opt):"
	@echo "  mlir/:            $$(find mlir -name '*.mlir' 2>/dev/null | wc -l) tests"
	@echo ""
	@echo "E2E compilation tests (mlir-opt + llc + cc):"
	@echo "  e2e/:             $$(find e2e -name '*.mlir' 2>/dev/null | wc -l) tests"
	@echo ""
	@echo "  ========================================"
	@flt=$$(find . -name '*.flt' | wc -l); \
	mlir=$$(find mlir -name '*.mlir' 2>/dev/null | wc -l); \
	e2e=$$(find e2e -name '*.mlir' 2>/dev/null | wc -l); \
	echo "  TOTAL:            $$((flt + mlir + e2e)) tests"

# Clean temporary files
clean:
	@find . -name '*.tmp' -delete
	@find . -name '*.out' -delete
	@echo "Cleaned temporary files"

# Help
help:
	@echo "LangBackend Test Suite"
	@echo ""
	@echo "Usage:"
	@echo "  make              Run all tests (interpreter + MLIR IR + E2E)"
	@echo "  make check        Build FunLang and run all tests"
	@echo "  make mlir         Run MLIR IR validation tests only"
	@echo "  make e2e          Run E2E compilation tests only"
	@echo "  make phase2-core  Run Phase 2 tests (arithmetic, let, control flow)"
	@echo "  make phase3-functions  Run Phase 3 tests (functions, recursion)"
	@echo "  make phase4-closures   Run Phase 4 tests (closures, HOF)"
	@echo "  make phase5-dialect    Run Phase 5 tests (dialect operations)"
	@echo "  make phase6-patterns   Run Phase 6 tests (lists, patterns, tuples)"
	@echo "  make count        Show test counts"
	@echo "  make clean        Remove temporary files"
	@echo "  make help         Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - fslit: Interpreter tests"
	@echo "  - mlir-opt, mlir-translate, llc, cc: MLIR/E2E tests"
