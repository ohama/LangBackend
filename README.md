# LangBackend

FunLang 언어의 백엔드 구현 프로젝트.

[LangTutorial](https://github.com/ohama/LangTutorial)에서 구현한 FunLang 인터프리터를 기반으로, 컴파일러 백엔드와 언어 도구를 개발합니다.

## 목표

- 코드 생성 (Code Generation)
- 최적화 (Optimization)
- 런타임/VM 구현
- LSP (Language Server Protocol) 지원

## 프로젝트 구조

```
LangBackend/
├── .claude/              # Claude 설정 (submodule)
├── LangTutorial/         # FunLang 인터프리터 (submodule)
└── src/                  # 백엔드 구현 (예정)
```

## FunLang 언어 개요

LangTutorial에서 구현된 FunLang은 다음 기능을 지원합니다:

- 사칙연산, 비교/논리 연산
- 변수 바인딩 (`let`, `let rec`)
- 조건문 (`if-then-else`)
- 함수 정의/호출, 클로저, 재귀
- 튜플, 리스트
- 패턴 매칭
- Hindley-Milner 타입 추론

## 관련 프로젝트

- [LangTutorial](https://github.com/ohama/LangTutorial) - FunLang 인터프리터 구현 튜토리얼
- [LangLSP](https://github.com/ohama/LangLSP) - FunLang LSP 서버

## 라이선스

MIT
