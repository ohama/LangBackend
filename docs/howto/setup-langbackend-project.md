---
created: 2026-02-02
description: LangBackend 프로젝트 초기 설정 및 submodule 구성
---

# LangBackend 프로젝트 설정

FunLang 백엔드 프로젝트를 생성하고 submodule을 구성하는 방법.

## The Insight

새 프로젝트를 만들 때 git 설정, SSH alias를 사용한 origin 지정, submodule 추가를 순서대로 진행해야 한다. 기존 디렉토리가 있으면 백업 후 submodule로 교체한다.

## Why This Matters

- git config 없이 커밋하면 잘못된 사용자 정보가 기록됨
- HTTPS URL 대신 SSH alias를 사용해야 여러 GitHub 계정 환경에서 인증 문제 없음
- 기존 디렉토리를 삭제하지 않고 submodule 추가하면 에러 발생

## Recognition Pattern

- 새 프로젝트 생성
- 여러 GitHub 계정 사용 환경
- Claude-Config, LangTutorial 등 공통 submodule 필요

## The Approach

1. git init 및 사용자 설정
2. SSH alias로 origin 지정
3. 기존 디렉토리 백업 → 삭제 → submodule 추가 → 복원
4. README.md 작성

### Step 1: Git 초기화 및 사용자 설정

```bash
git init

git config --local user.name "ohama"
git config --local user.email "ohama100@gmail.com"
```

### Step 2: Origin 설정

SSH alias를 사용하여 remote origin 설정.

```bash
git remote add origin git@github-ohama:ohama/LangBackend.git

# 확인
git remote -v
```

결과:
```
origin  git@github-ohama:ohama/LangBackend.git (fetch)
origin  git@github-ohama:ohama/LangBackend.git (push)
```

### Step 3: .claude Submodule 추가

기존 `.claude/` 디렉토리가 있는 경우 백업 후 추가.

```bash
# 백업
mv .claude/settings.local.json /tmp/

# 삭제
rm -rf .claude

# submodule 추가
git submodule add git@github-ohama:ohama/Claude-Config.git .claude

# 백업 복원
mv /tmp/settings.local.json .claude/
```

### Step 4: LangTutorial Submodule 추가

```bash
git submodule add git@github-ohama:ohama/LangTutorial.git LangTutorial
```

### Step 5: 확인

```bash
git submodule status
```

결과:
```
 eeb2c049f64fd196ec14f1b5c8685a515a87f54b .claude (heads/master)
 e4f2af2dc78c625a61b6682abfee7419a62610d4 LangTutorial (milestone-v3.0-59-ge4f2af2)
```

## Example

최종 `.gitmodules` 파일:

```ini
[submodule ".claude"]
	path = .claude
	url = git@github-ohama:ohama/Claude-Config.git
[submodule "LangTutorial"]
	path = LangTutorial
	url = git@github-ohama:ohama/LangTutorial.git
```

최종 디렉토리 구조:

```
LangBackend/
├── .claude/              # Claude 설정 (submodule)
├── .git/
├── .gitmodules
├── LangTutorial/         # FunLang 인터프리터 (submodule)
└── README.md
```

## 체크리스트

- [ ] git init 완료
- [ ] git config (user.name, user.email) 설정
- [ ] SSH alias로 origin 설정
- [ ] .claude submodule 추가 (settings.local.json 보존)
- [ ] LangTutorial submodule 추가
- [ ] README.md 작성

## 관련 문서

- [setup-git-submodules-with-ssh-alias](../../../LangLSP/docs/howto/setup-git-submodules-with-ssh-alias.md) - SSH alias로 submodule 설정
