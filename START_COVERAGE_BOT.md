# 🚀 Coverage Bot 즉시 시작 가이드

## 📋 빠른 체크리스트

### 1️⃣ GitHub Secrets 설정 (필수)
**Settings → Secrets and variables → Actions** 에서:

| Secret | 값 | 설명 |
|--------|---|------|
| `GH_PAT_COVERAGE_BOT` | Personal Access Token | repo, workflow 권한 필요 |
| `COV_TARGET` | `60` | Phase 3 목표 (나중에 75로 변경) |
| `COV_GAIN_MIN` | `0.5` | 최소 향상폭 |

### 2️⃣ Bot 수동 실행 (첫 시동)
```bash
# GitHub CLI로 실행
gh workflow run coverage-bot.yml

# 또는 GitHub UI에서:
# Actions → 🤖 Self-Healing Coverage Bot → Run workflow
```

### 3️⃣ 결과 확인
- 약 5-10분 후 새 PR 생성됨
- Branch: `feature/auto-coverage-YYYYMMDD`
- 자동으로 커버리지 향상 테스트 포함

---

## 🔧 로컬 테스트 (선택사항)

```bash
# 로컬에서 봇 동작 테스트
export COV_TARGET=60
export COV_GAIN_MIN=0.5
python scripts/coverage_autofix.py
```

---

## 📊 현재 상태

- **현재 커버리지**: 45.01% (branch coverage 포함)
- **Phase 3 목표**: 60%
- **예상 소요**: 3-5회 자동 PR

---

## ⚡ 즉시 실행 명령어

```bash
# 1. PR 머지
gh pr merge 3 --squash

# 2. 최신 main pull
git checkout main && git pull

# 3. Bot 실행
gh workflow run coverage-bot.yml

# 4. 실행 상태 확인
gh run list --workflow=coverage-bot.yml
```

---

## 🎯 예상 시나리오

1. **오늘**: Bot이 45% → 47-48% PR 생성
2. **내일**: 자동 실행으로 48% → 50-52% 
3. **3일차**: 52% → 55-57%
4. **4-5일**: 60% 도달! 🎉

그 후 `COV_TARGET=75`로 변경하면 Phase 4 자동 시작!

---

## ❓ 문제 해결

### Bot이 PR을 만들지 않는다면?
1. Secrets 확인 (특히 PAT)
2. Actions 로그 확인
3. 수동으로 스크립트 실행해서 에러 확인

### 커버리지가 오르지 않는다면?
- `COV_GAIN_MIN`을 `0.3`으로 낮춰보기
- 자동 생성된 테스트 품질 확인

---

**지금 바로 시작하세요! 🤖**