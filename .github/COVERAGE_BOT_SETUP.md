# 🤖 Coverage Bot 설정 가이드

## 🔑 필수 Secrets 설정

### 1. Personal Access Token (PAT) 생성

1. GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)**
2. **Generate new token (classic)** 클릭
3. 설정:
   - **Note**: `coverage-bot`
   - **Expiration**: 90 days (또는 No expiration)
   - **Scopes**: 
     - ✅ `repo` (모든 하위 항목)
     - ✅ `workflow`
4. **Generate token** → 토큰 복사

### 2. Repository Secrets 추가

**Settings → Secrets and variables → Actions → New repository secret**

| Name | Value | Required |
|------|-------|----------|
| `GH_PAT_COVERAGE_BOT` | 위에서 복사한 PAT | ✅ |
| `COV_TARGET` | `60` | ✅ |
| `COV_GAIN_MIN` | `0.5` | ❌ (기본값 0.5) |
| `GDRIVE_FILE_ID` | Google Drive 파일 ID | ❌ |
| `GDRIVE_SHA256` | 파일 SHA-256 체크섬 | ❌ |
| `SLACK_COV_WEBHOOK` | Slack Webhook URL | ❌ |

## 🚀 Bot 실행 방법

### 방법 1: GitHub UI
1. **Actions** 탭 → **🤖 Self-Healing Coverage Bot**
2. **Run workflow** 버튼
3. **Run workflow** 확인

### 방법 2: GitHub CLI
```bash
gh workflow run coverage-bot.yml
```

### 방법 3: 자동 스크립트
```bash
bash scripts/quick_bot_start.sh
```

## 📊 Coverage 목표 단계

| Phase | Target | Secrets 설정 |
|-------|--------|--------------|
| Phase 3 | 60% | `COV_TARGET=60` |
| Phase 4 | 75% | `COV_TARGET=75` |
| Phase 5 | 85% | `COV_TARGET=85` |

## 🔄 자동 실행 일정

- **매일**: 3:30 AM UTC (12:30 PM KST)
- **수동**: 언제든지 workflow_dispatch

## 📈 예상 진행 상황

```
Day 1: 45% → 47% (첫 PR)
Day 2: 47% → 49% 
Day 3: 49% → 52%
Day 4: 52% → 55%
Day 5: 55% → 58%
Day 6: 58% → 60% ✅ Phase 3 완료!
```

## ⚙️ CI Threshold 조정

커버리지가 안정적으로 유지되면:

```yaml
# .github/workflows/ci.yml
env:
  COV_FAIL_UNDER: 35  # → 45 → 50 → 60 순차 상향
```

## 🔍 문제 해결

### Bot이 실행되지 않음
- PAT 권한 확인 (repo, workflow)
- Secrets 이름 정확히 확인

### PR이 생성되지 않음
- 커버리지 향상이 0.5% 미만일 수 있음
- `COV_GAIN_MIN=0.3`으로 낮춰보기

### 테스트가 실패함
- 자동 생성된 테스트의 mock 확인
- `tests/conftest.py`의 전역 mock 설정 확인

---

**준비되셨나요? 지금 바로 시작하세요! 🚀**