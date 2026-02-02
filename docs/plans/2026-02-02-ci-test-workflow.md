# CI Test Workflow 設計

> 日期: 2026-02-02
> 狀態: 已確認

## 目標

建立 CI workflow，確保 PR 與主要分支的程式碼品質。

## 觸發條件

- Push 到 `main` 或 `dev` 分支
- 所有 PR（開啟、更新、重新開啟）

## 檢查項目

| 檢查 | 工具 | 說明 |
|------|------|------|
| Lint | ruff | 程式碼風格與 lint 檢查 |
| Type Check | mypy | 靜態型別檢查 |
| Test | pytest | 單元測試 |

## Job 結構

```
┌─────────────┐     ┌─────────────┐
│    lint     │     │    test     │
│ (ruff+mypy) │     │  (pytest)   │
│   py3.11    │     │ py3.11/12/13│
└─────────────┘     └─────────────┘
      ↓                   ↓
      └───── 並行執行 ─────┘
```

### lint job

- Python 版本: 3.11（固定，因結果不受版本影響）
- 執行順序: ruff check → mypy
- 失敗即停止

### test job

- Python 版本矩陣: 3.11, 3.12, 3.13
- 作業系統: Ubuntu (latest)
- 執行: pytest

## 技術決策

1. **使用 uv 安裝依賴** - 與現有 docs.yml 一致，速度快
2. **Lint 與 Test 並行** - 減少整體執行時間
3. **Lint 只跑一次** - ruff/mypy 結果與 Python 版本無關

## 檔案

- `.github/workflows/ci.yml`

## 未來擴展

- [ ] 測試覆蓋率報告 (pytest-cov)
- [ ] 安全性檢查 (bandit)
- [ ] Smoke test（安裝後驗證）
