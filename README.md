# Pattern Quant Trading System

自動交易模型 - 基於技術形態識別的量化交易系統

## 功能特點

- 杯柄形態識別
- 趨勢分析與過濾
- 風險管理
- 回測引擎
- 策略優化
- 視覺化儀表板

## 本地安裝

```bash
pip install -e .
```

## 本地運行

```bash
python run.py
```

或使用批次檔：

```bash
run.bat
```

或直接運行 Streamlit：

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud 部署

### 快速部署步驟

1. **Fork 或 Clone 此倉庫到你的 GitHub 帳號**

2. **前往 [Streamlit Cloud](https://share.streamlit.io/)**

3. **點擊 "New app" 並選擇：**
   - Repository: `你的用戶名/Thetoicxdude`
   - Branch: `master`
   - Main file path: `streamlit_app.py`

4. **點擊 "Deploy"**

5. **等待部署完成**（通常需要 2-5 分鐘）

### 配置說明

- `streamlit_app.py` - Streamlit Cloud 入口文件
- `requirements.txt` - Python 依賴套件
- `.streamlit/config.toml` - Streamlit 配置
- `packages.txt` - 系統級依賴（如需要）

### 環境變數（可選）

在 Streamlit Cloud 的 App settings 中可以設置：

- `DATABASE_URL` - 資料庫連接字串（使用真實資料庫時）
- `TOTAL_CAPITAL` - 總資金（預設: 1000000）
- `REFRESH_INTERVAL` - 刷新間隔秒數（預設: 30）

## 專案結構

- `pattern_quant/core/` - 核心演算法
- `pattern_quant/data/` - 資料獲取
- `pattern_quant/db/` - 資料庫管理
- `pattern_quant/risk/` - 風險管理
- `pattern_quant/ui/` - 使用者介面
- `pattern_quant/strategy/` - 交易策略
- `pattern_quant/evolution/` - 演化優化
- `pattern_quant/optimization/` - 參數優化
- `tests/` - 測試檔案

## 技術棧

- **前端**: Streamlit, Plotly
- **數據處理**: Pandas, NumPy, SciPy
- **數據源**: yfinance
- **資料庫**: PostgreSQL (可選)
- **測試**: pytest, hypothesis
