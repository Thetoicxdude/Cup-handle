# Pattern Quant Trading System

自動交易模型 - 基於技術形態識別的量化交易系統

## 功能特點

- 杯柄形態識別
- 趨勢分析與過濾
- 風險管理
- 回測引擎
- 策略優化
- 視覺化儀表板

## 安裝

```bash
pip install -e .
```

## 使用

```bash
python run.py
```

或使用批次檔：

```bash
run.bat
```

## 專案結構

- `pattern_quant/core/` - 核心演算法
- `pattern_quant/data/` - 資料獲取
- `pattern_quant/db/` - 資料庫管理
- `pattern_quant/risk/` - 風險管理
- `pattern_quant/ui/` - 使用者介面
- `tests/` - 測試檔案
