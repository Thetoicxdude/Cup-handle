"""
雙引擎策略模組 (Dual-Engine Strategy Module)

此模組提供市場適應性策略功能，根據市場狀態自動切換趨勢策略與均值回歸策略。

主要組件：
- MarketStateClassifier: 市場狀態分類器，根據 ADX 與 BBW 判定市場狀態
- TrendStrategy: 趨勢策略，在趨勢市場中執行型態突破交易
- MeanReversionStrategy: 均值回歸策略，在震盪市場中執行布林均值回歸交易
- SignalResolver: 信號解決器，處理策略信號衝突
- ATRPositionSizer: ATR 倉位計算器，用於趨勢策略
- FixedFractionalSizer: 固定金額倉位計算器，用於震盪策略
- DualEngineStrategy: 雙引擎策略主類別，整合所有組件

使用範例：
    from pattern_quant.strategy import (
        DualEngineStrategy,
        DualEngineConfig,
        MarketState,
    )
    
    # 建立配置
    config = DualEngineConfig(enabled=True)
    
    # 建立策略引擎
    engine = DualEngineStrategy(config=config)
    
    # 執行分析
    result = engine.analyze(
        symbol="AAPL",
        prices=prices,
        highs=highs,
        lows=lows,
        volumes=volumes,
    )
    
    # 檢查市場狀態
    if result.market_state.state == MarketState.TREND:
        print("趨勢市場")
"""

# 資料模型
from pattern_quant.strategy.models import (
    MarketState,
    ADXResult,
    BBWResult,
    MarketStateResult,
    TrendSignal,
    MeanReversionSignal,
    CandlePattern,
    DualEngineConfig,
)

# 配置管理
from pattern_quant.strategy.config import DualEngineConfigManager

# 市場狀態分類器
from pattern_quant.strategy.market_classifier import MarketStateClassifier

# 策略類別
from pattern_quant.strategy.trend_strategy import TrendStrategy, TrendStrategyConfig
from pattern_quant.strategy.mean_reversion_strategy import MeanReversionStrategy

# 信號解決器
from pattern_quant.strategy.signal_resolver import SignalResolver, ResolvedSignal, ConflictType

# 倉位計算器
from pattern_quant.strategy.position_sizer import (
    ATRPositionSizer,
    FixedFractionalSizer,
    PositionSizeResult,
)

# 雙引擎主類別
from pattern_quant.strategy.dual_engine import DualEngineStrategy, DualEngineAnalysisResult

__all__ = [
    # 資料模型 (Data Models)
    "MarketState",
    "ADXResult",
    "BBWResult",
    "MarketStateResult",
    "TrendSignal",
    "MeanReversionSignal",
    "CandlePattern",
    "DualEngineConfig",
    # 配置管理 (Config Management)
    "DualEngineConfigManager",
    # 市場狀態分類器 (Market State Classifier)
    "MarketStateClassifier",
    # 策略類別 (Strategies)
    "TrendStrategy",
    "TrendStrategyConfig",
    "MeanReversionStrategy",
    # 信號解決器 (Signal Resolver)
    "SignalResolver",
    "ResolvedSignal",
    "ConflictType",
    # 倉位計算器 (Position Sizers)
    "ATRPositionSizer",
    "FixedFractionalSizer",
    "PositionSizeResult",
    # 雙引擎主類別 (Main Engine)
    "DualEngineStrategy",
    "DualEngineAnalysisResult",
]
