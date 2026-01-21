"""
Signal Optimization Layer - 訊號優化層

此模組提供技術指標計算、評分卡機制與自動調參功能。
"""

from .models import (
    RSIResult,
    MACDResult,
    BollingerResult,
    StochasticResult,
    IndicatorSnapshot,
)
from .indicator_pool import IndicatorPool
from .factor_config import (
    IndicatorType,
    RSIConfig,
    VolumeConfig,
    MACDConfig,
    EMAConfig,
    BollingerConfig,
    FactorConfig,
    FactorConfigManager,
)
from .signal_optimizer import (
    SignalStrength,
    ScoreDetail,
    OptimizedSignal,
    SignalOptimizer,
)
from .auto_tuner import (
    BacktestResult,
    TuningProgress,
    AutoTuner,
)

__all__ = [
    "RSIResult",
    "MACDResult",
    "BollingerResult",
    "StochasticResult",
    "IndicatorSnapshot",
    "IndicatorPool",
    "IndicatorType",
    "RSIConfig",
    "VolumeConfig",
    "MACDConfig",
    "EMAConfig",
    "BollingerConfig",
    "FactorConfig",
    "FactorConfigManager",
    "SignalStrength",
    "ScoreDetail",
    "OptimizedSignal",
    "SignalOptimizer",
    "BacktestResult",
    "TuningProgress",
    "AutoTuner",
]
