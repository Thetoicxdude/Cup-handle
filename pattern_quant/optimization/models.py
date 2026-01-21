"""
指標計算結果資料模型

定義各技術指標的計算結果資料類別。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RSIResult:
    """RSI 計算結果"""
    value: float                    # 當前 RSI 值 (0-100)
    is_overbought: bool             # 是否超買
    is_oversold: bool               # 是否超賣
    trend_zone: bool                # 是否在趨勢區間
    support_bounce: bool            # 是否從支撐區反彈


@dataclass
class MACDResult:
    """MACD 計算結果"""
    macd_line: float                # MACD 線
    signal_line: float              # 訊號線
    histogram: float                # 柱狀圖
    above_zero: bool                # 是否在零軸之上
    golden_cross: bool              # 是否剛發生黃金交叉


@dataclass
class BollingerResult:
    """布林通道計算結果"""
    upper: float                    # 上軌
    middle: float                   # 中軌 (SMA)
    lower: float                    # 下軌
    bandwidth: float                # 帶寬
    squeeze: bool                   # 是否處於壓縮狀態
    breakout_upper: bool            # 是否突破上軌


@dataclass
class StochasticResult:
    """KD 指標計算結果"""
    k_value: float                  # K 值
    d_value: float                  # D 值
    is_overbought: bool             # 是否超買
    is_oversold: bool               # 是否超賣


@dataclass
class IndicatorSnapshot:
    """指標快照 - 某時點所有指標的狀態"""
    rsi: Optional[RSIResult] = None
    macd: Optional[MACDResult] = None
    bollinger: Optional[BollingerResult] = None
    stochastic: Optional[StochasticResult] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    atr: Optional[float] = None
    volume_ratio: Optional[float] = None   # 當前成交量 / 20日均量
    rsi_divergence: Optional[bool] = None  # RSI 背離
