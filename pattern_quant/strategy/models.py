"""
雙引擎策略資料模型

定義市場狀態分類、策略信號與配置的資料類別。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MarketState(Enum):
    """市場狀態
    
    根據 ADX 與 BBW 指標判定的市場狀態類型。
    
    Attributes:
        TREND: 強趨勢狀態 (ADX > 25)
        RANGE: 震盪盤整狀態 (ADX < 20 且 BBW 穩定)
        NOISE: 混沌轉換狀態 (20 ≤ ADX ≤ 25)
    """
    TREND = "trend"
    RANGE = "range"
    NOISE = "noise"


@dataclass
class ADXResult:
    """ADX 計算結果
    
    Attributes:
        adx: ADX 值 (0-100)，衡量趨勢強度
        plus_di: +DI 值，正向趨勢指標
        minus_di: -DI 值，負向趨勢指標
    """
    adx: float
    plus_di: float
    minus_di: float


@dataclass
class BBWResult:
    """BBW (布林帶寬) 計算結果
    
    Attributes:
        bandwidth: 當前帶寬，公式為 (上軌 - 下軌) / 中軌
        avg_bandwidth: 歷史 20 日平均帶寬
        is_squeeze: 是否處於壓縮狀態 (BBW < 歷史平均 × 0.5)
        change_rate: 帶寬變化率
    """
    bandwidth: float
    avg_bandwidth: float
    is_squeeze: bool
    change_rate: float


@dataclass
class MarketStateResult:
    """市場狀態判定結果
    
    Attributes:
        state: 市場狀態 (TREND, RANGE, NOISE)
        allocation_weight: 資金權重 (0-1)
            - TREND: 100% (1.0)
            - RANGE: 60% (0.6)
            - NOISE: 0% (0.0)
        adx_result: ADX 計算詳情
        bbw_result: BBW 計算詳情
        confidence: 判定信心度 (0-1)
    """
    state: MarketState
    allocation_weight: float
    adx_result: ADXResult
    bbw_result: BBWResult
    confidence: float


@dataclass
class TrendSignal:
    """趨勢策略信號
    
    在趨勢市場中執行型態突破交易時產生的信號。
    
    Attributes:
        symbol: 股票代碼
        signal_type: 信號類型
            - "breakout_long": 突破做多
            - "stop_loss": 止損出場
            - "trailing_stop": 移動止損出場
        entry_price: 進場價格
        stop_loss_price: 止損價格 (型態支撐位)
        neckline_price: 頸線價格 (突破關鍵價位)
        pattern_score: 型態分數 (0-100)
        risk_reward_ratio: 風險回報比
    """
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss_price: float
    neckline_price: float
    pattern_score: float
    risk_reward_ratio: float


@dataclass
class CandlePattern:
    """K 線型態
    
    識別的 K 線反轉型態，用於均值回歸策略的進場確認。
    
    Attributes:
        pattern_type: 型態類型
            - "hammer": 錘頭線
            - "inverted_hammer": 倒錘頭線
            - "doji": 十字線
        confidence: 可信度 (0-1)
    """
    pattern_type: str
    confidence: float


@dataclass
class MeanReversionSignal:
    """均值回歸策略信號
    
    在震盪市場中執行布林均值回歸交易時產生的信號。
    
    Attributes:
        symbol: 股票代碼
        signal_type: 信號類型
            - "long_entry": 做多進場
            - "partial_exit": 部分出場 (50%)
            - "full_exit": 全部出場
            - "risk_override": 風控覆蓋強制出場
        entry_price: 進場價格 (布林下軌)
        target_1: 第一目標位 (布林中軌)
        target_2: 第二目標位 (布林上軌)
        stop_loss_price: 止損價格
        confirmation: 確認方式
            - "rsi_oversold": RSI 超賣確認
            - "hammer": 錘頭線確認
            - "both": 雙重確認
    """
    symbol: str
    signal_type: str
    entry_price: float
    target_1: float
    target_2: float
    stop_loss_price: float
    confirmation: str


@dataclass
class DualEngineConfig:
    """雙引擎策略配置
    
    控制雙引擎策略模組的所有可配置參數。
    
    Attributes:
        enabled: 是否啟用雙引擎模式
        
        # ADX 閾值
        adx_trend_threshold: 趨勢判定閾值 (ADX > 此值 = Trend)
        adx_range_threshold: 震盪判定閾值 (ADX < 此值 = Range)
        
        # 資金權重
        trend_allocation: 趨勢狀態資金權重 (預設 100%)
        range_allocation: 震盪狀態資金權重 (預設 60%)
        noise_allocation: 混沌狀態資金權重 (預設 0%)
        
        # 趨勢策略參數
        trend_score_threshold: 型態分數閾值
        trend_risk_reward: 目標風險回報比
        trend_trailing_activation: 移動止損啟動獲利比例
        
        # 均值回歸策略參數
        reversion_rsi_oversold: RSI 超賣閾值
        reversion_partial_exit: 部分出場比例
        reversion_adx_override: ADX 風控覆蓋閾值
        
        # 倉位計算參數
        trend_risk_per_trade: 趨勢策略單筆風險比例 (預設 1%)
        reversion_position_ratio: 震盪策略倉位比例 (預設 5%)
        
        # BBW 參數
        bbw_stability_threshold: BBW 穩定性閾值 (變化率 < 此值視為穩定)
    """
    enabled: bool = False
    
    # ADX 閾值
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    
    # 資金權重
    trend_allocation: float = 1.0
    range_allocation: float = 0.6
    noise_allocation: float = 0.0
    
    # 趨勢策略參數
    trend_score_threshold: float = 80.0
    trend_risk_reward: float = 3.0
    trend_trailing_activation: float = 0.10
    
    # 均值回歸策略參數
    reversion_rsi_oversold: float = 30.0
    reversion_partial_exit: float = 0.5
    reversion_adx_override: float = 25.0
    
    # 倉位計算參數
    trend_risk_per_trade: float = 0.01
    reversion_position_ratio: float = 0.05
    
    # BBW 參數
    bbw_stability_threshold: float = 0.10
