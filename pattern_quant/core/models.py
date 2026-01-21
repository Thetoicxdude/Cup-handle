"""Core data models for AI PatternQuant

This module defines all core dataclasses and enums used throughout the system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalStatus(Enum):
    """交易訊號狀態"""
    WAITING_BREAKOUT = "waiting_breakout"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class ExitReason(Enum):
    """出場原因"""
    HARD_STOP_LOSS = "hard_stop_loss"
    TECHNICAL_STOP_LOSS = "technical_stop_loss"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"


@dataclass
class Extremum:
    """局部極值點
    
    Attributes:
        index: 在序列中的位置
        price: 價格值
        is_peak: True=波峰, False=波谷
    """
    index: int
    price: float
    is_peak: bool


@dataclass
class CupPattern:
    """茶杯型態
    
    Attributes:
        left_peak_index: 左峰索引位置
        left_peak_price: 左峰價格
        right_peak_index: 右峰索引位置
        right_peak_price: 右峰價格
        bottom_index: 杯底索引位置
        bottom_price: 杯底價格
        r_squared: 二次擬合的 R² 值
        depth_ratio: 杯身深度比例
        symmetry_score: 左右峰對稱性分數
    """
    left_peak_index: int
    left_peak_price: float
    right_peak_index: int
    right_peak_price: float
    bottom_index: int
    bottom_price: float
    r_squared: float
    depth_ratio: float
    symmetry_score: float


@dataclass
class HandlePattern:
    """柄部型態
    
    Attributes:
        start_index: 柄部起點索引
        end_index: 柄部終點索引
        lowest_price: 柄部最低價格
        volume_slope: 成交量線性回歸斜率
    """
    start_index: int
    end_index: int
    lowest_price: float
    volume_slope: float


@dataclass
class MatchScore:
    """型態吻合分數
    
    Attributes:
        total_score: 綜合分數 (0-100)
        r_squared_score: 擬合度分數
        symmetry_score: 對稱性分數
        volume_score: 成交量萎縮分數
        depth_score: 深度合理性分數
    """
    total_score: float
    r_squared_score: float
    symmetry_score: float
    volume_score: float
    depth_score: float


@dataclass
class OHLCV:
    """K 線數據
    
    Attributes:
        time: 時間戳
        symbol: 股票代碼
        open: 開盤價
        high: 最高價
        low: 最低價
        close: 收盤價
        volume: 成交量
    """
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class PatternResult:
    """型態識別結果
    
    Attributes:
        symbol: 股票代碼
        pattern_type: 型態類型
        cup: 茶杯型態（可選）
        handle: 柄部型態（可選）
        score: 吻合分數（可選）
        is_valid: 是否為有效型態
        rejection_reason: 拒絕原因（可選）
    """
    symbol: str
    pattern_type: str
    cup: Optional[CupPattern]
    handle: Optional[HandlePattern]
    score: Optional[MatchScore]
    is_valid: bool
    rejection_reason: Optional[str]


@dataclass
class TradeSignal:
    """交易訊號
    
    Attributes:
        symbol: 股票代碼
        pattern_type: 型態類型
        match_score: 吻合分數
        breakout_price: 突破價位 (右杯緣 + 0.5%)
        stop_loss_price: 止損價位
        status: 訊號狀態
        created_at: 建立時間
        expected_profit_ratio: 預期獲利比
    """
    symbol: str
    pattern_type: str
    match_score: float
    breakout_price: float
    stop_loss_price: float
    status: SignalStatus
    created_at: datetime
    expected_profit_ratio: float


@dataclass
class Position:
    """持倉
    
    Attributes:
        symbol: 股票代碼
        quantity: 持倉數量
        entry_price: 進場價格
        current_price: 當前價格
        sector: 板塊
        entry_time: 進場時間
        stop_loss_price: 止損價位
        trailing_stop_active: 移動止盈是否啟動
        trailing_stop_price: 移動止盈價位
    """
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    sector: str
    entry_time: datetime
    stop_loss_price: float
    trailing_stop_active: bool
    trailing_stop_price: float
