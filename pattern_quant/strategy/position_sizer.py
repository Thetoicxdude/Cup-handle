"""
倉位計算器模組

提供兩種倉位計算策略：
1. ATRPositionSizer: 基於 ATR 波動率倒數模型，用於趨勢策略
2. FixedFractionalSizer: 固定金額模型，用於震盪策略

Requirements: 7.1, 7.2, 7.3, 8.1, 8.2, 8.3
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class PositionSizeResult:
    """倉位計算結果
    
    Attributes:
        shares: 建議股數
        position_value: 倉位市值
        risk_amount: 風險金額
        sizing_method: 計算方法 ("atr_volatility" 或 "fixed_fractional")
    """
    shares: int
    position_value: float
    risk_amount: float
    sizing_method: str


class ATRPositionSizer:
    """ATR 波動率倒數模型 - 用於趨勢策略
    
    根據 ATR (Average True Range) 計算倉位大小，確保每筆交易的風險
    固定為總資金的特定比例（預設 1%）。
    
    公式: shares = (capital * risk_ratio) / (atr * multiplier)
    
    Attributes:
        risk_per_trade: 單筆交易風險比例 (預設 1%)
    
    Requirements: 7.1, 7.2, 7.3
    """
    
    def __init__(self, risk_per_trade: float = 0.01):
        """初始化 ATR 倉位計算器
        
        Args:
            risk_per_trade: 單筆交易風險比例 (預設 1%)
            
        Raises:
            ValueError: 當 risk_per_trade 不在有效範圍內
        """
        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError("risk_per_trade must be between 0 and 1 (exclusive of 0)")
        self.risk_per_trade = risk_per_trade
    
    def calculate(
        self,
        total_capital: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0
    ) -> PositionSizeResult:
        """計算趨勢策略倉位
        
        使用 ATR 波動率倒數模型計算建議股數，確保單筆交易風險
        不超過總資金的配置比例。
        
        公式: shares = (capital * risk_ratio) / (atr * multiplier)
        
        Args:
            total_capital: 總資金
            entry_price: 進場價格
            atr: ATR 值 (Average True Range)
            atr_multiplier: ATR 乘數，用於計算止損距離 (預設 2.0)
            
        Returns:
            PositionSizeResult 包含建議股數、倉位市值與風險金額
            
        Requirements: 7.1, 7.2, 7.3
        """
        # 驗證輸入參數
        if total_capital <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                sizing_method="atr_volatility"
            )
        
        if entry_price <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                sizing_method="atr_volatility"
            )
        
        if atr <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                sizing_method="atr_volatility"
            )
        
        if atr_multiplier <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                sizing_method="atr_volatility"
            )
        
        # 計算最大風險金額
        max_risk_amount = total_capital * self.risk_per_trade
        
        # 計算每股風險（基於 ATR 和乘數）
        risk_per_share = atr * atr_multiplier
        
        # 計算建議股數
        # shares = max_risk_amount / risk_per_share
        shares = math.floor(max_risk_amount / risk_per_share)
        
        # 確保股數不為負
        shares = max(0, shares)
        
        # 計算實際倉位市值
        position_value = shares * entry_price
        
        # 計算實際風險金額
        actual_risk_amount = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk_amount,
            sizing_method="atr_volatility"
        )


class FixedFractionalSizer:
    """固定金額模型 - 用於震盪策略
    
    每次投入總資金的固定比例（預設 5%）。
    
    公式: shares = (capital * position_ratio) / entry_price
    
    Attributes:
        position_ratio: 倉位比例 (預設 5%)
    
    Requirements: 8.1, 8.2, 8.3
    """
    
    def __init__(self, position_ratio: float = 0.05):
        """初始化固定金額倉位計算器
        
        Args:
            position_ratio: 倉位比例 (預設 5%)
            
        Raises:
            ValueError: 當 position_ratio 不在有效範圍內
        """
        if position_ratio <= 0 or position_ratio > 1:
            raise ValueError("position_ratio must be between 0 and 1 (exclusive of 0)")
        self.position_ratio = position_ratio
    
    def calculate(
        self,
        total_capital: float,
        entry_price: float
    ) -> PositionSizeResult:
        """計算震盪策略倉位
        
        使用固定金額模型計算建議股數，投入總資金的固定比例。
        
        公式: shares = (capital * position_ratio) / entry_price
        
        Args:
            total_capital: 總資金
            entry_price: 進場價格
            
        Returns:
            PositionSizeResult 包含建議股數、倉位市值與投入金額
            
        Requirements: 8.1, 8.2, 8.3
        """
        # 驗證輸入參數
        if total_capital <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                sizing_method="fixed_fractional"
            )
        
        if entry_price <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0.0,
                risk_amount=0.0,
                sizing_method="fixed_fractional"
            )
        
        # 計算目標投入金額
        target_position_value = total_capital * self.position_ratio
        
        # 計算建議股數
        # shares = target_position_value / entry_price
        shares = math.floor(target_position_value / entry_price)
        
        # 確保股數不為負
        shares = max(0, shares)
        
        # 計算實際倉位市值
        actual_position_value = shares * entry_price
        
        return PositionSizeResult(
            shares=shares,
            position_value=actual_position_value,
            risk_amount=actual_position_value,  # 對於固定金額模型，風險金額等於倉位市值
            sizing_method="fixed_fractional"
        )
