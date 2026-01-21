"""Risk Manager - 風控管理器

This module implements the RiskManager class for managing trading risk,
including position sizing, sector concentration, circuit breaker, stop loss,
and trailing stop functionality.
"""

from typing import List, Tuple
from pattern_quant.core.models import Position


class RiskManager:
    """風控管理器
    
    管理資金風險與交易限制，包含：
    - 持倉數量計算
    - 板塊集中度檢查
    - 斷路器機制
    - 止損檢查（硬止損與技術止損）
    - 移動止盈
    
    Attributes:
        max_position_ratio: 單筆最大比例 (預設 2%)
        max_sector_ratio: 單板塊最大比例 (預設 20%)
        circuit_breaker_threshold: 斷路器閾值 (預設 2%)
    """
    
    def __init__(
        self,
        max_position_ratio: float = 0.02,
        max_sector_ratio: float = 0.20,
        circuit_breaker_threshold: float = 0.02
    ):
        """初始化風控管理器
        
        Args:
            max_position_ratio: 單筆最大比例 (預設 2%)
            max_sector_ratio: 單板塊最大比例 (預設 20%)
            circuit_breaker_threshold: 斷路器閾值 (預設 2%)
        """
        if max_position_ratio <= 0 or max_position_ratio > 1:
            raise ValueError("max_position_ratio must be between 0 and 1")
        if max_sector_ratio <= 0 or max_sector_ratio > 1:
            raise ValueError("max_sector_ratio must be between 0 and 1")
        if circuit_breaker_threshold <= 0 or circuit_breaker_threshold > 1:
            raise ValueError("circuit_breaker_threshold must be between 0 and 1")
            
        self.max_position_ratio = max_position_ratio
        self.max_sector_ratio = max_sector_ratio
        self.circuit_breaker_threshold = circuit_breaker_threshold

    def calculate_position_size(
        self,
        total_capital: float,
        entry_price: float,
        stop_loss_price: float
    ) -> int:
        """計算建議持倉數量
        
        根據總資金、進場價格與止損價格計算建議的持倉數量。
        確保單筆交易不超過總資金的配置比例上限。
        
        Args:
            total_capital: 總資金
            entry_price: 進場價格
            stop_loss_price: 止損價格
            
        Returns:
            建議持倉數量（股數），若無法計算則返回 0
            
        Requirements: 9.1
        """
        if total_capital <= 0 or entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        if stop_loss_price >= entry_price:
            return 0
        
        # 計算單筆最大可用資金
        max_position_value = total_capital * self.max_position_ratio
        
        # 計算基於最大資金的股數
        max_shares_by_capital = int(max_position_value / entry_price)
        
        # 計算基於風險的股數（每股風險 = 進場價 - 止損價）
        risk_per_share = entry_price - stop_loss_price
        max_risk = total_capital * self.max_position_ratio
        max_shares_by_risk = int(max_risk / risk_per_share)
        
        # 取兩者較小值，確保不超過資金限制
        position_size = min(max_shares_by_capital, max_shares_by_risk)
        
        return max(0, position_size)

    def check_sector_concentration(
        self,
        positions: List[Position],
        new_sector: str,
        new_position_value: float,
        total_capital: float
    ) -> bool:
        """檢查板塊集中度是否超標
        
        檢查新增持倉後，該板塊的總持倉比例是否超過配置的上限。
        
        Args:
            positions: 現有持倉列表
            new_sector: 新持倉的板塊
            new_position_value: 新持倉的市值
            total_capital: 總資金
            
        Returns:
            True 如果板塊集中度超標，False 如果未超標
            
        Requirements: 9.2, 9.3
        """
        if total_capital <= 0:
            return True
        
        # 計算該板塊現有持倉市值
        sector_value = sum(
            pos.quantity * pos.current_price
            for pos in positions
            if pos.sector == new_sector
        )
        
        # 加上新持倉市值
        total_sector_value = sector_value + new_position_value
        
        # 計算板塊集中度
        sector_ratio = total_sector_value / total_capital
        
        return sector_ratio > self.max_sector_ratio

    def check_circuit_breaker(
        self,
        market_index_change: float
    ) -> bool:
        """檢查是否觸發斷路器
        
        當大盤指數當日下跌超過配置閾值時，觸發斷路器暫停所有買入動作。
        
        Args:
            market_index_change: 大盤指數當日漲跌幅（負值表示下跌）
            
        Returns:
            True 如果觸發斷路器（應暫停買入），False 如果未觸發
            
        Requirements: 10.1
        """
        # 當跌幅超過閾值時觸發（market_index_change 為負值表示下跌）
        return market_index_change < -self.circuit_breaker_threshold

    def check_stop_loss(
        self,
        position: Position,
        current_price: float,
        hard_stop_ratio: float = 0.05
    ) -> Tuple[bool, str]:
        """檢查是否觸發止損
        
        檢查兩種止損條件：
        1. 硬止損：當前價格低於進場價格 × (1 - 硬止損比例)
        2. 技術止損：當前價格跌破持倉的止損價位
        
        Args:
            position: 持倉資訊
            current_price: 當前價格
            hard_stop_ratio: 硬止損比例（預設 5%）
            
        Returns:
            (should_exit, reason) - 是否應出場及原因
            
        Requirements: 8.1, 8.2
        """
        if current_price <= 0:
            return False, ""
        
        # 檢查硬止損
        hard_stop_price = position.entry_price * (1 - hard_stop_ratio)
        if current_price < hard_stop_price:
            return True, "hard_stop_loss"
        
        # 檢查技術止損（跌破止損價位）
        if current_price < position.stop_loss_price:
            return True, "technical_stop_loss"
        
        # 檢查移動止盈觸發
        if position.trailing_stop_active and current_price < position.trailing_stop_price:
            return True, "trailing_stop"
        
        return False, ""

    def update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        profit_threshold: float = 0.10,
        trailing_ratio: float = 0.03
    ) -> Position:
        """更新移動止盈
        
        當持倉獲利超過閾值時啟動移動止盈機制。
        移動止盈價位會隨著價格上漲而調整，但不會下降。
        
        Args:
            position: 持倉資訊
            current_price: 當前價格
            profit_threshold: 啟動移動止盈的獲利閾值（預設 10%）
            trailing_ratio: 移動止盈回調比例（預設 3%）
            
        Returns:
            更新後的持倉資訊
            
        Requirements: 8.3, 8.4
        """
        if current_price <= 0:
            return position
        
        # 計算當前獲利比例
        profit_ratio = (current_price - position.entry_price) / position.entry_price
        
        # 檢查是否應啟動移動止盈
        if profit_ratio >= profit_threshold:
            # 計算新的移動止盈價位
            new_trailing_stop = current_price * (1 - trailing_ratio)
            
            if not position.trailing_stop_active:
                # 首次啟動移動止盈
                return Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    sector=position.sector,
                    entry_time=position.entry_time,
                    stop_loss_price=position.stop_loss_price,
                    trailing_stop_active=True,
                    trailing_stop_price=new_trailing_stop
                )
            else:
                # 移動止盈已啟動，只有當新價位更高時才更新
                updated_trailing_price = max(position.trailing_stop_price, new_trailing_stop)
                return Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    sector=position.sector,
                    entry_time=position.entry_time,
                    stop_loss_price=position.stop_loss_price,
                    trailing_stop_active=True,
                    trailing_stop_price=updated_trailing_price
                )
        
        # 獲利未達閾值，僅更新當前價格
        return Position(
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=current_price,
            sector=position.sector,
            entry_time=position.entry_time,
            stop_loss_price=position.stop_loss_price,
            trailing_stop_active=position.trailing_stop_active,
            trailing_stop_price=position.trailing_stop_price
        )
