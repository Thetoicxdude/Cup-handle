"""
雙引擎策略配置管理器 (Dual Engine Config Manager)

管理雙引擎策略的配置，支援配置的讀取、儲存與預設值。
"""

from dataclasses import asdict
from typing import Dict, List, Optional, Any
import json

from pattern_quant.strategy.models import DualEngineConfig


class DualEngineConfigManager:
    """
    雙引擎策略配置管理器
    
    管理股票的雙引擎策略配置，支援：
    - 全局配置（適用於所有股票）
    - 個股配置（覆蓋全局配置）
    - 配置的讀取、儲存與預設值回退
    
    目前使用記憶體儲存，未來可擴展至資料庫持久化。
    """
    
    # 全局配置的特殊鍵值
    GLOBAL_CONFIG_KEY = "__global__"
    
    def __init__(self, db_connection: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            db_connection: 資料庫連線字串（目前未使用，保留供未來擴展）
        """
        self._db_connection = db_connection
        self._configs: Dict[str, DualEngineConfig] = {}
    
    def get_config(self, symbol: Optional[str] = None) -> DualEngineConfig:
        """
        獲取雙引擎策略配置
        
        優先順序：
        1. 個股配置（若 symbol 有指定且存在）
        2. 全局配置（若存在）
        3. 預設配置
        
        Args:
            symbol: 股票代碼，None 表示獲取全局配置
            
        Returns:
            對應的配置物件
        """
        # 嘗試獲取個股配置
        if symbol and symbol in self._configs:
            return self._configs[symbol]
        
        # 嘗試獲取全局配置
        if self.GLOBAL_CONFIG_KEY in self._configs:
            return self._configs[self.GLOBAL_CONFIG_KEY]
        
        # 返回預設配置
        return self.get_default_config()
    
    def save_config(
        self, 
        config: DualEngineConfig, 
        symbol: Optional[str] = None
    ) -> bool:
        """
        儲存雙引擎策略配置
        
        Args:
            config: 要儲存的配置物件
            symbol: 股票代碼，None 表示儲存為全局配置
            
        Returns:
            儲存是否成功
        """
        try:
            key = symbol if symbol else self.GLOBAL_CONFIG_KEY
            self._configs[key] = config
            return True
        except Exception:
            return False
    
    def get_default_config(self) -> DualEngineConfig:
        """
        返回預設配置
        
        預設值符合設計文件中的規格：
        - ADX 趨勢閾值: 25
        - ADX 震盪閾值: 20
        - 趨勢資金權重: 100%
        - 震盪資金權重: 60%
        - 混沌資金權重: 0%
        
        Returns:
            使用預設值的配置物件
        """
        return DualEngineConfig()
    
    def list_configured_symbols(self) -> List[str]:
        """
        列出所有有自訂配置的股票
        
        Returns:
            股票代碼列表（不包含全局配置鍵）
        """
        return [
            key for key in self._configs.keys() 
            if key != self.GLOBAL_CONFIG_KEY
        ]
    
    def delete_config(self, symbol: Optional[str] = None) -> bool:
        """
        刪除配置
        
        Args:
            symbol: 股票代碼，None 表示刪除全局配置
            
        Returns:
            刪除是否成功
        """
        key = symbol if symbol else self.GLOBAL_CONFIG_KEY
        if key in self._configs:
            del self._configs[key]
            return True
        return False
    
    def has_custom_config(self, symbol: Optional[str] = None) -> bool:
        """
        檢查是否有自訂配置
        
        Args:
            symbol: 股票代碼，None 表示檢查全局配置
            
        Returns:
            是否有自訂配置
        """
        key = symbol if symbol else self.GLOBAL_CONFIG_KEY
        return key in self._configs
    
    def has_global_config(self) -> bool:
        """
        檢查是否有全局配置
        
        Returns:
            是否有全局配置
        """
        return self.GLOBAL_CONFIG_KEY in self._configs

    
    def to_dict(self, config: DualEngineConfig) -> Dict[str, Any]:
        """
        將配置轉換為字典格式
        
        Args:
            config: 配置物件
            
        Returns:
            字典格式的配置
        """
        return asdict(config)
    
    def from_dict(self, data: Dict[str, Any]) -> DualEngineConfig:
        """
        從字典建立配置物件
        
        Args:
            data: 字典格式的配置資料
            
        Returns:
            配置物件
        """
        return DualEngineConfig(
            enabled=data.get("enabled", False),
            adx_trend_threshold=float(data.get("adx_trend_threshold", 25.0)),
            adx_range_threshold=float(data.get("adx_range_threshold", 20.0)),
            trend_allocation=float(data.get("trend_allocation", 1.0)),
            range_allocation=float(data.get("range_allocation", 0.6)),
            noise_allocation=float(data.get("noise_allocation", 0.0)),
            trend_score_threshold=float(data.get("trend_score_threshold", 80.0)),
            trend_risk_reward=float(data.get("trend_risk_reward", 3.0)),
            trend_trailing_activation=float(data.get("trend_trailing_activation", 0.10)),
            reversion_rsi_oversold=float(data.get("reversion_rsi_oversold", 30.0)),
            reversion_partial_exit=float(data.get("reversion_partial_exit", 0.5)),
            reversion_adx_override=float(data.get("reversion_adx_override", 25.0)),
            trend_risk_per_trade=float(data.get("trend_risk_per_trade", 0.01)),
            reversion_position_ratio=float(data.get("reversion_position_ratio", 0.05)),
            bbw_stability_threshold=float(data.get("bbw_stability_threshold", 0.10)),
        )
    
    def to_json(self, config: DualEngineConfig) -> str:
        """
        將配置轉換為 JSON 字串
        
        Args:
            config: 配置物件
            
        Returns:
            JSON 字串
        """
        return json.dumps(self.to_dict(config), ensure_ascii=False, indent=2)
    
    def from_json(self, json_str: str) -> DualEngineConfig:
        """
        從 JSON 字串建立配置物件
        
        Args:
            json_str: JSON 字串
            
        Returns:
            配置物件
        """
        data = json.loads(json_str)
        return self.from_dict(data)
    
    def update_adx_thresholds(
        self,
        trend_threshold: float,
        range_threshold: float,
        symbol: Optional[str] = None
    ) -> bool:
        """
        更新 ADX 閾值設定
        
        這是策略實驗室中常用的操作，提供便捷方法。
        
        Args:
            trend_threshold: 趨勢判定閾值 (ADX > 此值 = Trend)
            range_threshold: 震盪判定閾值 (ADX < 此值 = Range)
            symbol: 股票代碼，None 表示更新全局配置
            
        Returns:
            更新是否成功
        """
        config = self.get_config(symbol)
        config.adx_trend_threshold = trend_threshold
        config.adx_range_threshold = range_threshold
        return self.save_config(config, symbol)
    
    def update_allocation_weights(
        self,
        trend_allocation: float,
        range_allocation: float,
        noise_allocation: float = 0.0,
        symbol: Optional[str] = None
    ) -> bool:
        """
        更新資金權重設定
        
        這是策略實驗室中常用的操作，提供便捷方法。
        
        Args:
            trend_allocation: 趨勢狀態資金權重 (0-1)
            range_allocation: 震盪狀態資金權重 (0-1)
            noise_allocation: 混沌狀態資金權重 (0-1)
            symbol: 股票代碼，None 表示更新全局配置
            
        Returns:
            更新是否成功
        """
        config = self.get_config(symbol)
        config.trend_allocation = trend_allocation
        config.range_allocation = range_allocation
        config.noise_allocation = noise_allocation
        return self.save_config(config, symbol)
    
    def set_enabled(self, enabled: bool, symbol: Optional[str] = None) -> bool:
        """
        設定雙引擎模式啟用狀態
        
        Args:
            enabled: 是否啟用
            symbol: 股票代碼，None 表示設定全局配置
            
        Returns:
            設定是否成功
        """
        config = self.get_config(symbol)
        config.enabled = enabled
        return self.save_config(config, symbol)
