"""
因子配置系統 (Factor Config)

管理每支股票的指標配置，包含各指標的啟用狀態、權重與詳細參數。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class IndicatorType(Enum):
    """指標類型列舉"""
    RSI = "rsi"
    VOLUME = "volume"
    MACD = "macd"
    EMA = "ema"
    BOLLINGER = "bollinger"


@dataclass
class RSIConfig:
    """RSI 詳細配置
    
    評分邏輯說明：
    - 趨勢區間 (40-70)：健康的上升動能，加分
    - 超買 (>=80)：風險較高，扣分
    - 超賣反彈 (30-40)：潛在買入機會，加分
    - 極度超賣 (<30)：可能繼續下跌，不加分也不扣分
    """
    enabled: bool = True
    weight: float = 1.0
    period: int = 14
    overbought: float = 80.0
    oversold: float = 30.0
    trend_lower: float = 40.0           # 放寬趨勢區間下限（原 50）
    trend_upper: float = 70.0           # 調整趨勢區間上限（原 75）
    check_divergence: bool = True
    trend_zone_bonus: float = 8.0       # 趨勢區間加分（原 10，略降）
    overbought_penalty: float = -15.0   # 超買扣分（原 -20，減輕）
    weak_penalty: float = -5.0          # 動能不足扣分（原 -10，大幅減輕）
    support_bounce_bonus: float = 12.0  # 支撐反彈加分（原 15，略降）
    divergence_penalty: float = -10.0   # 背離扣分（原 -15，減輕）


@dataclass
class VolumeConfig:
    """成交量配置
    
    評分邏輯說明：
    - 放量突破：確認趨勢，加分
    - 縮量：不一定是壞事（縮量整理），輕微扣分
    """
    enabled: bool = True
    weight: float = 1.0
    high_volume_threshold: float = 1.3  # 放量倍數（原 1.5，降低門檻）
    high_volume_bonus: float = 10.0     # 放量加分（原 15，略降）
    low_volume_penalty: float = -2.0    # 縮量扣分（原 -5，大幅減輕）


@dataclass
class MACDConfig:
    """MACD 配置
    
    評分邏輯說明：
    - 零軸之上且柱狀圖為正：強勢，加分
    - 零軸之下：弱勢，但不一定是壞事（可能是底部），輕微扣分
    - 黃金交叉：趨勢轉強，加分
    """
    enabled: bool = True
    weight: float = 1.0
    above_zero_bonus: float = 8.0       # 零軸之上加分（原 10，略降）
    below_zero_penalty: float = -5.0    # 零軸之下扣分（原 -10，大幅減輕）
    golden_cross_bonus: float = 12.0    # 黃金交叉加分（原 15，略降）


@dataclass
class EMAConfig:
    """均線配置
    
    評分邏輯說明：
    - 價格在均線之上：趨勢向上，加分
    - 價格在均線之下：可能是回調買點，輕微扣分
    """
    enabled: bool = True
    weight: float = 1.0
    above_ema20_bonus: float = 5.0      # EMA20 之上加分
    above_ema50_bonus: float = 8.0      # EMA50 之上加分（原 10，略降）
    below_ema20_penalty: float = -3.0   # EMA20 之下扣分（原 -10，大幅減輕）


@dataclass
class BollingerConfig:
    """布林通道配置
    
    評分邏輯說明：
    - 波動率壓縮後突破：強勢突破，加分
    - 壓縮 + RSI 配合：確認突破，加分
    """
    enabled: bool = True
    weight: float = 1.0
    squeeze_threshold: float = 0.5
    squeeze_breakout_bonus: float = 15.0    # 壓縮突破加分（原 20，略降）
    squeeze_rsi_combo_bonus: float = 8.0    # 壓縮 + RSI 組合加分（原 10，略降）


@dataclass
class FactorConfig:
    """完整因子配置
    
    重要說明：
    - buy_threshold：最終分數超過此值才會觸發強烈買入訊號
    - 型態分數通常在 60-80 之間
    - 因子調整後的分數可能增加或減少 20-40 分
    - 建議 buy_threshold 設在 65-75 之間
    """
    symbol: str
    rsi: RSIConfig = field(default_factory=RSIConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    macd: MACDConfig = field(default_factory=MACDConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    bollinger: BollingerConfig = field(default_factory=BollingerConfig)
    buy_threshold: float = 70.0         # 買入閾值（原 80，降低以增加交易機會）
    watch_threshold: float = 60.0       # 觀望閾值（與 score_threshold 同步）
    use_atr_stop_loss: bool = False     # 是否使用 ATR 動態止損
    atr_multiplier: float = 2.0         # ATR 止損倍數

    def to_dict(self) -> Dict[str, Any]:
        """將配置轉換為字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorConfig":
        """從字典建立配置物件"""
        symbol = data.get("symbol", "")
        
        rsi_data = data.get("rsi", {})
        volume_data = data.get("volume", {})
        macd_data = data.get("macd", {})
        ema_data = data.get("ema", {})
        bollinger_data = data.get("bollinger", {})
        
        return cls(
            symbol=symbol,
            rsi=RSIConfig(**rsi_data) if rsi_data else RSIConfig(),
            volume=VolumeConfig(**volume_data) if volume_data else VolumeConfig(),
            macd=MACDConfig(**macd_data) if macd_data else MACDConfig(),
            ema=EMAConfig(**ema_data) if ema_data else EMAConfig(),
            bollinger=BollingerConfig(**bollinger_data) if bollinger_data else BollingerConfig(),
            buy_threshold=float(data.get("buy_threshold", 80)),
            watch_threshold=float(data.get("watch_threshold", 60)),
            use_atr_stop_loss=data.get("use_atr_stop_loss", False),
            atr_multiplier=float(data.get("atr_multiplier", 2.0)),
        )

    def to_json(self) -> str:
        """將配置轉換為 JSON 字串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "FactorConfig":
        """從 JSON 字串建立配置物件"""
        data = json.loads(json_str)
        return cls.from_dict(data)



class FactorConfigManager:
    """
    因子配置管理器
    
    管理股票的因子配置，支援儲存、讀取與預設配置回退。
    目前使用記憶體儲存，未來可擴展至資料庫持久化。
    """
    
    def __init__(self, db_connection: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            db_connection: 資料庫連線字串（目前未使用，保留供未來擴展）
        """
        self._db_connection = db_connection
        self._configs: Dict[str, FactorConfig] = {}
    
    def get_config(self, symbol: str) -> FactorConfig:
        """
        獲取股票的因子配置
        
        Args:
            symbol: 股票代碼
            
        Returns:
            該股票的配置，若無專屬配置則返回預設配置
        """
        if symbol in self._configs:
            return self._configs[symbol]
        return self.get_default_config(symbol)
    
    def save_config(self, config: FactorConfig) -> bool:
        """
        儲存因子配置
        
        Args:
            config: 要儲存的配置物件
            
        Returns:
            儲存是否成功
        """
        try:
            self._configs[config.symbol] = config
            return True
        except Exception:
            return False
    
    def get_default_config(self, symbol: str) -> FactorConfig:
        """
        返回預設配置
        
        Args:
            symbol: 股票代碼
            
        Returns:
            使用預設值的配置物件
        """
        return FactorConfig(symbol=symbol)
    
    def list_configured_symbols(self) -> List[str]:
        """
        列出所有有自訂配置的股票
        
        Returns:
            股票代碼列表
        """
        return list(self._configs.keys())
    
    def delete_config(self, symbol: str) -> bool:
        """
        刪除股票的自訂配置
        
        Args:
            symbol: 股票代碼
            
        Returns:
            刪除是否成功
        """
        if symbol in self._configs:
            del self._configs[symbol]
            return True
        return False
    
    def has_custom_config(self, symbol: str) -> bool:
        """
        檢查股票是否有自訂配置
        
        Args:
            symbol: 股票代碼
            
        Returns:
            是否有自訂配置
        """
        return symbol in self._configs
