"""
基因組資料模型 (Genome Data Models)

定義生物演化優化引擎的核心資料結構，包含基因組、基因片段與個體。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import json
import math


class GeneType(Enum):
    """基因類型
    
    Attributes:
        FLOAT: 連續變量（浮點數）
        INTEGER: 離散變量（整數）
    """
    FLOAT = "float"
    INTEGER = "integer"


@dataclass
class GeneBounds:
    """基因邊界定義
    
    定義單一基因的數值範圍與類型約束。
    
    Attributes:
        min_value: 最小值
        max_value: 最大值
        gene_type: 基因類型（FLOAT 或 INTEGER）
    """
    min_value: float
    max_value: float
    gene_type: GeneType
    
    def validate(self, value: float) -> bool:
        """驗證數值是否在邊界內
        
        Args:
            value: 要驗證的數值
            
        Returns:
            數值是否在 [min_value, max_value] 範圍內
        """
        return self.min_value <= value <= self.max_value
    
    def clamp(self, value: float) -> float:
        """將數值限制在邊界內
        
        Args:
            value: 要限制的數值
            
        Returns:
            限制後的數值，若為 INTEGER 類型則四捨五入
        """
        clamped = max(self.min_value, min(self.max_value, value))
        if self.gene_type == GeneType.INTEGER:
            return float(round(clamped))
        return clamped


@dataclass
class DualEngineControlGenes:
    """基因片段 A：雙引擎核心控制
    
    控制市場狀態判定與資金分配的基因。
    
    Attributes:
        trend_threshold: ADX 趨勢判定閾值 (20-40)
        range_threshold: ADX 震盪判定閾值 (10-25)
        trend_allocation: 趨勢模式資金權重 (0.5-1.0)
        range_allocation: 震盪模式資金權重 (0.3-0.8)
        volatility_stability: BBW 變化率容忍值 (0.05-0.20)
    """
    trend_threshold: float
    range_threshold: float
    trend_allocation: float
    range_allocation: float
    volatility_stability: float
    
    def to_list(self) -> List[float]:
        """轉換為列表"""
        return [
            self.trend_threshold,
            self.range_threshold,
            self.trend_allocation,
            self.range_allocation,
            self.volatility_stability,
        ]
    
    @classmethod
    def from_list(cls, values: List[float]) -> "DualEngineControlGenes":
        """從列表建立"""
        if len(values) != 5:
            raise ValueError(f"Expected 5 values, got {len(values)}")
        return cls(
            trend_threshold=values[0],
            range_threshold=values[1],
            trend_allocation=values[2],
            range_allocation=values[3],
            volatility_stability=values[4],
        )
    
    def to_dict(self) -> Dict[str, float]:
        """轉換為字典"""
        return {
            "trend_threshold": self.trend_threshold,
            "range_threshold": self.range_threshold,
            "trend_allocation": self.trend_allocation,
            "range_allocation": self.range_allocation,
            "volatility_stability": self.volatility_stability,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "DualEngineControlGenes":
        """從字典建立"""
        return cls(
            trend_threshold=data["trend_threshold"],
            range_threshold=data["range_threshold"],
            trend_allocation=data["trend_allocation"],
            range_allocation=data["range_allocation"],
            volatility_stability=data["volatility_stability"],
        )


@dataclass
class FactorWeightGenes:
    """基因片段 B：因子權重分配
    
    控制各技術指標權重與買入閾值的基因。
    
    Attributes:
        rsi_weight: RSI 權重 (0.0-2.0)
        volume_weight: 成交量權重 (0.0-2.0)
        macd_weight: MACD 權重 (0.0-2.0)
        ema_weight: 均線權重 (0.0-2.0)
        bollinger_weight: 布林通道權重 (0.0-2.0)
        score_threshold: 總分買入閾值 (50-90)
    """
    rsi_weight: float
    volume_weight: float
    macd_weight: float
    ema_weight: float
    bollinger_weight: float
    score_threshold: float
    
    def to_list(self) -> List[float]:
        """轉換為列表"""
        return [
            self.rsi_weight,
            self.volume_weight,
            self.macd_weight,
            self.ema_weight,
            self.bollinger_weight,
            self.score_threshold,
        ]
    
    @classmethod
    def from_list(cls, values: List[float]) -> "FactorWeightGenes":
        """從列表建立"""
        if len(values) != 6:
            raise ValueError(f"Expected 6 values, got {len(values)}")
        return cls(
            rsi_weight=values[0],
            volume_weight=values[1],
            macd_weight=values[2],
            ema_weight=values[3],
            bollinger_weight=values[4],
            score_threshold=values[5],
        )
    
    def to_dict(self) -> Dict[str, float]:
        """轉換為字典"""
        return {
            "rsi_weight": self.rsi_weight,
            "volume_weight": self.volume_weight,
            "macd_weight": self.macd_weight,
            "ema_weight": self.ema_weight,
            "bollinger_weight": self.bollinger_weight,
            "score_threshold": self.score_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "FactorWeightGenes":
        """從字典建立"""
        return cls(
            rsi_weight=data["rsi_weight"],
            volume_weight=data["volume_weight"],
            macd_weight=data["macd_weight"],
            ema_weight=data["ema_weight"],
            bollinger_weight=data["bollinger_weight"],
            score_threshold=data["score_threshold"],
        )
    
    def get_weights(self) -> List[float]:
        """獲取所有權重值（不含 score_threshold）"""
        return [
            self.rsi_weight,
            self.volume_weight,
            self.macd_weight,
            self.ema_weight,
            self.bollinger_weight,
        ]


@dataclass
class MicroIndicatorGenes:
    """基因片段 C：微觀指標參數
    
    控制各技術指標詳細參數的基因。
    
    Attributes:
        rsi_period: RSI 週期 (5-21)
        rsi_overbought: RSI 超買線 (65-85)
        rsi_oversold: RSI 超賣線 (15-35)
        volume_spike_multiplier: 成交量突變倍數 (1.2-3.0)
        macd_bonus: MACD 訊號加成 (5-20)
        bollinger_squeeze_threshold: 布林壓縮閾值 (0.02-0.10)
    """
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    volume_spike_multiplier: float
    macd_bonus: float
    bollinger_squeeze_threshold: float
    
    def to_list(self) -> List[float]:
        """轉換為列表"""
        return [
            float(self.rsi_period),
            self.rsi_overbought,
            self.rsi_oversold,
            self.volume_spike_multiplier,
            self.macd_bonus,
            self.bollinger_squeeze_threshold,
        ]
    
    @classmethod
    def from_list(cls, values: List[float]) -> "MicroIndicatorGenes":
        """從列表建立"""
        if len(values) != 6:
            raise ValueError(f"Expected 6 values, got {len(values)}")
        return cls(
            rsi_period=int(round(values[0])),
            rsi_overbought=values[1],
            rsi_oversold=values[2],
            volume_spike_multiplier=values[3],
            macd_bonus=values[4],
            bollinger_squeeze_threshold=values[5],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "volume_spike_multiplier": self.volume_spike_multiplier,
            "macd_bonus": self.macd_bonus,
            "bollinger_squeeze_threshold": self.bollinger_squeeze_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MicroIndicatorGenes":
        """從字典建立"""
        return cls(
            rsi_period=int(data["rsi_period"]),
            rsi_overbought=float(data["rsi_overbought"]),
            rsi_oversold=float(data["rsi_oversold"]),
            volume_spike_multiplier=float(data["volume_spike_multiplier"]),
            macd_bonus=float(data["macd_bonus"]),
            bollinger_squeeze_threshold=float(data["bollinger_squeeze_threshold"]),
        )


# 染色體基因索引常數
CHROMOSOME_GENE_NAMES: List[str] = [
    # Segment A: Dual Engine Control (0-4)
    "trend_threshold",
    "range_threshold",
    "trend_allocation",
    "range_allocation",
    "volatility_stability",
    # Segment B: Factor Weights (5-10)
    "rsi_weight",
    "volume_weight",
    "macd_weight",
    "ema_weight",
    "bollinger_weight",
    "score_threshold",
    # Segment C: Micro Indicators (11-16)
    "rsi_period",
    "rsi_overbought",
    "rsi_oversold",
    "volume_spike_multiplier",
    "macd_bonus",
    "bollinger_squeeze_threshold",
]

CHROMOSOME_LENGTH = len(CHROMOSOME_GENE_NAMES)  # 17


@dataclass
class Genome:
    """完整基因組
    
    包含三個基因片段的完整策略參數編碼。
    
    Attributes:
        dual_engine: 雙引擎控制基因片段
        factor_weights: 因子權重基因片段
        micro_indicators: 微觀指標基因片段
    """
    dual_engine: DualEngineControlGenes
    factor_weights: FactorWeightGenes
    micro_indicators: MicroIndicatorGenes
    
    def to_chromosome(self) -> List[float]:
        """轉換為染色體陣列
        
        將基因組轉換為線性的浮點數陣列，用於遺傳操作。
        
        Returns:
            長度為 17 的浮點數列表
        """
        chromosome = []
        chromosome.extend(self.dual_engine.to_list())
        chromosome.extend(self.factor_weights.to_list())
        chromosome.extend(self.micro_indicators.to_list())
        return chromosome
    
    @classmethod
    def from_chromosome(cls, chromosome: List[float]) -> "Genome":
        """從染色體陣列建立基因組
        
        Args:
            chromosome: 長度為 17 的浮點數列表
            
        Returns:
            重建的基因組物件
            
        Raises:
            ValueError: 若染色體長度不正確
        """
        if len(chromosome) != CHROMOSOME_LENGTH:
            raise ValueError(
                f"Chromosome length must be {CHROMOSOME_LENGTH}, got {len(chromosome)}"
            )
        
        dual_engine = DualEngineControlGenes.from_list(chromosome[0:5])
        factor_weights = FactorWeightGenes.from_list(chromosome[5:11])
        micro_indicators = MicroIndicatorGenes.from_list(chromosome[11:17])
        
        return cls(
            dual_engine=dual_engine,
            factor_weights=factor_weights,
            micro_indicators=micro_indicators,
        )
    
    def normalize_weights(self) -> "Genome":
        """歸一化因子權重
        
        將所有因子權重歸一化，使其總和為 1.0。
        若所有權重為零，則平均分配。
        
        Returns:
            新的基因組物件，權重已歸一化
        """
        weights = self.factor_weights.get_weights()
        total = sum(weights)
        
        if total == 0 or math.isclose(total, 0.0, abs_tol=1e-10):
            # 若所有權重為零，平均分配
            normalized = [0.2] * 5
        else:
            normalized = [w / total for w in weights]
        
        new_factor_weights = FactorWeightGenes(
            rsi_weight=normalized[0],
            volume_weight=normalized[1],
            macd_weight=normalized[2],
            ema_weight=normalized[3],
            bollinger_weight=normalized[4],
            score_threshold=self.factor_weights.score_threshold,
        )
        
        return Genome(
            dual_engine=self.dual_engine,
            factor_weights=new_factor_weights,
            micro_indicators=self.micro_indicators,
        )
    
    def validate_constraints(self) -> bool:
        """驗證邏輯約束
        
        檢查基因組是否滿足所有邏輯約束：
        1. trend_threshold > range_threshold
        2. 所有權重非負
        3. rsi_period >= 2
        
        Returns:
            是否滿足所有約束
        """
        # 約束 1: trend_threshold > range_threshold
        if self.dual_engine.trend_threshold <= self.dual_engine.range_threshold:
            return False
        
        # 約束 2: 所有權重非負
        weights = self.factor_weights.get_weights()
        if any(w < 0 for w in weights):
            return False
        
        # 約束 3: rsi_period >= 2
        if self.micro_indicators.rsi_period < 2:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "dual_engine": self.dual_engine.to_dict(),
            "factor_weights": self.factor_weights.to_dict(),
            "micro_indicators": self.micro_indicators.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        """從字典建立基因組"""
        return cls(
            dual_engine=DualEngineControlGenes.from_dict(data["dual_engine"]),
            factor_weights=FactorWeightGenes.from_dict(data["factor_weights"]),
            micro_indicators=MicroIndicatorGenes.from_dict(data["micro_indicators"]),
        )
    
    def to_json(self) -> str:
        """序列化為 JSON 字串
        
        Returns:
            JSON 格式的字串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str, bounds: Optional[Dict[str, "GeneBounds"]] = None) -> "Genome":
        """從 JSON 字串反序列化
        
        Args:
            json_str: JSON 格式的字串
            bounds: 可選的邊界定義，用於驗證與 clamping
            
        Returns:
            重建的基因組物件
            
        Raises:
            ValueError: 若 JSON 格式不正確或數值超出邊界
        """
        data = json.loads(json_str)
        genome = cls.from_dict(data)
        
        # 若提供邊界，進行驗證與 clamping
        if bounds is not None:
            genome = genome._clamp_to_bounds(bounds)
        
        return genome
    
    def _clamp_to_bounds(self, bounds: Dict[str, "GeneBounds"]) -> "Genome":
        """將基因組限制在邊界內
        
        Args:
            bounds: 基因邊界定義
            
        Returns:
            限制後的新基因組
        """
        chromosome = self.to_chromosome()
        clamped = []
        
        for i, gene_name in enumerate(CHROMOSOME_GENE_NAMES):
            if gene_name in bounds:
                clamped.append(bounds[gene_name].clamp(chromosome[i]))
            else:
                clamped.append(chromosome[i])
        
        return Genome.from_chromosome(clamped)


@dataclass
class Individual:
    """演化個體
    
    代表種群中的一個個體，包含基因組與適應度資訊。
    
    Attributes:
        genome: 個體的基因組
        fitness: 適應度分數（預設為 0.0）
        generation: 所屬世代（預設為 0）
    """
    genome: Genome
    fitness: float = 0.0
    generation: int = 0
    
    def __lt__(self, other: "Individual") -> bool:
        """比較適應度（用於排序）
        
        Args:
            other: 另一個個體
            
        Returns:
            self.fitness < other.fitness
        """
        return self.fitness < other.fitness
    
    def __eq__(self, other: object) -> bool:
        """比較兩個個體是否相等"""
        if not isinstance(other, Individual):
            return False
        return (
            self.fitness == other.fitness
            and self.generation == other.generation
            and self._genome_equals(self.genome, other.genome)
        )
    
    @staticmethod
    def _genome_equals(g1: Genome, g2: Genome, tolerance: float = 1e-9) -> bool:
        """比較兩個基因組是否相等（考慮浮點數誤差）"""
        c1 = g1.to_chromosome()
        c2 = g2.to_chromosome()
        
        if len(c1) != len(c2):
            return False
        
        for v1, v2 in zip(c1, c2):
            if not math.isclose(v1, v2, abs_tol=tolerance):
                return False
        
        return True
    
    def copy(self) -> "Individual":
        """建立個體的深拷貝"""
        return Individual(
            genome=Genome.from_chromosome(self.genome.to_chromosome()),
            fitness=self.fitness,
            generation=self.generation,
        )


# 預設基因邊界定義
DEFAULT_GENOME_BOUNDS: Dict[str, GeneBounds] = {
    # 雙引擎控制基因 (Segment A)
    "trend_threshold": GeneBounds(20.0, 40.0, GeneType.FLOAT),
    "range_threshold": GeneBounds(10.0, 25.0, GeneType.FLOAT),
    "trend_allocation": GeneBounds(0.5, 1.0, GeneType.FLOAT),
    "range_allocation": GeneBounds(0.3, 0.8, GeneType.FLOAT),
    "volatility_stability": GeneBounds(0.05, 0.20, GeneType.FLOAT),
    
    # 因子權重基因 (Segment B)
    "rsi_weight": GeneBounds(0.0, 2.0, GeneType.FLOAT),
    "volume_weight": GeneBounds(0.0, 2.0, GeneType.FLOAT),
    "macd_weight": GeneBounds(0.0, 2.0, GeneType.FLOAT),
    "ema_weight": GeneBounds(0.0, 2.0, GeneType.FLOAT),
    "bollinger_weight": GeneBounds(0.0, 2.0, GeneType.FLOAT),
    "score_threshold": GeneBounds(50.0, 90.0, GeneType.FLOAT),
    
    # 微觀指標基因 (Segment C)
    "rsi_period": GeneBounds(5, 21, GeneType.INTEGER),
    "rsi_overbought": GeneBounds(65.0, 85.0, GeneType.FLOAT),
    "rsi_oversold": GeneBounds(15.0, 35.0, GeneType.FLOAT),
    "volume_spike_multiplier": GeneBounds(1.2, 3.0, GeneType.FLOAT),
    "macd_bonus": GeneBounds(5.0, 20.0, GeneType.FLOAT),
    "bollinger_squeeze_threshold": GeneBounds(0.02, 0.10, GeneType.FLOAT),
}


def validate_genome_bounds(genome: Genome, bounds: Dict[str, GeneBounds]) -> bool:
    """驗證基因組是否在邊界內
    
    Args:
        genome: 要驗證的基因組
        bounds: 基因邊界定義
        
    Returns:
        所有基因是否都在邊界內
    """
    chromosome = genome.to_chromosome()
    
    for i, gene_name in enumerate(CHROMOSOME_GENE_NAMES):
        if gene_name in bounds:
            if not bounds[gene_name].validate(chromosome[i]):
                return False
    
    return True


def genome_equals(g1: Genome, g2: Genome, tolerance: float = 1e-9) -> bool:
    """比較兩個基因組是否相等（考慮浮點數誤差）
    
    Args:
        g1: 第一個基因組
        g2: 第二個基因組
        tolerance: 浮點數比較容差
        
    Returns:
        兩個基因組是否相等
    """
    return Individual._genome_equals(g1, g2, tolerance)
