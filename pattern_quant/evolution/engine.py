"""
生物演化優化引擎 (Evolutionary Engine)

整合所有演化組件的主引擎，提供完整的演化優化與滾動視窗驗證功能。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any

from .models import (
    GeneBounds,
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
)
from .population import PopulationGenerator
from .selection import SelectionOperator
from .crossover import CrossoverOperator
from .mutation import MutationOperator
from .fitness import FitnessEvaluator, FitnessObjective
from .generation import GenerationController, GenerationStats, EvolutionHistory
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardAnalyzer,
)


@dataclass
class EvolutionConfig:
    """演化配置
    
    控制演化優化引擎的所有可配置參數。
    
    Attributes:
        population_size: 種群大小 (50-100)
        max_generations: 最大世代數 (10-50)
        tournament_size: 競賽選擇大小
        elitism_rate: 精英保留率 (0.05-0.20)
        crossover_rate: 交叉率 (0.6-0.9)
        mutation_rate: 突變率 (0.01-0.05)
        mutation_strength: 突變強度
        fitness_objective: 適應度目標函數
        min_trades_threshold: 最低交易次數閾值
        convergence_threshold: 收斂閾值
        convergence_patience: 收斂耐心值
    """
    population_size: int = 50
    max_generations: int = 20
    tournament_size: int = 3
    elitism_rate: float = 0.1
    crossover_rate: float = 0.8
    mutation_rate: float = 0.02
    mutation_strength: float = 0.1
    fitness_objective: FitnessObjective = FitnessObjective.SHARPE_RATIO
    min_trades_threshold: int = 10
    convergence_threshold: float = 0.001
    convergence_patience: int = 5
    
    def validate(self) -> bool:
        """驗證配置是否有效
        
        Returns:
            配置是否有效
        """
        return (
            2 <= self.population_size <= 500
            and 1 <= self.max_generations <= 200
            and self.tournament_size >= 1
            and 0.0 <= self.elitism_rate <= 0.50
            and 0.0 <= self.crossover_rate <= 1.0
            and 0.0 <= self.mutation_rate <= 1.0
            and self.mutation_strength > 0
            and self.min_trades_threshold >= 0
            and self.convergence_threshold > 0
            and self.convergence_patience >= 1
        )


class EvolutionaryEngine:
    """生物演化優化引擎
    
    整合所有演化組件的主引擎，提供：
    - 單次演化優化 (optimize)
    - 滾動視窗優化 (walk_forward_optimize)
    - 基因組轉換為策略配置 (genome_to_configs)
    
    Attributes:
        config: 演化配置
        genome_bounds: 基因邊界定義
    """
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        genome_bounds: Optional[Dict[str, GeneBounds]] = None,
    ):
        """初始化演化引擎
        
        Args:
            config: 演化配置，預設使用 EvolutionConfig()
            genome_bounds: 基因邊界定義，預設使用 DEFAULT_GENOME_BOUNDS
            
        Raises:
            ValueError: 若配置無效
        """
        self.config = config or EvolutionConfig()
        self.genome_bounds = genome_bounds or DEFAULT_GENOME_BOUNDS
        
        if not self.config.validate():
            raise ValueError("Invalid EvolutionConfig")
        
        # 初始化演化組件
        self._population_generator = PopulationGenerator(
            genome_bounds=self.genome_bounds,
            population_size=self.config.population_size,
        )
        
        self._fitness_evaluator = FitnessEvaluator(
            objective=self.config.fitness_objective,
            min_trades_threshold=self.config.min_trades_threshold,
        )
        
        self._selection_operator = SelectionOperator(
            tournament_size=self.config.tournament_size,
            elitism_rate=self.config.elitism_rate,
        )
        
        self._crossover_operator = CrossoverOperator(
            crossover_rate=self.config.crossover_rate,
            genome_bounds=self.genome_bounds,
        )
        
        self._mutation_operator = MutationOperator(
            mutation_rate=self.config.mutation_rate,
            mutation_strength=self.config.mutation_strength,
            genome_bounds=self.genome_bounds,
        )
    
    def optimize(
        self,
        symbol: str,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
        progress_callback: Optional[Callable[[int, GenerationStats], None]] = None,
    ) -> EvolutionHistory:
        """執行單次演化優化
        
        在給定的價格數據上執行遺傳演算法優化，尋找最佳策略參數。
        
        Args:
            symbol: 股票代碼（用於日誌記錄）
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            progress_callback: 進度回調函數，接收 (generation, stats) 參數
            
        Returns:
            演化歷史記錄
            
        Raises:
            ValueError: 若價格數據為空
        """
        if not prices:
            raise ValueError("Prices cannot be empty")
        
        # 建立世代控制器
        generation_controller = GenerationController(
            max_generations=self.config.max_generations,
            convergence_threshold=self.config.convergence_threshold,
            convergence_patience=self.config.convergence_patience,
            progress_callback=progress_callback,
        )
        
        # 生成初始種群
        initial_population = self._population_generator.generate_population()
        
        # 執行演化
        history = generation_controller.evolve(
            initial_population=initial_population,
            fitness_evaluator=self._fitness_evaluator,
            selection_operator=self._selection_operator,
            crossover_operator=self._crossover_operator,
            mutation_operator=self._mutation_operator,
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )
        
        return history

    
    def walk_forward_optimize(
        self,
        symbol: str,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
        walk_forward_config: Optional[WalkForwardConfig] = None,
        progress_callback: Optional[Callable[[int, int, WalkForwardResult], None]] = None,
    ) -> WalkForwardSummary:
        """執行滾動視窗優化
        
        在多個滾動視窗上執行演化優化與驗證，確保參數的泛化能力。
        
        Args:
            symbol: 股票代碼（用於日誌記錄）
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            walk_forward_config: 滾動視窗配置，預設使用 WalkForwardConfig()
            progress_callback: 進度回調函數，接收 (window_index, total_windows, result) 參數
            
        Returns:
            滾動視窗總結
            
        Raises:
            ValueError: 若價格數據不足
        """
        if not prices:
            raise ValueError("Prices cannot be empty")
        
        wf_config = walk_forward_config or WalkForwardConfig()
        
        # 建立滾動視窗分析器
        analyzer = WalkForwardAnalyzer(
            config=wf_config,
            population_size=self.config.population_size,
            max_generations=self.config.max_generations,
            fitness_objective=self.config.fitness_objective,
            tournament_size=self.config.tournament_size,
            elitism_rate=self.config.elitism_rate,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            mutation_strength=self.config.mutation_strength,
            min_trades_threshold=self.config.min_trades_threshold,
            convergence_threshold=self.config.convergence_threshold,
            convergence_patience=self.config.convergence_patience,
        )
        
        # 執行滾動視窗分析
        summary = analyzer.analyze(
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes,
            progress_callback=progress_callback,
        )
        
        return summary
    
    def genome_to_configs(
        self,
        genome: Genome,
        symbol: str,
    ) -> Tuple[Any, Any]:
        """將基因組轉換為策略配置
        
        將演化出的最佳基因組轉換為 DualEngineConfig 和 FactorConfig，
        以便直接應用於交易策略。
        
        Args:
            genome: 要轉換的基因組
            symbol: 股票代碼
            
        Returns:
            (DualEngineConfig, FactorConfig) 元組
        """
        # 延遲導入以避免循環依賴
        from pattern_quant.strategy.models import DualEngineConfig
        from pattern_quant.optimization.factor_config import (
            FactorConfig,
            RSIConfig,
            VolumeConfig,
            MACDConfig,
            EMAConfig,
            BollingerConfig,
        )
        
        # 歸一化權重
        normalized_genome = genome.normalize_weights()
        
        # 建立 DualEngineConfig
        dual_engine_config = DualEngineConfig(
            enabled=True,
            adx_trend_threshold=normalized_genome.dual_engine.trend_threshold,
            adx_range_threshold=normalized_genome.dual_engine.range_threshold,
            trend_allocation=normalized_genome.dual_engine.trend_allocation,
            range_allocation=normalized_genome.dual_engine.range_allocation,
            noise_allocation=0.0,  # 混沌狀態不交易
            bbw_stability_threshold=normalized_genome.dual_engine.volatility_stability,
        )
        
        # 建立 FactorConfig
        factor_config = FactorConfig(
            symbol=symbol,
            rsi=RSIConfig(
                enabled=True,
                weight=normalized_genome.factor_weights.rsi_weight,
                period=normalized_genome.micro_indicators.rsi_period,
                overbought=normalized_genome.micro_indicators.rsi_overbought,
                oversold=normalized_genome.micro_indicators.rsi_oversold,
            ),
            volume=VolumeConfig(
                enabled=True,
                weight=normalized_genome.factor_weights.volume_weight,
                high_volume_threshold=normalized_genome.micro_indicators.volume_spike_multiplier,
            ),
            macd=MACDConfig(
                enabled=True,
                weight=normalized_genome.factor_weights.macd_weight,
                golden_cross_bonus=normalized_genome.micro_indicators.macd_bonus,
            ),
            ema=EMAConfig(
                enabled=True,
                weight=normalized_genome.factor_weights.ema_weight,
            ),
            bollinger=BollingerConfig(
                enabled=True,
                weight=normalized_genome.factor_weights.bollinger_weight,
                squeeze_threshold=normalized_genome.micro_indicators.bollinger_squeeze_threshold,
            ),
            buy_threshold=normalized_genome.factor_weights.score_threshold,
            watch_threshold=normalized_genome.factor_weights.score_threshold - 10.0,
        )
        
        return dual_engine_config, factor_config
