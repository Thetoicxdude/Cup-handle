"""
滾動視窗分析器 (Walk-Forward Analyzer)

負責執行滾動視窗驗證，確保演化出的參數在未見數據上的有效性。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
import math

from .models import (
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
)
from .population import PopulationGenerator
from .selection import SelectionOperator
from .crossover import CrossoverOperator
from .mutation import MutationOperator
from .fitness import FitnessEvaluator, FitnessObjective, FitnessResult
from .generation import GenerationController, EvolutionHistory


@dataclass
class WalkForwardConfig:
    """滾動視窗配置
    
    定義訓練視窗、測試視窗與步進長度的配置。
    
    Attributes:
        in_sample_days: 訓練視窗長度（天數），預設為 252（約一年）
        out_of_sample_days: 測試視窗長度（天數），預設為 63（約一季）
        step_size_days: 步進長度（天數），預設為 21（約一個月）
    """
    in_sample_days: int = 252
    out_of_sample_days: int = 63
    step_size_days: int = 21
    
    def validate(self) -> bool:
        """驗證配置是否有效
        
        Returns:
            配置是否有效
        """
        return (
            self.in_sample_days > 0
            and self.out_of_sample_days > 0
            and self.step_size_days > 0
            and self.step_size_days <= self.out_of_sample_days
        )


@dataclass
class WalkForwardResult:
    """滾動視窗結果
    
    記錄單一滾動視窗的演化與驗證結果。
    
    Attributes:
        window_index: 視窗索引（從 0 開始）
        in_sample_start: 訓練視窗起始索引
        in_sample_end: 訓練視窗結束索引（不含）
        out_of_sample_start: 測試視窗起始索引
        out_of_sample_end: 測試視窗結束索引（不含）
        best_genome: 訓練視窗中演化出的最佳基因組
        in_sample_fitness: 訓練視窗的適應度分數
        out_of_sample_fitness: 測試視窗的適應度分數
        out_of_sample_trades: 測試視窗的交易次數
        out_of_sample_return: 測試視窗的報酬率
    """
    window_index: int
    in_sample_start: int
    in_sample_end: int
    out_of_sample_start: int
    out_of_sample_end: int
    best_genome: Genome
    in_sample_fitness: float
    out_of_sample_fitness: float
    out_of_sample_trades: int
    out_of_sample_return: float


@dataclass
class WalkForwardSummary:
    """滾動視窗總結
    
    彙總所有滾動視窗的結果與整體績效指標。
    
    Attributes:
        windows: 各視窗結果列表
        aggregate_return: 彙總報酬率
        aggregate_sharpe: 彙總夏普比率
        aggregate_win_rate: 彙總勝率
        robustness_score: 穩健性評分（IS vs OOS 表現一致性）
    """
    windows: List[WalkForwardResult]
    aggregate_return: float
    aggregate_sharpe: float
    aggregate_win_rate: float
    robustness_score: float


class WalkForwardAnalyzer:
    """滾動視窗分析器
    
    執行滾動視窗驗證，在訓練視窗上執行演化優化，
    然後在測試視窗上驗證最佳參數的有效性。
    
    Attributes:
        config: 滾動視窗配置
        population_size: 種群大小
        max_generations: 最大世代數
        fitness_objective: 適應度目標函數
        tournament_size: 競賽選擇大小
        elitism_rate: 精英保留率
        crossover_rate: 交叉率
        mutation_rate: 突變率
        mutation_strength: 突變強度
        min_trades_threshold: 最低交易次數閾值
        convergence_threshold: 收斂閾值
        convergence_patience: 收斂耐心值
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        population_size: int = 50,
        max_generations: int = 20,
        fitness_objective: FitnessObjective = FitnessObjective.SHARPE_RATIO,
        tournament_size: int = 3,
        elitism_rate: float = 0.1,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.02,
        mutation_strength: float = 0.1,
        min_trades_threshold: int = 10,
        convergence_threshold: float = 0.001,
        convergence_patience: int = 5,
    ):
        """初始化滾動視窗分析器
        
        Args:
            config: 滾動視窗配置
            population_size: 種群大小，預設為 50
            max_generations: 最大世代數，預設為 20
            fitness_objective: 適應度目標函數，預設為 SHARPE_RATIO
            tournament_size: 競賽選擇大小，預設為 3
            elitism_rate: 精英保留率，預設為 0.1
            crossover_rate: 交叉率，預設為 0.8
            mutation_rate: 突變率，預設為 0.02
            mutation_strength: 突變強度，預設為 0.1
            min_trades_threshold: 最低交易次數閾值，預設為 10
            convergence_threshold: 收斂閾值，預設為 0.001
            convergence_patience: 收斂耐心值，預設為 5
            
        Raises:
            ValueError: 若配置無效
        """
        if not config.validate():
            raise ValueError("Invalid WalkForwardConfig")
        
        self.config = config
        self.population_size = population_size
        self.max_generations = max_generations
        self.fitness_objective = fitness_objective
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.min_trades_threshold = min_trades_threshold
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
    
    def analyze(
        self,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
        progress_callback: Optional[Callable[[int, int, WalkForwardResult], None]] = None,
    ) -> WalkForwardSummary:
        """執行滾動視窗分析
        
        在多個滾動視窗上執行演化優化與驗證。
        
        Args:
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            progress_callback: 進度回調函數，接收 (window_index, total_windows, result) 參數
            
        Returns:
            滾動視窗總結
            
        Raises:
            ValueError: 若數據長度不足
        """
        # 計算所需最小數據長度
        min_data_length = self.config.in_sample_days + self.config.out_of_sample_days
        if len(prices) < min_data_length:
            raise ValueError(
                f"Insufficient data: need at least {min_data_length} data points, "
                f"got {len(prices)}"
            )
        
        # 計算視窗數量
        total_windows = self._calculate_total_windows(len(prices))
        
        if total_windows == 0:
            raise ValueError("Cannot create any walk-forward windows with given data")
        
        # 執行滾動視窗分析
        results: List[WalkForwardResult] = []
        
        for window_idx in range(total_windows):
            # 計算視窗邊界
            is_start, is_end, oos_start, oos_end = self._calculate_window_bounds(
                window_idx, len(prices)
            )
            
            # 截取訓練視窗數據
            is_prices = self._get_window_data(prices, is_start, is_end)
            is_highs = self._get_window_data(highs, is_start, is_end) if highs else None
            is_lows = self._get_window_data(lows, is_start, is_end) if lows else None
            is_volumes = self._get_window_data(volumes, is_start, is_end) if volumes else None
            
            # 在訓練視窗上執行演化
            evolution_history = self._run_evolution(
                is_prices, is_highs, is_lows, is_volumes
            )
            
            best_genome = evolution_history.final_best.genome
            in_sample_fitness = evolution_history.final_best.fitness
            
            # 截取測試視窗數據
            oos_prices = self._get_window_data(prices, oos_start, oos_end)
            oos_highs = self._get_window_data(highs, oos_start, oos_end) if highs else None
            oos_lows = self._get_window_data(lows, oos_start, oos_end) if lows else None
            oos_volumes = self._get_window_data(volumes, oos_start, oos_end) if volumes else None
            
            # 在測試視窗上評估最佳基因組
            oos_result = self._evaluate_genome(
                best_genome, oos_prices, oos_highs, oos_lows, oos_volumes
            )
            
            # 建立視窗結果
            result = WalkForwardResult(
                window_index=window_idx,
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_of_sample_start=oos_start,
                out_of_sample_end=oos_end,
                best_genome=best_genome,
                in_sample_fitness=in_sample_fitness,
                out_of_sample_fitness=oos_result.fitness_score,
                out_of_sample_trades=oos_result.total_trades,
                out_of_sample_return=oos_result.total_return,
            )
            
            results.append(result)
            
            # 進度回調
            if progress_callback is not None:
                progress_callback(window_idx, total_windows, result)
        
        # 計算彙總指標
        summary = self._calculate_summary(results)
        
        return summary
    
    def _calculate_total_windows(self, data_length: int) -> int:
        """計算可建立的視窗總數
        
        Args:
            data_length: 數據長度
            
        Returns:
            視窗總數
        """
        # 第一個視窗需要 in_sample + out_of_sample 的數據
        # 之後每個視窗需要額外 step_size 的數據
        min_required = self.config.in_sample_days + self.config.out_of_sample_days
        
        if data_length < min_required:
            return 0
        
        # 計算可以滾動多少次
        remaining = data_length - min_required
        additional_windows = remaining // self.config.step_size_days
        
        return 1 + additional_windows
    
    def _calculate_window_bounds(
        self,
        window_index: int,
        data_length: int,
    ) -> tuple:
        """計算視窗邊界
        
        Args:
            window_index: 視窗索引
            data_length: 數據長度
            
        Returns:
            (in_sample_start, in_sample_end, out_of_sample_start, out_of_sample_end)
        """
        # 訓練視窗起始位置隨視窗索引推進
        is_start = window_index * self.config.step_size_days
        is_end = is_start + self.config.in_sample_days
        
        # 測試視窗緊接在訓練視窗之後
        oos_start = is_end
        oos_end = min(oos_start + self.config.out_of_sample_days, data_length)
        
        return is_start, is_end, oos_start, oos_end
    
    def _get_window_data(
        self,
        data: Optional[List[float]],
        start_idx: int,
        end_idx: int,
    ) -> Optional[List[float]]:
        """截取視窗數據
        
        Args:
            data: 原始數據序列
            start_idx: 起始索引
            end_idx: 結束索引（不含）
            
        Returns:
            截取的數據，若原始數據為 None 則返回 None
        """
        if data is None:
            return None
        
        return data[start_idx:end_idx]
    
    def _run_evolution(
        self,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> EvolutionHistory:
        """在訓練視窗上執行演化
        
        Args:
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            
        Returns:
            演化歷史
        """
        # 建立演化組件
        population_generator = PopulationGenerator(
            genome_bounds=DEFAULT_GENOME_BOUNDS,
            population_size=self.population_size,
        )
        
        fitness_evaluator = FitnessEvaluator(
            objective=self.fitness_objective,
            min_trades_threshold=self.min_trades_threshold,
        )
        
        selection_operator = SelectionOperator(
            tournament_size=self.tournament_size,
            elitism_rate=self.elitism_rate,
        )
        
        crossover_operator = CrossoverOperator(
            crossover_rate=self.crossover_rate,
            genome_bounds=DEFAULT_GENOME_BOUNDS,
        )
        
        mutation_operator = MutationOperator(
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength,
            genome_bounds=DEFAULT_GENOME_BOUNDS,
        )
        
        generation_controller = GenerationController(
            max_generations=self.max_generations,
            convergence_threshold=self.convergence_threshold,
            convergence_patience=self.convergence_patience,
        )
        
        # 生成初始種群
        initial_population = population_generator.generate_population()
        
        # 執行演化
        history = generation_controller.evolve(
            initial_population=initial_population,
            fitness_evaluator=fitness_evaluator,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )
        
        return history
    
    def _evaluate_genome(
        self,
        genome: Genome,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> FitnessResult:
        """評估基因組在測試視窗上的表現
        
        Args:
            genome: 要評估的基因組
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            
        Returns:
            適應度評估結果
        """
        fitness_evaluator = FitnessEvaluator(
            objective=self.fitness_objective,
            min_trades_threshold=self.min_trades_threshold,
        )
        
        individual = Individual(genome=genome)
        
        return fitness_evaluator.evaluate(
            individual, prices, highs, lows, volumes
        )
    
    def _calculate_summary(
        self,
        results: List[WalkForwardResult],
    ) -> WalkForwardSummary:
        """計算滾動視窗總結
        
        Args:
            results: 各視窗結果列表
            
        Returns:
            滾動視窗總結
        """
        if not results:
            return WalkForwardSummary(
                windows=[],
                aggregate_return=0.0,
                aggregate_sharpe=0.0,
                aggregate_win_rate=0.0,
                robustness_score=0.0,
            )
        
        # 計算彙總報酬率（複利）
        aggregate_return = 1.0
        for result in results:
            aggregate_return *= (1 + result.out_of_sample_return)
        aggregate_return -= 1.0
        
        # 計算彙總夏普比率（簡化：使用 OOS 適應度的平均值）
        oos_fitness_values = [r.out_of_sample_fitness for r in results]
        aggregate_sharpe = sum(oos_fitness_values) / len(oos_fitness_values)
        
        # 計算彙總勝率（正報酬視窗的比例）
        winning_windows = sum(1 for r in results if r.out_of_sample_return > 0)
        aggregate_win_rate = winning_windows / len(results)
        
        # 計算穩健性評分（IS vs OOS 表現一致性）
        robustness_score = self._calculate_robustness_score(results)
        
        return WalkForwardSummary(
            windows=results,
            aggregate_return=aggregate_return,
            aggregate_sharpe=aggregate_sharpe,
            aggregate_win_rate=aggregate_win_rate,
            robustness_score=robustness_score,
        )
    
    def _calculate_robustness_score(
        self,
        results: List[WalkForwardResult],
    ) -> float:
        """計算穩健性評分
        
        穩健性評分衡量訓練視窗與測試視窗表現的一致性。
        使用 OOS/IS 適應度比率的平均值，並限制在 [0, 1] 範圍內。
        
        Args:
            results: 各視窗結果列表
            
        Returns:
            穩健性評分 (0-1)，越高表示越穩健
        """
        if not results:
            return 0.0
        
        ratios = []
        for result in results:
            if result.in_sample_fitness > 0:
                # 計算 OOS/IS 比率
                ratio = result.out_of_sample_fitness / result.in_sample_fitness
                # 限制比率在合理範圍內
                ratio = max(0.0, min(2.0, ratio))
                ratios.append(ratio)
            elif result.out_of_sample_fitness >= 0:
                # IS 為零或負，但 OOS 非負，給予中等評分
                ratios.append(0.5)
            else:
                # 兩者都為負或零
                ratios.append(0.0)
        
        if not ratios:
            return 0.0
        
        # 計算平均比率
        avg_ratio = sum(ratios) / len(ratios)
        
        # 轉換為 0-1 評分
        # 比率為 1.0 時評分最高（IS 與 OOS 表現一致）
        # 比率偏離 1.0 時評分降低
        if avg_ratio <= 1.0:
            # 比率 0-1 映射到評分 0-1
            robustness = avg_ratio
        else:
            # 比率 1-2 映射到評分 1-0.5
            # OOS 優於 IS 也是好事，但過度優化可能有問題
            robustness = 1.0 - (avg_ratio - 1.0) * 0.5
        
        return max(0.0, min(1.0, robustness))
