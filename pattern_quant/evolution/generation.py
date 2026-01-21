"""
世代控制器 (Generation Controller)

負責控制演化迭代流程，整合所有演化算子執行完整的遺傳演算法。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
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
from .fitness import FitnessEvaluator, FitnessResult


@dataclass
class GenerationStats:
    """世代統計
    
    記錄單一世代的統計資訊。
    
    Attributes:
        generation: 世代編號
        best_fitness: 最佳適應度
        average_fitness: 平均適應度
        worst_fitness: 最差適應度
        best_genome: 最佳個體的基因組
    """
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    best_genome: Genome


@dataclass
class EvolutionHistory:
    """演化歷史
    
    記錄完整演化過程的歷史資訊。
    
    Attributes:
        generations: 各世代統計列表
        final_best: 最終最佳個體
        total_generations: 總世代數
        converged: 是否收斂
    """
    generations: List[GenerationStats]
    final_best: Individual
    total_generations: int
    converged: bool


class GenerationController:
    """世代控制器
    
    負責控制演化迭代流程，整合選擇、交叉、突變算子執行完整的遺傳演算法。
    
    Attributes:
        max_generations: 最大世代數
        convergence_threshold: 收斂閾值
        convergence_patience: 收斂耐心值（連續無改善世代數）
        progress_callback: 進度回調函數
    """
    
    def __init__(
        self,
        max_generations: int = 20,
        convergence_threshold: float = 0.001,
        convergence_patience: int = 5,
        progress_callback: Optional[Callable[[int, GenerationStats], None]] = None,
    ):
        """初始化世代控制器
        
        Args:
            max_generations: 最大世代數，預設為 20
            convergence_threshold: 收斂閾值，預設為 0.001
            convergence_patience: 收斂耐心值，預設為 5
            progress_callback: 進度回調函數，接收 (generation, stats) 參數
            
        Raises:
            ValueError: 若 max_generations 不在 [10, 50] 範圍內
        """
        if max_generations < 10 or max_generations > 50:
            raise ValueError(
                f"Max generations must be between 10 and 50, got {max_generations}"
            )
        
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.progress_callback = progress_callback
    
    def evolve(
        self,
        initial_population: List[Individual],
        fitness_evaluator: FitnessEvaluator,
        selection_operator: SelectionOperator,
        crossover_operator: CrossoverOperator,
        mutation_operator: MutationOperator,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> EvolutionHistory:
        """執行演化流程
        
        執行完整的遺傳演算法迭代，包含評估、選擇、交叉、突變等步驟。
        
        Args:
            initial_population: 初始種群
            fitness_evaluator: 適應度評估器
            selection_operator: 選擇算子
            crossover_operator: 交叉算子
            mutation_operator: 突變算子
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            
        Returns:
            演化歷史記錄
            
        Raises:
            ValueError: 若初始種群為空
        """
        if not initial_population:
            raise ValueError("Initial population cannot be empty")
        
        # 初始化
        population = initial_population
        history: List[GenerationStats] = []
        converged = False
        
        # 演化迭代
        for gen in range(self.max_generations):
            # 1. 評估適應度
            population = fitness_evaluator.evaluate_population(
                population, prices, highs, lows, volumes
            )
            
            # 2. 記錄統計
            stats = self._calculate_stats(gen, population)
            history.append(stats)
            
            # 3. 進度回調
            if self.progress_callback is not None:
                self.progress_callback(gen, stats)
            
            # 4. 檢查收斂
            if self._check_convergence(history):
                converged = True
                break
            
            # 5. 若非最後一代，執行演化操作
            if gen < self.max_generations - 1:
                population = self._evolve_generation(
                    population,
                    selection_operator,
                    crossover_operator,
                    mutation_operator,
                    gen + 1,
                )
        
        # 找出最終最佳個體
        final_best = max(population, key=lambda ind: ind.fitness)
        
        return EvolutionHistory(
            generations=history,
            final_best=final_best,
            total_generations=len(history),
            converged=converged,
        )
    
    def _evolve_generation(
        self,
        population: List[Individual],
        selection_operator: SelectionOperator,
        crossover_operator: CrossoverOperator,
        mutation_operator: MutationOperator,
        next_generation: int,
    ) -> List[Individual]:
        """執行單一世代的演化操作
        
        Args:
            population: 當前種群
            selection_operator: 選擇算子
            crossover_operator: 交叉算子
            mutation_operator: 突變算子
            next_generation: 下一世代編號
            
        Returns:
            新一代種群
        """
        population_size = len(population)
        
        # 1. 精英保留
        elite = selection_operator.get_elite(population)
        
        # 2. 選擇親代
        num_offspring_needed = population_size - len(elite)
        num_parents = max(2, num_offspring_needed)
        parents = selection_operator.select_parents(population, num_parents)
        
        # 3. 交叉產生子代
        offspring = crossover_operator.crossover_population(
            parents, num_offspring_needed
        )
        
        # 4. 突變
        mutated_offspring = mutation_operator.mutate_population(offspring)
        
        # 5. 更新世代編號
        for ind in mutated_offspring:
            ind.generation = next_generation
        
        for ind in elite:
            ind.generation = next_generation
        
        # 6. 組合新種群
        new_population = elite + mutated_offspring
        
        return new_population[:population_size]
    
    def _calculate_stats(
        self,
        generation: int,
        population: List[Individual],
    ) -> GenerationStats:
        """計算世代統計
        
        Args:
            generation: 世代編號
            population: 種群
            
        Returns:
            世代統計資訊
        """
        fitness_values = [ind.fitness for ind in population]
        
        best_fitness = max(fitness_values)
        worst_fitness = min(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        
        # 找出最佳個體
        best_individual = max(population, key=lambda ind: ind.fitness)
        
        return GenerationStats(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            best_genome=Genome.from_chromosome(
                best_individual.genome.to_chromosome()
            ),
        )
    
    def _check_convergence(
        self,
        history: List[GenerationStats],
    ) -> bool:
        """檢查是否收斂
        
        若連續 convergence_patience 個世代的最佳適應度改善
        低於 convergence_threshold，則判定為收斂。
        
        Args:
            history: 演化歷史
            
        Returns:
            是否收斂
        """
        if len(history) < self.convergence_patience + 1:
            return False
        
        # 取得最近 convergence_patience + 1 個世代的最佳適應度
        recent_best = [
            stats.best_fitness
            for stats in history[-(self.convergence_patience + 1):]
        ]
        
        # 計算改善量
        improvements = []
        for i in range(1, len(recent_best)):
            improvement = recent_best[i] - recent_best[i - 1]
            improvements.append(improvement)
        
        # 檢查是否所有改善都低於閾值
        all_below_threshold = all(
            abs(imp) < self.convergence_threshold
            for imp in improvements
        )
        
        return all_below_threshold
