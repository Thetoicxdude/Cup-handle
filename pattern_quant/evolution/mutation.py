"""
突變算子 (Mutation Operator)

負責執行基因突變操作，對子代引入隨機擾動以避免陷入局部最優解。
"""

import random
from typing import Dict, List, Optional

from .models import (
    GeneType,
    GeneBounds,
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
    CHROMOSOME_GENE_NAMES,
    CHROMOSOME_LENGTH,
)


class MutationOperator:
    """突變算子
    
    實作高斯突變 (Gaussian Mutation) 機制，對基因引入隨機擾動。
    
    Attributes:
        mutation_rate: 每個基因的突變機率 (1%-5%)
        mutation_strength: 高斯突變的標準差係數 (相對於基因範圍)
        genome_bounds: 基因邊界定義
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.02,
        mutation_strength: float = 0.1,
        genome_bounds: Optional[Dict[str, GeneBounds]] = None,
    ):
        """初始化突變算子
        
        Args:
            mutation_rate: 每個基因的突變機率，預設為 0.02 (2%)
            mutation_strength: 高斯突變的標準差係數，預設為 0.1
            genome_bounds: 基因邊界定義，預設使用 DEFAULT_GENOME_BOUNDS
            
        Raises:
            ValueError: 若 mutation_rate 不在 [0.01, 0.05] 範圍內
        """
        if mutation_rate < 0.01 or mutation_rate > 0.05:
            raise ValueError(
                f"Mutation rate must be between 0.01 and 0.05, got {mutation_rate}"
            )
        
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.genome_bounds = genome_bounds or DEFAULT_GENOME_BOUNDS
    
    def gaussian_mutate(
        self,
        individual: Individual,
    ) -> Individual:
        """高斯突變
        
        對個體的每個基因以 mutation_rate 的機率進行高斯突變。
        突變時，從常態分布中取樣一個隨機值加到基因上。
        若突變後的值超出邊界，則 clamp 到最近的邊界值。
        
        Args:
            individual: 要突變的個體
            
        Returns:
            突變後的新個體（原個體不變）
        """
        # 取得染色體
        chromosome = individual.genome.to_chromosome()
        mutated_chromosome = chromosome.copy()
        
        # 對每個基因進行突變判定
        for i, gene_name in enumerate(CHROMOSOME_GENE_NAMES):
            if random.random() < self.mutation_rate:
                # 執行高斯突變
                bounds = self.genome_bounds[gene_name]
                gene_range = bounds.max_value - bounds.min_value
                
                # 計算標準差（相對於基因範圍）
                std = gene_range * self.mutation_strength
                
                # 從常態分布取樣擾動值
                perturbation = random.gauss(0, std)
                
                # 加上擾動
                new_value = mutated_chromosome[i] + perturbation
                
                # Clamp 到邊界內
                mutated_chromosome[i] = bounds.clamp(new_value)
        
        # 建立新個體
        mutated_individual = Individual(
            genome=Genome.from_chromosome(mutated_chromosome),
            fitness=0.0,  # 重置適應度
            generation=individual.generation,
        )
        
        return mutated_individual
    
    def mutate_population(
        self,
        population: List[Individual],
    ) -> List[Individual]:
        """對種群執行突變
        
        對種群中的每個個體執行高斯突變。
        
        Args:
            population: 種群列表
            
        Returns:
            突變後的種群列表（原種群不變）
            
        Raises:
            ValueError: 若種群為空
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        mutated_population = []
        
        for individual in population:
            mutated_individual = self.gaussian_mutate(individual)
            mutated_population.append(mutated_individual)
        
        return mutated_population
