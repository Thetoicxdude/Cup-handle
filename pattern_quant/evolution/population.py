"""
種群生成器 (Population Generator)

負責生成初始種群並確保多樣性。
"""

import random
from typing import Dict, List, Optional
import math

from .models import (
    GeneType,
    GeneBounds,
    DualEngineControlGenes,
    FactorWeightGenes,
    MicroIndicatorGenes,
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
    CHROMOSOME_GENE_NAMES,
    CHROMOSOME_LENGTH,
    validate_genome_bounds,
)


class PopulationGenerator:
    """種群生成器
    
    負責生成初始種群並確保多樣性。
    
    Attributes:
        genome_bounds: 基因邊界定義
        population_size: 種群大小
        diversity_threshold: 多樣性閾值
    """
    
    def __init__(
        self,
        genome_bounds: Optional[Dict[str, GeneBounds]] = None,
        population_size: int = 50,
        diversity_threshold: float = 0.3,
    ):
        """初始化種群生成器
        
        Args:
            genome_bounds: 基因邊界定義，預設使用 DEFAULT_GENOME_BOUNDS
            population_size: 種群大小，預設 50
            diversity_threshold: 多樣性閾值，預設 0.3
            
        Raises:
            ValueError: 若 population_size 不在 [50, 100] 範圍內
        """
        if population_size < 50 or population_size > 100:
            raise ValueError(
                f"Population size must be between 50 and 100, got {population_size}"
            )
        
        self.genome_bounds = genome_bounds or DEFAULT_GENOME_BOUNDS
        self.population_size = population_size
        self.diversity_threshold = diversity_threshold
    
    def generate_random_individual(self, generation: int = 0) -> Individual:
        """生成隨機個體
        
        在定義的參數邊界內隨機生成一個個體。
        
        Args:
            generation: 所屬世代，預設為 0
            
        Returns:
            隨機生成的個體
        """
        chromosome = []
        
        for gene_name in CHROMOSOME_GENE_NAMES:
            bounds = self.genome_bounds[gene_name]
            
            if bounds.gene_type == GeneType.INTEGER:
                # 整數類型：在範圍內隨機選擇整數
                value = float(random.randint(
                    int(bounds.min_value),
                    int(bounds.max_value)
                ))
            else:
                # 浮點類型：在範圍內均勻分布
                value = random.uniform(bounds.min_value, bounds.max_value)
            
            chromosome.append(value)
        
        genome = Genome.from_chromosome(chromosome)
        
        return Individual(
            genome=genome,
            fitness=0.0,
            generation=generation,
        )
    
    def generate_population(self, generation: int = 0) -> List[Individual]:
        """生成初始種群
        
        生成指定大小的初始種群，並確保多樣性。
        
        Args:
            generation: 所屬世代，預設為 0
            
        Returns:
            初始種群列表
        """
        population = []
        
        for _ in range(self.population_size):
            individual = self.generate_random_individual(generation)
            population.append(individual)
        
        # 確保種群多樣性
        population = self.ensure_diversity(population)
        
        return population
    
    def check_diversity(self, population: List[Individual]) -> float:
        """計算種群多樣性指標
        
        使用基因標準差的平均值來衡量多樣性。
        多樣性指標範圍為 [0, 1]，值越高表示多樣性越好。
        
        Args:
            population: 種群列表
            
        Returns:
            多樣性指標 (0-1)
        """
        if len(population) < 2:
            return 0.0
        
        # 收集所有染色體
        chromosomes = [ind.genome.to_chromosome() for ind in population]
        
        # 計算每個基因的標準差（相對於其範圍）
        normalized_stds = []
        
        for i, gene_name in enumerate(CHROMOSOME_GENE_NAMES):
            bounds = self.genome_bounds[gene_name]
            gene_range = bounds.max_value - bounds.min_value
            
            if gene_range == 0:
                continue
            
            # 收集該基因在所有個體中的值
            gene_values = [chrom[i] for chrom in chromosomes]
            
            # 計算標準差
            mean = sum(gene_values) / len(gene_values)
            variance = sum((v - mean) ** 2 for v in gene_values) / len(gene_values)
            std = math.sqrt(variance)
            
            # 歸一化標準差（相對於範圍）
            # 均勻分布的標準差約為 range / sqrt(12) ≈ 0.289 * range
            # 我們用這個作為參考來歸一化
            expected_std = gene_range / math.sqrt(12)
            normalized_std = min(std / expected_std, 1.0) if expected_std > 0 else 0.0
            normalized_stds.append(normalized_std)
        
        # 返回平均歸一化標準差
        if not normalized_stds:
            return 0.0
        
        return sum(normalized_stds) / len(normalized_stds)
    
    def ensure_diversity(self, population: List[Individual]) -> List[Individual]:
        """確保種群多樣性
        
        若種群多樣性低於閾值，則重新生成部分個體以提高多樣性。
        
        Args:
            population: 原始種群
            
        Returns:
            多樣性改善後的種群
        """
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            diversity = self.check_diversity(population)
            
            if diversity >= self.diversity_threshold:
                break
            
            # 多樣性不足，重新生成部分個體
            # 保留前半部分，重新生成後半部分
            num_to_regenerate = len(population) // 2
            
            # 保留原有個體
            kept = population[:len(population) - num_to_regenerate]
            
            # 生成新個體
            new_individuals = []
            for _ in range(num_to_regenerate):
                new_ind = self.generate_random_individual(
                    generation=population[0].generation if population else 0
                )
                new_individuals.append(new_ind)
            
            population = kept + new_individuals
            attempt += 1
        
        return population


def validate_population_bounds(
    population: List[Individual],
    bounds: Optional[Dict[str, GeneBounds]] = None
) -> bool:
    """驗證種群中所有個體是否在邊界內
    
    Args:
        population: 種群列表
        bounds: 基因邊界定義，預設使用 DEFAULT_GENOME_BOUNDS
        
    Returns:
        所有個體是否都在邊界內
    """
    bounds = bounds or DEFAULT_GENOME_BOUNDS
    
    for individual in population:
        if not validate_genome_bounds(individual.genome, bounds):
            return False
    
    return True
