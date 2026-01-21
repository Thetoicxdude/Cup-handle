"""
交叉算子 (Crossover Operator)

負責執行基因交叉操作，透過基因重組產生子代。
"""

import random
from typing import Dict, List, Optional, Tuple

from .models import (
    GeneBounds,
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
    CHROMOSOME_GENE_NAMES,
    CHROMOSOME_LENGTH,
)


class CrossoverOperator:
    """交叉算子
    
    實作雙點交叉 (Two-Point Crossover) 機制，透過基因重組產生子代。
    
    Attributes:
        crossover_rate: 交叉機率 (0.6-0.9)
        genome_bounds: 基因邊界定義
    """
    
    def __init__(
        self,
        crossover_rate: float = 0.8,
        genome_bounds: Optional[Dict[str, GeneBounds]] = None,
    ):
        """初始化交叉算子
        
        Args:
            crossover_rate: 交叉機率，預設為 0.8
            genome_bounds: 基因邊界定義，預設使用 DEFAULT_GENOME_BOUNDS
            
        Raises:
            ValueError: 若 crossover_rate 不在 [0.6, 0.9] 範圍內
        """
        if crossover_rate < 0.6 or crossover_rate > 0.9:
            raise ValueError(
                f"Crossover rate must be between 0.6 and 0.9, got {crossover_rate}"
            )
        
        self.crossover_rate = crossover_rate
        self.genome_bounds = genome_bounds or DEFAULT_GENOME_BOUNDS
    
    def two_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Tuple[Individual, Individual]:
        """雙點交叉
        
        隨機選擇兩個切點，交換親代之間的基因片段產生兩個子代。
        子代的基因完全來自親代，不會產生新的數值。
        
        Args:
            parent1: 第一個親代
            parent2: 第二個親代
            
        Returns:
            兩個子代個體的元組 (offspring1, offspring2)
        """
        # 取得親代染色體
        chrom1 = parent1.genome.to_chromosome()
        chrom2 = parent2.genome.to_chromosome()
        
        # 根據交叉機率決定是否執行交叉
        if random.random() > self.crossover_rate:
            # 不執行交叉，直接複製親代
            next_gen = max(parent1.generation, parent2.generation) + 1
            offspring1 = Individual(
                genome=Genome.from_chromosome(chrom1.copy()),
                fitness=0.0,
                generation=next_gen,
            )
            offspring2 = Individual(
                genome=Genome.from_chromosome(chrom2.copy()),
                fitness=0.0,
                generation=next_gen,
            )
            return offspring1, offspring2
        
        # 隨機選擇兩個切點 (0 <= point1 < point2 <= CHROMOSOME_LENGTH)
        point1 = random.randint(0, CHROMOSOME_LENGTH - 1)
        point2 = random.randint(point1 + 1, CHROMOSOME_LENGTH)
        
        # 建立子代染色體
        # offspring1: parent1[0:point1] + parent2[point1:point2] + parent1[point2:]
        # offspring2: parent2[0:point1] + parent1[point1:point2] + parent2[point2:]
        offspring_chrom1 = chrom1[:point1] + chrom2[point1:point2] + chrom1[point2:]
        offspring_chrom2 = chrom2[:point1] + chrom1[point1:point2] + chrom2[point2:]
        
        # 驗證並修正子代基因邊界
        offspring_chrom1 = self._validate_and_clamp(offspring_chrom1)
        offspring_chrom2 = self._validate_and_clamp(offspring_chrom2)
        
        # 建立子代個體
        offspring1 = Individual(
            genome=Genome.from_chromosome(offspring_chrom1),
            fitness=0.0,
            generation=max(parent1.generation, parent2.generation) + 1,
        )
        offspring2 = Individual(
            genome=Genome.from_chromosome(offspring_chrom2),
            fitness=0.0,
            generation=max(parent1.generation, parent2.generation) + 1,
        )
        
        return offspring1, offspring2
    
    def _validate_and_clamp(self, chromosome: List[float]) -> List[float]:
        """驗證並修正染色體基因邊界
        
        確保所有基因值都在定義的邊界內。
        
        Args:
            chromosome: 染色體陣列
            
        Returns:
            修正後的染色體陣列
        """
        clamped = []
        for i, gene_name in enumerate(CHROMOSOME_GENE_NAMES):
            if gene_name in self.genome_bounds:
                bounds = self.genome_bounds[gene_name]
                clamped.append(bounds.clamp(chromosome[i]))
            else:
                clamped.append(chromosome[i])
        return clamped
    
    def crossover_population(
        self,
        parents: List[Individual],
        offspring_count: int,
    ) -> List[Individual]:
        """對親代執行交叉產生子代
        
        從親代列表中隨機配對，執行交叉操作產生指定數量的子代。
        
        Args:
            parents: 親代列表
            offspring_count: 需要產生的子代數量
            
        Returns:
            子代列表
            
        Raises:
            ValueError: 若親代列表為空
            ValueError: 若 offspring_count < 1
        """
        if not parents:
            raise ValueError("Parents list cannot be empty")
        
        if offspring_count < 1:
            raise ValueError(
                f"Offspring count must be at least 1, got {offspring_count}"
            )
        
        offspring = []
        
        while len(offspring) < offspring_count:
            # 隨機選擇兩個親代（允許重複選擇）
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # 執行交叉
            child1, child2 = self.two_point_crossover(parent1, parent2)
            
            # 加入子代列表
            offspring.append(child1)
            if len(offspring) < offspring_count:
                offspring.append(child2)
        
        return offspring[:offspring_count]
