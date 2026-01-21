"""
選擇算子 (Selection Operator)

負責從種群中選擇親代進行繁衍，實作競賽選擇與精英保留機制。
"""

import random
from typing import List, Optional

from .models import Individual


class SelectionOperator:
    """選擇算子
    
    實作競賽選擇 (Tournament Selection) 與精英保留 (Elitism) 機制。
    
    Attributes:
        tournament_size: 競賽選擇的參與者數量 (k)
        elitism_rate: 精英保留比例 (5%-20%)
    """
    
    def __init__(
        self,
        tournament_size: int = 3,
        elitism_rate: float = 0.1,
    ):
        """初始化選擇算子
        
        Args:
            tournament_size: 競賽選擇的參與者數量，預設為 3
            elitism_rate: 精英保留比例，預設為 0.1 (10%)
            
        Raises:
            ValueError: 若 tournament_size < 1
            ValueError: 若 elitism_rate 不在 [0.05, 0.20] 範圍內
        """
        if tournament_size < 1:
            raise ValueError(
                f"Tournament size must be at least 1, got {tournament_size}"
            )
        
        if elitism_rate < 0.05 or elitism_rate > 0.20:
            raise ValueError(
                f"Elitism rate must be between 0.05 and 0.20, got {elitism_rate}"
            )
        
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
    
    def tournament_select(
        self,
        population: List[Individual],
    ) -> Individual:
        """競賽選擇
        
        從種群中隨機選擇 k 個個體，返回其中適應度最高者。
        
        Args:
            population: 種群列表
            
        Returns:
            競賽勝出的個體（適應度最高者）
            
        Raises:
            ValueError: 若種群為空
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        # 確保競賽大小不超過種群大小
        actual_tournament_size = min(self.tournament_size, len(population))
        
        # 隨機選擇 k 個參與者
        participants = random.sample(population, actual_tournament_size)
        
        # 返回適應度最高的個體
        winner = max(participants, key=lambda ind: ind.fitness)
        
        return winner
    
    def select_parents(
        self,
        population: List[Individual],
        num_parents: int,
    ) -> List[Individual]:
        """選擇親代
        
        使用競賽選擇從種群中選擇指定數量的親代。
        
        Args:
            population: 種群列表
            num_parents: 需要選擇的親代數量
            
        Returns:
            選中的親代列表
            
        Raises:
            ValueError: 若種群為空
            ValueError: 若 num_parents < 1
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        if num_parents < 1:
            raise ValueError(
                f"Number of parents must be at least 1, got {num_parents}"
            )
        
        parents = []
        for _ in range(num_parents):
            parent = self.tournament_select(population)
            parents.append(parent)
        
        return parents
    
    def get_elite(
        self,
        population: List[Individual],
    ) -> List[Individual]:
        """獲取精英個體
        
        根據精英保留比例，返回種群中適應度最高的個體。
        精英個體將直接進入下一代，不經過交叉和突變。
        
        Args:
            population: 種群列表
            
        Returns:
            精英個體列表（深拷貝）
            
        Raises:
            ValueError: 若種群為空
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        # 計算精英數量（至少保留 1 個）
        num_elite = max(1, int(len(population) * self.elitism_rate))
        
        # 按適應度降序排序
        sorted_population = sorted(
            population,
            key=lambda ind: ind.fitness,
            reverse=True
        )
        
        # 返回前 num_elite 個個體的深拷貝
        elite = [ind.copy() for ind in sorted_population[:num_elite]]
        
        return elite
