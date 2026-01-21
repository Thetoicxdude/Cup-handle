"""
適應度評估器 (Fitness Evaluator)

負責評估個體的適應度分數，整合回測引擎進行策略評估。
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable, Any

from .models import (
    Genome,
    Individual,
)


class FitnessObjective(Enum):
    """適應度目標函數
    
    定義不同的優化目標，用於計算適應度分數。
    
    Attributes:
        SHARPE_RATIO: 夏普比率，風險調整後收益
        SORTINO_RATIO: 索提諾比率，下行風險調整後收益
        NET_PROFIT: 淨利潤，最大化總收益
        MIN_MAX_DRAWDOWN: 最小化最大回撤，防禦型策略
    """
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    NET_PROFIT = "net_profit"
    MIN_MAX_DRAWDOWN = "min_max_drawdown"


@dataclass
class FitnessResult:
    """適應度評估結果
    
    包含適應度分數及相關績效指標。
    
    Attributes:
        fitness_score: 適應度分數（根據目標函數計算）
        total_trades: 總交易次數
        win_rate: 勝率 (0-1)
        total_return: 總報酬率
        max_drawdown: 最大回撤
        sharpe_ratio: 夏普比率
        sortino_ratio: 索提諾比率
    """
    fitness_score: float
    total_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float


class FitnessEvaluator:
    """適應度評估器
    
    負責評估個體的適應度分數，整合回測引擎進行策略評估。
    
    Attributes:
        objective: 適應度目標函數
        min_trades_threshold: 最低交易次數閾值
        risk_free_rate: 無風險利率（年化）
        trading_days_per_year: 每年交易日數
    """
    
    def __init__(
        self,
        objective: FitnessObjective = FitnessObjective.SHARPE_RATIO,
        min_trades_threshold: int = 10,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        backtest_engine: Optional[Any] = None,
    ):
        """初始化適應度評估器
        
        Args:
            objective: 適應度目標函數，預設為 SHARPE_RATIO
            min_trades_threshold: 最低交易次數閾值，預設為 10
            risk_free_rate: 無風險利率（年化），預設為 0.02 (2%)
            trading_days_per_year: 每年交易日數，預設為 252
            backtest_engine: 回測引擎實例（可選）
        """
        self.objective = objective
        self.min_trades_threshold = min_trades_threshold
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.backtest_engine = backtest_engine
    
    def evaluate(
        self,
        individual: Individual,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> FitnessResult:
        """評估單一個體的適應度
        
        執行回測並計算適應度分數。若交易次數低於閾值，
        則適應度分數為零。
        
        Args:
            individual: 要評估的個體
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            
        Returns:
            適應度評估結果
        """
        # 歸一化權重
        normalized_genome = individual.genome.normalize_weights()
        
        # 驗證約束條件
        if not normalized_genome.validate_constraints():
            # 約束違反，返回零適應度
            return FitnessResult(
                fitness_score=0.0,
                total_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
            )
        
        # 執行回測（簡化版本，實際應整合回測引擎）
        returns = self._simulate_returns(normalized_genome, prices, highs, lows, volumes)
        
        # 計算績效指標
        total_trades = len([r for r in returns if r != 0.0])
        
        # 低交易次數懲罰
        if total_trades < self.min_trades_threshold:
            return FitnessResult(
                fitness_score=0.0,
                total_trades=total_trades,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
            )
        
        # 計算各項指標
        win_rate = self._calculate_win_rate(returns)
        total_return = self._calculate_total_return(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # 根據目標函數計算適應度分數
        fitness_score = self._calculate_fitness_score(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_return=total_return,
            max_drawdown=max_drawdown,
        )
        
        # 確保適應度分數為有限數值
        if not math.isfinite(fitness_score):
            fitness_score = 0.0
        
        return FitnessResult(
            fitness_score=fitness_score,
            total_trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
        )
    
    def evaluate_population(
        self,
        population: List[Individual],
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> List[Individual]:
        """評估整個種群的適應度
        
        對種群中的每個個體進行適應度評估，並更新其 fitness 屬性。
        
        Args:
            population: 種群列表
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            
        Returns:
            更新適應度後的種群列表
            
        Raises:
            ValueError: 若種群為空
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        evaluated_population = []
        
        for individual in population:
            result = self.evaluate(individual, prices, highs, lows, volumes)
            
            # 建立新個體並更新適應度
            evaluated_individual = Individual(
                genome=individual.genome,
                fitness=result.fitness_score,
                generation=individual.generation,
            )
            evaluated_population.append(evaluated_individual)
        
        return evaluated_population
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """計算夏普比率
        
        夏普比率 = (平均報酬 - 無風險利率) / 報酬標準差
        
        Args:
            returns: 報酬率序列
            
        Returns:
            夏普比率，若無法計算則返回 0.0
        """
        if len(returns) < 2:
            return 0.0
        
        # 過濾非零報酬
        non_zero_returns = [r for r in returns if r != 0.0]
        if len(non_zero_returns) < 2:
            return 0.0
        
        # 計算平均報酬
        mean_return = sum(non_zero_returns) / len(non_zero_returns)
        
        # 計算標準差
        variance = sum((r - mean_return) ** 2 for r in non_zero_returns) / len(non_zero_returns)
        std_return = math.sqrt(variance)
        
        if std_return == 0 or math.isclose(std_return, 0.0, abs_tol=1e-10):
            return 0.0
        
        # 計算日化無風險利率
        daily_risk_free = self.risk_free_rate / self.trading_days_per_year
        
        # 計算夏普比率（年化）
        sharpe = (mean_return - daily_risk_free) / std_return
        sharpe_annualized = sharpe * math.sqrt(self.trading_days_per_year)
        
        # 確保返回有限數值
        if not math.isfinite(sharpe_annualized):
            return 0.0
        
        return sharpe_annualized
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """計算索提諾比率
        
        索提諾比率 = (平均報酬 - 無風險利率) / 下行標準差
        只考慮負報酬的波動性。
        
        Args:
            returns: 報酬率序列
            
        Returns:
            索提諾比率，若無法計算則返回 0.0
        """
        if len(returns) < 2:
            return 0.0
        
        # 過濾非零報酬
        non_zero_returns = [r for r in returns if r != 0.0]
        if len(non_zero_returns) < 2:
            return 0.0
        
        # 計算平均報酬
        mean_return = sum(non_zero_returns) / len(non_zero_returns)
        
        # 計算日化無風險利率
        daily_risk_free = self.risk_free_rate / self.trading_days_per_year
        
        # 計算下行標準差（只考慮低於目標的報酬）
        target = daily_risk_free
        downside_returns = [min(0, r - target) ** 2 for r in non_zero_returns]
        
        if not downside_returns:
            return 0.0
        
        downside_variance = sum(downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance)
        
        if downside_std == 0 or math.isclose(downside_std, 0.0, abs_tol=1e-10):
            # 沒有下行風險，返回高值（但有限）
            if mean_return > daily_risk_free:
                return 10.0  # 設定上限
            return 0.0
        
        # 計算索提諾比率（年化）
        sortino = (mean_return - daily_risk_free) / downside_std
        sortino_annualized = sortino * math.sqrt(self.trading_days_per_year)
        
        # 確保返回有限數值
        if not math.isfinite(sortino_annualized):
            return 0.0
        
        return sortino_annualized
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """計算勝率
        
        Args:
            returns: 報酬率序列
            
        Returns:
            勝率 (0-1)
        """
        non_zero_returns = [r for r in returns if r != 0.0]
        if not non_zero_returns:
            return 0.0
        
        wins = sum(1 for r in non_zero_returns if r > 0)
        return wins / len(non_zero_returns)
    
    def _calculate_total_return(self, returns: List[float]) -> float:
        """計算總報酬率
        
        使用複利計算總報酬率。
        
        Args:
            returns: 報酬率序列
            
        Returns:
            總報酬率
        """
        if not returns:
            return 0.0
        
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
        
        return cumulative - 1.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """計算最大回撤
        
        Args:
            returns: 報酬率序列
            
        Returns:
            最大回撤（正值表示損失）
        """
        if not returns:
            return 0.0
        
        # 計算累積淨值曲線
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        
        for r in returns:
            cumulative *= (1 + r)
            if cumulative > peak:
                peak = cumulative
            
            drawdown = (peak - cumulative) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_fitness_score(
        self,
        sharpe_ratio: float,
        sortino_ratio: float,
        total_return: float,
        max_drawdown: float,
    ) -> float:
        """根據目標函數計算適應度分數
        
        Args:
            sharpe_ratio: 夏普比率
            sortino_ratio: 索提諾比率
            total_return: 總報酬率
            max_drawdown: 最大回撤
            
        Returns:
            適應度分數
        """
        if self.objective == FitnessObjective.SHARPE_RATIO:
            return sharpe_ratio
        
        elif self.objective == FitnessObjective.SORTINO_RATIO:
            return sortino_ratio
        
        elif self.objective == FitnessObjective.NET_PROFIT:
            return total_return
        
        elif self.objective == FitnessObjective.MIN_MAX_DRAWDOWN:
            # 最小化回撤，轉換為最大化問題
            # 使用 1 - max_drawdown 作為適應度
            # 回撤越小，適應度越高
            return 1.0 - max_drawdown
        
        else:
            # 預設使用夏普比率
            return sharpe_ratio
    
    def _simulate_returns(
        self,
        genome: Genome,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> List[float]:
        """模擬交易報酬
        
        根據基因組參數模擬交易策略的報酬序列。
        這是一個簡化版本，實際應整合完整的回測引擎。
        
        Args:
            genome: 基因組（已歸一化）
            prices: 收盤價序列
            highs: 最高價序列（可選）
            lows: 最低價序列（可選）
            volumes: 成交量序列（可選）
            
        Returns:
            報酬率序列
        """
        if len(prices) < 2:
            return []
        
        # 計算價格變化率
        price_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                price_returns.append(ret)
            else:
                price_returns.append(0.0)
        
        # 簡化的信號生成邏輯
        # 實際應使用完整的策略引擎
        returns = []
        position = 0  # 0: 無持倉, 1: 多頭
        
        # 使用基因組參數影響交易決策
        score_threshold = genome.factor_weights.score_threshold
        trend_threshold = genome.dual_engine.trend_threshold
        
        # 簡化的 RSI 計算
        rsi_period = genome.micro_indicators.rsi_period
        rsi_overbought = genome.micro_indicators.rsi_overbought
        rsi_oversold = genome.micro_indicators.rsi_oversold
        
        for i in range(len(price_returns)):
            # 簡化的交易邏輯
            if i < rsi_period:
                returns.append(0.0)
                continue
            
            # 計算簡化的動量指標
            recent_returns = price_returns[max(0, i - rsi_period):i]
            if recent_returns:
                avg_return = sum(recent_returns) / len(recent_returns)
            else:
                avg_return = 0.0
            
            # 簡化的信號生成
            # 使用 score_threshold 作為進場閾值的參考
            entry_threshold = (score_threshold - 50) / 100  # 轉換為 0-0.4 範圍
            
            if position == 0:
                # 無持倉，檢查進場條件
                if avg_return > entry_threshold * 0.01:
                    position = 1
                    returns.append(0.0)
                else:
                    returns.append(0.0)
            else:
                # 有持倉，計算報酬
                returns.append(price_returns[i])
                
                # 檢查出場條件
                if avg_return < -entry_threshold * 0.01:
                    position = 0
        
        return returns
