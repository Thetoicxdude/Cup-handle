"""Data Source Abstract Interface - 數據源抽象介面

This module defines the abstract interface for data sources that provide
stock market data to the system.

Requirements: 6.1
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List


class DataSource(ABC):
    """數據源抽象介面
    
    定義數據源必須實作的方法，用於抓取股票 OHLCV 數據與大盤指數。
    具體實作可以是券商 API、Yahoo Finance、或其他數據提供商。
    """
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[dict]:
        """抓取 OHLCV 數據
        
        Args:
            symbol: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            OHLCV 數據列表，每筆數據為 dict 格式：
            {
                'time': datetime,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int
            }
            
        Raises:
            ConnectionError: 連線失敗
            ValueError: 無效的參數
        """
        pass
    
    @abstractmethod
    def fetch_market_index(self, index_symbol: str) -> float:
        """抓取大盤指數當日漲跌幅
        
        Args:
            index_symbol: 指數代碼（如 ^TWII 台灣加權指數）
            
        Returns:
            當日漲跌幅（百分比，如 -2.5 表示下跌 2.5%）
            
        Raises:
            ConnectionError: 連線失敗
            ValueError: 無效的指數代碼
        """
        pass
