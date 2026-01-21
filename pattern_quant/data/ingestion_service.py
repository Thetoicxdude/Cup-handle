"""Data Ingestion Service - 數據攝取服務

This module implements the data ingestion service that fetches stock data
from configured data sources and stores it in the database.

Requirements: 6.2, 6.3, 6.4
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Any

from .data_source import DataSource

logger = logging.getLogger(__name__)


class DatabaseConnection(Protocol):
    """資料庫連線協議
    
    定義資料庫連線必須實作的方法。
    """
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """執行 SQL 查詢"""
        ...
    
    def executemany(self, query: str, params_list: List[tuple]) -> Any:
        """批次執行 SQL 查詢"""
        ...
    
    def commit(self) -> None:
        """提交交易"""
        ...


class DataIngestionService:
    """數據攝取服務
    
    負責從數據源抓取股票數據並存入資料庫。
    實作錯誤處理與重試邏輯。
    
    Attributes:
        data_source: 數據源實例
        db_connection: 資料庫連線
        fetch_interval_seconds: 抓取間隔（秒）
        max_retries: 最大重試次數
        retry_delay_seconds: 重試延遲（秒）
    """
    
    def __init__(
        self,
        data_source: DataSource,
        db_connection: Optional[DatabaseConnection] = None,
        fetch_interval_seconds: int = 60,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0
    ):
        """初始化數據攝取服務
        
        Args:
            data_source: 數據源實例
            db_connection: 資料庫連線（可選，用於存儲數據）
            fetch_interval_seconds: 抓取間隔（秒）
            max_retries: 最大重試次數
            retry_delay_seconds: 重試延遲（秒）
        """
        self.data_source = data_source
        self.db_connection = db_connection
        self.fetch_interval_seconds = fetch_interval_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._pending_data: List[tuple] = []  # 本地佇列，用於暫存失敗的數據

    def ingest_watchlist(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, str]:
        """抓取監控名單的數據
        
        對監控名單中的每支股票抓取 OHLCV 數據。
        實作錯誤處理，失敗的股票會在下一週期重試。
        
        Args:
            symbols: 股票代碼列表
            start_date: 開始日期（預設為今天）
            end_date: 結束日期（預設為今天）
            
        Returns:
            結果字典 {symbol: "success" | "failure: <reason>"}
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date
        
        results: Dict[str, str] = {}
        
        for symbol in symbols:
            try:
                data = self._fetch_with_retry(symbol, start_date, end_date)
                
                if data:
                    # 嘗試存入資料庫
                    if self.db_connection is not None:
                        store_success = self.store_ohlcv(symbol, data)
                        if store_success:
                            results[symbol] = "success"
                        else:
                            results[symbol] = "failure: database storage failed"
                    else:
                        # 無資料庫連線時，僅記錄成功抓取
                        results[symbol] = "success"
                        logger.info(f"Fetched {len(data)} records for {symbol} (no DB storage)")
                else:
                    results[symbol] = "failure: no data returned"
                    
            except Exception as e:
                error_msg = f"failure: {str(e)}"
                results[symbol] = error_msg
                logger.error(f"Failed to ingest data for {symbol}: {e}")
        
        return results
    
    def _fetch_with_retry(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[dict]:
        """帶重試邏輯的數據抓取
        
        Args:
            symbol: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            OHLCV 數據列表
            
        Raises:
            Exception: 重試次數用盡後仍失敗
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                data = self.data_source.fetch_ohlcv(symbol, start_date, end_date)
                return data
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{self.max_retries} failed for {symbol}: {e}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds)
        
        # 所有重試都失敗
        logger.error(f"All {self.max_retries} fetch attempts failed for {symbol}")
        raise last_exception if last_exception else Exception("Unknown fetch error")

    def store_ohlcv(self, symbol: str, data: List[dict]) -> bool:
        """存入資料庫
        
        將 OHLCV 數據存入 TimescaleDB。
        實作重試邏輯，失敗後暫存至本地佇列。
        
        Args:
            symbol: 股票代碼
            data: OHLCV 數據列表
            
        Returns:
            True 如果存儲成功，False 如果失敗
        """
        if self.db_connection is None:
            logger.warning("No database connection configured, skipping storage")
            return False
        
        if not data:
            logger.warning(f"No data to store for {symbol}")
            return True  # 空數據視為成功
        
        # 準備批次插入的數據
        insert_query = """
            INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, symbol) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        
        params_list = [
            (
                record.get('time'),
                symbol,
                record.get('open'),
                record.get('high'),
                record.get('low'),
                record.get('close'),
                record.get('volume')
            )
            for record in data
        ]
        
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                self.db_connection.executemany(insert_query, params_list)
                self.db_connection.commit()
                logger.info(f"Successfully stored {len(data)} records for {symbol}")
                return True
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Store attempt {attempt + 1}/{self.max_retries} failed for {symbol}: {e}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds)
        
        # 所有重試都失敗，暫存至本地佇列
        logger.error(
            f"All {self.max_retries} store attempts failed for {symbol}, "
            f"queuing {len(data)} records for later retry"
        )
        self._pending_data.extend(params_list)
        self._trigger_storage_alert(symbol, last_exception)
        
        return False
    
    def _trigger_storage_alert(self, symbol: str, exception: Optional[Exception]) -> None:
        """觸發存儲失敗告警
        
        Args:
            symbol: 股票代碼
            exception: 異常物件
        """
        # 記錄嚴重錯誤，實際系統可整合告警服務（如 PagerDuty、Slack）
        logger.critical(
            f"ALERT: Database storage failed for {symbol}. "
            f"Error: {exception}. "
            f"Pending records: {len(self._pending_data)}"
        )
    
    def retry_pending_data(self) -> int:
        """重試暫存的數據
        
        嘗試將本地佇列中的數據存入資料庫。
        
        Returns:
            成功存儲的記錄數
        """
        if not self._pending_data or self.db_connection is None:
            return 0
        
        insert_query = """
            INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, symbol) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        
        try:
            self.db_connection.executemany(insert_query, self._pending_data)
            self.db_connection.commit()
            stored_count = len(self._pending_data)
            self._pending_data.clear()
            logger.info(f"Successfully stored {stored_count} pending records")
            return stored_count
        except Exception as e:
            logger.error(f"Failed to store pending data: {e}")
            return 0
    
    @property
    def pending_count(self) -> int:
        """返回待處理的記錄數"""
        return len(self._pending_data)
