"""State Manager for Dashboard Persistence

This module provides the StateManager class for managing persistent dashboard state
using SQLite. Enables multi-user state sharing and simulation persistence.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from pattern_quant.db.state_schema import get_state_schema_statements

logger = logging.getLogger(__name__)


@dataclass
class SimulationInfo:
    """模擬資訊摘要"""
    id: int
    name: str
    status: str
    symbols: List[str]
    update_interval: int
    initial_capital: float
    created_at: str
    last_heartbeat: str
    is_alive: bool  # 根據心跳判斷是否仍在運行


@dataclass
class SimulationState:
    """模擬狀態快照"""
    simulation_id: int
    capital: float
    positions: Dict[str, Any]
    trades: List[Dict[str, Any]]
    logs: List[str]
    active_signals: List[Dict[str, Any]]
    evolution_history: List[Dict[str, Any]]
    updated_at: str


class StateManager:
    """Dashboard State Manager
    
    使用 SQLite 管理面板狀態的持久化，支援：
    - 模擬狀態的儲存與載入
    - 多使用者狀態共享
    - 心跳機制偵測活躍模擬
    - 舊狀態清理
    
    Attributes:
        db_path: SQLite 資料庫檔案路徑
        heartbeat_timeout: 心跳逾時秒數 (預設 120 秒)
    """
    
    def __init__(self, db_path: Optional[str] = None, heartbeat_timeout: int = 120):
        """初始化 StateManager
        
        Args:
            db_path: SQLite 資料庫路徑，預設為 data/state.db
            heartbeat_timeout: 判斷模擬是否存活的心跳逾時秒數
        """
        if db_path is None:
            # 使用專案目錄下的 data 資料夾
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(project_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "state.db")
        
        self.db_path = db_path
        self.heartbeat_timeout = heartbeat_timeout
        self._initialize_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """取得資料庫連線"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _initialize_schema(self) -> bool:
        """初始化資料庫結構"""
        try:
            with self._get_connection() as conn:
                for statement in get_state_schema_statements():
                    conn.executescript(statement)
                conn.commit()
            logger.info(f"State database initialized at {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize state schema: {e}")
            return False
    
    # ==================== 模擬管理 ====================
    
    def create_simulation(
        self,
        name: str,
        symbols: List[str],
        parameters: Dict[str, Any],
        update_interval: int = 300,
        initial_capital: float = 1000000.0
    ) -> Optional[int]:
        """建立新的模擬記錄
        
        Args:
            name: 模擬名稱
            symbols: 監控標的列表
            parameters: 策略參數
            update_interval: 更新間隔（秒）
            initial_capital: 初始資金
            
        Returns:
            模擬 ID，失敗則返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO live_simulations 
                    (name, symbols, parameters, update_interval, initial_capital, status)
                    VALUES (?, ?, ?, ?, ?, 'running')
                    """,
                    (
                        name,
                        json.dumps(symbols, ensure_ascii=False),
                        json.dumps(parameters, ensure_ascii=False, default=str),
                        update_interval,
                        initial_capital
                    )
                )
                conn.commit()
                sim_id = cursor.lastrowid
                
                # 建立初始狀態記錄
                self._create_initial_state(conn, sim_id, initial_capital)
                
                logger.info(f"Created simulation {sim_id}: {name}")
                return sim_id
        except Exception as e:
            logger.error(f"Failed to create simulation: {e}")
            return None
    
    def _create_initial_state(self, conn: sqlite3.Connection, simulation_id: int, capital: float):
        """建立初始狀態記錄"""
        conn.execute(
            """
            INSERT INTO simulation_state 
            (simulation_id, capital, positions, trades, logs, active_signals, evolution_history)
            VALUES (?, ?, '{}', '[]', '[]', '[]', '[]')
            """,
            (simulation_id, capital)
        )
        conn.commit()
    
    def get_simulation(self, simulation_id: int) -> Optional[SimulationInfo]:
        """取得模擬資訊
        
        Args:
            simulation_id: 模擬 ID
            
        Returns:
            SimulationInfo 或 None
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM live_simulations WHERE id = ?",
                    (simulation_id,)
                ).fetchone()
                
                if row:
                    return self._row_to_simulation_info(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get simulation {simulation_id}: {e}")
            return None
    
    def get_active_simulations(self) -> List[SimulationInfo]:
        """取得所有進行中的模擬（狀態為 running 且心跳正常）
        
        Returns:
            活躍模擬列表
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM live_simulations 
                    WHERE status = 'running'
                    ORDER BY created_at DESC
                    """
                ).fetchall()
                
                return [self._row_to_simulation_info(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get active simulations: {e}")
            return []
    
    def get_all_simulations(self) -> List[SimulationInfo]:
        """取得所有模擬記錄
        
        Returns:
            所有模擬列表
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM live_simulations ORDER BY created_at DESC"
                ).fetchall()
                
                return [self._row_to_simulation_info(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all simulations: {e}")
            return []
    
    def _row_to_simulation_info(self, row: sqlite3.Row) -> SimulationInfo:
        """將資料庫行轉換為 SimulationInfo"""
        # 檢查心跳是否逾時
        last_heartbeat = datetime.strptime(row['last_heartbeat'], '%Y-%m-%d %H:%M:%S')
        is_alive = (datetime.now() - last_heartbeat).total_seconds() < self.heartbeat_timeout
        
        return SimulationInfo(
            id=row['id'],
            name=row['name'],
            status=row['status'],
            symbols=json.loads(row['symbols']),
            update_interval=row['update_interval'],
            initial_capital=row['initial_capital'],
            created_at=row['created_at'],
            last_heartbeat=row['last_heartbeat'],
            is_alive=is_alive and row['status'] == 'running'
        )
    
    def update_heartbeat(self, simulation_id: int) -> bool:
        """更新模擬心跳時間戳
        
        Args:
            simulation_id: 模擬 ID
            
        Returns:
            是否成功
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE live_simulations 
                    SET last_heartbeat = datetime('now', 'localtime')
                    WHERE id = ?
                    """,
                    (simulation_id,)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {simulation_id}: {e}")
            return False
    
    def update_simulation_status(self, simulation_id: int, status: str) -> bool:
        """更新模擬狀態
        
        Args:
            simulation_id: 模擬 ID
            status: 新狀態 (running, paused, stopped)
            
        Returns:
            是否成功
        """
        try:
            with self._get_connection() as conn:
                if status == 'stopped':
                    conn.execute(
                        """
                        UPDATE live_simulations 
                        SET status = ?, stopped_at = datetime('now', 'localtime')
                        WHERE id = ?
                        """,
                        (status, simulation_id)
                    )
                else:
                    conn.execute(
                        """
                        UPDATE live_simulations 
                        SET status = ?
                        WHERE id = ?
                        """,
                        (status, simulation_id)
                    )
                conn.commit()
                logger.info(f"Updated simulation {simulation_id} status to {status}")
                return True
        except Exception as e:
            logger.error(f"Failed to update simulation status {simulation_id}: {e}")
            return False
    
    def stop_simulation(self, simulation_id: int) -> bool:
        """停止模擬
        
        Args:
            simulation_id: 模擬 ID
            
        Returns:
            是否成功
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE live_simulations 
                    SET status = 'stopped', stopped_at = datetime('now', 'localtime')
                    WHERE id = ?
                    """,
                    (simulation_id,)
                )
                conn.commit()
                logger.info(f"Stopped simulation {simulation_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to stop simulation {simulation_id}: {e}")
            return False
    
    def delete_simulation(self, simulation_id: int) -> bool:
        """刪除模擬及其狀態記錄
        
        Args:
            simulation_id: 模擬 ID
            
        Returns:
            是否成功
        """
        try:
            with self._get_connection() as conn:
                # CASCADE 會自動刪除相關的 simulation_state 記錄
                conn.execute(
                    "DELETE FROM live_simulations WHERE id = ?",
                    (simulation_id,)
                )
                conn.commit()
                logger.info(f"Deleted simulation {simulation_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete simulation {simulation_id}: {e}")
            return False
    
    def delete_old_simulations(self, days: int = 7) -> int:
        """刪除指定天數前的舊模擬
        
        Args:
            days: 保留最近幾天的記錄
            
        Returns:
            刪除的記錄數
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM live_simulations 
                    WHERE status = 'stopped' 
                    AND stopped_at < datetime('now', 'localtime', ?)
                    """,
                    (f'-{days} days',)
                )
                conn.commit()
                count = cursor.rowcount
                logger.info(f"Deleted {count} old simulations")
                return count
        except Exception as e:
            logger.error(f"Failed to delete old simulations: {e}")
            return 0
    
    # ==================== 狀態管理 ====================
    
    def save_state(
        self,
        simulation_id: int,
        capital: float,
        positions: Dict[str, Any],
        trades: List[Dict[str, Any]],
        logs: List[str],
        active_signals: Optional[List[Dict[str, Any]]] = None,
        evolution_history: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """儲存模擬狀態
        
        Args:
            simulation_id: 模擬 ID
            capital: 當前資金
            positions: 持倉資訊
            trades: 交易記錄
            logs: 日誌
            active_signals: 活躍訊號
            evolution_history: 演化歷史
            
        Returns:
            是否成功
        """
        try:
            with self._get_connection() as conn:
                # 更新或插入狀態記錄
                conn.execute(
                    """
                    INSERT OR REPLACE INTO simulation_state 
                    (id, simulation_id, capital, positions, trades, logs, active_signals, evolution_history, updated_at)
                    SELECT 
                        COALESCE(
                            (SELECT id FROM simulation_state WHERE simulation_id = ?),
                            NULL
                        ),
                        ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime')
                    """,
                    (
                        simulation_id,
                        simulation_id,
                        capital,
                        json.dumps(positions, ensure_ascii=False, default=str),
                        json.dumps(trades, ensure_ascii=False, default=str),
                        json.dumps(logs, ensure_ascii=False),
                        json.dumps(active_signals or [], ensure_ascii=False, default=str),
                        json.dumps(evolution_history or [], ensure_ascii=False, default=str)
                    )
                )
                
                # 同時更新心跳
                conn.execute(
                    """
                    UPDATE live_simulations 
                    SET last_heartbeat = datetime('now', 'localtime')
                    WHERE id = ?
                    """,
                    (simulation_id,)
                )
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save state for simulation {simulation_id}: {e}")
            return False
    
    def load_state(self, simulation_id: int) -> Optional[SimulationState]:
        """載入模擬狀態
        
        Args:
            simulation_id: 模擬 ID
            
        Returns:
            SimulationState 或 None
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM simulation_state 
                    WHERE simulation_id = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (simulation_id,)
                ).fetchone()
                
                if row:
                    return SimulationState(
                        simulation_id=row['simulation_id'],
                        capital=row['capital'],
                        positions=json.loads(row['positions'] or '{}'),
                        trades=json.loads(row['trades'] or '[]'),
                        logs=json.loads(row['logs'] or '[]'),
                        active_signals=json.loads(row['active_signals'] or '[]'),
                        evolution_history=json.loads(row['evolution_history'] or '[]'),
                        updated_at=row['updated_at']
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to load state for simulation {simulation_id}: {e}")
            return None
    
    # ==================== 設定管理 ====================
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """取得應用程式設定
        
        Args:
            key: 設定鍵
            default: 預設值
            
        Returns:
            設定值或預設值
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM app_settings WHERE key = ?",
                    (key,)
                ).fetchone()
                
                if row:
                    return json.loads(row['value'])
                return default
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return default
    
    def set_setting(self, key: str, value: Any) -> bool:
        """設定應用程式設定
        
        Args:
            key: 設定鍵
            value: 設定值
            
        Returns:
            是否成功
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO app_settings (key, value, updated_at)
                    VALUES (?, ?, datetime('now', 'localtime'))
                    """,
                    (key, json.dumps(value, ensure_ascii=False, default=str))
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to set setting {key}: {e}")
            return False
    
    # ==================== 工具方法 ====================
    
    def get_db_size(self) -> int:
        """取得資料庫檔案大小（bytes）"""
        try:
            return os.path.getsize(self.db_path)
        except:
            return 0
    
    def vacuum(self) -> bool:
        """壓縮資料庫檔案以釋放空間"""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False


# 全域單例
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """取得 StateManager 單例"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
