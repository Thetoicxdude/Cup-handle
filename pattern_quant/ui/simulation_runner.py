
import threading
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback

from pattern_quant.core.backtest_engine import RealDataBacktestEngine, StrategyParameters
from pattern_quant.db.state_manager import get_state_manager, SimulationInfo, SimulationState

class SimulationRunner:
    """Background runner for live simulations.
    
    This class is designed to run as a singleton using st.cache_resource.
    It manages a background thread that continuously updates active simulations,
    ensuring they continue running even if the browser tab is closed.
    """
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._is_running = False
        self._lock = threading.Lock()
        self.state_manager = get_state_manager()
        
        # Cache for backtest engines to avoid recreating them constantly
        # Key: simulation_id, Value: RealDataBacktestEngine instance
        self._engines: Dict[int, RealDataBacktestEngine] = {}
        
    def start(self):
        """Start the background simulation loop."""
        with self._lock:
            if not self._is_running:
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()
                self._is_running = True
                print("Background SimulationRunner started.")

    def stop(self):
        """Stop the background simulation loop."""
        with self._lock:
            if self._is_running:
                self._stop_event.set()
                self._thread.join(timeout=5.0)
                self._is_running = False
                print("Background SimulationRunner stopped.")

    def is_running(self) -> bool:
        """Check if the runner is active."""
        return self._is_running
        
    def _run_loop(self):
        """Main loop that iterates through active simulations."""
        print("[SimRunner] Background loop started")
        loop_count = 0
        while not self._stop_event.is_set():
            loop_count += 1
            try:
                active_count = self._process_active_simulations()
                if loop_count % 6 == 0:  # Log every 30 seconds (6 * 5s)
                    print(f"[SimRunner] Loop #{loop_count}, processed {active_count} active simulations")
            except Exception as e:
                print(f"Error in SimulationRunner loop: {e}")
                traceback.print_exc()
            
            # Wait 5 seconds between cycles - heartbeat updates happen each cycle
            self._wait_with_stop_check(5.0)

    def _wait_with_stop_check(self, seconds: float):
        """Wait for specified seconds or until stop event is set."""
        self._stop_event.wait(timeout=seconds)

    def _process_active_simulations(self) -> int:
        """Fetch and update all active simulations."""
        active_sims = self.state_manager.get_active_simulations()
        
        for sim in active_sims:
            if self._stop_event.is_set():
                break
                
            try:
                self._update_single_simulation(sim)
            except Exception as e:
                print(f"Error updating simulation {sim.id}: {e}")
                traceback.print_exc()
        
        return len(active_sims)

    def _update_single_simulation(self, sim: SimulationInfo):
        """Update logic for a single simulation."""
        import json
        import yfinance as yf
        
        print(f"[SimRunner] Processing simulation {sim.id}: {sim.name}")
        
        # 1. ALWAYS update heartbeat first to keep the simulation "alive"
        self.state_manager.update_heartbeat(sim.id)
        
        # 2. Load latest state
        state = self.state_manager.load_state(sim.id)
        if not state:
            print(f"[SimRunner] No state found for simulation {sim.id}")
            return
        
        # 3. Check if it's time for a FULL update (fetch data, etc.) based on update_interval
        # Use the state's updated_at timestamp, not the heartbeat (which we just updated)
        last_state_update = state.updated_at
        do_full_update = True
        
        # Force immediate update if no signals yet (first run)
        has_signals = state.active_signals and len(state.active_signals) > 0
        
        if last_state_update and has_signals:
            if isinstance(last_state_update, str):
                try:
                    last_state_update = datetime.strptime(last_state_update, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        last_state_update = datetime.fromisoformat(last_state_update)
                    except:
                        last_state_update = datetime.min
            
            elapsed = (datetime.now() - last_state_update).total_seconds()
            interval = sim.update_interval if sim.update_interval else 60
            
            if elapsed < interval:
                print(f"[SimRunner] Heartbeat updated for {sim.id}, but skipping full update ({elapsed:.0f}s < {interval}s)")
                do_full_update = False
        else:
            # First run or no signals yet - do full update immediately
            print(f"[SimRunner] First update for simulation {sim.id} - fetching data immediately")
        
        if not do_full_update:
            return
        
        print(f"[SimRunner] Updating simulation {sim.id}...")
        
        # 4. Fetch current prices for all symbols
        current_prices = {}
        price_details = {}  # For storing detailed price info
        for symbol in sim.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")  # Get 5 days for trend analysis
                if not hist.empty:
                    close = float(hist['Close'].iloc[-1])
                    current_prices[symbol] = close
                    
                    # Calculate price change
                    prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else close
                    change = close - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0
                    
                    # Get high/low
                    high = float(hist['High'].iloc[-1])
                    low = float(hist['Low'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                    
                    price_details[symbol] = {
                        'price': close,
                        'change': change,
                        'change_pct': change_pct,
                        'high': high,
                        'low': low,
                        'volume': volume,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    print(f"[SimRunner] {symbol}: ${close:.2f} ({change_pct:+.2f}%)")
            except Exception as e:
                print(f"[SimRunner] Failed to fetch {symbol}: {e}")
        
        if not current_prices:
            print(f"[SimRunner] No price data fetched, skipping update")
            return
        
        # 5. Update positions with current prices
        capital = state.capital
        positions = state.positions.copy() if state.positions else {}
        trades = state.trades.copy() if state.trades else []
        logs = state.logs.copy() if state.logs else []
        
        # Update current prices in positions
        for symbol, pos in positions.items():
            if isinstance(pos, dict) and symbol in current_prices:
                pos['current_price'] = current_prices[symbol]
        
        # 6. Analyze trading decisions and record reasons
        decision_logs = []
        for symbol in sim.symbols:
            if symbol not in current_prices:
                continue
            
            price = current_prices[symbol]
            has_position = symbol in positions and positions[symbol]
            
            # Get parameters (simplified - in full implementation would parse from sim parameters)
            # For now just explain what we're checking
            if has_position:
                pos = positions[symbol]
                entry_price = pos.get('entry_price', 0) if isinstance(pos, dict) else 0
                stop_loss = pos.get('stop_loss_price', 0) if isinstance(pos, dict) else 0
                pnl_pct = ((price - entry_price) / entry_price * 100) if entry_price else 0
                
                if price <= stop_loss:
                    decision_logs.append(f"⚠️ {symbol}: 價格 ${price:.2f} 觸及止損 ${stop_loss:.2f}")
                elif pnl_pct >= 10:  # Example profit threshold
                    decision_logs.append(f"🎯 {symbol}: 獲利 {pnl_pct:.1f}%，接近止盈目標")
                else:
                    decision_logs.append(f"📊 {symbol}: 持倉中，當前損益 {pnl_pct:+.1f}%")
            else:
                # Not in position - analyze entry signals
                change_pct = price_details.get(symbol, {}).get('change_pct', 0)
                decision_logs.append(f"🔍 {symbol}: ${price:.2f} ({change_pct:+.1f}%) - 等待入場型態訊號")
        
        # Add log entries
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] 更新完成 - {len(current_prices)} 個標的"
        logs.append(log_entry)
        
        for decision in decision_logs:
            logs.append(f"[{timestamp}] {decision}")
        
        if len(logs) > 100:
            logs = logs[-100:]
        
        # 7. Store active signals with price details for UI display
        active_signals = []
        for symbol, details in price_details.items():
            active_signals.append({
                'symbol': symbol,
                'price': details['price'],
                'change': details['change'],
                'change_pct': details['change_pct'],
                'high': details['high'],
                'low': details['low'],
                'volume': details['volume'],
                'timestamp': details['timestamp'],
                'has_position': symbol in positions and positions[symbol] is not None
            })
        
        # 8. Save updated state
        self.state_manager.save_state(
            simulation_id=sim.id,
            capital=capital,
            positions=positions,
            trades=trades,
            logs=logs,
            active_signals=active_signals,
            evolution_history=state.evolution_history
        )
        
        # 9. Update heartbeat
        self.state_manager.update_heartbeat(sim.id)
        print(f"[SimRunner] Simulation {sim.id} updated successfully")

    def _get_or_create_engine(self, sim: SimulationInfo) -> RealDataBacktestEngine:
        """Retrieve cached engine or create a new one."""
        if sim.id in self._engines:
            return self._engines[sim.id]
        
        # Create new engine
        engine = RealDataBacktestEngine()
        self._engines[sim.id] = engine
        return engine

@st.cache_resource
def get_simulation_runner() -> SimulationRunner:
    """Get the singleton SimulationRunner instance."""
    runner = SimulationRunner()
    runner.start()
    return runner
