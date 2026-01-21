"""Yahoo Finance Data Source - 真實股票數據源

This module provides real stock market data using yfinance API.
Supports fetching OHLCV data and market index information.

Requirements: 6.1
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from pattern_quant.data.data_source import DataSource

logger = logging.getLogger(__name__)


class YFinanceDataSource(DataSource):
    """Yahoo Finance 數據源
    
    使用 yfinance 套件抓取真實股票數據。
    支援美股、台股（需加 .TW 後綴）等多個市場。
    
    Attributes:
        cache: 簡單的記憶體快取，避免重複請求
        cache_ttl: 快取存活時間（秒）
    """
    
    def __init__(self, cache_ttl: int = 300):
        """初始化 Yahoo Finance 數據源
        
        Args:
            cache_ttl: 快取存活時間（秒），預設 5 分鐘
        """
        self.cache = {}
        self.cache_ttl = cache_ttl
        self._yf = None
    
    def _get_yfinance(self):
        """延遲載入 yfinance 模組"""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance 套件未安裝。請執行: pip install yfinance"
                )
        return self._yf
    
    def _get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """生成快取鍵值"""
        return f"{symbol}_{start_date.date()}_{end_date.date()}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """檢查快取是否有效"""
        if cache_key not in self.cache:
            return False
        cached_time, _ = self.cache[cache_key]
        return (datetime.now() - cached_time).total_seconds() < self.cache_ttl
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[dict]:
        """抓取 OHLCV 數據
        
        Args:
            symbol: 股票代碼（美股直接輸入如 AAPL，台股需加 .TW 如 2330.TW）
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            OHLCV 數據列表
            
        Raises:
            ConnectionError: 連線失敗
            ValueError: 無效的參數或找不到股票
        """
        yf = self._get_yfinance()
        
        # 檢查快取
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        if self._is_cache_valid(cache_key):
            logger.debug(f"使用快取數據: {symbol}")
            _, data = self.cache[cache_key]
            return data
        
        try:
            logger.info(f"從 Yahoo Finance 抓取 {symbol} 數據...")
            
            # 建立 Ticker 物件
            ticker = yf.Ticker(symbol)
            
            # 抓取歷史數據
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if df.empty:
                logger.warning(f"找不到 {symbol} 的數據")
                return []
            
            # 轉換為標準格式
            result = []
            for idx, row in df.iterrows():
                result.append({
                    'time': idx.to_pydatetime(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            # 存入快取
            self.cache[cache_key] = (datetime.now(), result)
            
            logger.info(f"成功抓取 {symbol}: {len(result)} 筆數據")
            return result
            
        except Exception as e:
            logger.error(f"抓取 {symbol} 數據失敗: {e}")
            raise ConnectionError(f"無法抓取 {symbol} 數據: {e}")
    
    def fetch_market_index(self, index_symbol: str) -> float:
        """抓取大盤指數當日漲跌幅
        
        Args:
            index_symbol: 指數代碼
                - ^GSPC: S&P 500
                - ^DJI: 道瓊工業指數
                - ^IXIC: 納斯達克
                - ^TWII: 台灣加權指數
                
        Returns:
            當日漲跌幅（百分比）
        """
        yf = self._get_yfinance()
        
        try:
            ticker = yf.Ticker(index_symbol)
            
            # 抓取最近 2 天數據計算漲跌幅
            df = ticker.history(period='2d')
            
            if len(df) < 2:
                # 如果只有一天數據，嘗試用 info
                info = ticker.info
                if 'regularMarketChangePercent' in info:
                    return float(info['regularMarketChangePercent'])
                return 0.0
            
            # 計算漲跌幅
            prev_close = df['Close'].iloc[-2]
            curr_close = df['Close'].iloc[-1]
            change_pct = (curr_close - prev_close) / prev_close * 100
            
            return float(change_pct)
            
        except Exception as e:
            logger.error(f"抓取指數 {index_symbol} 失敗: {e}")
            raise ConnectionError(f"無法抓取指數 {index_symbol}: {e}")
    
    def get_stock_info(self, symbol: str) -> dict:
        """取得股票基本資訊
        
        Args:
            symbol: 股票代碼
            
        Returns:
            股票資訊字典，包含名稱、板塊、市值等
        """
        yf = self._get_yfinance()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('shortName', info.get('longName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
            }
        except Exception as e:
            logger.warning(f"取得 {symbol} 資訊失敗: {e}")
            return {
                'symbol': symbol,
                'name': symbol,
                'sector': 'Unknown',
                'industry': 'Unknown',
            }
    
    def search_symbols(self, query: str, limit: int = 10) -> List[dict]:
        """搜尋股票代碼
        
        Args:
            query: 搜尋關鍵字
            limit: 最大結果數量
            
        Returns:
            符合的股票列表
        """
        # yfinance 沒有內建搜尋功能，這裡提供常用股票列表
        common_stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            {'symbol': 'V', 'name': 'Visa Inc.'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson'},
            {'symbol': 'WMT', 'name': 'Walmart Inc.'},
            {'symbol': 'PG', 'name': 'Procter & Gamble Co.'},
            {'symbol': 'MA', 'name': 'Mastercard Inc.'},
            {'symbol': 'UNH', 'name': 'UnitedHealth Group Inc.'},
            {'symbol': 'HD', 'name': 'Home Depot Inc.'},
            # 台股
            {'symbol': '2330.TW', 'name': '台積電'},
            {'symbol': '2317.TW', 'name': '鴻海'},
            {'symbol': '2454.TW', 'name': '聯發科'},
            {'symbol': '2308.TW', 'name': '台達電'},
            {'symbol': '2881.TW', 'name': '富邦金'},
        ]
        
        query_lower = query.lower()
        results = [
            s for s in common_stocks
            if query_lower in s['symbol'].lower() or query_lower in s['name'].lower()
        ]
        
        return results[:limit]
    
    def clear_cache(self):
        """清除所有快取"""
        self.cache.clear()
        logger.info("快取已清除")


# 預設的熱門股票監控名單
DEFAULT_WATCHLIST = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'V', 'JNJ'
]

# 台股監控名單
TW_WATCHLIST = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2881.TW',
    '2882.TW', '2303.TW', '2412.TW', '2886.TW', '1301.TW'
]

# ============ 擴展資產類別 ============

# 美股 - 科技股
US_TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'AMD', 'INTC', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'UBER', 'ABNB'
]

# 美股 - 金融股
US_FINANCE_STOCKS = [
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA'
]

# 美股 - 醫療保健
US_HEALTHCARE_STOCKS = [
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY'
]

# 美股 - 消費品
US_CONSUMER_STOCKS = [
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT'
]

# 美股 - 能源
US_ENERGY_STOCKS = [
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'
]

# ETF - 指數型
INDEX_ETFS = [
    'SPY',   # S&P 500
    'QQQ',   # 納斯達克 100
    'DIA',   # 道瓊工業
    'IWM',   # 羅素 2000
    'VTI',   # 全美股市
    'VOO',   # Vanguard S&P 500
    'IVV',   # iShares S&P 500
    'VEA',   # 已開發市場
    'VWO',   # 新興市場
    'EFA',   # MSCI EAFE
]

# ETF - 產業型
SECTOR_ETFS = [
    'XLK',   # 科技
    'XLF',   # 金融
    'XLV',   # 醫療保健
    'XLE',   # 能源
    'XLY',   # 非必需消費
    'XLP',   # 必需消費
    'XLI',   # 工業
    'XLB',   # 原物料
    'XLU',   # 公用事業
    'XLRE',  # 房地產
]

# ETF - 槓桿型
LEVERAGED_ETFS = [
    'TQQQ',  # 3x 納斯達克
    'SQQQ',  # -3x 納斯達克
    'SPXL',  # 3x S&P 500
    'SPXS',  # -3x S&P 500
    'UPRO',  # 3x S&P 500
    'SOXL',  # 3x 半導體
    'SOXS',  # -3x 半導體
    'FNGU',  # 3x FANG+
    'LABU',  # 3x 生技
]

# 債券 ETF
BOND_ETFS = [
    'TLT',   # 20+ 年美國國債
    'IEF',   # 7-10 年美國國債
    'SHY',   # 1-3 年美國國債
    'BND',   # 總體債券市場
    'AGG',   # 美國綜合債券
    'LQD',   # 投資級公司債
    'HYG',   # 高收益債券
    'TIP',   # 抗通膨債券
    'EMB',   # 新興市場債券
    'MUB',   # 市政債券
]

# 商品 ETF
COMMODITY_ETFS = [
    'GLD',   # 黃金
    'SLV',   # 白銀
    'USO',   # 原油
    'UNG',   # 天然氣
    'DBA',   # 農產品
    'DBC',   # 商品指數
    'PDBC',  # 多元商品
    'PPLT',  # 鉑金
    'PALL',  # 鈀金
    'CPER',  # 銅
]

# 期貨相關 ETF (透過 ETF 追蹤期貨)
FUTURES_ETFS = [
    'VXX',   # VIX 短期期貨
    'UVXY',  # 1.5x VIX 短期期貨
    'SVXY',  # -0.5x VIX 短期期貨
    'KOLD',  # -2x 天然氣
    'BOIL',  # 2x 天然氣
    'UCO',   # 2x 原油
    'SCO',   # -2x 原油
]

# 加密貨幣 (Yahoo Finance 支援)
CRYPTO_SYMBOLS = [
    'BTC-USD',   # 比特幣
    'ETH-USD',   # 以太坊
    'BNB-USD',   # 幣安幣
    'XRP-USD',   # 瑞波幣
    'SOL-USD',   # Solana
    'ADA-USD',   # Cardano
    'DOGE-USD',  # 狗狗幣
    'DOT-USD',   # Polkadot
    'AVAX-USD',  # Avalanche
    'MATIC-USD', # Polygon
    'LINK-USD',  # Chainlink
    'LTC-USD',   # 萊特幣
]

# 加密貨幣 ETF
CRYPTO_ETFS = [
    'BITO',  # ProShares 比特幣期貨 ETF
    'BTF',   # Valkyrie 比特幣期貨 ETF
    'GBTC',  # Grayscale 比特幣信託
    'ETHE',  # Grayscale 以太坊信託
    'IBIT',  # iShares 比特幣信託
    'FBTC',  # Fidelity 比特幣 ETF
]

# 國際市場 ETF
INTERNATIONAL_ETFS = [
    'EWJ',   # 日本
    'FXI',   # 中國大型股
    'EWZ',   # 巴西
    'EWY',   # 韓國
    'EWT',   # 台灣
    'EWG',   # 德國
    'EWU',   # 英國
    'EWA',   # 澳洲
    'EWC',   # 加拿大
    'INDA',  # 印度
]

# 房地產 REITs
REIT_ETFS = [
    'VNQ',   # Vanguard 房地產
    'IYR',   # iShares 房地產
    'SCHH',  # Schwab 房地產
    'RWR',   # SPDR 房地產
    'XLRE',  # 房地產精選
]

# 所有資產類別字典
ASSET_CATEGORIES = {
    '美股-科技': US_TECH_STOCKS,
    '美股-金融': US_FINANCE_STOCKS,
    '美股-醫療': US_HEALTHCARE_STOCKS,
    '美股-消費': US_CONSUMER_STOCKS,
    '美股-能源': US_ENERGY_STOCKS,
    '台股': TW_WATCHLIST,
    'ETF-指數': INDEX_ETFS,
    'ETF-產業': SECTOR_ETFS,
    'ETF-槓桿': LEVERAGED_ETFS,
    '債券ETF': BOND_ETFS,
    '商品ETF': COMMODITY_ETFS,
    '期貨ETF': FUTURES_ETFS,
    '加密貨幣': CRYPTO_SYMBOLS,
    '加密貨幣ETF': CRYPTO_ETFS,
    '國際市場': INTERNATIONAL_ETFS,
    'REITs': REIT_ETFS,
}
