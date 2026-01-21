#!/usr/bin/env python
"""AI PatternQuant 啟動程式

雙擊此檔案或執行 python run.py 即可啟動系統。
"""

import subprocess
import sys
import os


def check_dependencies():
    """檢查必要的套件是否已安裝"""
    required = ['streamlit', 'yfinance', 'plotly', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"缺少套件: {', '.join(missing)}")
        print("正在安裝...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user'] + missing)
        print("安裝完成！")


def main():
    """啟動 AI PatternQuant"""
    print("=" * 50)
    print("  AI PatternQuant 幾何特徵量化交易系統")
    print("=" * 50)
    print()
    
    # 檢查依賴
    print("檢查套件依賴...")
    check_dependencies()
    print("✓ 套件檢查完成")
    print()
    
    # 取得專案目錄
    project_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_dir, 'pattern_quant', 'ui', 'app.py')
    
    if not os.path.exists(app_path):
        print(f"錯誤: 找不到應用程式 {app_path}")
        input("按 Enter 鍵退出...")
        return
    
    print("正在啟動 Streamlit 伺服器...")
    print("瀏覽器將自動開啟，如果沒有請手動訪問: http://localhost:8501")
    print()
    print("按 Ctrl+C 停止伺服器")
    print("-" * 50)
    
    # 啟動 Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            app_path,
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false'
        ], cwd=project_dir)
    except KeyboardInterrupt:
        print("\n伺服器已停止")


if __name__ == '__main__':
    main()
