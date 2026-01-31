import shutil
from pathlib import Path
import json

def cleanup_files():
    # 1. 設定の読み込み
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("config.json が見つからないため、削除をスキップします。")
        return

    # 2. 削除対象のパスを取得
    # ※ configのキー名はご自身の環境に合わせて調整してください
    output_bag = Path(config['main']['output_bag'])
    pcd_directory = Path(config['main']['output_directory'])

    print("\n--- Cleanup Process ---")

    # 3. ROSバッグファイルの削除
    if output_bag.exists():
        try:
            output_bag.unlink()
            print(f"Deleted bag file: {output_bag}")
        except Exception as e:
            print(f"Failed to delete bag: {e}")
    else:
        print("No bag file to delete.")

    # 4. PCDディレクトリの削除 (中身ごと削除)
    if pcd_directory.exists() and pcd_directory.is_dir():
        try:
            # shutil.rmtree はディレクトリ内が空でなくても全て削除します
            shutil.rmtree(pcd_directory)
            print(f"Deleted directory and all contents: {pcd_directory}")
        except Exception as e:
            print(f"Failed to delete directory: {e}")
    else:
        print("No directory to delete.")