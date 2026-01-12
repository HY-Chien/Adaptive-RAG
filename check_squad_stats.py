import json
import os
from collections import Counter, defaultdict

def analyze_prediction_results(file_path):
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 '{file_path}'")
        return

    print(f"正在讀取並分析: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 準備計數器
        stats = defaultdict(Counter)
        total_counts = Counter()
        
        # *** 關鍵修改：這裡要用 .values() 或 .items() ***
        # 因為這個檔案是 { "ID": {內容}, "ID": {內容} } 的結構
        for q_id, info in data.items():
            # 防呆：確保 info 是字典
            if not isinstance(info, dict):
                continue

            d_name = info.get('dataset_name', 'Unknown')
            # 注意：這裡欄位名稱是 'prediction' 而不是 'answer'
            pred = info.get('prediction', 'Unknown') 
            
            stats[d_name][pred] += 1
            total_counts[pred] += 1
            
        # --- 輸出報表 ---
        print("\n" + "="*85)
        print(f"{'Dataset (資料集)':<20} | {'Total':<6} | {'A (Simple)':<15} | {'B (Single)':<15} | {'C (Multi)':<15}")
        print("-" * 85)

        for d_name in sorted(stats.keys()):
            c = stats[d_name]
            total = sum(c.values())
            
            # 計算百分比
            pa = (c.get('A', 0)/total*100) if total else 0
            pb = (c.get('B', 0)/total*100) if total else 0
            pc = (c.get('C', 0)/total*100) if total else 0
            
            print(f"{d_name:<20} | {total:<6} | {c.get('A', 0):<4} ({pa:>5.1f}%) | {c.get('B', 0):<4} ({pb:>5.1f}%) | {c.get('C', 0):<4} ({pc:>5.1f}%)")

        print("-" * 85)
        
        # 總體統計
        t_total = sum(total_counts.values())
        ta = (total_counts.get('A', 0)/t_total*100) if t_total else 0
        tb = (total_counts.get('B', 0)/t_total*100) if t_total else 0
        tc = (total_counts.get('C', 0)/t_total*100) if t_total else 0
        
        print(f"{'>>> ALL PREDICTIONS':<20} | {t_total:<6} | {total_counts.get('A', 0):<4} ({ta:>5.1f}%) | {total_counts.get('B', 0):<4} ({tb:>5.1f}%) | {total_counts.get('C', 0):<4} ({tc:>5.1f}%)")
        print("="*85 + "\n")

    except Exception as e:
        print(f"發生未預期的錯誤: {e}")

# ==========================================
# 請確認您的檔案路徑
# ==========================================
target_file = "classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/25/2025_12_13/10_29_41/predict/dict_id_pred_results.json"

if __name__ == "__main__":
    analyze_prediction_results(target_file)