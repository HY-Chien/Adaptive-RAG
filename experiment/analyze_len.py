import json
import jsonlines
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. 設定路徑 (注意：這裡都加了 ../ 因為我們在 experiment 資料夾內)
# ==========================================

# ★ 請確認這行路徑是您最新的預測結果 (epoch 30, 時間戳記對應您的 log)
PRED_DIR = '../classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40/predict/'

# 原始資料的根目錄
DATA_DIR = '../processed_data'

# 要分析的資料集
DATASETS = ['nq', 'trivia', 'squad', 'musique', 'hotpotqa', '2wikimultihopqa']

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_questions_from_jsonl(dataset_name):
    """從原始 jsonl 檔案中讀取問題文本"""
    # 路徑修正：往上一層找 processed_data
    path = os.path.join(DATA_DIR, dataset_name, 'test_subsampled.jsonl')
    qid_to_text = {}
    
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return {}

    with jsonlines.open(path) as reader:
        for obj in reader:
            # 嘗試抓取可能的 ID 欄位
            qid = obj.get('question_id') or obj.get('id') or obj.get('_id')
            # 嘗試抓取可能的 Question 欄位
            q_text = obj.get('question') or obj.get('question_text')
            
            if qid and q_text:
                qid_to_text[str(qid)] = q_text
    return qid_to_text

def main():
    # 1. 讀取預測結果
    pred_file = os.path.join(PRED_DIR, 'dict_id_pred_results.json')
    if not os.path.exists(pred_file):
        print(f"Error: 找不到預測檔案，請檢查路徑是否正確：\n{pred_file}")
        return

    print(f"Loading predictions from {pred_file}...")
    preds = load_json(pred_file)
    
    # 2. 讀取所有資料集的問題文本
    print("Loading original questions text...")
    all_questions = {}
    for ds in DATASETS:
        q_map = load_questions_from_jsonl(ds)
        all_questions.update(q_map)
        print(f"  - {ds}: loaded {len(q_map)} questions")

    # 3. 整合資料
    data_list = []
    missing_count = 0
    
    for qid, info in preds.items():
        label = info['prediction'] # A, B, or C
        ds_name = info['dataset_name']
        
        # 找回問題文本
        q_text = all_questions.get(qid)
        
        if q_text:
            # 計算長度 (以空白分割計算字數)
            length = len(q_text.split())
            data_list.append({
                'qid': qid,
                'dataset': ds_name,
                'label': label,
                'length': length,
                'text': q_text
            })
        else:
            missing_count += 1

    print(f"Total analyzed: {len(data_list)} samples (Missing text: {missing_count})")
    
    if len(data_list) == 0:
        print("No data found to plot. Exiting.")
        return

    df = pd.DataFrame(data_list)
    
    # 4. 畫圖分析
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6)) #稍微加寬一點

    # 子圖 1: 整體長度分佈 (Boxplot)
    plt.subplot(1, 2, 1)
    order = ['A', 'B', 'C'] # A=Easy, B=Medium, C=Hard
    sns.boxplot(x='label', y='length', data=df, order=order, palette="Set2")
    plt.title('Question Length Distribution by Predicted Complexity')
    plt.xlabel('Predicted Label (A:Zero, B:Single, C:Multi)')
    plt.ylabel('Question Length (Words)')

    # 子圖 2: 每個資料集的平均長度 (Barplot)
    plt.subplot(1, 2, 2)
    sns.barplot(x='dataset', y='length', hue='label', data=df, hue_order=order, palette="Set2", errorbar=('ci', 95))
    plt.title('Average Length by Dataset & Label')
    plt.xticks(rotation=45)
    plt.ylabel('Avg Question Length')
    plt.legend(title='Complexity')

    plt.tight_layout()
    output_img = 'analysis_length_correlation.png'
    plt.savefig(output_img)
    print(f"\nAnalysis plot saved to: {os.path.abspath(output_img)}")
    
    # 5. 印出統計數據到終端機
    print("\n=== Statistics (Mean Length & Count) ===")
    stats = df.groupby('label')['length'].agg(['mean', 'std', 'count', 'min', 'max'])
    print(stats)

if __name__ == "__main__":
    main()
    