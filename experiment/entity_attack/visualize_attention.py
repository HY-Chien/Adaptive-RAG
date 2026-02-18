import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= 設定區 =================
# 1. 模型路徑
MODEL_PATH = "/home/S113065528/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40"

# 2. Log 檔案路徑
LOG_FILE = "nq_subject_attack_log_100.json"

# 3. 輸出的圖片檔名 (可以加上 ID 以便區分)
OUTPUT_IMG = "attention_heatmap_nq_4366.png"

# 4. ★★★ 修改這裡：指定你要找的 ID ★★★
TARGET_ID = "single_nq_dev_4366"

# ==========================================

def load_example_by_id():
    if not os.path.exists(LOG_FILE):
        print(f"Error: 找不到 Log 檔 {LOG_FILE}，請確認檔案位置。")
        return None, None

    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("Error: Log 檔是空的")
        return None, None
        
    # 搜尋指定的 ID
    found_entry = None
    for entry in data:
        if entry.get('id') == TARGET_ID:
            found_entry = entry
            break
            
    if found_entry is None:
        print(f"Error: 在 Log 檔中找不到 ID 為 '{TARGET_ID}' 的資料。")
        return None, None

    # 讀取資料內容
    q1 = found_entry.get('original_question')
    q2 = found_entry.get('perturbed_question')
    
    # 抓取變動資訊 (用於顯示)
    change_info = found_entry.get('changes', [{}])[0]
    orig_word = change_info.get('original_word', 'unknown')
    new_word = change_info.get('replaced_with', 'unknown')

    print(f"\nSuccessfully Found ID: {TARGET_ID}")
    print(f"Target Entity Change: '{orig_word}' -> '{new_word}'")
    print(f"Original:  {q1}")
    print(f"Perturbed: {q2}")
    
    return q1, q2

def visualize_cross_attention(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids
    
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]])

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            decoder_input_ids=decoder_input_ids,
            output_attentions=True
        )

    # 抓取最後一層的 Cross Attention
    last_layer_attn = outputs.cross_attentions[-1]
    
    # 平均所有 Head 的權重 [Batch, Heads, Seq_len_out, Seq_len_in] -> [Seq_len_in]
    attn_avg = last_layer_attn.mean(dim=1).squeeze(0).mean(dim=0)
    
    # 轉為 numpy
    attn_weights = attn_avg.cpu().numpy().flatten()
    
    # 處理 Tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    clean_tokens = [t.replace(' ', '') for t in tokens]
    
    return clean_tokens, attn_weights

def plot_heatmap(tokens_1, weights_1, tokens_2, weights_2, q1_text, q2_text):
    plt.figure(figsize=(22, 10))
    sns.set(font_scale=1.4)

    # 確保不會因為權重差異太大而導致顏色看不清，統一用 0~最大值
    max_val = max(np.max(weights_1), np.max(weights_2))

    # 畫第一張圖 (原始問題)
    plt.subplot(2, 1, 1)
    # 這裡將 weights 包成 list of list 以符合 heatmap 格式 (1, len)
    sns.heatmap([weights_1], xticklabels=tokens_1, yticklabels=['Attn'], 
                cmap="Reds", cbar=True, annot=False, square=True,
                vmin=0, vmax=max_val) # 固定色階範圍方便比較
    plt.title(f"Original: {q1_text}", fontsize=16)
    plt.xticks(rotation=45, ha='right')

    # 畫第二張圖 (修改後問題)
    plt.subplot(2, 1, 2)
    sns.heatmap([weights_2], xticklabels=tokens_2, yticklabels=['Attn'], 
                cmap="Reds", cbar=True, annot=False, square=True,
                vmin=0, vmax=max_val) # 固定色階範圍方便比較
    plt.title(f"Perturbed: {q2_text}", fontsize=16)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"\nHeatmap saved to {os.path.abspath(OUTPUT_IMG)}")

def main():
    # 1. 讀取資料 (By ID)
    q1, q2 = load_example_by_id()
    if not q1 or not q2:
        return

    # 2. 載入模型
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: 找不到模型路徑，請檢查路徑：\n{MODEL_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.eval()
    
    # 3. 畫圖
    print("Visualizing Attention...")
    tokens1, weights1 = visualize_cross_attention(model, tokenizer, q1)
    tokens2, weights2 = visualize_cross_attention(model, tokenizer, q2)
    
    # 移除最後的 </s> token (通常是最後一個)
    if tokens1 and tokens1[-1] == '</s>':
        tokens1 = tokens1[:-1]
        weights1 = weights1[:-1]
    if tokens2 and tokens2[-1] == '</s>':
        tokens2 = tokens2[:-1]
        weights2 = weights2[:-1]

    plot_heatmap(tokens1, weights1, tokens2, weights2, q1, q2)

if __name__ == "__main__":
    main()