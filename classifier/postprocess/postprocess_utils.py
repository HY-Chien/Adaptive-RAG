import json, jsonlines
import os, sys

def load_json(json_file_path):
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    return json_data

def save_json(json_file_path, json_data):
    if not os.path.exists(os.path.dirname(json_file_path)): 
        os.makedirs(os.path.dirname(json_file_path)) 
    
    with open(json_file_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4, sort_keys=True)
    print(json_file_path)


def save_prediction_with_classified_label(total_qid_to_classification_pred, dataset_name, stepNum_result_file, dataName_to_multi_one_zero_file, output_path):
    qid_to_classification_pred = {}
    qid_to_classification_pred_option = {}
    total_stepNum = 0

    # === 優化 1: 建立快取字典，避免在迴圈內重複讀取同一個檔案 ===
    loaded_files_cache = {}
    
    # 預先讀取 StepNum 檔案 (如果有需要的話)
    step_num_data = {}
    if os.path.exists(stepNum_result_file):
        try:
            step_num_data = load_json(stepNum_result_file)
        except Exception as e:
            print(f"Warning: Failed to load stepNum file: {e}")

    for qid in total_qid_to_classification_pred.keys():
        
        if dataset_name != total_qid_to_classification_pred[qid]['dataset_name']:
            continue

        predicted_option = total_qid_to_classification_pred[qid]['prediction']
        
        # 計算 StepNum
        stepNum = 0
        if predicted_option == 'C':
            # 從預先讀取的資料中拿，避免重複讀檔
            stepNum = step_num_data.get(qid, 0) 
        elif predicted_option == 'B':
            stepNum = 1
        elif predicted_option == 'A':
            stepNum = 0

        # === 核心修改: 安全讀取預測結果 ===
        try:
            # 1. 取得對應的檔案路徑
            target_file_path = dataName_to_multi_one_zero_file[dataset_name][predicted_option]
            
            # 2. 檢查是否已經讀過這個檔案 (優化效能)
            if target_file_path not in loaded_files_cache:
                if os.path.exists(target_file_path):
                    loaded_files_cache[target_file_path] = load_json(target_file_path)
                else:
                    # 如果檔案不存在，存一個空字典，避免之後報錯
                    print(f"Warning: File not found: {target_file_path}")
                    loaded_files_cache[target_file_path] = {}

            # 3. 從快取中取得資料
            current_pred_data = loaded_files_cache[target_file_path]

            # 4. === 關鍵修正 ===: 檢查 ID 是否存在
            if qid not in current_pred_data:
                # 如果這個 ID 在預測檔裡找不到 (例如 NQ 的情況)，就跳過
                # print(f"Skipping QID {qid}: Not found in {predicted_option} prediction file.")
                continue

            pred = current_pred_data[qid]
            
            # 存入結果
            qid_to_classification_pred[qid] = pred
            qid_to_classification_pred_option[qid] = {'prediction' : pred, 'option' : predicted_option, 'stepNum' : stepNum}
            total_stepNum = total_stepNum + stepNum

        except Exception as e:
            print(f"Error processing QID {qid}: {e}")
            continue
    
    print('==============')
    save_json(os.path.join(output_path, dataset_name , dataset_name+'.json'), qid_to_classification_pred)
    save_json(os.path.join(output_path, dataset_name, dataset_name+'_option.json'), qid_to_classification_pred_option)
    print('StepNum')
    print(dataset_name + ': ' +str(total_stepNum))