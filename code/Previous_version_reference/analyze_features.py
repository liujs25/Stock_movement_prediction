import os
import sys
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import joblib  # 确保安装了 joblib: pip install joblib
from datetime import datetime

# 保持与训练脚本一致的路径引用
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.data_process import DataProcessor

def tprint(*args, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}]', *args, **kwargs)

def analyze_model_features(model_path, data_dir):
    tprint(f"Starting feature analysis for model: {model_path}")

    # 1. 初始化 DataProcessor 重新构建特征名称列表
    processor = DataProcessor(data_dir)
    feature_names = processor.final_feature_names
    tprint(f"Reconstructed feature name list. Total dimensions: {len(feature_names)}")

    # 2. 兼容性加载模型
    bst = None
    try:
        # 尝试使用 joblib 加载 (因为报错信息显示是 Pickle 格式)
        loaded_obj = joblib.load(model_path)
        
        # 判断加载回来的是包装类还是原始 Booster
        if hasattr(loaded_obj, 'model'):
            bst = loaded_obj.model
        elif isinstance(loaded_obj, xgb.Booster):
            bst = loaded_obj
        else:
            # 某些情况下可能是 sklearn 接口对象
            bst = loaded_obj.get_booster()
            
        tprint("Model loaded successfully via joblib.")
    except Exception as e:
        tprint(f"Joblib load failed, trying native XGBoost load... Error: {e}")
        try:
            bst = xgb.Booster()
            bst.load_model(model_path)
            tprint("Model loaded successfully via native XGBoost loader.")
        except Exception as e2:
            tprint(f"All loading methods failed. Error: {e2}")
            return

    # 3. 提取特征重要性 (Importance Type: Gain)
    importance_gain = bst.get_score(importance_type='gain')
    
    # 4. 建立映射
    mapped_importance = []
    for k, v in importance_gain.items():
        # XGBoost 的 key 通常是 'f123'
        try:
            idx = int(k[1:]) 
            if idx < len(feature_names):
                mapped_importance.append((feature_names[idx], v))
            else:
                mapped_importance.append((f"idx_{idx}", v))
        except ValueError:
            # 如果 key 本身就是名称
            mapped_importance.append((k, v))

    # 5. 排序并取前 20 名
    sorted_importance = sorted(mapped_importance, key=lambda x: x[1], reverse=True)
    top_20 = sorted_importance[:20]

    # 6. 打印结果
    if not top_20:
        tprint("Warning: No feature importance scores found. Did the model actually use any features?")
        return

    tprint("="*60)
    tprint(f"{'Feature Name':<40} | {'Gain Score':<15}")
    tprint("-"*60)
    for name, score in top_20:
        print(f"{name:<40} | {score:>15.4f}")
    tprint("="*60)

    # 7. 可视化绘制
    plot_importance(top_20)

def plot_importance(top_20):
    names = [x[0] for x in top_20][::-1] 
    scores = [x[1] for x in top_20][::-1]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(names, scores, color='skyblue', edgecolor='navy')
    plt.xlabel('Average Gain (Signal Strength)')
    plt.title('Top 20 Features: What is the model looking at?')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                 va='center', ha='left', fontsize=9, color='darkblue')

    output_fig = "./Experiments/feature_importance_analysis.png"
    plt.tight_layout()
    plt.savefig(output_fig)
    tprint(f"Analysis plot saved to: {output_fig}")
    # plt.show() # 如果在服务器运行，建议注释掉这行以免阻塞

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze XGBoost Model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    
    args = parser.parse_args()
    analyze_model_features(args.model_path, args.data_dir)