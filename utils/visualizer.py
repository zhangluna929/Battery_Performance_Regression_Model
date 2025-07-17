import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_results(y_test, predictions, model_names):
    """
    可视化模型的预测结果与真实值的对比图。

    参数:
    - y_test: 真实的测试集目标值.
    - predictions: 一个包含各个模型预测结果的列表.
    - model_names: 一个包含各个模型名称的列表.
    """
    logging.info("正在生成可视化结果...")
    
    plt.figure(figsize=(14, 7))
    
    # 绘制所有模型预测值 vs 真实值的散点图
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        sns.scatterplot(x=y_test, y=pred, label=name, color=colors[i % len(colors)], alpha=0.6)

    # 绘制理想情况下的对角线
    min_val = min(y_test.min(), pd.Series(predictions).apply(min).min())
    max_val = max(y_test.max(), pd.Series(predictions).apply(max).max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')

    plt.title('模型预测值 vs. 真实值', fontsize=16)
    plt.xlabel('真实值 (RUL)', fontsize=12)
    plt.ylabel('预测值 (RUL)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 