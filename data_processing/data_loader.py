import pandas as pd
import logging

def load_and_preprocess_data(filepath):
    """加载数据并进行基本预处理"""
    logging.info(f"从 '{filepath}' 加载数据...")
    df = pd.read_csv(filepath)
    logging.info(f"数据加载成功，维度: {df.shape}")

    # 处理缺失值 (以防万一)
    df.dropna(inplace=True)
    logging.info(f"处理缺失值后，维度: {df.shape}")
    
    return df 