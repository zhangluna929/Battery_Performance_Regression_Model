import logging
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def feature_engineering(df):
    """执行高级特征工程"""
    logging.info("开始执行特征工程...")
    
    # 按 battery_id 分组，以便计算每个电池的时间序列特征
    df_grouped = df.groupby('battery_id')
    
    # 1. 物理机理特征 (基于变化率)
    df['capacity_ah_diff'] = df_grouped['capacity_ah'].diff().fillna(0)
    df['resistance_ohm_diff'] = df_grouped['resistance_ohm'].diff().fillna(0)
    
    # 2. 时间序列特征 (移动平均)
    df['capacity_moving_avg_5'] = df_grouped['capacity_ah'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['resistance_moving_avg_5'] = df_grouped['resistance_ohm'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # 3. 构造新的交互特征
    df['temp_X_discharge_rate'] = df['temperature_c'] * df['discharge_rate_c']

    logging.info(f"特征工程完成，新增特征: ['capacity_ah_diff', 'resistance_ohm_diff', 'capacity_moving_avg_5', 'resistance_moving_avg_5', 'temp_X_discharge_rate']")
    logging.info(f"数据最终维度: {df.shape}")
    
    # 再次处理因特征工程可能产生的NaN值
    df.fillna(0, inplace=True)
    
    return df

# 特征选择
def feature_selection(X, y, k=10):
    logging.info(f"Performing feature selection. Selecting top {k} features.")
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    logging.info(f"Selected features: {selected_features}")
    return X_new, selected_features 