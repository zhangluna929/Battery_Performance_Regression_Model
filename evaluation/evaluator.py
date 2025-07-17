import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# 模型评估
def evaluate_model(model, X_test, y_test, model_name=""):
    """
    评估模型的性能.
    
    参数:
    - model: 已训练的模型.
    - X_test: 测试集特征.
    - y_test: 测试集目标.
    - model_name: 模型的名称（用于日志记录）.
    """
    logging.info(f"正在评估模型: {model_name}")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"[{model_name}] 均方误差 (MSE): {mse:.4f}")
    logging.info(f"[{model_name}] 平均绝对误差 (MAE): {mae:.4f}")
    logging.info(f"[{model_name}] 均方根误差 (RMSE): {rmse:.4f}")
    logging.info(f"[{model_name}] R² 分数: {r2:.4f}")
    
    return mse, mae, rmse, r2

# 集成预测
def ensemble_predict(models, X_test):
    """
    通过对多个模型的预测进行平均来生成集成预测。

    参数:
    - models: 已训练模型的列表.
    - X_test: 测试集特征.

    返回:
    - ensemble_pred: 集成预测结果.
    """
    if not models:
        logging.warning("模型列表为空，无法进行集成预测。")
        return None

    logging.info("正在生成集成预测...")
    predictions = [model.predict(X_test) for model in models]
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred 