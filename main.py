import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

from data_processing.data_loader import load_and_preprocess_data
from feature_engineering.feature_generator import feature_engineering
from models.random_forest import build_random_forest_model
from models.xgboost import build_xgboost_model
from models.gaussian_process import build_gpr_model
from models.stacking_model import build_stacking_model
from training.trainer import tune_model, cross_validation
from evaluation.evaluator import evaluate_model
from evaluation.explainer import explain_model_with_shap
from utils.visualizer import visualize_results
from utils.logger import setup_logging
from utils.config_loader import load_config
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor

def main():
    """主函数，运行整个模型训练和评估流程"""
    setup_logging()
    config = load_config()
    logging.info("配置加载成功。")
    
    logging.info("启动模型训练流程...")

    # 1. 加载数据
    # 注意：这里的文件路径是硬编码的，后续我们会用配置文件来管理
    df = load_and_preprocess_data('data/battery_performance_data.csv')

    # 2. 特征工程
    df_featured = feature_engineering(df.copy())

    # 3. 定义特征和目标
    X = df_featured.drop(columns=config['features']['features_to_exclude'])
    y = df_featured[config['features']['target_column']]
    
    feature_names = X.columns

    # 4. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("数据标准化完成。")
    
    # 5. 特征选择
    # 使用递归特征消除 (RFE)
    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    X_selected = selector.fit_transform(X_scaled, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = feature_names[selected_indices]
    logging.info(f"已选择 {len(selected_features)} 个最佳特征: {selected_features.tolist()}")

    # 6. 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )

    # 7. 构建和训练模型
    
    # 定义超参数搜索空间
    rf_param_space = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(5, 50),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 4)
    }
    
    xgb_param_space = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'subsample': Real(0.6, 1.0, 'uniform'),
        'colsample_bytree': Real(0.6, 1.0, 'uniform')
    }

    # 使用贝叶斯优化调整模型
    rf_model_tuned = tune_model(RandomForestRegressor(random_state=42), X_train, y_train, rf_param_space, n_iter=32)
    xgb_model_tuned = tune_model(XGBRegressor(random_state=42), X_train, y_train, xgb_param_space, n_iter=32)
    
    gpr_model = build_gpr_model(X_train, y_train)
    models = [rf_model_tuned, xgb_model_tuned, gpr_model]

    # 8. 模型评估
    logging.info("评估调优后的随机森林模型...")
    evaluate_model(rf_model_tuned, X_test, y_test, model_name="Tuned Random Forest")
    
    logging.info("评估调优后的XGBoost模型...")
    evaluate_model(xgb_model_tuned, X_test, y_test, model_name="Tuned XGBoost")

    logging.info("评估高斯过程回归模型...")
    evaluate_model(gpr_model, X_test, y_test, model_name="Gaussian Process")

    # 9. 构建和评估 Stacking 集成模型
    # 我们将调优后的模型作为 Stacking 的基模型
    base_models = [rf_model_tuned, xgb_model_tuned]
    base_model_names = ["Tuned RF", "Tuned XGB"]
    stacking_model = build_stacking_model(X_train, y_train, base_models, base_model_names)
    
    logging.info("评估 Stacking 集成模型...")
    evaluate_model(stacking_model, X_test, y_test, model_name="Stacking Ensemble")

    # 10. 可视化结果
    all_predictions = [rf_model_tuned.predict(X_test), 
                       xgb_model_tuned.predict(X_test), 
                       gpr_model.predict(X_test), 
                       stacking_model.predict(X_test)]
    model_names = ["Tuned Random Forest", "Tuned XGBoost", "Gaussian Process", "Stacking Ensemble"]
    visualize_results(y_test, all_predictions, model_names)

    # 11. 模型可解释性分析 (使用 SHAP)
    # 我们将对性能最好的 Stacking 模型进行深入分析
    # 注意：SHAP 分析可能需要一些时间
    explain_model_with_shap(stacking_model, X_train, selected_features)


    logging.info("模型训练和评估完成。")

if __name__ == '__main__':
    main()