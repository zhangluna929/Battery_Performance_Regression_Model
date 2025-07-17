import shap
import logging
import matplotlib.pyplot as plt

def explain_model_with_shap(model, X, feature_names):
    """
    使用 SHAP 来解释模型的预测。

    这将生成一个 SHAP 摘要图，显示特征的全局重要性。

    参数:
    - model: 需要解释的模型.
    - X: 用于解释的数据集 (例如，训练集或测试集).
    - feature_names: 特征名称列表.
    """
    logging.info(f"正在为模型 {type(model).__name__} 生成 SHAP 可解释性分析...")

    try:
        # SHAP 需要一个统一的接口来处理不同类型的模型
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # 1. SHAP 摘要图 (全局特征重要性)
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot for {type(model).__name__}')
        plt.tight_layout()
        plt.show()

        logging.info("SHAP 摘要图已生成。")

    except Exception as e:
        logging.error(f"生成 SHAP 分析时出错: {e}")
        logging.warning("对于某些模型（如未使用特定树结构的模型），标准的SHAP树解释器可能不适用。")
        logging.warning("请尝试使用 KernelExplainer 或其他适合您模型的解释器。") 