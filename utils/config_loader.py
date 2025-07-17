import yaml

def load_config(config_path='config/config.yaml'):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config 