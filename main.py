"""
Use this script to train, validate and test models.
"""
from src.session.model_manager import ModelManager
from src.config.seeds import set_seed

if __name__ == '__main__':
    set_seed(9999)

    manager = ModelManager(models_dir='./models', config_path="./config.yaml")
    manager.start()
