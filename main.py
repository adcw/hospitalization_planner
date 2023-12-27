"""
Use this script to train, validate and test models.
"""
from src.session.model_manager import ModelManager
from src.config.seeds import set_seed

if __name__ == '__main__':
    # set_seed(9999)
    set_seed(1111)

    manager = ModelManager(models_dir='./models', config_path="./config.yaml", test_perc=0.05)
    manager.start()
