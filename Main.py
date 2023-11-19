"""
Use this script to train, validate and test models.
"""
from src.session.ModelManager import ModelManager

if __name__ == '__main__':
    manager = ModelManager(models_dir='./models', config_path="./config.yaml")
    manager.start()
