from data_loader.data_loader import DataLoader
from model.default_model import CreateModel
from trainer.default_trainer import CreateTrainer
import config

print('Load data.')
data_loader = DataLoader(config.PATH['data'])

print('Create model.')
default_model = CreateModel(config.MODEL_CONFIG)

print('Create trainer.')
trainer = CreateTrainer(default_model.model, data_loader, config.HYPER_PARAMTER)

print('Start training model.')
trainer.train()
default_model.save_model(config.PATH['save'])

print('Start evaluate model.')
trainer.evaluate()