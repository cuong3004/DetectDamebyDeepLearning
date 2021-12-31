from litmodel import LitClassification
from dataloader import train_loader, valid_loader, test_loader, target_list
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from callbacks import input_monitor, checkpoint_callback, early_stop_callback
import torchvision
import torch


num_classes = len(set(target_list))
version = "b0"
lr = 1e-4
epoch = 20

model_mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
model_mobilenet.classifier[1] = torch.nn.Linear(1280,num_classes)
model = LitClassification(model_mobilenet, lr)


for x, y in train_loader:
    print(x.shape)
    break

callbacks = [input_monitor, checkpoint_callback, early_stop_callback]

# training

trainer = pl.Trainer(gpus=1, callbacks = callbacks, max_epochs=epoch)
trainer.fit(model, train_loader, valid_loader)

# trainer.test(test_loader)