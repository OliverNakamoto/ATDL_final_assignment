
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Dice
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import time
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

from data_loading import LungSegmentationDataset, BreastTumorDataset
from models import LinearHead
from utils import plot_image_prediction_and_mask, print_average_parameter_value, print_average_grad_parameter_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = 3  
learning_rate = 1e-5
num_epochs = 100
batch_size = 64
num_layers = 1
input_dim = 384 * num_layers
weight_decay = 1e-3

base_path = "/dtu/p1/osanch/fine_tune/data"  

pattern = r'\((\d+)\)\.png$'
dataset = []
for path in os.listdir(base_path):
    class_of_tumor = path
    for filename in os.listdir(os.path.join(base_path, path)):
        match = re.search(pattern, filename)
        if match:
            index = match.group(1)
            dataset.append((class_of_tumor, index))

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((448, 448)),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.1953, 0.1925, 0.1942]),
])
transform_train_mask = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
])

train_val, test = train_test_split(dataset, test_size=0.2, random_state=42)

train_dataset = BreastTumorDataset(train_val, base_dir=base_path, transform=transform_train, target_transform=transform_train_mask)
val_dataset = BreastTumorDataset(test, base_dir=base_path, transform=transform_train, target_transform=transform_train_mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

df = pd.DataFrame(columns=["r", "loss", "dice_metric", "trainable", "best_model", "time"])
rs = [255]  #list of 'r' values to test

for r in rs:
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = LinearHead(dinov2_vits14, r, input_dim, num_classes).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader.dataset), eta_min=0.0001)
    dice_metric = Dice(num_classes=num_classes, ignore_index=0, threshold=0.5).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
       if p.requires_grad:
           print(p)
    print("trainable parameters", num_trainable_params)
    iteration = {"r": r, "loss": [], "dice_metric": [], "trainable": num_trainable_params, "best_model": None, "time": None}

    dice_metric_max = 0
    start = time.time()

    for epoch in range(num_epochs):
        print("Current Epoch: ", epoch)
        for p in model.parameters():
            if p.requires_grad:
                print(p)
        model.train()
        running_loss = 0.0
        for i, (imgs, patch_labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            patch_labels = patch_labels.to(device)

            outputs = model(imgs, num_layers)
            outputs = outputs.view(-1, num_classes)
            patch_labels = patch_labels.view(-1)

            # Using sigmoid focal loss
            targets_one_hot = F.one_hot(patch_labels, num_classes=num_classes).float()
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        dice_metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for i, (imgs, mask) in enumerate(val_loader):
                imgs = imgs.to(device)
                mask = mask.to(device).squeeze(1)

                outputs = model(imgs, num_layers)
                outputs = outputs.view(-1, num_classes)

                preds = torch.argmax(outputs, dim=1)
                preds = preds.view(-1, 32, 32)
                resize = transforms.Resize(mask.shape[1:], interpolation=transforms.InterpolationMode.NEAREST_EXACT)
                preds = resize(preds)

                dice_metric.update(preds, mask)

            dice_metric_avg = dice_metric.compute().item()
            print("Dice average for epoch: ", dice_metric_avg)
            if dice_metric_avg > dice_metric_max:
                dice_metric_max = dice_metric_avg
                torch.save(model.state_dict(), f"model_{r}_CANCER.pth")
            iteration["dice_metric"].append(dice_metric_avg)

    end = time.time()
    length = end - start
    iteration["time"] = length
    iteration_df = pd.DataFrame([iteration])
    df = pd.concat([df, iteration_df], ignore_index=True)

df.to_csv('results.csv', index=False)
