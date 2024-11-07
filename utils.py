
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np

def plot_image_prediction_and_mask(image, prediction, mask):
    image_rgb = image.permute(1, 2, 0).cpu().numpy()[:, :, 0]
    prediction = prediction.cpu().numpy()
    mask = mask.cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb, cmap='gray')
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.show()

def print_average_parameter_value(model):
    total_sum = 0
    total_params = 0

    for param in model.parameters():
        total_sum += param.sum().item()
        total_params += param.numel()

    average_value = total_sum / total_params if total_params > 0 else 0
    print(f"Average parameter value: {average_value:.4f}")

def print_average_grad_parameter_value(model):
    total_sum = 0
    total_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total_sum += param.sum().item()
            total_params += param.numel()

    average_value = total_sum / total_params if total_params > 0 else 0
    print(f"Average value of parameters with requires_grad=True: {average_value:.5f}")
