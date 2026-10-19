import torch
from torch.utils.data import ConcatDataset

from data import load_cifar, splits
from evalutils import get_accuracy
from utils import set_seed, get_device
from train import train_model

if __name__ == "__main__":

    set_seed(42)

    device = get_device()
    print(f"Using device {device}")

    train_retain, _, test_retain, test_forget = splits(augment=True) # augmented train, clean test
    test = ConcatDataset([test_retain, test_forget])  
    print(f"Training on {len(train_retain)} images (no cats)..")

    model = train_model(train_retain, device=device)

    print("\n--Test accuracy--")
    print(f"Retain classes: {get_accuracy(model, test_retain, device):.4f}")
    print(f"Forget class: {get_accuracy(model, test_forget, device):.4f}")
    print(f"All classes: {get_accuracy(model, test, device):.4f}\n")

    torch.save(model.state_dict(), "checkpoints/retrained_model.pt")
    print("Retrained model checkpoint saved.")