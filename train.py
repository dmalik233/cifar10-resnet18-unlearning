import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_cifar, splits
from model import ResNet18
from evalutils import get_accuracy
from utils import set_seed, get_device

num_epochs = 30
lr = 0.1
batch_size = 128
weight_decay = 5e-4

def train_model(train_set, num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                weight_decay=weight_decay, device=None):

    model = ResNet18().to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | loss = {epoch_loss/len(train_set):.4f}")
    return model

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Using device {device}\n")

    train, _ = load_cifar(augment=True)   # augmented for training
    _, test = load_cifar()                # clean for evaluation
    _, _, test_retain, test_forget = splits()

    model = train_model(train, device=device)

    print("\n--Test accuracy--")
    print(f"Retain classes: {get_accuracy(model, test_retain, device):.4f}")
    print(f"Forget class: {get_accuracy(model, test_forget, device):.4f}")
    print(f"All classes: {get_accuracy(model, test, device):.4f}\n")

    torch.save(model.state_dict(), "checkpoints/original_model.pt")
    print("Original model checkpoint saved.")