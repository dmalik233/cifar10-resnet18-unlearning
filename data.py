import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

forget_class = 3  # cat
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

def load_cifar(root="data", augment=False):

    clean = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # augmentation only applies to the training set, not the test set
    train = datasets.CIFAR10(root, train=True, download=True, transform=aug if augment else clean)
    test = datasets.CIFAR10(root, train=False, download=True, transform=clean)

    return train, test

def class_split(dataset, forget_class=forget_class):
    targets = torch.tensor(dataset.targets)  # a plain list in CIFAR
    forget_index = torch.where(targets == forget_class)[0].tolist()
    retain_index = torch.where(targets != forget_class)[0].tolist()

    return Subset(dataset, retain_index), Subset(dataset, forget_index)

def splits(root="data", forget_class=forget_class, augment=False):
    train, test = load_cifar(root, augment)
    train_retain, train_forget = class_split(train, forget_class)
    test_retain, test_forget = class_split(test, forget_class)
    return train_retain, train_forget, test_retain, test_forget

if __name__ == "__main__":
    train_retain, train_forget, test_retain, test_forget = splits()
    print(f"Class to forget: {forget_class}")
    print(f"Train- Retain: {len(train_retain)}, Forget: {len(train_forget)}")
    print(f"Test- Retain: {len(test_retain)}, Forget: {len(test_forget)}")