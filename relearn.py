import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from data import splits
from evalutils import get_accuracy
from utils import set_seed, get_device, load_model


def relearn(model, few_cats, train_retain, test_forget, device, steps=300, lr=1e-4,
            eval_every=10, cat_batch=16, retain_batch=64):
    
    """
    Tests how quickly a model relearns the forgotten class when given
    a fraction of the forgotten data, alongside normal retain data.
    If a method only masked the final layer but kept the features,
    the model will relearn the forgotten class very quickly.
    """ 

    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    loader_cats = DataLoader(few_cats, batch_size=cat_batch, shuffle=True, drop_last=True)
    loader_retain = DataLoader(train_retain, batch_size=retain_batch, shuffle=True, drop_last=True)

    accs = [get_accuracy(model, test_forget, device)]
    step = 0

    while step < steps:
        for (xc, yc), (xr, yr) in zip(loader_cats, loader_retain):
            model.train()
            x = torch.cat([xc, xr]).to(device)
            y = torch.cat([yc, yr]).to(device)

            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % eval_every == 0:
                accs.append(get_accuracy(model, test_forget, device))
            if step == steps:
                break

    return accs


def reauc(accs, orig_acc):
    mean_acc = (0.5 * accs[0] + sum(accs[1:-1]) + 0.5 * accs[-1]) / (len(accs) - 1)
    return mean_acc / orig_acc


if __name__ == "__main__":
    device = get_device()
    train_retain, train_forget, _, test_forget = splits()

    set_seed(42)
    few_cats = Subset(train_forget, torch.randperm(len(train_forget))[:1000].tolist())

    original = load_model("checkpoints/original_model.pt", device)
    orig_acc = get_accuracy(original, test_forget, device)
    print(f"Original (baseline): {orig_acc:.4f}")

    models = {
        "Retrained": "checkpoints/retrained_model.pt",
        "Gradient ascent": "checkpoints/unlearn_gradient_ascent.pt",
        "Bad teacher": "checkpoints/unlearn_bad_teacher.pt",
        "Fisher Method (hard)": "checkpoints/unlearn_fisher_prune_hard.pt",
        "Fisher Method (reset)": "checkpoints/unlearn_fisher_prune_reset.pt",
    }

    steps_x = list(range(0, 301, 10))
    plt.axhline(y=orig_acc, color="black", linestyle="--", label="Original (baseline)")

    for name, path in models.items():
        set_seed(42)
        accs = relearn(load_model(path, device), few_cats, train_retain, test_forget, device)

        score = reauc(accs, orig_acc)
        hit = next((s for s, a in zip(steps_x, accs) if a >= 0.75), ">300")
        print(f"{name}: start {accs[0]:.4f} | end {accs[-1]:.4f} | steps to 75%: {hit} | ReAUC: {score:.4f}")

        plt.plot(steps_x, accs, label=name)

    plt.xlabel("fine-tuning steps")
    plt.ylabel("test accuracy on cats")
    plt.title("Relearning from 1000 cats")
    plt.legend()
    plt.savefig("results/relearn.png")
    plt.close()