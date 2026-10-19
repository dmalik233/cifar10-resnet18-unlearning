import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import splits, forget_class
from evalutils import get_accuracy
from utils import set_seed, get_device, load_model

def recalibrate_bn(model, loader_retain, device):
    """
    Wipes existing BatchNorm running stats (eliminating residual forget-class
    distribution info) and computes cumulative running_mean and running_var
    over the retain subset using forward passes with all weights frozen.
    """
    bn_layers = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            bn_layers.append((m, m.momentum))
            m.momentum = None  # Cumulative moving average (1/n) instead of EMA
            m.train()

    with torch.no_grad():
        for xr, _ in loader_retain:
            model(xr.to(device))

    for m, orig_mom in bn_layers:
        m.momentum = orig_mom

    model.eval()

def grad_asc(model, train_forget, train_retain, device, max_batch=100, stop_acc=0.01,
             lr_forget=1e-4, lr_retain=1e-3, forget_batch=64, retain_batch=64,
             retain_size=5000, check_every=5, retain_print_every=20):

    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()

    # retain subset = forget set (5k images)
    indices = torch.randperm(len(train_retain))[:retain_size].tolist()
    retain_subset = Subset(train_retain, indices)

    loader_forget = DataLoader(train_forget, batch_size=forget_batch, shuffle=True)
    loader_retain = DataLoader(retain_subset, batch_size=retain_batch, shuffle=True)

    print(f"Pre calibration train forget acc: {get_accuracy(model, train_forget, device):.4f}")
    print("Recalibrating BatchNorm on retain subset before unlearning...")
    recalibrate_bn(model, loader_retain, device)
    
    print(f"Post calibration train forget acc: {get_accuracy(model, train_forget, device):.4f}")

    # Separate SGD optimizers
    optim_forget = torch.optim.SGD(model.parameters(), lr=lr_forget)
    optim_retain = torch.optim.SGD(model.parameters(), lr=lr_retain, momentum=0.9, weight_decay=5e-4)

    batch_count = 0
    print(f"Max batches: {max_batch}, stop at forget accuracy: {stop_acc}")

    while True:
        for (xf, yf), (xr, yr) in zip(loader_forget, loader_retain):
            xf, yf = xf.to(device), yf.to(device)
            xr, yr = xr.to(device), yr.to(device)
            batch_count += 1

            # model in eval so stats don't get poisoned by cats
            model.eval()

            # Ascent on forget
            loss_f = -criterion(model(xf), yf)
            optim_forget.zero_grad()
            loss_f.backward()
            optim_forget.step()

            # Descent on retain
            loss_r = criterion(model(xr), yr)
            optim_retain.zero_grad()
            loss_r.backward()
            optim_retain.step()

            if batch_count % check_every == 0 or batch_count == max_batch:
                forget_acc = get_accuracy(model, train_forget, device)

                if batch_count % retain_print_every == 0:
                    retain_acc = get_accuracy(model, retain_subset, device)
                    print(f"Batch {batch_count} | forget acc: {forget_acc:.4f} | retain acc: {retain_acc:.4f}")

                if forget_acc <= stop_acc or batch_count == max_batch:
                    reason = f"Target forget accuracy {stop_acc} reached." if forget_acc <= stop_acc else "Max batch count reached."
                    print(reason)
                    print(f"Stopping at batch {batch_count} -> forget acc: {forget_acc:.4f}")
                    
                    full_retain_acc = get_accuracy(model, train_retain, device)
                    print(f"Final full retain acc: {full_retain_acc:.4f}")
                    return model.eval()

if __name__=="__main__":
    set_seed(42)
    device = get_device()
    train_retain, train_forget, _, _ = splits()

    original = load_model("checkpoints/original_model.pt", device)

    print("\n--Gradient ascent based unlearning--")
    model = grad_asc(original, train_forget, train_retain, device)
    torch.save(model.state_dict(), "checkpoints/unlearn_gradient_ascent.pt")
    print("Gradient ascent model checkpoint saved.")