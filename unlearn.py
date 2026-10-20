import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from model import ResNet18
from data import splits, forget_class
from evalutils import get_accuracy
from utils import set_seed, get_device, load_model

def recalibrate_bn(model, loader_retain, device):
    """
    Clear existing BatchNorm running stats (eliminating forget-class
    distribution info) and compute cumulative running_mean and running_var
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


def random_pred_generator(batch_size, num_classes=10, noise_std=0.02, device=None):
    # uniform probs + gaussian noise for dumb teacher targets
    probs = torch.full((batch_size, num_classes), 1.0 / num_classes, device=device)
    probs = probs + torch.randn_like(probs) * noise_std
    probs = probs.clamp(min=1e-6)
    return probs / probs.sum(dim=1, keepdim=True)


def bad_teacher(model, train_forget, train_retain, device,
                epochs=1, lr=0.001, batch_size=128, retain_ratio=0.3, noise_std=0.02):

    student = copy.deepcopy(model).to(device)
    good = copy.deepcopy(model).to(device).eval()

    # 30% of retain set
    num_retain = int(len(train_retain) * retain_ratio)
    retain_indices = torch.randperm(len(train_retain))[:num_retain].tolist()
    retain_subset = Subset(train_retain, retain_indices)

    # flag lu (unlearning label): 0 for retain, 1 for forget
    combined_data = []
    for idx in retain_indices:
        x, y = train_retain[idx]
        combined_data.append((x, y, 0.0))

    for i in range(len(train_forget)):
        x, y = train_forget[i]
        combined_data.append((x, y, 1.0))

    loader = DataLoader(combined_data, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(f"Unlearning on {len(combined_data)} mixed samples. Total epochs- {epochs}")

    student.train()
    for epoch in range(epochs):
        for x, _, lu in loader:
            x = x.to(device)
            lu = lu.to(device, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                target_good = F.softmax(good(x), dim=1)
                target_dumb = random_pred_generator(x.size(0), num_classes=10, noise_std=noise_std, device=device)
                overall_target = (1.0 - lu) * target_good + lu * target_dumb

            student_log_probs = F.log_softmax(student(x), dim=1)
            loss = F.kl_div(student_log_probs, overall_target, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        forget_acc = get_accuracy(student, train_forget, device)
        retain_acc = get_accuracy(student, retain_subset, device)
        print(f"Epoch {epoch + 1}/{epochs} | forget acc: {forget_acc:.4f} | retain acc: {retain_acc:.4f}")

    print("Recalibrating BatchNorm on retain subset...")
    recalibrate_bn(student, DataLoader(retain_subset, batch_size=128), device)

    full_retain_acc = get_accuracy(student, train_retain, device)
    print(f"Final full retain acc: {full_retain_acc:.4f}")
    return student.eval()


if __name__=="__main__":
    set_seed(42)
    device = get_device()
    train_retain, train_forget, _, _ = splits()

    original = load_model("checkpoints/original_model.pt", device)

    print("\n--Gradient ascent based unlearning--")
    model = grad_asc(original, train_forget, train_retain, device)
    torch.save(model.state_dict(), "checkpoints/unlearn_gradient_ascent.pt")
    print("Gradient ascent model checkpoint saved.")

    print("\n--Bad teacher based unlearning--")
    model = bad_teacher(original, train_forget, train_retain, device)
    torch.save(model.state_dict(), "checkpoints/unlearn_bad_teacher.pt")
    print("Bad teacher model checkpoint saved.")