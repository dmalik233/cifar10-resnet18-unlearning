import numpy as np
from torch.utils.data import ConcatDataset

from data import splits, class_names, forget_class
from evalutils import get_accuracy, get_probs, pred_distribution, js_div, save_confusion_matrix
from utils import get_device, load_model, set_seed


def report(m_type, model, reference, test_retain, test_forget, device):

    set_seed(42)
    print(f"\n--{m_type} model--")
    print(f"Retain accuracy: {get_accuracy(model, test_retain, device):.4f}")
    print(f"Forget accuracy: {get_accuracy(model, test_forget, device):.4f}")

    js_retain = js_div(get_probs(reference, test_retain, device), get_probs(model, test_retain, device))
    js_forget = js_div(get_probs(reference, test_forget, device), get_probs(model, test_forget, device))

    #comparison against the ideal baseline (full retrain case)
    print(f"\nJS div - {m_type} to retrained (retain): {js_retain:.4f}")
    print(f"JS div - {m_type} to retrained (forget): {js_forget:.4f}")

    distribution = pred_distribution(model, test_forget, device)
    print(f"\nDistribution of {class_names[forget_class]}s:")
    for name, p in zip(class_names, distribution):
        print(f"  {name:<11s} {p*100:5.1f}%")

    full_test = ConcatDataset([test_retain, test_forget])
    file_name = m_type.lower().replace(" ", "_")
    save_confusion_matrix(model, full_test, device, m_type, f"results/cm_{file_name}.png")

if __name__ == "__main__":

    set_seed(42)
    device = get_device()
    _, _, test_retain, test_forget = splits()
    original = load_model("checkpoints/original_model.pt", device)
    retrained = load_model("checkpoints/retrained_model.pt", device)
    grad_ascent = load_model("checkpoints/unlearn_gradient_ascent.pt", device)

    report("Original", original, retrained, test_retain, test_forget, device)
    report("Retrained", retrained, retrained, test_retain, test_forget, device)
    report("Gradient ascent", grad_ascent, retrained, test_retain, test_forget, device)
    