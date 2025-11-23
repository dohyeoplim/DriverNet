import os
import matplotlib.pyplot as plt
from typing import Optional

def save_confusion_matrix(
    cm_tensor,
    save_path: str,
    title: str = "",
    class_names: Optional[list] = None,
    figsize=(8, 8),
    dpi=150,
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = cm_tensor.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if class_names is not None:
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
    else:
        plt.xticks(range(cm.shape[1]))
        plt.yticks(range(cm.shape[0]))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
