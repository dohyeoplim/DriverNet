import torch
import torch.nn.functional as F
import math

def kl_div_with_temperature(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    T = temperature
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return (T * T) * F.kl_div(log_p_s, p_t.detach(), reduction="batchmean")

def get_hard_example_weights(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    teacher_entropy_weight: float,
    student_error_weight: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_entropy = -torch.sum(teacher_probs * torch.log(teacher_probs.clamp_min(1e-8)), dim=1)
        max_entropy = math.log(num_classes)
        normalized_entropy = teacher_entropy / max_entropy

        ce_per_sample = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing, reduction='none')
        max_ce = ce_per_sample.max()
        normalized_student_error = ce_per_sample / (max_ce + 1e-8)

        sample_weights = 1.0 + teacher_entropy_weight * normalized_entropy + student_error_weight * normalized_student_error

        return sample_weights.detach()

def entropy_gated_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    normalized_entropy: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        conf_gate = (1.0 - normalized_entropy).clamp(min=0.0)

    kd_per_sample = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1), F.softmax(teacher_logits / temperature, dim=-1), reduction="none").sum(dim=-1)

    return (kd_per_sample * conf_gate).mean()
