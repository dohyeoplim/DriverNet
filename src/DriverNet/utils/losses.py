import math
import torch
import torch.nn.functional as F

_TEACHER_ENTROPY_WEIGHT = 1.5
_STUDENT_ERROR_WEIGHT = 1.2
_EPS = 1e-8

def kl_div_with_temperature(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    t = temperature
    student_scaled = student_logits / t
    teacher_scaled = teacher_logits / t
    log_p_s = F.log_softmax(student_scaled, dim=-1)
    p_t = F.softmax(teacher_scaled, dim=-1)
    return (t * t) * F.kl_div(log_p_s, p_t.detach(), reduction="batchmean")

def get_hard_example_weights(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_entropy = -torch.sum(
            teacher_probs * torch.log(teacher_probs.clamp_min(_EPS)),
            dim=1,
        )
        max_entropy = math.log(num_classes)
        normalized_entropy = teacher_entropy / max_entropy
        ce_per_sample = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing, reduction="none")
        max_ce = ce_per_sample.max().clamp_min(_EPS)
        normalized_error = ce_per_sample / max_ce
        sample_weights = 1.0 + _TEACHER_ENTROPY_WEIGHT * normalized_entropy + _STUDENT_ERROR_WEIGHT * normalized_error
        return sample_weights

def entropy_gated_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    normalized_entropy: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        gate = (1.0 - normalized_entropy).clamp(min=0.0, max=1.0)
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    log_p_s = F.log_softmax(student_scaled, dim=-1)
    p_t = F.softmax(teacher_scaled, dim=-1)
    kd_per_sample = F.kl_div(log_p_s, p_t.detach(), reduction="none").sum(dim=-1)
    return (kd_per_sample * gate).mean()
