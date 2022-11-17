import torch
from torch.nn import functional as F


class RegularizationLoss(torch.nn.Module):
    def __init__(self, lambda_p: float, max_steps: int = 12, prior_type="geometric"):
        super().__init__()
        p_g = torch.zeros((max_steps,))
        if prior_type == "geometric":
            not_halted = 1.0
            for k in range(max_steps):
                p_g[k] = not_halted * lambda_p
                not_halted = not_halted * (1 - lambda_p)
            p_g[-1] = 1 - p_g[:-1].sum(-1)
        elif prior_type == "normal":
            mu = 1 / lambda_p
            domain = torch.arange(1, max_steps + 1)
            density = torch.exp(-((domain - mu) ** 2))
            p_g = density / density.sum()
        assert 0.99 < p_g.sum().item() < 1.01
        self.p_g = torch.nn.Parameter(p_g, requires_grad=False)
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, p):
        p_g = self.p_g[None, : p.size(1)].expand_as(p)
        return self.kl_div(p.log(), p_g)


def kl_with_temperature(
    s_logits, t_logits, temperature, two_sided: bool = False, reduction="none"
):
    if t_logits.size(-1) > 1:
        if two_sided:
            distillation_loss = (
                kl_with_temperature(
                    s_logits, t_logits, temperature, reduction=reduction
                )
                + kl_with_temperature(
                    t_logits, s_logits, temperature, reduction=reduction
                )
            ) / 2
        else:
            distillation_loss = (
                F.kl_div(
                    torch.log_softmax(s_logits / temperature, dim=-1),
                    torch.softmax(t_logits / temperature, dim=-1),
                    reduction=reduction,
                )
                * temperature**2
            )
    else:
        distillation_loss = F.mse_loss(s_logits, t_logits, reduction=reduction)
    return distillation_loss
