import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, device, reduction: str = "mean",
                use_pair_mask: bool = True,
                zero_diag_neg: bool = True):
        super(Loss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.reduction = reduction
        self.use_pair_mask = use_pair_mask
        self.zero_diag_neg = zero_diag_neg
    def wmse_loss(self, input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        # assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        # assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res

    def corr_criterion(
            self,
            P,
            Y,
            G,
            C_pos,
            C_neg,
    ):
        """
        Args:
            P: prediction probabilities, shape [N, C], value range [0, 1]
            Y: ground-truth multi-label matrix, shape [N, C], 0/1
               Y=1 means positive label.
               Y=0 may mean negative or missing, which should be distinguished by G.
            G: observed-label mask, shape [N, C]
               G=1 means this label is observed, including positive and explicit negative.
               G=0 means this label is missing / unannotated.
            C_pos: positive correlation matrix, shape [C, C]
            C_neg: negative/exclusive correlation matrix, shape [C, C]

        Returns:
            loss_pos, loss_neg
        """

        if P.dim() != 2:
            raise ValueError(f"P must be [N, C], but got shape {P.shape}")
        if Y.dim() != 2:
            raise ValueError(f"Y must be [N, C], but got shape {Y.shape}")
        if G.dim() != 2:
            raise ValueError(f"G must be [N, C], but got shape {G.shape}")
        if C_pos.dim() != 2 or C_neg.dim() != 2:
            raise ValueError("C_pos and C_neg must be [C, C] matrices")
        if P.shape != Y.shape:
            raise ValueError("P and Y shape mismatch")
        if P.shape != G.shape:
            raise ValueError("P and G shape mismatch")
        if P.size(1) != C_pos.size(0) or P.size(1) != C_pos.size(1):
            raise ValueError("P and C_pos shape mismatch")
        if P.size(1) != C_neg.size(0) or P.size(1) != C_neg.size(1):
            raise ValueError("P and C_neg shape mismatch")

        N, C = P.shape
        device = P.device
        dtype = P.dtype
        eps = 1e-8

        Y = Y.to(device=device, dtype=dtype)
        G = G.to(device=device, dtype=dtype)
        C_pos = C_pos.to(device=device, dtype=dtype)
        C_neg = C_neg.to(device=device, dtype=dtype)

        # remove self-correlation
        C_pos = C_pos.clone()
        C_pos.fill_diagonal_(0.0)

        if self.zero_diag_neg:
            C_neg = C_neg.clone()
            C_neg.fill_diagonal_(0.0)

        # ============================================================
        # Positive correlation activation loss
        #
        # L_pos = - sum_{i,j,k}
        #          Y_ij * G_ij * C_pos[j,k] * (1 - G_ik) * log(P_ik)
        # ============================================================

        # source label j should be observed positive label
        Yj = Y.unsqueeze(2)  # [N, C, 1]
        Gj = G.unsqueeze(2)  # [N, C, 1]

        # target label k should be unobserved / missing
        unknown_k = (1.0 - G).unsqueeze(1)  # [N, 1, C]

        Pk = P.unsqueeze(1)  # [N, 1, C]
        Cpos = C_pos.unsqueeze(0)  # [1, C, C]

        pos_weight = Yj * Gj * Cpos * unknown_k  # [N, C, C]

        pos_term = -pos_weight * torch.log(Pk.clamp(min=eps))

        pos_norm = pos_weight.sum().clamp(min=1.0)
        loss_pos = pos_term.sum() / pos_norm

        # ============================================================
        # Negative correlation suppression loss
        #
        #
        # L_neg = sum_{i,j,k}
        #          Y_ij * G_ij * (1 - Y_ik)
        #          * C_neg[j,k] * sg(P_ij) * P_ik
        #
        # ============================================================

        Pj = P.unsqueeze(2)  # [N, C, 1]
        Pk = P.unsqueeze(1)  # [N, 1, C]

        Yj = Y.unsqueeze(2)  # [N, C, 1]
        Yk = Y.unsqueeze(1)  # [N, 1, C]

        Gj = G.unsqueeze(2)  # [N, C, 1]

        Cneg = C_neg.unsqueeze(0)  # [1, C, C]

        neg_weight = Yj * Gj * (1.0 - Yk) * Cneg  # [N, C, C]

        neg_term = neg_weight * Pj.detach() * Pk

        neg_norm = neg_weight.sum().clamp(min=1.0)
        loss_neg = neg_term.sum() / neg_norm

        return loss_pos, loss_neg
