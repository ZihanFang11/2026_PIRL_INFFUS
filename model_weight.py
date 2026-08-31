
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class PIRL(nn.Module):
    def __init__(
        self,
        n_input,
        h_dim,
        Nlabel,
        init_theta: float = 0.01,
        ista_T: int = 5,
        rho: float = 0.1,
        eps: float = 1e-8,
        alpha: float = 1,
        gamma: float =  0.1,
        beta: float = 1,
    ):
        super(PIRL, self).__init__()
        self.T = ista_T
        self.n_view = len(n_input)
        self.d_list = n_input
        self.h_dim = h_dim
        self.n_labels = Nlabel
        self.eps = eps

        # ---- learnable thresholds ----
        self.raw_theta_z = nn.Parameter(torch.tensor(init_theta, dtype=torch.float32))
        self.raw_theta_h = nn.Parameter(torch.tensor(init_theta, dtype=torch.float32))

        # ---- unfolding parameters ----
        # H_t^{(v)} A_t^{(v)}
        self.A = nn.ModuleList([
            nn.Linear(h_dim, h_dim, bias=False)
            for _ in range(self.T)
        ])

        # Z_t C_t
        self.C = nn.ModuleList([
            nn.Linear(h_dim, h_dim, bias=False)
            for _ in range(self.T)
        ])

        # X^{(v)} D_t^{(v)}
        self.D = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(dv, h_dim, bias=False) for _ in range(self.T)
            ])
            for dv in self.d_list
        ])

        self.Q = nn.ModuleList([nn.Linear(self.n_labels, self.h_dim, bias=False)   for _ in range(self.T)])

        # ---- scalar coefficients for each layer and view ----
        self.alpha_h = nn.Parameter(torch.ones(self.T, self.n_view) * alpha)
        self.gamma_h = nn.Parameter(torch.ones(self.T, self.n_view) * gamma)
        self.beta_z = nn.Parameter(torch.ones(self.T, self.n_view) *beta)

        self.norm_z = nn.LayerNorm(h_dim)
        self.norm_h = nn.LayerNorm(h_dim)

        self.regression = Linear(h_dim, Nlabel)

        # hedge weights across unfolded layers
        self.register_buffer("pi", torch.full((self.T,), 1.0 / self.T))
        self.rho = rho

        self.fusion_logits = nn.Parameter(torch.ones( self.n_view)/ self.n_view)


    def soft_threshold_z(self, u):

        # print('z',self.raw_theta_z,torch.mean(u))
        return F.relu(u - self.raw_theta_z) - F.relu(-u - self.raw_theta_z)

    def soft_threshold_h(self, u):
        #
        return F.relu(u - self.raw_theta_h) - F.relu(-u - self.raw_theta_h)

    def fusion(self, msg_list, we):
        """
        Mask-aware fusion with global learnable view weights.

        Args:
            msg_list: list of [N, H]
            we: [N, V], availability mask or prior weight for each sample-view pair

        Returns:
            fused: [N, H]
            norm_weight: [N, V]
        """
        # global learnable weights over V views
        alpha = torch.softmax(self.fusion_logits, dim=0)  # [V]

        # combine global learnable weights with sample-wise mask / prior weights
        weight = we * alpha.unsqueeze(0)  # [N, V]

        denom = weight.sum(dim=1, keepdim=True).clamp_min(self.eps)

        fused = 0.0
        for v, msg in enumerate(msg_list):
            fused = fused + weight[:, v:v + 1] * msg

        fused = fused / denom
        return fused
    @torch.no_grad()


    def _hedge_update(self, layer_losses):
        losses = layer_losses.detach().to(self.pi.device)
        new_pi = self.pi * torch.exp(-self.rho * losses)
        new_pi = new_pi / new_pi.sum().clamp_min(1e-12)
        self.pi.copy_(new_pi)

    def _build_initial_Z(self, Xs, we, teacher_label=None, teacher_forcing=0.0):
        """
        Build initial Z_0 from mask-aware multi-view feature fusion + label guidance.
        Q is view-specific: Q[t][v].
        """
        init_feature_msgs = []
        for v in range(self.n_view):
            view_term = self.D[v][0](Xs[v])  # [n, h]
            init_feature_msgs.append(view_term)

        Z_feat = self.fusion(init_feature_msgs, we)  # [n, h]

        logits0 = self.regression(torch.relu(Z_feat))
        pre_label = torch.sigmoid(logits0)

        if teacher_label is not None and teacher_forcing > 0.0:
            guide_label = teacher_forcing * teacher_label.float() + (1.0 - teacher_forcing) * pre_label
        else:
            guide_label = pre_label

        init_label_msgs = []
        for v in range(self.n_view):
            label_msg_v = self.Q[0](guide_label)  # [n, h]
            init_label_msgs.append(label_msg_v)

        Z_label = self.fusion(init_label_msgs, we)  # [n, h]

        Z0 = self.soft_threshold_z(self.norm_z(Z_feat + Z_label))
        return Z0, pre_label

    def _run_impl(self, Xs, lap, we, teacher_label=None, teacher_forcing=0.0):
        """
        Xs: list of tensors, each [n, d_v]
        lap: list of sparse/dense Laplacian, each [n, n]
        we: [n, V], availability mask or weights
        """
        n = Xs[0].shape[0]
        device = Xs[0].device

        if we is None:
            we = torch.ones(n, self.n_view, device=device, dtype=Xs[0].dtype)

        # initial Z
        Z, pre_label = self._build_initial_Z(
            Xs=Xs,
            we=we,
            teacher_label=teacher_label,
            teacher_forcing=teacher_forcing
        )

        # initialize H^{(v)} = Z
        H = {v: Z.clone() for v in range(self.n_view)}

        layer_probs = []

        for t in range(self.T):
            # ========= Step 1: update H^{(v)} =========
            for v in range(self.n_view):
                mask_v = we[:, v:v + 1]  # [n,1]

                # H_t^{(v)} A_t^{(v)}
                h_self = self.A[t](H[v])  # [n,h]
                view_term = self.D[v][t](Xs[v])  # X^{(v)} D_t^{(v)}
                # L^{(v)} H_t^{(v)}
                lap_h = lap[v].mm(H[v]) if lap[v].is_sparse else torch.matmul(lap[v], H[v])

                alpha_tv = F.softplus(self.alpha_h[t, v])
                gam_tv = F.softplus(self.gamma_h[t, v])

                # masked H update:
                # M^(v) * (H A +X^{(v)} D_t^{(v)}  - lambda L H) + gamma Z
                H_tmp = mask_v * (h_self +view_term -  gam_tv* lap_h) +  alpha_tv* Z
                H[v] = self.soft_threshold_h(self.norm_h(H_tmp))

            # ========= Step 2: build label guidance =========
            if teacher_label is not None and teacher_forcing > 0.0:
                guide_label = teacher_forcing * teacher_label.float() + (1.0 - teacher_forcing) * pre_label
            else:
                guide_label = pre_label



            # ========= Step 3: update Z =========
            z_msgs = []
            for v in range(self.n_view):
                # masked aggregation term:
                beta_tv = F.softplus(self.beta_z[t, v])  # rho_t^{(v)}
                msg_v =   beta_tv  * H[v]
                z_msgs.append(msg_v)
            Z_tmp = self.C[t](Z)+  self.Q[t](guide_label)+self.fusion(z_msgs, we)


            Z = self.soft_threshold_z(self.norm_z(Z_tmp))

            logits_t = self.regression(torch.relu(Z))   # [n,C]
            pre_label = torch.sigmoid(logits_t)         # [n,C]
            layer_probs.append(pre_label)

        # ========= Hedge aggregation across layers =========
        pi = self.pi.to(layer_probs[0].device).view(self.T, 1, 1)  # [T,1,1]
        layer_probs_t = torch.stack(layer_probs, dim=0)            # [T,n,C]
        final_prob = (pi * layer_probs_t).sum(dim=0)               # [n,C]
        # print('h', self.raw_theta_h,self.raw_theta_z,Z)
        return final_prob, layer_probs

    def forward(self, mul_X, lap, we=None, Label=None, teacher_forcing=0.0):
        return self._run_impl(
            Xs=mul_X,
            lap=lap,
            we=we,
            teacher_label=Label,
            teacher_forcing=teacher_forcing
        )

    @torch.no_grad()
    def test(self, mul_X, lap, we=None):
        return self._run_impl(
            Xs=mul_X,
            lap=lap,
            we=we,
            teacher_label=None,
            teacher_forcing=0.0
        )