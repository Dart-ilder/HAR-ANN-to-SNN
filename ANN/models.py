
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch.functional as SF 
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_absolute_error



class SelfAttention(nn.Module):
    """
    Implements multi-hop self-attention (Linformer style) producing embeddings and attention matrices.
    """
    def __init__(
        self,
        hidden_dim: int,
        size: int,
        num_hops: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.size = size
        self.num_hops = num_hops

        self.W1 = nn.Parameter(torch.Tensor(size, hidden_dim))
        self.W2 = nn.Parameter(torch.Tensor(num_hops, size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, _ = x.size()
        x_inv = x.clone().permute(0, 2, 1)
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        W1, W2 = torch.tile(W1, [B, 1, 1]), torch.tile(W2, [B, 1, 1])
        score = torch.tanh(torch.matmul(self.W1, x_inv))
        weights = F.softmax(torch.matmul(self.W2, score), dim=-1)
        embed = torch.matmul(weights, x)
        flat = embed.view(B, -1)

        return flat, weights



class ConvAttention(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        num_labels: int,
        num_conv_filters: int,
        size: int,
        num_hops=10,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
       
        self.save_hyperparameters()
        # CNN to extract features per timestep
        self.conv = nn.Conv2d(1, num_conv_filters, (1, input_shape[1]))

        self.attention = SelfAttention(
            hidden_dim=num_conv_filters,
            size=size,
            num_hops=num_hops,
        )

        # Classifier: flattened attention embedding to num_labels
        self.fc = nn.Sequential(
            nn.Linear(num_hops * num_conv_filters, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        B = x.size(0)
        # x: (batch, seq_len, feat_dim, 1)
        x = x.permute(0, 3, 1, 2)                     # (batch,1,seq_len,feat_dim)
        x = self.conv(x)                              # (batch,filters,seq_len,1)
        x = x.squeeze(3).permute(0, 2, 1)             # (batch,seq_len,filters)
        attn_out, _ = self.attention(x)               # (batch,lstm_units*num_hops)
          
        return self.fc(attn_out)  

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy_score(y.cpu(), preds.cpu()), prog_bar=True)
        self.log('val_recall', recall_score(y.cpu(), preds.cpu(), average='macro', zero_division=0))
        self.log('val_f1', f1_score(y.cpu(), preds.cpu(), average='macro', zero_division=0))
        return {'preds': preds.cpu(), 'targets': y.cpu()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    


class ConvConv(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        num_labels: int,
        num_conv_filters: int,
        size: int,
        num_hops=10,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
       
        self.save_hyperparameters()
        # CNN to extract features per timestep
        self.conv = nn.Conv2d(1, num_conv_filters, (1, input_shape[1]))
        self.temp_conv = nn.Sequential(
            nn.Conv2d(num_conv_filters, 32, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)), 
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0),), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)), 
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)), 
            nn.ReLU()            
        )

        self.fc = nn.Sequential(
            nn.Linear(64*125, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        B = x.size(0)
        # x: (batch, seq_len, feat_dim, 1)
        x = x.permute(0, 3, 1, 2)                     # (batch,1,seq_len,feat_dim)
        x = self.conv(x)                              # (batch,filters,seq_len,1)
        x = self.temp_conv(x)
        x = nn.Flatten()(x)            
          
        return self.fc(x)  

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy_score(y.cpu(), preds.cpu()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy_score(y.cpu(), preds.cpu()), prog_bar=True)
        self.log('val_recall', recall_score(y.cpu(), preds.cpu(), average='macro', zero_division=0))
        self.log('val_f1', f1_score(y.cpu(), preds.cpu(), average='macro', zero_division=0))
        return {'preds': preds.cpu(), 'targets': y.cpu()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



# --------------------------------- helpers ----------------------------------
@torch.no_grad()
def collect_max_act(model: nn.Module, loader: torch.utils.data.DataLoader, device='cuda', 
                    num_batches: int = 100):
    """
    Run `num_batches` forward passes and record the maximum absolute activation
    per output channel of every Conv/Linear layer. Returns a dict whose keys
    are layer names in *forward order*:  ('conv', 'temp_conv.0', ... , 'fc.0').
    """
    model.eval().to(device)
    max_act = {}
    hooks = []

    # register hooks to capture outputs
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            max_act[name] = None
            def _hook(module, inp, out, n=name):
                a = out.detach().abs().amax(dim=0)
                nonlocal max_act
                max_act[n] = a if max_act[n] is None else torch.maximum(max_act[n], a)
            hooks.append(m.register_forward_hook(_hook))

    it = iter(loader)
    for _ in range(min(num_batches, len(loader))):
        x, _ = next(it);  model(x.to(device))

    for h in hooks: h.remove()
    return {k: v.cpu() for k, v in max_act.items()}

# ---------------------------- SSPC-BM neuron --------------------------------
class SSPCBlock(nn.Module):
    """
    Single-spike phase-coding neuron with base manipulation (Alg. 1 of the paper).
    """
    def __init__(self, weight, bias, base_in, base_out, vth_shift, max_in, max_out):
        super().__init__()
        # channel-wise re-scale (Eq. 2, Sec. II-C)
        w = weight * (max_in.view(1, -1, 1, 1) / max_out.view(-1, 1, 1, 1))
        b = bias / max_out
        self.W = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)
        self.register_buffer('Vth', torch.full((w.size(0),), vth_shift))
        self.register_buffer('v',   torch.zeros(w.size(0)))
        self.register_buffer('fired', torch.zeros(w.size(0), dtype=torch.bool))
        self.base_in, self.base_out = base_in, base_out

    def forward(self, x):                                 # x already spikes (0/1)
        if self.fired.all():                              # early exit after 1st spike
            return torch.zeros(x.size(0), self.W.size(0), device=x.device)
        self.v.mul_(self.base_in)
        self.v.add_(torch.einsum('o i h w, b i h w -> b o', self.W, x) + self.b)
        s = (self.v >= self.Vth) & ~self.fired            # newly fired this phase
        self.fired |= s
        self.Vth.mul_(self.base_in / self.base_out)       # dynamic threshold
        return s.float()

# -------------------------- SpikingConvConv wrapper -------------------------
class SpikingConvConv(pl.LightningModule):
    """
    Inference-only wrapper around a *trained* ConvConv using SSPC-BM.
    """
    def __init__(self, convconv: ConvConv, max_act: dict, T=16, Q=1.3):
        super().__init__()
        self.T, self.Q = T, Q
        self.vth_shift = (Q + 1) / (2 * Q)                # rounding threshold
        self.build_from(convconv, max_act)

    def build_from(self, net, max_act):
        """
        net …… trained ConvConv
        max_act …… dict from collect_max_act()
        """
        self.blocks = nn.ModuleList()
        # 1st Conv2d (base 2  →  Q)
        self.blocks.append(
            SSPCBlock(net.conv.weight, net.conv.bias, 2.0, self.Q,
                      self.vth_shift, max_act['input'], max_act['conv'])
        )
        # temp_conv[0]  (Q → Q)
        self.blocks.append(
            SSPCBlock(net.temp_conv[0].weight, net.temp_conv[0].bias,
                      self.Q, self.Q, self.vth_shift,
                      max_act['conv'], max_act['temp_conv.0'])
        )
        # temp_conv[1]  (Q → Q)
        self.blocks.append(
            SSPCBlock(net.temp_conv[1].weight, net.temp_conv[1].bias,
                      self.Q, self.Q, self.vth_shift,
                      max_act['temp_conv.0'], max_act['temp_conv.1'])
        )
        # temp_conv[3]  (Q → Q)
        self.blocks.append(
            SSPCBlock(net.temp_conv[3].weight, net.temp_conv[3].bias,
                      self.Q, self.Q, self.vth_shift,
                      max_act['temp_conv.1'], max_act['temp_conv.3'])
        )
        # temp_conv[4]  (Q → Q)
        self.blocks.append(
            SSPCBlock(net.temp_conv[4].weight, net.temp_conv[4].bias,
                      self.Q, self.Q, self.vth_shift,
                      max_act['temp_conv.3'], max_act['temp_conv.4'])
        )
        # fc[0]  (Q → Q)
        self.blocks.append(
            SSPCBlock(net.fc[0].weight.T.view(128, 64, 1, 125),
                      net.fc[0].bias, self.Q, self.Q, self.vth_shift,
                      max_act['temp_conv.4'], max_act['fc.0'])
        )
        # final linear stays analog (mean membrane potential over T)
        self.fc_out = nn.Linear(128, net.fc[2].out_features, bias=True)
        self.fc_out.load_state_dict(net.fc[2].state_dict())

    # ---------- forward over T phases ----------
    def forward(self, x):
        B = x.size(0)
        spikes = SF.rate_to_binary(x, num_steps=self.T)      # (B,T,seq,feat,1)
        logits = torch.zeros(B, self.fc_out.out_features, device=x.device)

        for t in range(self.T):
            s = spikes[:, t].permute(0, 3, 1, 2)            # (B,1,seq,feat)
            for blk in self.blocks[:5]:
                s = blk(s.unsqueeze(-1) if s.dim() == 4 else s)  # keep 4-D conv shape
            flat = s.view(B, -1)                            # (B,64*125)
            s_fc = self.blocks[5](flat.unsqueeze(-1).unsqueeze(-1))  # to (B,128)
            logits += self.fc_out(s_fc)
        return logits / self.T                              # average across phases
