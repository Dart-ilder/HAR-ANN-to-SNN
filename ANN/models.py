
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.embed_size = size
        # CNN to extract features per timestep
        self.conv = nn.Conv2d(1, num_conv_filters, (1, input_shape[1]))

        self.attention = SelfAttention(
            hidden_dim=num_conv_filters,
            size=self.embed_size,
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
    