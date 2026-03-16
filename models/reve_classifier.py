import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class REVEClassifier(nn.Module):
    def __init__(self, cfg, num_classes=3):
        super().__init__()
        self.pos_bank = AutoModel.from_pretrained(
            cfg.model.positions,
            trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            cfg.model.pretrained,
            trust_remote_code=True
        )

        encoder_dim = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, cfg.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(cfg.model.hidden_dim, num_classes)
        )

        self.ch_names = None

    def set_channel_info(self, ch_names):
        self.ch_names = ch_names

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def apply_lora(self, lora_cfg):
        config = LoraConfig(
            r=lora_cfg.lora_r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=["query", "value"],
            bias="none"
        )
        self.encoder = get_peft_model(self.encoder, config)
        self.encoder.print_trainable_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch, 62, 800) float tensor
        Returns:
            logits: (batch, num_classes)
        """
        positions = self.pos_bank(self.ch_names)                    # (62, 3)
        positions = positions.expand(x.size(0), -1, -1)             # (batch, 62, 3)
        outputs = self.encoder(x, positions)
        hidden = outputs.last_hidden_state[:, 0, :]                 # CLS token
        return self.classifier(hidden)
