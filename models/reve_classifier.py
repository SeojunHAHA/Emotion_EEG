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

        encoder_dim = self.encoder.config.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, cfg.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(cfg.model.hidden_dim, num_classes)
        )

        self.ch_names = None
        self.valid_ch_idx = None

    def set_channel_info(self, ch_names):
        # pos_bank에 있는 채널만 필터링
        valid_names = []
        valid_idx = []
        for i, ch in enumerate(ch_names):
            if ch in self.pos_bank.mapping:
                valid_names.append(ch)
                valid_idx.append(i)
        self.ch_names = valid_names
        self.valid_ch_idx = valid_idx
        print(f"Using {len(valid_names)}/{len(ch_names)} channels")

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
            target_modules=["to_qkv", "to_out"],
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
        x = x[:, self.valid_ch_idx, :]                              # (batch, valid_ch, time)
        positions = self.pos_bank(self.ch_names)                    # (valid_ch, 3)
        positions = positions.expand(x.size(0), -1, -1)             # (batch, valid_ch, 3)
        layer_outputs = self.encoder(x, positions, return_output=True)  # list of (batch, ch*patches, embed_dim)
        hidden = layer_outputs[-1].mean(dim=1)                         # 마지막 레이어 mean pooling → (batch, embed_dim)
        return self.classifier(hidden)
