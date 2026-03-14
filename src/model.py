import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class DeepMLPHead(nn.Module):
    """
    PyTorch version of the old notebook's custom classification head.

    Do NOT add softmax here.
    CrossEntropyLoss expects raw logits.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _make_head(in_features: int, num_classes: int, head_type: str = "linear"):
    if head_type == "linear":
        return nn.Linear(in_features, num_classes)
    elif head_type == "deep_mlp":
        return DeepMLPHead(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")


def build_model(num_classes: int, head_type: str = "linear"):
    """
    For the head ablation experiment, use torchvision ResNet50 only.
    This keeps the backbone fixed and avoids timm head-shape issues.
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = _make_head(in_features, num_classes, head_type=head_type)

    backbone_name = f"torchvision:resnet50 + {head_type}"
    return model, backbone_name


def freeze_backbone(model):
    """
    Freeze all parameters, then unfreeze the classification head.
    Works for both linear and deep_mlp heads attached to model.fc.
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    n_trainable = sum(p.requires_grad for p in model.parameters())
    if n_trainable == 0:
        raise ValueError("freeze_backbone() finished but no trainable parameters were found.")


def unfreeze_last_n_blocks(model, n: int = 30):
    """
    Generic fine-tuning strategy:
    unfreeze the last N parameter tensors.
    """
    params = list(model.parameters())
    for p in params[-n:]:
        p.requires_grad = True