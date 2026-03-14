import numpy as np
import torch
from PIL import Image

from .model import build_model


@torch.no_grad()
def predict_one_image(model, img_path: str, eval_tfms, class_names, device):
    """
    Predict a single image.
    Returns:
        predicted_label, confidence, probability_array
    """
    img = Image.open(img_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))

    return class_names[pred], float(probs[pred]), probs


def load_model_for_inference(checkpoint_path: str, class_names, device, head_type: str = "linear"):
    """
    Build model and load checkpoint for inference.
    """
    model, backbone_name = build_model(
        num_classes=len(class_names),
        head_type=head_type,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()
    return model, backbone_name