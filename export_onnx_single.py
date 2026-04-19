import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import io

DEVICE = torch.device("cpu")
BASE_DIR = Path(__file__).parent
def export_single(model_path, onnx_path, architecture, img_size):
    print(f"Експортуємо {model_path} → {onnx_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    classes = checkpoint["classes"]

    model = architecture(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    # Старий спосіб — все в один файл
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            dynamo=False,  # ← ключовий параметр
        )

    size_mb = Path(onnx_path).stat().st_size / 1024 / 1024
    print(f"✅ Збережено: {onnx_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    export_single(
        "best_model.pth",
        BASE_DIR / "model_b0_single.onnx",
        models.efficientnet_b0,
        224
    )

    export_single(
        "best_model_b3.pth",
        BASE_DIR / "model_b3_single.onnx",
        models.efficientnet_b3,
        300
    )

    print("\nГотово!")