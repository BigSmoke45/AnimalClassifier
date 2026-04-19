import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 300

def load_model(model_path="best_model_b3.pth"):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(DEVICE)
    return model, classes

def predict_image(model, classes, img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return classes[probs.argmax()], probs.max() * 100

# Деградації якості
def apply_blur(img, level):
    """Розмиття — імітує розфокус камери"""
    return img.filter(ImageFilter.GaussianBlur(radius=level))

def apply_noise(img, level):
    """Шум — імітує погане освітлення / дешеву камеру"""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, level, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_darkness(img, level):
    """Темрява — імітує погане освітлення"""
    arr = np.array(img).astype(np.float32)
    arr = np.clip(arr * level, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_resolution(img, level):
    """Низька роздільність — імітує далеку камеру"""
    small = img.resize((level, level), Image.BILINEAR)
    return small.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

def run_quality_test(model, classes, data_dir, n_samples=50):
    data_dir = Path(data_dir)

    # Збираємо випадкові фото з кожного класу
    all_images = []
    for cls in classes:
        folder = data_dir / cls
        files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        sampled = random.sample(files, min(n_samples, len(files)))
        all_images.extend([(f, cls) for f in sampled])

    print(f"Тестуємо на {len(all_images)} фото\n")

    # Рівні деградації
    degradations = {
        "Оригінал":    lambda img: img,
        "Blur слабкий":   lambda img: apply_blur(img, 2),
        "Blur сильний":   lambda img: apply_blur(img, 6),
        "Шум слабкий":    lambda img: apply_noise(img, 20),
        "Шум сильний":    lambda img: apply_noise(img, 60),
        "Темрява 50%":    lambda img: apply_darkness(img, 0.5),
        "Темрява 20%":    lambda img: apply_darkness(img, 0.2),
        "Роздільність 64px": lambda img: apply_resolution(img, 64),
        "Роздільність 32px": lambda img: apply_resolution(img, 32),
        "Роздільність 16px": lambda img: apply_resolution(img, 16),
    }

    results = {}

    for deg_name, deg_fn in degradations.items():
        correct = 0
        total = len(all_images)

        for img_path, true_cls in all_images:
            img = Image.open(img_path).convert("RGB")
            degraded = deg_fn(img)
            pred_cls, confidence = predict_image(model, classes, degraded)
            if pred_cls == true_cls:
                correct += 1

        acc = correct / total * 100
        results[deg_name] = acc
        print(f"{deg_name:<25} {acc:5.1f}%  {'█' * int(acc/5)}")

    # Графік
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    accs = list(results.values())
    colors = ["green" if a >= 95 else "orange" if a >= 80 else "red" for a in accs]

    bars = ax.barh(names, accs, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy %", fontsize=12)
    ax.set_title("Точність моделі при різній якості зображень", fontsize=14)
    ax.axvline(x=95, color="green", linestyle="--", alpha=0.5, label="95% поріг")
    ax.axvline(x=80, color="orange", linestyle="--", alpha=0.5, label="80% поріг")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=11)

    ax.legend()
    plt.tight_layout()
    plt.savefig("quality_test.png", dpi=150)
    plt.show()
    print("\nЗбережено: quality_test.png")

    return results


def save_examples(data_dir, classes, n=2):
    """Зберігає приклади деградацій для кожного класу"""
    data_dir = Path(data_dir)

    degradations = {
        "original": lambda img: img,
        "blur_weak": lambda img: apply_blur(img, 2),
        "blur_strong": lambda img: apply_blur(img, 6),
        "noise_weak": lambda img: apply_noise(img, 20),
        "noise_strong": lambda img: apply_noise(img, 60),
        "dark_50": lambda img: apply_darkness(img, 0.5),
        "dark_20": lambda img: apply_darkness(img, 0.2),
        "res_64": lambda img: apply_resolution(img, 64),
        "res_32": lambda img: apply_resolution(img, 32),
        "res_16": lambda img: apply_resolution(img, 16),
    }

    output_dir = Path("quality_examples")
    output_dir.mkdir(exist_ok=True)

    for cls in classes:
        folder = data_dir / cls
        files = list(folder.glob("*.jpg")) + \
                list(folder.glob("*.jpeg")) + \
                list(folder.glob("*.png"))
        sampled = random.sample(files, min(n, len(files)))

        for img_path in sampled:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))

            # Один великий колаж для цього фото
            n_deg = len(degradations)
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()

            for ax, (deg_name, deg_fn) in zip(axes, degradations.items()):
                degraded = deg_fn(img_resized)
                ax.imshow(degraded)
                ax.set_title(deg_name, fontsize=10)
                ax.axis("off")

            fig.suptitle(f"Клас: {cls} | {img_path.name}", fontsize=13)
            plt.tight_layout()

            save_path = output_dir / f"{cls}_{img_path.stem}_examples.png"
            plt.savefig(save_path, dpi=100)
            plt.close()
            print(f"Збережено: {save_path}")

    print(f"\nВсі приклади в папці: {output_dir}/")
    
if __name__ == "__main__":
    model, classes = load_model()
    print(f"Класи: {classes}\n")
    results = run_quality_test(model, classes, "data/animals", n_samples=100)

    save_examples("data/animals", classes, n=2)