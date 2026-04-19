# Animal Classifier

Система розпізнавання тварин на основі EfficientNet з підтримкою двох моделей.

## 🚀 Demo
[Hugging Face Space](https://huggingface.co/spaces/bigsmoke45/animal-classifier)

## 📊 Моделі

| Модель | Параметри | Точність | Розмір |
|--------|-----------|----------|--------|
| EfficientNet-B0 | 4M | 97.5% | 15MB |
| EfficientNet-B3 | 12M | 98.8% | 41MB |

## Класи
- 🐕 Собака
- 🐴 Кінь  
- 🐘 Слон
- 🐱 Кіт

## Технології
- PyTorch + EfficientNet (навчання)
- ONNX Runtime (inference)
- FastAPI (backend)
- Hugging Face Spaces (деплой)

## Стійкість до деградації якості

| Тип деградації | B0 | B3 |
|----------------|----|----|
| Оригінал | 98.3% | 98.5% |
| Blur сильний | 85.8% | 81.5% |
| Шум сильний | 90.5% | 90.8% |
| Темрява 20% | 94.8% | 98.0% |
| Роздільність 16px | 67.0% | 68.0% |

## Архітектура

Датасет (Animals-10) https://www.kaggle.com/datasets/alessiocorrado99/animals10/data

↓

Fine-tuning EfficientNet (PyTorch)

↓

ONNX Export

↓

FastAPI Server + Web UI

## 🚀 Запуск локально

```bash
pip install fastapi uvicorn onnxruntime pillow numpy python-multipart
uvicorn server:app --reload
```

Відкрити http://localhost:8000

## 📁 Структура

├── server.py          # FastAPI backend

├── static/

│   └── index.html     # Web інтерфейс

├── model_b0_single.onnx  # EfficientNet-B0

├── model_b3_single.onnx  # EfficientNet-B3

├── Dockerfile

└── requirements.txt
