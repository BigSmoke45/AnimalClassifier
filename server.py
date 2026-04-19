import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import io

CLASSES = ['cane', 'cavallo', 'elefante', 'gatto']
TRANSLATE = {
    'cane': 'Собака',
    'cavallo': 'Кінь',
    'elefante': 'Слон',
    'gatto': 'Кіт',
}

IMG_SIZES = {
    "b0": 224,
    "b3": 300,
}

MODEL_PATHS = {
    "b0": "model_b0_single.onnx",
    "b3": "model_b3_single.onnx",
}

# Завантажуємо обидві сесії при старті
sessions = {}
for key, path in MODEL_PATHS.items():
    sessions[key] = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )
print("Моделі завантажені!")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess(img: Image.Image, img_size: int) -> np.ndarray:
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("b0")):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_size = IMG_SIZES.get(model, 224)
    tensor = preprocess(img, img_size)

    session = sessions[model]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    probs = softmax(outputs[0][0])

    results = [
        {"class": cls, "confidence": float(prob) * 100}
        for cls, prob in zip(CLASSES, probs)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "prediction": results[0]["class"],
        "confidence": results[0]["confidence"],
        "all": results,
    }