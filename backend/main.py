import csv
import io
import math
import re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat

app = FastAPI(title="Ayurvedic Skin Disease Image Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

records = []
with open("data.csv", mode="r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        records.append(row)

CONDITION_PROFILES = {
    "Acne and Pimples": {"r_ratio": 0.42, "g_ratio": 0.30, "b_ratio": 0.28, "brightness": 145, "contrast": 45},
    "Dry and Flaky Skin": {"r_ratio": 0.36, "g_ratio": 0.33, "b_ratio": 0.31, "brightness": 175, "contrast": 35},
    "Eczema and Dermatitis": {"r_ratio": 0.44, "g_ratio": 0.29, "b_ratio": 0.27, "brightness": 135, "contrast": 50},
    "Hyperpigmentation and Dark Spots": {"r_ratio": 0.39, "g_ratio": 0.32, "b_ratio": 0.29, "brightness": 95, "contrast": 40},
    "Fungal Ringworm Infection": {"r_ratio": 0.45, "g_ratio": 0.28, "b_ratio": 0.27, "brightness": 130, "contrast": 55},
    "Psoriasis Plaques": {"r_ratio": 0.38, "g_ratio": 0.32, "b_ratio": 0.30, "brightness": 180, "contrast": 60},
    "Sunburn and Heat Rash": {"r_ratio": 0.47, "g_ratio": 0.27, "b_ratio": 0.26, "brightness": 150, "contrast": 30},
    "Oily Congested Pores": {"r_ratio": 0.37, "g_ratio": 0.33, "b_ratio": 0.30, "brightness": 160, "contrast": 28},
}

def extract_image_features(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    w, h = img.size
    center_crop = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    stat = ImageStat.Stat(center_crop)

    r_mean, g_mean, b_mean = stat.mean
    total = (r_mean + g_mean + b_mean) or 1.0
    brightness = sum(stat.mean) / 3.0
    contrast = sum(stat.stddev) / 3.0

    return {
        "r_ratio": r_mean / total,
        "g_ratio": g_mean / total,
        "b_ratio": b_mean / total,
        "brightness": brightness,
        "contrast": contrast,
    }

def compute_visual_similarity(feat, prof):
    color_dist = math.sqrt(
        ((feat["r_ratio"] - prof["r_ratio"]) * 10) ** 2 +
        ((feat["g_ratio"] - prof["g_ratio"]) * 10) ** 2 +
        ((feat["b_ratio"] - prof["b_ratio"]) * 10) ** 2
    )
    bright_dist = abs(feat["brightness"] - prof["brightness"]) / 100.0
    contrast_dist = abs(feat["contrast"] - prof["contrast"]) / 50.0
    total_dist = color_dist + bright_dist + contrast_dist
    return max(0.0, 1.0 / (1.0 + total_dist))

@app.post("/analyze-image")
async def analyze_skin_image(
    file: UploadFile = File(...),
    skin_type: str = Form("Oily"),
    notes: str = Form("")
):
    image_bytes = await file.read()
    features = extract_image_features(image_bytes)
    note_words = set(re.findall(r"[a-z]+", notes.lower()))

    scored = []
    for row in records:
        cond = row["condition"]
        prof = CONDITION_PROFILES.get(cond)
        if not prof:
            continue

        score = compute_visual_similarity(features, prof)

        if row["skin_type"].lower() == skin_type.lower():
            score += 0.12

        symptom_words = set(re.findall(r"[a-z]+", row["symptoms"].lower()))
        if note_words & symptom_words:
            score += 0.15

        scored.append((min(score, 0.98), row))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, row in scored[:2]:
        results.append({
            "condition": row["condition"],
            "ayurvedic_name": row["ayurvedic_name"],
            "dosha": row["dosha"],
            "confidence": round(score * 100, 1),
            "herbs": row["herbs"],
            "formulation": row["formulation"],
            "usage": row["usage"],
            "diet_advice": row["diet_advice"],
            "precaution": row["precaution"],
        })

    return {
        "image_metrics": {
            "redness_index": round(features["r_ratio"] * 100, 1),
            "brightness": round(features["brightness"], 1),
            "texture_variance": round(features["contrast"], 1),
        },
        "recommendations": results,
    }