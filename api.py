import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

from utils import load_processed_data, recommend_for_user

# ---------- Глобальные переменные, заполняемые при старте ----------
model = None
train_matrix = None
user_to_idx = {}
idx_to_user = {}
idx_to_product = {}
product_info_df = None          # DataFrame с информацией о товарах
product_info_dict = {}          # Быстрый словарь product_id → {name, aisle, ...}

def load_resources():
    """Загружает модель, матрицы и маппинги один раз при запуске."""
    global model, train_matrix, user_to_idx, idx_to_user, idx_to_product
    global product_info_df, product_info_dict

    # 1. Данные из processed
    data = load_processed_data("data/processed")
    train_matrix = data["train_matrix"]
    mappings = data["mappings"]
    user_to_idx = mappings["user_to_idx"]
    idx_to_user = mappings["idx_to_user"]
    idx_to_product = mappings["idx_to_product"]
    product_info_df = data["product_info"]

    # 2. Модель
    with open("model/als_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 3. Быстрый словарь product_id → название/категория
    product_info_dict = (
        product_info_df.set_index("product_id")
        .to_dict(orient="index")
    )

    print("✅ Модель и данные загружены, сервис готов.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """События запуска/остановки FastAPI (современный способ)."""
    load_resources()
    yield

app = FastAPI(title="Recommendation API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/recommend/{user_id}")
def get_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50, description="Число рекомендаций")
):
    """
    Возвращает персональные рекомендации для пользователя.

    - **user_id** – исходный ID пользователя из датасета
    - **k** – количество рекомендаций (по умолчанию 10)
    """
    # Проверяем, известен ли пользователь
    if user_id not in user_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Пользователь {user_id} не найден в тренировочных данных"
        )

    # Преобразуем во внутренний индекс
    user_idx = user_to_idx[user_id]

    # Получаем рекомендации: списки индексов товаров и их скоров
    item_indices, scores = recommend_for_user(
        model=model,
        user_id=user_idx,
        train_matrix=train_matrix,
        k=k
    )

    # Преобразуем индексы обратно в product_id и собираем информацию
    recommendations = []
    for item_idx, score in zip(item_indices, scores):
        product_id = idx_to_product.get(item_idx)
        info = product_info_dict.get(product_id, {})
        recommendations.append({
            "product_id": int(product_id),
            "product_name": info.get("product_name", "Unknown"),
            "aisle": info.get("aisle", ""),
            "department": info.get("department", ""),
            "score": round(float(score), 5)
        })

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }


@app.get("/health")
def health():
    """Проверка работоспособности."""
    return {"status": "ok"}