import joblib
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    features: list[str]


dict_to_correspondance = {0: "not_harassement", 1: "harassement"}

app = FastAPI()
tokenizer = joblib.load("data/models/vectorizer_model_for_classifying.joblib")
model = joblib.load("data/models/random_forest_model_for_classifying.joblib")


@app.post("/predict")
async def predict(item: Item):
    data_to_predict = tokenizer.transform(item.features)
    pred = model.predict(data_to_predict)[0]
    return {"prediction": dict_to_correspondance[pred]}
