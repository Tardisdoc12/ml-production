################################################################################
# filename: main.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 12/05,2025
################################################################################

import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Trainer

from core_function import compute_metrics, get_dataset, get_model, trainer
from preprocessing import get_train_val_test_from_dataframe, treat_data

################################################################################


def treat_dataset():
    dataset = get_dataset("data/raw/cyberbullying_tweets.csv")
    _, _, _ = treat_data(dataset)


def train_model(compute_metrics_str: str = "all"):
    dataframe = pd.read_csv("data/processed/processed_tweet.csv")
    train_dataset, _, test_dataset = get_train_val_test_from_dataframe(dataframe)
    train_dataset_clean = train_dataset.remove_columns(
        ["tweet_text", "cyberbullying_type"]
    )
    print(train_dataset_clean)
    train_dataset_clean = train_dataset_clean.map(str_to_tensor)
    test_dataset_clean = test_dataset.remove_columns(
        ["tweet_text", "cyberbullying_type"]
    )
    test_dataset_clean = test_dataset_clean.map(str_to_tensor)
    model_name = "vinai/bertweet-base"
    if compute_metrics_str == "all":
        trainer(model_name, train_dataset_clean, test_dataset_clean, "Train")
    else:
        trainer(
            model_name,
            train_dataset_clean,
            test_dataset_clean,
            "Train",
            callback_metrics=compute_metrics,
        )


def test_model(epoch: int = 7155, path_save: str = "figures/results_distilbert.json"):
    path_output = "Train/checkpoint-" + str(epoch)
    tokenizer, model = get_model("", path_output)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    dataframe = pd.read_csv("data/processed/processed_tweet.csv")
    _, val_dataset, _ = get_train_val_test_from_dataframe(dataframe)
    val_dataset_clean = val_dataset.remove_columns(["tweet_text", "cyberbullying_type"])
    val_dataset_clean = val_dataset_clean.map(str_to_tensor)

    results = trainer.predict(val_dataset_clean)
    y_pred = results.predictions.argmax(axis=1)
    y_true = results.label_ids
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    print(
        "🧪 Prédit 0    🧪 Prédit 1\n",
        f"🎯 Réel 0 (négatif)    ✅ TN = {matrix[0][0]}    ❌ FP = {matrix[0][1]}\n",
        f"🎯 Réel 1 (positif)    ❌ FN = {matrix[1][0]}    ✅ TP = {matrix[1][1]}",
    )
    result = {"metrics": report, "confusion_matrix": matrix.tolist()}
    with open(path_save, "w") as file:
        json.dump(result, file)


def str_to_tensor(example, max_len=128):
    for key in ["input_ids", "attention_mask"]:
        ids = list(map(int, example[key].strip("[]").split()))

        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids += [0] * (max_len - len(ids))  # padding

        example[key] = torch.tensor(ids)
    return example


test_model()

################################################################################
# End of File
################################################################################
