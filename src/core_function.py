################################################################################
# filename: core_function.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 12/05,2025
################################################################################

# import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)
from transformers.training_args import TrainingArguments

################################################################################


def get_dataset(path: str) -> dict:
    """
    Return dataset
    """
    return load_dataset(path.split(".")[-1], data_files=path)


################################################################################


def get_model(model_name: str, from_output_dir=None) -> tuple:
    """
    Return tokenizer and model
    """
    if not from_output_dir is None:
        model_name = from_output_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


################################################################################


def compute_all_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = predictions[:, 1]  # classe positive

    # 👇 seuil personnalisé
    threshold = 0.7
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


################################################################################


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = predictions[:, 1]  # probas pour classe 1 (haineux)

    # Inverser le seuil : ici on veut détecter la classe 0 (non haineux)
    threshold = 0.7
    preds = (probs < threshold).astype(int)  # 1 si proba < 0.7 ⇒ prédire classe 0

    return {
        "f1": f1_score(labels, preds, pos_label=0),
    }


################################################################################


def trainer(
    model_name: str,
    train_dataset: dict,
    eval_dataset: dict,
    output_dir: str,
    callback_metrics=compute_all_metrics,
) -> None:
    """
    Train model
    """
    tokenizer, model = get_model(model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=callback_metrics,
    )
    trainer.train()


################################################################################
# End of File
################################################################################
