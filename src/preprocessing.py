################################################################################
# filename: preprocessing.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 12/05,2025
################################################################################

import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer

################################################################################
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_function(example):
    return tokenizer(
        example["tweet_text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # la limite pour BERTweet est 128 tokens
    )


def treat_data(dataset_to_treat: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    dataset_to_treat = dataset_to_treat.map(transform_text_to_int)
    dataset_to_treat = dataset_to_treat.map(tokenize_function)

    train_df = dataset_to_treat["train"].to_pandas()
    train_df.to_csv("data/processed/processed_tweet.csv", index=False)
    get_repartition(dataset_to_treat)

    # Assume que la colonne 'label' contient les classes
    train_dataset, val_dataset, test_dataset = get_train_val_test_from_dataframe(
        train_df
    )
    return train_dataset, val_dataset, test_dataset


################################################################################


def get_train_val_test_from_dataframe(train_df):
    y = train_df["cyberbullying_type"]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, temp_idx in splitter.split(train_df, y):
        train_split = train_df.iloc[train_idx]
        temp_split = train_df.iloc[temp_idx]

    # 50% val, 50% test dans les 20% restants (10% val / 10% test globalement)
    y_temp = temp_split["cyberbullying_type"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in splitter.split(temp_split, y_temp):
        val_split = temp_split.iloc[val_idx]
        test_split = temp_split.iloc[test_idx]

    # Convertir en objets Dataset
    train_dataset = Dataset.from_pandas(train_split.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_split.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_split.reset_index(drop=True))
    global label2id
    label2id = {
        label: i for i, label in enumerate(set(train_dataset["cyberbullying_type"]))
    }
    train_dataset = train_dataset.map(add_labels)
    val_dataset = val_dataset.map(add_labels)
    test_dataset = test_dataset.map(add_labels)
    return train_dataset, val_dataset, test_dataset


################################################################################


def add_labels(example):
    example["labels"] = label2id[example["cyberbullying_type"]]
    return example


################################################################################


def transform_text_to_int(example):
    example["cyberbullying_type"] = (
        0 if example["cyberbullying_type"] == "not_cyberbullying" else 1
    )
    return example


################################################################################


# def tokenize_function(example):
#     model_name = "vinai/bertweet-base" #"distilbert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
#     return tokenizer(example["tweet_text"], padding="max_length", truncation=True)


################################################################################


def get_repartition(dataset_to_draw: Dataset) -> None:
    df = dataset_to_draw["train"].to_pandas()
    label_counts = df["cyberbullying_type"].value_counts().sort_index()

    # Créer l'histogramme
    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.index, label_counts.values, color=["skyblue", "salmon"])
    plt.xticks([0, 1], ["Non haineux (0)", "Haineux (1)"])
    plt.xlabel("Classe")
    plt.ylabel("Nombre de tweets")
    plt.title("Répartition des tweets haineux vs non-haineux")

    # Sauvegarder le graphique
    plt.savefig("figures/histogramme_labels.png")
    plt.close()


################################################################################
# End of File
################################################################################
