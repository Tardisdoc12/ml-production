import os

from randomforest import train_rainforest
from training_hugging_face import test_model, train_model, treat_data


def launch_all_training():
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    if not os.path.exists("data/processed/processed_tweet.csv"):
        treat_data()
    train_rainforest()
    train_model()
    test_model("figures/results_distilbert_all_metrics.json")
    train_model("")
    test_model("figures/results_distilbert_f1_score.json")


if __name__ == "__main__":
    launch_all_training()
