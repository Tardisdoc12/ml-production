################################################################################
# filename: main.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 12/05,2025
################################################################################

from core_function import get_dataset, trainer
from preprocessing import treat_data

################################################################################


def main():
    dataset = get_dataset("data/raw/cyberbullying_tweets.csv")
    train_dataset, val_dataset, test_dataset = treat_data(dataset)
    model_name = "distilbert-base-uncased"
    trainer(model_name, train_dataset, test_dataset, "Train")


if "__main__" == __name__:
    main()

################################################################################
# End of File
################################################################################
