from pathlib import Path

from data_preprocessing.csv_to_dataframe import read_dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = read_dataset(Path("./datasets/raw/marketing_campaign.csv"), "\t")
    print(dataset)
