import pandas as pd


def curriculum_cvd_dataset(levels="No levels", percentage=100):
    """Select the dataset

    Args:
        levels (str, optional): The level of difficulty of the data. Can be "two" (easy and hard), "three" (easy, medium, and hard), or "No levels" (no level splitting). The default value is "No levels".

        percentage (int, optional): The percentage of the data to use. The default value is 100.

    Returns:
        data, target: The data and target variables.
    """

    with open("../../datasets/CVD_curriculum.csv") as f:
        dataset = pd.read_csv(f, delimiter=",")

        if levels == "two":
            # Split the data into two levels of difficulty: easy and hard.
            dataset["Level"] = dataset["Level"].replace({"Medium": "Easy"})
            unique_values = {}
            for value in dataset["Level"].unique():
                head = dataset[dataset["Level"] == value].head(int(len(dataset) * percentage / 2))
                unique_values[value] = head

            dataset = pd.concat([unique_values['Easy'], unique_values['Hard']], axis=0)
        elif levels == "three":
            # Split the data into three levels of difficulty: easy, medium, and hard.
            unique_values = {}
            for value in dataset["Level"].unique():
                head = dataset[dataset["Level"] == value].head(int(len(dataset) * percentage / 3))
                unique_values[value] = head

            dataset = pd.concat([unique_values['Easy'], unique_values['Medium'], unique_values['Hard']], axis=0)
        else:
            # Do not split the data by difficulty.
            dataset = dataset.sample(frac=1, random_state=0)
            dataset = dataset.head(int(len(dataset) * percentage))

        data = dataset.iloc[:, 1:20]
        target = dataset["Heart_Disease"]

    return data, target


def fault_detection_dataset():
    with open("../../datasets/fault_detection.csv") as f:
        dataset = pd.read_csv(f, delimiter=";").sample(frac=1, random_state=0).head(300)
        data = dataset.iloc[:, :60]
        target = dataset["Fault_lag"]

    return data, target
