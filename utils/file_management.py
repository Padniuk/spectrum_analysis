import pandas as pd
import os


def file_reader(file_path):
    try:
        df = pd.read_csv(file_path)
        time = pd.to_numeric(df["Time"][1:])
        channel_a = pd.to_numeric(df["Channel A"][1:])
        channel_b = pd.to_numeric(df["Channel B"][1:])
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, sep=";", decimal=",")
        time = pd.to_numeric(df["Time"][1:].str.replace(",", "."))
        channel_a = pd.to_numeric(df["Channel A"][1:].str.replace(",", "."))
        channel_b = pd.to_numeric(df["Channel B"][1:].str.replace(",", "."))

    return time, channel_a, channel_b


def file_writer(folder_path, data):
    output_dir = os.path.join(folder_path, "tmp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, _ in data[0].items():
        if os.path.exists(os.path.join(output_dir, f"{key}.txt")):
            os.remove(os.path.join(output_dir, f"{key}.txt"))

    for res in data:
        for key, parameters in res.items():
            with open(os.path.join(output_dir, f"{key}.txt"), "a") as file:
                file.write(",".join(map(str, parameters)) + "\n")
