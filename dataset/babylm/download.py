from datasets import load_dataset

dataset = load_dataset("KylinC/babylm_10M")
path_to_save = "./babylm_10M"
dataset.save_to_disk(path_to_save)
