import pickle

dataset_path = "/home/sudeep/Documents/datasets/metaworld-expert-v1.0/assembly-v2-goal-observable.pickle"

# read the pickle file
with open(dataset_path, "rb") as f:
    data = pickle.load(f)
    print(data)