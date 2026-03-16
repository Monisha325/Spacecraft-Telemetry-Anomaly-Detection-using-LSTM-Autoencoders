
import numpy as np
import os

def load_channel(channel_id, data_dir="data"):
    train_path = os.path.join(data_dir, "train", f"{channel_id}.npy")
    test_path = os.path.join(data_dir, "test", f"{channel_id}.npy")
    X_train = np.load(train_path)
    X_test = np.load(test_path)
    print(f"Loaded {channel_id} — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test
