from torch.utils.data import Dataset
import torch
import pandas as pd
from .download import check_and_download_data

class MTDataset(Dataset):
    """
    PyTorch Dataset for loading time series data from a CSV file.

    Args:
        data_set_type (str): Type of dataset to load. Can be specified as 'mtb_orig', 'mtb_w', 'mtb_wh'. Defaults to 'mtb_wh'.
        seq_length (int): Length of the input sequences.
        features (list, optional): List of feature columns to use. If None, all columns are used. Defaults to None.
        file_path (str): Path to the CSV file containing the dataset. If empty, automatically download the dataset.
        normalize (bool, optional): Whether to normalize the features (Z-score normalization). Defaults to True.
        split (str, optional): Dataset split to use, either 'train', 'val', or 'test'. Defaults to "train".
        forecast_horizon (int, optional): Number of steps to forecast into the future. Defaults to 3.
        split_ratio (tuple, optional): Proportion of data to use for training, validation, and testing. Must be a tuple of three elements. Defaults to (0.8, 0.1, 0.1).

    Raises:
        ValueError: If split_ratio does not contain exactly three elements.
        ValueError: If split is not one of 'train', 'val', or 'test'.

    Attributes:
        data (pd.DataFrame): Loaded data from the CSV file.
        features (list): List of feature columns used.
        features_data (np.ndarray): Feature data extracted from the DataFrame.
        target_data (np.ndarray): Target data extracted from the DataFrame.
        mean (np.ndarray): Mean of the features for normalization.
        std (np.ndarray): Standard deviation of the features for normalization.
        forecast_horizon (int): Number of steps to forecast into the future.
    """
    def __init__(self, data_set_type = 'mtb_wh', seq_length = 24, features=None, file_path = '', normalize=True, split="train", forecast_horizon=3, split_ratio=(0.8, 0.1, 0.1)):
        self.seq_length = seq_length
        if file_path == '':
            file_path = f'{check_and_download_data()}/{data_set_type}.csv'
        # Load CSV file into a Pandas DataFrame
        self.data = pd.read_csv(file_path)

        # If features are not specified, use all columns except the target column
        if features is None:
            self.features = [col for col in self.data.columns if col != 'relative_timestamp']
        else:
            self.features = features

        # Extract features and target
        self.features_data = self.data[self.features].values
        self.target_data = self.data[self.features].values
        # Check if the target data is 1-dimensional, if so, reshape it to 2-dimensional (n, 1)
        if self.features_data.ndim == 1:
            self.features_data = self.features_data.reshape(-1, 1)
        if self.target_data.ndim == 1:
            self.target_data = self.target_data.reshape(-1, 1)

        # Normalize the features if needed
        if normalize:
            self.mean = self.features_data.mean(axis=0)
            self.std = self.features_data.std(axis=0)
            self.features_data = (self.features_data - self.mean) / self.std

        # Split data into training and testing sets
        if len(split_ratio) != 3:
            raise ValueError("split_ratio must be a list of 3 elements.")
        train_size = int(len(self.data) * split_ratio[0])
        val_size = int(len(self.data) * split_ratio[1])
        if split == "train":
            self.features_data = self.features_data[:train_size]
            self.target_data = self.target_data[:train_size]
        elif split == "val":
            self.features_data = self.features_data[train_size:train_size+val_size]
            self.target_data = self.target_data[train_size:train_size+val_size]
        elif split == "test":
            self.features_data = self.features_data[train_size+val_size:]
            self.target_data = self.target_data[train_size+val_size:]
        else:
            raise ValueError("Invalid split value. Choose 'train' or 'test'.")
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        """
        Returns the total number of sequences that can be created.
        """
        return len(self.features_data) - self.seq_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        """
        Returns a single input-target pair.

        Args:
            idx (int): Index of the starting point of the sequence.

        Returns:
            Tuple (torch.Tensor, torch.Tensor): Input sequence and corresponding target.
        """
        # Input sequence of length seq_length
        input_seq = self.features_data[idx:idx + self.seq_length]

        # Target value at the end of the sequence
        target = self.target_data[idx + self.seq_length + self.forecast_horizon - 1]

        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)