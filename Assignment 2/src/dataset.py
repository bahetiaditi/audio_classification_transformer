import os
from pathlib import Path
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm

meta_file_path = '/Deep Learning/Assignment 2/Archive (6)/meta/esc50.csv'
path = Path('/Deep Learning/Assignment 2/Archive (6)')
df = pd.read_csv(meta_file_path)

wavs = list(path.glob('audio/*'))
waveform, sample_rate = torchaudio.load(str(wavs[0]))

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

ipd.Audio(waveform, rate=sample_rate)

class CustomDataset(Dataset):
    def __init__(self, dataset, **kwargs):
        self.data_directory = kwargs["data_directory"]
        self.data_frame = kwargs["data_frame"]
        self.validation_fold = kwargs["validation_fold"]
        self.testing_fold = kwargs["testing_fold"]
        self.esc_10_flag = kwargs["esc_10_flag"]
        self.file_column = kwargs["file_column"]
        self.label_column = kwargs["label_column"]
        self.sampling_rate = kwargs["sampling_rate"]
        self.new_sampling_rate = kwargs["new_sampling_rate"]
        self.sample_length_seconds = kwargs["sample_length_seconds"]

        if self.esc_10_flag:
            self.data_frame = self.data_frame.loc[self.data_frame['esc10'] == True]

        if dataset == "train":
            self.data_frame = self.data_frame.loc[
                (self.data_frame['fold'] != self.validation_fold) & (self.data_frame['fold'] != self.testing_fold)]
        elif dataset == "val":
            self.data_frame = self.data_frame.loc[self.data_frame['fold'] == self.validation_fold]
        elif dataset == "test":
            self.data_frame = self.data_frame.loc[self.data_frame['fold'] == self.testing_fold]

        self.categories = sorted(self.data_frame[self.label_column].unique())

        self.file_names = []
        self.labels = []

        self.category_to_index = {}
        self.index_to_category = {}

        for i, category in enumerate(self.categories):
            self.category_to_index[category] = i
            self.index_to_category[i] = category

        for ind in tqdm(range(len(self.data_frame))):
            row = self.data_frame.iloc[ind]
            file_path = self.data_directory / "audio" / row[self.file_column]
            self.file_names.append(file_path)
            self.labels.append(self.category_to_index[row[self.label_column]])

        self.resampler = torchaudio.transforms.Resample(self.sampling_rate, self.new_sampling_rate)

        if self.sample_length_seconds == 2:
            self.window_size = self.new_sampling_rate * 2
            self.step_size = int(self.new_sampling_rate * 0.75)
        else:
            self.window_size = self.new_sampling_rate
            self.step_size = int(self.new_sampling_rate * 0.5)

    def __getitem__(self, index):
        path = self.file_names[index]
        audio_file = torchaudio.load(str(path), format=None, normalize=True)
        audio_tensor = self.resampler(audio_file[0])
        splits = audio_tensor.unfold(1, self.window_size, self.step_size)
        samples = splits.permute(1, 0, 2)
        return samples, self.labels[index]

    def __len__(self):
        return len(self.file_names)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.data_module_kwargs = kwargs

    def setup(self, stage=None, current_fold=None):
        kwargs = dict(self.data_module_kwargs)
        kwargs.pop('validation_fold', None)
        kwargs.pop('testing_fold', None)
        test_fold = kwargs.get('testing_fold', 1)

        if stage in {'fit', None}:
            self.training_dataset = CustomDataset(dataset="train", validation_fold=current_fold, testing_fold=test_fold, **kwargs)
            self.validation_dataset = CustomDataset(dataset="val", validation_fold=current_fold, testing_fold=test_fold, **kwargs)

        if stage in {'test', None}:
            self.testing_dataset = CustomDataset(dataset="test", validation_fold=current_fold, testing_fold=test_fold, **kwargs)


    def train_dataloader(self):
        return DataLoader(self.training_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_function,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_function,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset,
                          batch_size=32,
                          shuffle=False,
                          collate_fn=self.collate_function,
                          num_workers=self.num_workers)

    def collate_function(self, batch):
     """
     Collate function to process a batch of examples and labels.

     Args:
        batch: a list of tuples (example, label), where
            example is a tensor of split 1-second sub-frame audio tensors per file,
            label is the label for the entire file.

     Returns:
         A tuple containing batches of examples and labels.
     """
     example_tensors = []
     label_tensors = []

     for single_example, single_label in batch:
       for each_subframe in single_example:
         example_tensors.append(each_subframe)
         label_tensors.append(single_label)

     stacked_examples = torch.stack(example_tensors)
     encoded_labels = torch.tensor(label_tensors, dtype=torch.long)

     return stacked_examples, encoded_labels