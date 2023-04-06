import collections

import torch
import glob
import numpy as np

import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import normalize
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


def pad_to_dim(v, dim):
    """Pads a tensor with zeros to a larger size.

    Args:
        v (tensor): the volume block (batch_size, days, channels, strikes, exps, types)
        dim (tuple): the size to resize to

    Returns:
        tensor: the resized volume block (batch_size, days, channels, strikes+, exps+, types)
    """
    n, c, d_diff, h_diff, w = dim - torch.Tensor([*v.shape])
    w_left, w_right = 0, 0
    h_top, h_bottom = 0, h_diff
    d_front, d_back = 0, d_diff
    pad = (
        int(w_left),
        int(w_right),
        int(h_top),
        int(h_bottom),
        int(d_front),
        int(d_back),
    )
    return torch.nn.functional.pad(v, pad=pad)


def pad_collate(batch):
    """Resizes and transposes all volumes within the batch to largest size, only necessary
    if dataset has unequal strikes/exps.

    Args:
        batch (list[list[tensor]]): a list of pairs of volumes of size (days,channels,strikes,exps,types)

    Returns:
        list[tensor]: a list of batched volumes (batch_size,channels,days,strikes,exps,types)
    """
    max_slice_in_batch_shape = torch.cat(
        [
            torch.stack([torch.Tensor([*symbol.shape]) for symbol in sample])
            for sample in batch
        ]
    ).max(0)[0]
    # TODO why am i dealing with lists here? what do i need to have the batch split like a,b = batch later on?
    logging.debug(f"before: {[p[0].shape for p in batch]}")

    # pad batch to largest timeseries
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            batch[i][j] = pad_to_dim(
                batch[i][j],
                dim=max_slice_in_batch_shape,
            )

    logging.debug(f"after: {[p[0].shape for p in batch]}")
    return [
        torch.transpose(
            torch.stack([s for s in np.array(batch, dtype=object)[:, 0]]), 1, 2
        ),
        torch.transpose(
            torch.stack([s for s in np.array(batch, dtype=object)[:, 1]]), 1, 2
        ),
    ]


def spatiotemporal_dates(
    dataset_path,
    symbols_dates,
    num_frames,
    stride,
    max_delta,
    skip_symbols=[],
    keep_symbols=[],
    return_type="dict",
):
    """Creates pairs of date "frames" to be used in contrastive loss.

    Example Output:
    {"GSKY": ['2022-02-11', '2022-02-14', '2022-02-15'], ['2022-02-18', '2022-02-22', '2022-02-23']}

    Args:
        symbols_dates (dict): a dict of symbols and their corresponding dates
        num_frames (int, optional): num contrastive frames to create. Defaults to 2.
        stride (int, optional): days to step within the frames. Defaults to 1.
        max_delta (int, optional): max days spread between the frames. Defaults to 3.

    Returns:
        dict: symbols with a list of frame tuples
    """

    date_sets = []
    skipped_symbols = []
    symbols = list(symbols_dates.keys())
    for symbol in symbols:
        if symbol in skip_symbols:
            continue
        if (len(keep_symbols) > 0 and symbol in keep_symbols) or (
            len(keep_symbols) == 0
        ):
            dates = symbols_dates[symbol]
            # randomly pop a few dates so it staggers each epoch
            # TODO once i reshuffle each epoch
            rand_pop = np.random.randint(low=0, high=5)
            if len(dates) <= num_frames * 2 + max_delta + rand_pop:
                # logging.warning(
                #     f"num_frames({num_frames}), delta({max_delta}), rand_pop({rand_pop}): {num_frames * 2 + max_delta + rand_pop}d sample >> {len(dates)}d dataset"
                # )
                logging.warning(f"skipping {symbol}: only {len(dates)} date(s)")
                skipped_symbols.append(symbol)
                continue

            set_idxs = [i for i in range(0, len(dates), stride)]
            num_sets = len(set_idxs)
            deltas = np.random.choice(
                np.arange(max_delta),
                p=np.arange(max_delta, 0, -1) / sum(np.arange(max_delta, 0, -1)),
                size=num_sets,
            )
            for delta, w1, w2 in [
                (
                    delta,
                    dates[idx : idx + num_frames],
                    dates[idx + num_frames + delta : idx + num_frames * 2 + delta],
                )
                for idx, delta in zip(range(0, len(dates), stride), deltas)
                if len(dates[idx + num_frames + delta : idx + num_frames * 2 + delta])
                == num_frames
            ]:
                logging.debug(
                    f"{symbol}, {delta}, {num_frames}, {len(dates)}, {[f for f in w1]} -> {[f for f in w2]}"
                )
                if return_type == "dict":
                    date_sets.append(
                        {
                            "delta": delta,
                            "window_one": [
                                f"{dataset_path}/{symbol}-{w}.npy" for w in w1
                            ],
                            "window_two": [
                                f"{dataset_path}/{symbol}-{w}.npy" for w in w2
                            ],
                        }
                    )
                else:
                    date_sets.append(
                        (
                            [f"{dataset_path}/{symbol}-{w}.npy" for w in w1],
                            [f"{dataset_path}/{symbol}-{w}.npy" for w in w2],
                        )
                    )
    return date_sets


def load_volume(slice_filepaths):
    uneven_volume = []
    for slice_filepath in slice_filepaths:
        slice = np.load(slice_filepath)
        uneven_volume.append(slice)
        logging.debug(f"loaded: {slice_filepath} size {slice.shape}")
    return uneven_volume


def load_uniform_volume(slice_filepaths):
    """load npy filepaths from disk that have all been presized to the same shape

    Args:
        slice_filepaths (list(str)): a list of filepaths like:


    Returns:
        torch.Tensor: a volume of size (c, f, h, w)
    """
    volume = np.stack(load_volume(slice_filepaths))
    volume = torch.Tensor(volume)
    volume = torch.permute(volume, (1, 0, 2, 3))
    return volume


def load_symbols_dates_from_dataset(dataset_path):
    filepaths = glob.glob(f"{dataset_path}/*.npy")
    symbols_dates = collections.defaultdict(list)
    for filepath in filepaths:
        symbol = filepath.split("/")[-1].split("-")[0]
        date = "-".join(filepath.split("/")[-1].split("-")[1:])[:-4]
        symbols_dates[symbol].append(date)

    for symbol in symbols_dates.keys():
        symbols_dates[symbol] = sorted(symbols_dates[symbol])

    logging.info(f"Symbols: {', '.join(symbols_dates.keys())}")
    dates = list(set([item for sublist in symbols_dates.values() for item in sublist]))
    logging.info(f"Dates: {', '.join(dates)}")
    return symbols_dates


class OcDataset(Dataset):
    def __init__(
        self,
        data_dir,
        pretext_task,
        num_frames,
        stride,
        max_delta,
        skip_symbols=[],
        keep_symbols=[],
    ):
        self.transform = T.Compose(
            [
                # torch.nn.Sequential()
                # TODO find mean/std from calcs not including
                # zeros, since there are many.
                # try this: https://discuss.pytorch.org/t/use-tensor-mean-method-but-ignore-0-values/60170/3
                NormalizeVideo(
                    mean=torch.Tensor(
                        [
                            3.8504e-01,
                            3.4533e02,
                            2.1859e-04,
                            1.2402e00,
                            8.6137e-01,
                            4.4128e02,
                            1.6343e-04,
                            1.2402e00,
                        ]
                    ),
                    std=torch.Tensor(
                        [
                            1.7997e00,
                            2.6223e03,
                            3.9207e-03,
                            3.3435e00,
                            2.6852e00,
                            2.8250e03,
                            5.6097e-03,
                            3.3435e00,
                        ]
                    ),
                )
                # NormalizeVideo(
                #     # channel_dim=0,
                #     # NOTE calc'd from batch of 2048 vols
                #     # of original 10 sample symbs
                #     mean=torch.tensor(
                #         [
                #             6.2324e01,
                #             9.6467e00,
                #             6.1295e01,
                #             7.7610e00,
                #             1.9308e-01,
                #             2.5376e02,
                #             1.4423e-04,
                #             1.0992e00,
                #             4.7027e01,
                #             1.0202e01,
                #             9.1426e00,
                #             9.7798e00,
                #             8.0463e00,
                #             1.0072e00,
                #             2.2545e02,
                #             8.9769e-05,
                #             1.0992e00,
                #             3.0351e01,
                #         ]
                #     ),
                #     std=torch.tensor(
                #         [
                #             2.5093e02,
                #             7.3349e01,
                #             2.4793e02,
                #             6.2451e01,
                #             1.1387e00,
                #             2.4505e03,
                #             3.4080e-03,
                #             3.6982e00,
                #             1.3704e03,
                #             4.8059e01,
                #             5.6805e01,
                #             4.7098e01,
                #             4.8254e01,
                #             3.4649e00,
                #             2.0212e03,
                #             3.6038e-03,
                #             3.6982e00,
                #             7.5986e02,
                #         ]
                # ),
                # )
            ]
        )
        symbols_dates = load_symbols_dates_from_dataset(data_dir)

        # spatiotemporal: https://arxiv.org/pdf/2008.03800v4.pdf
        # TODO convert this to a collate_fn so that the dates are randomly reshuffled
        # after each epoch. like: https://discuss.pytorch.org/t/dataloader-re-initialize-dataset-after-each-iteration/32658/8
        if pretext_task == "spatiotemporal":
            self.pairs = spatiotemporal_dates(
                dataset_path=data_dir,
                symbols_dates=symbols_dates,
                num_frames=num_frames,
                stride=stride,
                max_delta=max_delta,
                skip_symbols=skip_symbols,
                keep_symbols=keep_symbols,
            )
        else:
            raise ("no pretext task set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # from days, channels, strikes, exps -> channels, days, strikes, exps
        # print(idx)
        return {
            "window_one_filepaths": ",".join(self.pairs[idx]["window_one"]),
            "window_one": self.transform(
                load_uniform_volume(self.pairs[idx]["window_one"])
            ),
            "window_two_filepaths": ",".join(self.pairs[idx]["window_two"]),
            "window_two": self.transform(
                load_uniform_volume(self.pairs[idx]["window_two"])
            ),
        }