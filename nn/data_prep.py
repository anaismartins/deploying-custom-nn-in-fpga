import gc
import os
from telnetlib import DET
import time
from xml.sax.handler import DTDHandler

import numpy as np
import pandas as pd
import torch
from scipy.fft import fft
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.globals import (
    TRAIN_DATA_DIR,
    VALID_DATA_DIR,
    TEST_DATA_DIR,
    DETECTORS,
    FOLDER,
    OUTPUT_DIR,
)
from training.utils import print_or_output


def into_dataloader(d, out_file, batch, batch_n, minibatch_size, current_folder):
    # load the data

    if batch == "train":
        noise = np.load(
            # f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_files/10000/train_folder{current_folder}.npz"
            f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/train_folder{current_folder}_batch{batch_n}.npz"
        )[f"noise"]
    else:
        noise = np.load(
            # f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_files/10000/{batch}.npz"
            f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/{batch}_folder{current_folder}_batch{batch_n}.npz"
        )[f"noise"]
    # get zero vector of the same size

    if batch == "train":
        signal = np.load(
            # f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_files/10000/train_folder{current_folder}.npz"
            f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/train_folder{current_folder}_batch{batch_n}.npz"
        )[f"signal"]
    else:
        signal = np.load(
            # f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_files/10000/{batch}.npz"
            f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/{batch}_folder{current_folder}_batch{batch_n}.npz"
        )[f"signal"]
    # get vector of ones of the same size

    if np.isnan(noise).any():
        print_or_output("Noise contains NaN values", d, out_file)
        mask = np.isnan(noise).any(axis=(1, 2))
        # Filter out rows with NaNs
        noise = noise[~mask]
        print_or_output(f"New noise shape: {noise.shape}", d, out_file)

    if np.isnan(signal).any():
        print_or_output("Signal contains NaN values", d, out_file)
        mask = np.isnan(signal).any(axis=(1, 2))
        # Filter out rows with NaNs
        signal = signal[~mask]
        print_or_output(f"New signal shape: {signal.shape}", d, out_file)

    if (signal > 1).any() or (signal < -1).any():
        print(f"Signal out of range")
        mask = (signal > 1).any(axis=(1, 2)) | (signal < -1).any(axis=(1, 2))
        # Filter out rows with values out of range
        signal = signal[~mask]
        print(f"New signal shape: {signal.shape}")
    if (noise > 1).any() or (noise < -1).any():
        print(f"Noise out of range")
        mask = (noise > 1).any(axis=(1, 2)) | (noise < -1).any(axis=(1, 2))
        # Filter out rows with values out of range
        noise = noise[~mask]
        print(f"New noise shape: {noise.shape}")

    y_noise = np.zeros(noise.shape[0])
    y_signal = np.ones(signal.shape[0])

    # stack noise and signal into one dataset
    X = np.vstack((noise, signal))

    # delete the variables to free up memory
    del noise
    del signal
    torch.cuda.empty_cache()
    gc.collect()

    # stack the labels
    y = np.hstack((y_noise, y_signal))

    X = torch.tensor(X)
    y = torch.tensor(y)

    if torch.any(torch.isnan(X)):
        print_or_output("X contains NaN values", d, out_file)
        exit()
    if torch.any(torch.isnan(y)):
        print_or_output("y contains NaN values", d, out_file)
        exit()
    # used to fix nan in loss - not used now since we don't have any NaNs but might be useful in the future
    # X = torch.nan_to_num(X)
    # y = torch.nan_to_num(y)

    data = TensorDataset(X, y.type(torch.LongTensor))

    del X
    del y
    torch.cuda.empty_cache()
    gc.collect()

    if "test" not in batch:
        dataloader = DataLoader(
            data,
            shuffle=True,
            batch_size=minibatch_size,
        )
    else:
        dataloader = DataLoader(data, shuffle=False, batch_size=minibatch_size)

    del data
    torch.cuda.empty_cache()
    gc.collect()

    return dataloader


def get_all_time_data(new_file_folder, batch_array_size=100):
    print("Getting all time data")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if os.path.exists(TEST_DATA_DIR):
        d = "cluster"
    else:
        d = "pc"
    print(f"Running on: {d}")

    if d == "pc":
        output_dir = "./../output/"
        out_file = ""
        print(f"Using device: {device}\n", d, out_file)
    elif d == "cluster":
        output_dir = OUTPUT_DIR
        out_file = f"{output_dir}prints/{time.time()}.txt"
        with open(out_file, "w") as f:
            f.write(f"Using device: {device}\n")
    print(f"Output file: {out_file}")

    # train_order = []
    # for current_folder in range(0, FOLDER + 1):
    #     order.append(np.random.choice(np.arange(n), size=n, replace=False))

    for current_file in range(6, 7):
        # 0 - 4 for training
        # 5 for validation
        # 6 for testing
        print_or_output(f"File {current_file}", d, out_file)

        # all_data = {}

        if (
            current_file < 5
        ):  # for training --------------------------------------------------------------------------------------
            n_samples_train = 10000

            print_or_output(f"Training folder {current_file}", d, out_file)
            for batch in range(0, int(n_samples_train / batch_array_size)):
                all_data = {}
                print_or_output(f"Training batch {batch}", d, out_file)
                # curr_batch = order[current_file][
                #     batch * batch_array_size : (batch + 1) * batch_array_size
                # ]
                curr_batch = np.linspace(
                    batch * batch_array_size,
                    (batch + 1) * batch_array_size - 1,
                    batch_array_size,
                ).astype(int)

                noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
                signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

                for c in range(2):
                    for element_id in range(0, len(curr_batch)):
                        temp = np.empty((1, len(DETECTORS), 155648))

                        for detector in DETECTORS:
                            filename = (
                                str(
                                    current_file * n_samples_train
                                    + curr_batch[element_id]
                                ).zfill(8)
                                + "_"
                                + str(c)
                                + "_"
                                + detector
                                + ".npy"
                            )

                            data = np.load(
                                TRAIN_DATA_DIR + str(current_file) + "/" + filename
                            )

                            # H1 = 0
                            # L1 = 1
                            # V1 = 2
                            if "H1" in filename:
                                temp[0, 0] = data.copy()
                            elif "L1" in filename:
                                temp[0, 1] = data.copy()
                            elif "V1" in filename:
                                temp[0, 1] = data.copy()
                            del data
                            torch.cuda.empty_cache()
                            gc.collect()

                        if c == 0:
                            for j in range(len(DETECTORS)):
                                noise[element_id, j, :] = temp[0, j, :].copy()
                        elif c == 1:
                            for j in range(len(DETECTORS)):
                                signal[element_id, j, :] = temp[0, j, :].copy()

                        del temp
                        torch.cuda.empty_cache()
                        gc.collect()

                all_data[f"noise"] = noise
                all_data[f"signal"] = signal
                del noise
                del signal
                torch.cuda.empty_cache()
                gc.collect()

                np.savez(
                    f"{new_file_folder}train_folder{current_file}_batch{batch}",
                    **all_data,
                )
                # delete the variables to free up memory
                del all_data
                torch.cuda.empty_cache()
                gc.collect()

        elif (
            current_file == 5
        ):  # for validation --------------------------------------------------------------------------------------
            n_samples_val = 1300

            for folder in range(0, 5):
                print_or_output(f"Validation folder {folder}", d, out_file)

                for batch in range(0, int(n_samples_val / batch_array_size)):
                    all_data = {}
                    print_or_output(f"Validation batch {batch}", d, out_file)

                    # curr_batch = order[folder][
                    #     batch * batch_array_size
                    #     + int(n * 0.8) : (batch + 1) * batch_array_size
                    #     + int(n * 0.8)
                    # ]
                    curr_batch = np.linspace(
                        batch * batch_array_size,
                        (batch + 1) * batch_array_size - 1,
                        batch_array_size,
                    ).astype(int)

                    noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
                    signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

                    for c in range(2):
                        for element_id in range(0, len(curr_batch)):
                            temp = np.empty((1, len(DETECTORS), 155648))

                            for detector in DETECTORS:
                                filename = (
                                    str(
                                        folder * n_samples_val + curr_batch[element_id]
                                    ).zfill(8)
                                    + "_"
                                    + str(c)
                                    + "_"
                                    + detector
                                    + ".npy"
                                )

                                data = np.load(
                                    VALID_DATA_DIR + str(folder) + "/" + filename
                                )

                                # H1 = 0
                                # L1 = 1
                                # V1 = 2
                                if "H1" in filename:
                                    temp[0, 0] = data.copy()
                                elif "L1" in filename:
                                    temp[0, 1] = data.copy()
                                elif "V1" in filename:
                                    temp[0, 2] = data.copy()

                                del data
                                torch.cuda.empty_cache()
                                gc.collect()

                            if c == 0:
                                for j in range(len(DETECTORS)):
                                    noise[element_id, j, :] = temp[0, j, :].copy()
                            elif c == 1:
                                for j in range(len(DETECTORS)):
                                    signal[element_id, j, :] = temp[0, j, :].copy()

                            del temp
                            torch.cuda.empty_cache()
                            gc.collect()

                    all_data[f"noise"] = noise
                    all_data[f"signal"] = signal
                    del noise
                    del signal
                    torch.cuda.empty_cache()
                    gc.collect()

                    np.savez(
                        f"{new_file_folder}validation_folder{folder}_batch{batch}",
                        **all_data,
                    )
                    del all_data
                    torch.cuda.empty_cache()
                    gc.collect()

        elif (
            current_file == 6
        ):  # for testing --------------------------------------------------------------------------------------
            n_samples_test = 2200

            for folder in range(0, 5):
                print_or_output(f"Testing folder {folder}", d, out_file)

                for batch in range(0, int(n_samples_test / batch_array_size)):
                    all_data = {}
                    all_parameter_data = {}

                    print_or_output(f"Testing batch {batch}", d, out_file)
                    # curr_batch = order[folder][
                    #     batch * batch_array_size
                    #     + int(n * 0.9) : (batch + 1) * batch_array_size
                    #     + int(n * 0.9)
                    # ]
                    curr_batch = np.linspace(
                        batch * batch_array_size,
                        (batch + 1) * batch_array_size - 1,
                        batch_array_size,
                    ).astype(int)

                    noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
                    signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

                    noise_params = np.empty((len(curr_batch), 37))
                    signal_params = np.empty((len(curr_batch), 37))

                    for c in range(2):
                        for element_id in range(0, len(curr_batch)):
                            temp = np.empty((1, len(DETECTORS), 155648))

                            for detector in DETECTORS:
                                filename = (
                                    str(
                                        folder * n_samples_test + curr_batch[element_id]
                                    ).zfill(8)
                                    + "_"
                                    + str(c)
                                    + "_"
                                    + detector
                                    + ".npy"
                                )

                                data = np.load(
                                    TEST_DATA_DIR + str(folder) + "/" + filename
                                )

                                # H1 = 0
                                # L1 = 1
                                # V1 = 2
                                if "H1" in filename:
                                    temp[0, 0] = data.copy()
                                elif "L1" in filename:
                                    temp[0, 1] = data.copy()
                                elif "V1" in filename:
                                    temp[0, 2] = data.copy()

                                del data
                                torch.cuda.empty_cache()
                                gc.collect()

                            if c == 0:
                                for j in range(len(DETECTORS)):
                                    noise[element_id, j, :] = temp[0, j, :].copy()
                                    tmp = pd.read_csv(
                                        TEST_DATA_DIR
                                        + str(folder)
                                        + "/"
                                        + filename[:10]
                                        + ".csv"
                                    )
                                    noise_params[element_id] = tmp.to_numpy()
                            elif c == 1:
                                for j in range(len(DETECTORS)):
                                    signal[element_id, j, :] = temp[0, j, :].copy()
                                    tmp = pd.read_csv(
                                        TEST_DATA_DIR
                                        + str(folder)
                                        + "/"
                                        + filename[:10]
                                        + ".csv"
                                    )
                                    signal_params[element_id] = tmp.to_numpy()

                            del temp
                            torch.cuda.empty_cache()
                            gc.collect()

                    all_data[f"noise"] = noise
                    all_data[f"signal"] = signal
                    all_parameter_data[f"noise"] = noise_params
                    all_parameter_data[f"signal"] = signal_params
                    del noise
                    del signal
                    del noise_params
                    del signal_params
                    torch.cuda.empty_cache()
                    gc.collect()

                    np.savez(
                        f"{new_file_folder}test_folder{folder}_batch{batch}", **all_data
                    )
                    np.savez(
                        f"{new_file_folder}test_params_folder{folder}_batch{batch}",
                        **all_parameter_data,
                    )
                    del all_data
                    del all_parameter_data
                    torch.cuda.empty_cache()
                    gc.collect()


def get_test_data(new_file_folder, d, out_file):
    batch_array_size = 200

    n = 2200
    all_data = {}
    all_parameter_data = {}

    for folder in range(0, 5):
        print_or_output(f"Testing folder {folder}", d, out_file)

        for batch in range(0, int(n / batch_array_size)):
            print_or_output(f"Testing batch {batch}", d, out_file)
            # curr_batch = order[folder][
            #     batch * batch_array_size
            #     + int(n * 0.9) : (batch + 1) * batch_array_size
            #     + int(n * 0.9)
            # ]
            curr_batch = np.linspace(
                batch * batch_array_size,
                (batch + 1) * batch_array_size - 1,
                batch_array_size,
            ).astype(int)

            noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
            signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

            noise_params = np.empty((len(curr_batch), 37))
            signal_params = np.empty((len(curr_batch), 37))

            for c in range(2):
                for element_id in range(0, len(curr_batch)):
                    temp = np.empty((1, len(DETECTORS), 155648))

                    for detector in DETECTORS:
                        filename = (
                            str(folder * n + curr_batch[element_id]).zfill(8)
                            + "_"
                            + str(c)
                            + "_"
                            + detector
                            + ".npy"
                        )

                        data = np.load(TEST_DATA_DIR + str(folder) + "/" + filename)

                        # H1 = 0
                        # L1 = 1
                        # V1 = 2
                        if "H1" in filename:
                            temp[0, 0] = data.copy()
                        elif "L1" in filename:
                            temp[0, 1] = data.copy()
                        elif "V1" in filename:
                            temp[0, 2] = data.copy()

                        del data
                        torch.cuda.empty_cache()
                        gc.collect()

                    if c == 0:
                        for j in range(len(DETECTORS)):
                            noise[element_id, j, :] = temp[0, j, :].copy()
                            tmp = pd.read_csv(
                                TEST_DATA_DIR
                                + str(folder)
                                + "/"
                                + filename[:10]
                                + ".csv"
                            )
                            noise_params[element_id] = tmp.to_numpy()
                    elif c == 1:
                        for j in range(len(DETECTORS)):
                            signal[element_id, j, :] = temp[0, j, :].copy()
                            tmp = pd.read_csv(
                                TEST_DATA_DIR
                                + str(folder)
                                + "/"
                                + filename[:10]
                                + ".csv"
                            )
                            signal_params[element_id] = tmp.to_numpy()

                    del temp
                    torch.cuda.empty_cache()
                    gc.collect()

            all_data[f"noise_folder{folder}_batch{batch}"] = noise
            all_data[f"signal_folder{folder}_batch{batch}"] = signal
            all_parameter_data[f"noise_folder{folder}_batch{batch}"] = noise_params
            all_parameter_data[f"signal_folder{folder}_batch{batch}"] = signal_params

    np.savez(f"{new_file_folder}test", **all_data)
    np.savez(f"{new_file_folder}test_params", **all_parameter_data)
    del all_data
    del all_parameter_data
    torch.cuda.empty_cache()
    gc.collect()


def get_small_train_data(new_file_folder, d, out_file, batch_array_size):
    n = 10000

    for folder in range(0, 5):
        all_data = {}
        print_or_output(f"Folder {folder}", d, out_file)

        curr_batch = np.arange(0, batch_array_size).astype(int)

        noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
        signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

        for c in range(2):
            for element_id in range(0, len(curr_batch)):
                tmp = np.empty((1, len(DETECTORS), 155648))

                for detector in DETECTORS:
                    filename = (
                        f"{(folder * n + curr_batch[element_id]):08}_{c}_{detector}.npy"
                    )

                    data = np.load(f"{TRAIN_DATA_DIR}{folder}/{filename}")

                    if "H1" in filename:
                        tmp[0, 0] = data.copy()
                    elif "L1" in filename:
                        tmp[0, 1] = data.copy()
                    elif "V1" in filename:
                        tmp[0, 2] = data.copy()

                    del data
                    torch.cuda.empty_cache()
                    gc.collect()

                if c == 0:
                    for j in range(len(DETECTORS)):
                        noise[element_id, j, :] = tmp[0, j, :].copy()
                elif c == 1:
                    for j in range(len(DETECTORS)):
                        signal[element_id, j, :] = tmp[0, j, :].copy()

                del tmp
                torch.cuda.empty_cache()
                gc.collect()

        all_data[f"noise"] = noise
        all_data[f"signal"] = signal

        np.savez(f"{new_file_folder}folder{folder}", **all_data)
        del all_data
        torch.cuda.empty_cache()
        gc.collect()


def get_small_val_data(new_file_folder, d, out_file, batch_array_size):
    n = 1300

    for folder in range(0, 5):
        all_data = {}
        print_or_output(f"Folder {folder}", d, out_file)

        curr_batch = np.arange(0, batch_array_size).astype(int)

        noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
        signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

        for c in range(2):
            for element_id in range(0, len(curr_batch)):
                tmp = np.empty((1, len(DETECTORS), 155648))

                for detector in DETECTORS:
                    filename = (
                        f"{(folder * n + curr_batch[element_id]):08}_{c}_{detector}.npy"
                    )

                    data = np.load(f"{VALID_DATA_DIR}{folder}/{filename}")

                    if "H1" in filename:
                        tmp[0, 0] = data.copy()
                    elif "L1" in filename:
                        tmp[0, 1] = data.copy()
                    elif "V1" in filename:
                        tmp[0, 2] = data.copy()

                    del data
                    torch.cuda.empty_cache()
                    gc.collect()

                if c == 0:
                    for j in range(len(DETECTORS)):
                        noise[element_id, j, :] = tmp[0, j, :].copy()
                elif c == 1:
                    for j in range(len(DETECTORS)):
                        signal[element_id, j, :] = tmp[0, j, :].copy()

                del tmp
                torch.cuda.empty_cache()
                gc.collect()

        all_data[f"noise"] = noise
        all_data[f"signal"] = signal

        np.savez(f"{new_file_folder}folder{folder}", **all_data)
        del all_data
        torch.cuda.empty_cache()
        gc.collect()


def get_small_test_data(new_file_folder, d, out_file, batch_array_size):
    n = 2200

    for folder in range(0, 5):
        all_data = {}
        print_or_output(f"Folder {folder}", d, out_file)

        curr_batch = np.arange(0, batch_array_size).astype(int)

        noise = np.empty((len(curr_batch), len(DETECTORS), 155648))
        signal = np.empty((len(curr_batch), len(DETECTORS), 155648))

        for c in range(2):
            for element_id in range(0, len(curr_batch)):
                tmp = np.empty((1, len(DETECTORS), 155648))

                for detector in DETECTORS:
                    filename = (
                        f"{(folder * n + curr_batch[element_id]):08}_{c}_{detector}.npy"
                    )

                    data = np.load(f"{TEST_DATA_DIR}{folder}/{filename}")

                    if "H1" in filename:
                        tmp[0, 0] = data.copy()
                    elif "L1" in filename:
                        tmp[0, 1] = data.copy()
                    elif "V1" in filename:
                        tmp[0, 2] = data.copy()

                    del data
                    torch.cuda.empty_cache()
                    gc.collect()

                if c == 0:
                    for j in range(len(DETECTORS)):
                        noise[element_id, j, :] = tmp[0, j, :].copy()
                elif c == 1:
                    for j in range(len(DETECTORS)):
                        signal[element_id, j, :] = tmp[0, j, :].copy()

                del tmp
                torch.cuda.empty_cache()
                gc.collect()

        all_data[f"noise"] = noise
        all_data[f"signal"] = signal

        np.savez(f"{new_file_folder}folder{folder}", **all_data)
        del all_data
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if os.path.exists(TEST_DATA_DIR):
        d = "cluster"
    else:
        d = "pc"

    if d == "pc":
        output_dir = "./../output/"
        out_file = ""
        print(f"Using device: {device}\n")
    elif d == "cluster":
        output_dir = OUTPUT_DIR
        out_file = f"{output_dir}prints/{time.time()}.txt"
        with open(out_file, "w") as f:
            f.write(f"Using device: {device}\n")
    print(f"Output file: {out_file}")

    batch_array_size = 20

    new_file_folder = f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_data/"
    # new_file_folder = f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_small_test_data_{batch_array_size}/"
    # new_file_folder = f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_small_train_data/"
    # new_file_folder=f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_small_val_data/"
    # new_file_folder = f"/data/gravwav/amartins/earlywarningfpgas/parsed_data/new_test_data/"
    if not os.path.exists(new_file_folder):
        os.makedirs(new_file_folder)
        # get_all_time_data(new_file_folder, batch_array_size=100)
        # get_small_test_data(new_file_folder, d, out_file, batch_array_size)
        get_small_train_data(new_file_folder, d, out_file, batch_array_size)
    # get_small_val_data(new_file_folder, d, out_file)
    # get_test_data(new_file_folder, d, out_file)
