"""
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import math

_divider = "-------------------------------"


def preprocess_fn(data_path, batch):
    """
    Data pre-processing.
    Opens file with numpy array and turns it into int8.
    input arg: path of samples file
    return: numpy array
    """
    data = np.load(data_path)[batch]
    data = data.astype(np.int8)
    return data


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def runDPU(id, start, dpu, data):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()  # converts into NHWC format
    print("inputTensors:", inputTensors)
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    batchSize = input_ndim[0]
    n_of_samples = len(data)
    count = 0
    write_index = start
    ids = []
    ids_max = 1
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_samples:
        if count + batchSize <= n_of_samples:
            runSize = batchSize
        else:
            runSize = n_of_samples - count

        """prepare batch input/output """
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        """init input sample to input buffer """
        for j in range(runSize):
            print("input_ndim[0:]:", input_ndim[0:])
            print("input_ndim[1:]:", input_ndim[1:])
            sampleRun = inputData[0]
            sampleRun[j, ...] = data[(count + j) % n_of_samples].reshape(input_ndim[1:])
        """run with batch """
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count = count + runSize
        if count < n_of_samples:
            if len(ids) < ids_max - 1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            """store output vectors """
            for j in range(ids[index][1]):
                outputData[index][0][j] = sigmoid(outputData[index][0][j])
                out_q[write_index] = outputData[index][0][j]
                write_index += 1
        ids = []


def app(data_dir, threads, model_name, decision_threshold):

    listdata = os.listdir(data_dir)
    runTotal = len(listdata) * 100 * 2
    print("Total samples:", runTotal)

    global out_q
    out_q = [None] * runTotal

    if model_name == "GregNet2D":
        model = "/usr/share/vitis_ai_library/models/GregNet2D/GregNet2D.xmodel"
    elif model_name == "GregNet2D_nobatch":
        model = "/usr/share/vitis_ai_library/models/GregNet2D_nobatch/GregNet2D_nobatch.xmodel"
    elif model_name == "GregNet2DModified":
        model = "/usr/share/vitis_ai_library/models/GregNet2DModified/GregNet2DModified.xmodel"
    elif model_name == "GregNet2DModified_nobatch":
        model = "/usr/share/vitis_ai_library/models/GregNet2DModified_nobatch/GregNet2DModified_nobatch.xmodel"
    elif model_name == "GWaveNet2D":
        model = "/usr/share/vitis_ai_library/models/GWaveNet2D/GWaveNet2D.xmodel"
    elif model_name == "GWaveNet2DModified":
        model = "/usr/share/vitis_ai_library/models/GWaveNet2DModified/GWaveNet2DModified.xmodel"

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    print("subgraphs:", subgraphs)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    # input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    # input_scale = 2**input_fixpos

    """ preprocess samples """
    print(_divider)
    print("Pre-processing", runTotal, "samples...")

    data = []
    for i in range(len(listdata)):
        path = os.path.join(data_dir, listdata[i])
        tmp = preprocess_fn(path, "signal")
        # check if data array is empty
        if len(data) == 0:
            data = tmp
        else:
            data = np.concatenate((data, tmp), axis=0)

        tmp = preprocess_fn(path, "noise")
        data = np.concatenate((data, tmp), axis=0)

        print("Pre-processed", ((i + 1) * len(tmp) * 2), "samples...")

    """run threads """
    print(_divider)
    print("Starting", threads, "threads...")
    threadAll = []
    start = 0
    for i in range(threads):
        if i == threads - 1:
            end = len(data)
        else:
            end = start + (len(data) // threads)
        in_q = data[start:end]
        print("Thread", i, "processing", len(in_q), "samples...")
        print("Shape of in_q[0]:", in_q[0].shape)
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(_divider)
    print(
        "Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds"
        % (fps, runTotal, timetotal)
    )

    print(_divider)
    return


# only used if script is run as 'main' from command line
def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="data",
        help="Path to folder of samples. Default is data",
    )
    ap.add_argument(
        "-t", "--threads", type=int, default=1, help="Number of threads. Default is 1"
    )
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="GregNet2D",
        help="Path of xmodel. Default is GregNet2D",
    )
    ap.add_argument(
        "-dt",
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification. Default is 0.5",
    )
    args = ap.parse_args()

    print("Command line options:")
    print(" --data_dir           : ", args.data_dir)
    print(" --threads            : ", args.threads)
    print(" --model_name         : ", args.model_name)
    print(" --decision_threshold : ", args.decision_threshold)

    app(args.data_dir, args.threads, args.model_name, args.decision_threshold)


if __name__ == "__main__":
    main()
