import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
#from sklearn.utils import cpu_count
from joblib import cpu_count
from tqdm import tqdm

# PhysioNet Apnea-ECG dataset
# url: https://physionet.org/physiobank/database/apnea-ecg/

#base_dir = "dataset"
base_dir="testDataset"


fs = 100
sample = fs * 60  # 1 min's sample points

before = 2  # forward interval (min)
after = 2  # backward interval (min)
hr_min = 20
hr_max = 300

num_worker = 35 if cpu_count() > 35 else cpu_count() - 1  # Setting according to the number of CPU cores


def worker(name, labels):
    X = []
    y = []
    groups = []
    signals = wfdb.rdrecord(os.path.join(base_dir, name)).p_signal[:,0] # it looks first .hea file where it looks for the name and fromat of associated file.  
    # then looks for that file in the directory
    print(len(signals))
    print(signals[0:6000])

    for j in tqdm(range(len(labels[:5])), desc=name, file=sys.stdout):
        if j < before or (j + 1 + after) > len(signals) / float(sample):
             continue
    #     # debugging
    #     # print()
    #     # print("value of j",j)
    #     # print(len(signals))
    #     # print(len(labels))
    #     #print(signals)
    #
     #    signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]  # it takes 30000 samples of five minutes segment
    #     # Debugging
    #     print("length", len(signal))
    #     print(signal)
    #
    #     # signal filtering
    #     signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs),
    #                                      frequency=[3, 45], sampling_rate=fs)
    #
    #     # # Find R peaks
    #     rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
    #     rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
    #
    #
    #     # Remove abnormal R peaks signal
    #     if len(rpeaks) / (1 + after + before) < 40 or len(rpeaks) / (1 + after + before) > 200:
    #         continue
    #     # # Extract RRI, Ampl signal
    #     rri_tm  = rpeaks[1:] / float(fs)
    #     rri_signal = np.diff(rpeaks) / float(fs) # RR interval
    #
    #     rri_signal = medfilt(rri_signal, kernel_size=3)  # median filter
    #
    #     ampl_tm= rpeaks / float(fs)    # amplitude
    #     ampl_siganl= signal[rpeaks]
    #     hr = 60 / rri_signal
    #     # # Remove physiologically impossible HR signal
    #     if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
    #        # Save extracted signal
    #         X.append([(rri_tm, rri_signal), (ampl_tm, ampl_siganl)])
    #         y.append(0. if labels[j] == 'N' else 1.)
    #         groups.append(name)
    # return X, y, groups

# test purpose
names = ["a01","a02"]
labels=['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'N', 'N', 'N', 'N', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'N', 'N', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']

worker(names[0],labels)

if __name__ == "__main__":
    apnea_ecg = {}

    names = ["a01","a02"]
    #names = [
    #     "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    #     "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    #     "b01", "b02", "b03", "b04", "b05",
    #     "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    # ]

    o_train = []
    y_train = []
    groups_train = []
    print('Training...')  # this data is for training
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for i in range(len(names)-1):
            labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn").symbol
            print(labels)

            labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn")
            labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn", return_label_elements="label_store")
            labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn", return_label_elements="description")
            labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn", return_label_elements="symbol")
            print(labels.description)
            print(labels.label_store)
            print(labels)

            task_list.append(executor.submit(worker, names[i], labels))

    #
        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_train.extend(X)
            y_train.extend(y)
            groups_train.extend(groups)

    print()

    answers = {}
    with open(os.path.join(base_dir, "event-2-answers"), "r") as f:
        for answer in f.read().split("\n\n"):
           answers[answer[:3]] = list("".join(answer.split()[2::2]))
    #
    names = ["x01"]
    #     "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
    #     "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
    #     "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
    #     "x31", "x32", "x33", "x34", "x35"
    # ]

    o_test = []
    y_test = []
    groups_test = []
    print("Testing...")  # this data is for testing
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        print(len(names))
        for i in range(len(names)):
            labels = answers[names[i]]
            print(labels)
            task_list.append(executor.submit(worker, names[i], labels))

        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_test.extend(X)
            y_test.extend(y)
            groups_test.extend(groups)
    #
    apnea_ecg = dict(o_train=o_train, y_train=y_train, groups_train=groups_train, o_test=o_test, y_test=y_test,
                      groups_test=groups_test)
    with open(os.path.join(base_dir, "/TrainingTesting/apnea_data.pkl"), "wb") as f:
         pickle.dump(apnea_ecg, f, protocol=2)

    print("\nok!")
