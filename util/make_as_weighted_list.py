"""AudioSet metadata maker for M2D-AS

This utility requires `data/files_audioset.csv` as input.
Before you begin, make the list of AudioSet files as "data/files_audioset.csv" for the M2D pre-training by following "Example preprocessing steps (AudioSet)" in data/README.

In the M2D folder, you can create "data/files_as_weighted.csv" containing both sample path and labels (and also sample weights) with the following.

    python util/make_as_weighted_list.py

"""

from re import U
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import fire


def download_segment_csv():
    EVAL_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv'
    BALANCED_TRAIN_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv'
    UNBALANCED_TRAIN_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
    CLASS_LABEL_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'

    for subset_url in [EVAL_URL, BALANCED_TRAIN_URL, UNBALANCED_TRAIN_URL, CLASS_LABEL_URL]:
        subset_path = '/tmp/' + Path(subset_url).name
        if Path(subset_path).is_file():
            continue
        with open(subset_path, 'w') as f:
            subset_data = urllib.request.urlopen(subset_url).read().decode()
            f.write(subset_data)
            print('Wrote', subset_path)


def gen_weight(df, label_file):
    # Following AudioMAE https://github.com/facebookresearch/AudioMAE/blob/main/dataset/audioset/gen_weight.py

    def make_index_dict(label_csv):
        index_lookup = {}
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            line_count = 0
            for row in csv_reader:
                index_lookup[row['mid']] = row['index']
                line_count += 1
        return index_lookup

    index_dict = make_index_dict(label_file)
    label_count = np.zeros(527)

    for sample in df.label.values:
        sample_labels = sample.split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    label_weight = 1000.0 / (label_count + 100)

    sample_weight = np.zeros(len(df))
    for i, sample in enumerate(df.label.values):
        sample_labels = sample.split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            # summing up the weight of all appeared classes in the sample, note audioset is multiple-label classification
            sample_weight[i] += label_weight[label_idx]
    sample_weight = np.power(sample_weight, 1.0/1.5)  # making the weights softer
    df['weight'] = sample_weight
    return df


def make_metadata(org_list='data/files_audioset.csv', to_list='data/files_as_weighted.csv'):
    # download the original metadata.
    download_segment_csv()

    # load label maps.
    e_df = pd.read_csv('/tmp/eval_segments.csv', skiprows=2, sep=', ', engine='python')
    e_df['split'] = 'eval_segments'
    b_df = pd.read_csv('/tmp/balanced_train_segments.csv', skiprows=2, sep=', ', engine='python')
    b_df['split'] = 'balanced_train_segments'
    u_df = pd.read_csv('/tmp/unbalanced_train_segments.csv', skiprows=2, sep=', ', engine='python')
    u_df['split'] = 'unbalanced_train_segments'
    df = pd.concat([e_df, b_df, u_df])
    df = df[['# YTID', 'positive_labels', 'split']].copy()
    df.columns = ['ytid', 'label', 'split']
    # clean labels.
    def remove_quotations(s):
        assert s[0] == '"' and s[-1] == '"'
        return s[1:-1]
    df.label = df.label.apply(lambda s: remove_quotations(s))
    label_mapper = {ytid: label for ytid, label in df[['ytid', 'label']].values}

    # calculate weights for each sample in org_list, and store the results in to_list.
    org_df = pd.read_csv(org_list)  # assert: org_list has only one column "file_name"
    org_df['label'] = org_df.file_name.apply(lambda f: label_mapper[f.split('/')[-1][:11]])  # assign labels for each file_name
    new_df = gen_weight(org_df, '/tmp/class_labels_indices.csv')  # assign sample weights for each file_name
    new_df.to_csv(to_list, index=None)
    print('Created', to_list, 'based on', org_list)


fire.Fire(make_metadata)
