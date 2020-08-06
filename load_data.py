import string

import cv2
import numpy as np
import os
import random
import torch.utils.data as D
from datetime import datetime

from pairs_idx_wid_iam import WID2LABEL_TR, WID2LABEL_TE

CREATE_PAIRS = False

IMG_HEIGHT = 64
IMG_WIDTH = 216
MAX_CHARS = 10
# NUM_CHANNEL = 15
NUM_CHANNEL = 50
EXTRA_CHANNEL = NUM_CHANNEL + 1
# NUM_WRITERS = 500  # iam
NUM_WRITERS = 579  # merged
NORMAL = True
OUTPUT_MAX_LEN = MAX_CHARS + 2  # <GO>+groundtruth+<END>

img_base = '/home/hendrik/GANwriting/data/'
text_corpus = 'corpora_english/brown-azAZ.tr'

with open(text_corpus, 'r') as _f:
    text_corpus = _f.read().split()


def generate_date():
    date_formats = ['%d.%m.%y', '%d.%m.%Y', '%d-%m-%y']

    start = datetime.strptime('01.01.1000', '%d.%m.%Y')
    end = datetime.strptime('01.01.2020', '%d.%m.%Y')
    delta = end - start

    rand_date = start + delta * random.random()

    return rand_date.strftime(random.choice(date_formats))


def generate_num():
    string = str(random.randint(0, 100000))
    if random.choice([True, False]):
        # Add 0 padding for some nums without increasing max num length
        string = string[:-1].zfill(len(string))

    return string


dates = []
for i in range(len(text_corpus) // 4):
    date = generate_date()
    dates.append(date)
text_corpus.extend(dates)

for i in range(len(text_corpus) // 2):
    num = generate_num()
    text_corpus.append(num)

# src = 'Groundtruth/gan.iam.tr_va.gt.filter27'
# src = 'Groundtruth/train_with_numbers_n_dates_mixed_random_wid'
src = 'Groundtruth/train_numbers_gen_numbers_dates_mixed_with_wid'
tar = 'Groundtruth/gan.iam.test.gt.filter27'


def labelDictionary():
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
              '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '`']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter


num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())
vocab_size = num_classes + num_tokens


def edits1(word, min_len=2, max_len=MAX_CHARS):
    "All edits that are one edit away from `word`."
    letters = list(string.ascii_lowercase)
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    if len(word) <= min_len:
        return random.choice(list(set(transposes + replaces + inserts)))
    elif len(word) >= max_len:
        return random.choice(list(set(deletes + transposes + replaces)))
    else:
        return random.choice(list(set(deletes + transposes + replaces + inserts)))


class IAM_words(D.Dataset):
    def __init__(self, data_dict, oov):
        self.data_dict = data_dict
        self.oov = oov
        self.output_max_len = OUTPUT_MAX_LEN

    # word [0, 15, 27, 13, 32, 31, 1, 2, 2, 2]
    def new_ed1(self, word_ori):
        word = word_ori.copy()
        start = word.index(tokens['GO_TOKEN'])
        fin = word.index(tokens['END_TOKEN'])
        word = ''.join([index2letter[i - num_tokens] for i in word[start + 1: fin]])
        new_word = edits1(word)
        label = np.array(self.label_padding(new_word, num_tokens))
        return label

    def __getitem__(self, wid_idx_num):
        words = self.data_dict[wid_idx_num]
        '''shuffle images'''
        np.random.shuffle(words)

        wids = list()
        idxs = list()
        imgs = list()
        img_widths = list()
        labels = list()

        for word in words:
            wid, idx = word[0].split(',')
            img, img_width = self.read_image_single(idx)
            label = self.label_padding(' '.join(word[1:]), num_tokens)

            insert_pos = -1
            wids.insert(insert_pos, wid)
            idxs.insert(insert_pos, idx)
            imgs.insert(insert_pos, img)
            img_widths.insert(insert_pos, img_width)
            labels.insert(insert_pos, label)

        final_wid = wid_idx_num
        num_imgs = len(imgs)
        if num_imgs >= EXTRA_CHANNEL:
            final_img = np.stack(imgs[:EXTRA_CHANNEL], axis=0)  # 64, h, w
            final_idx = idxs[:EXTRA_CHANNEL]
            final_img_width = img_widths[:EXTRA_CHANNEL]
            final_label = labels[:EXTRA_CHANNEL]
        else:
            final_idx = idxs
            final_img = imgs
            final_img_width = img_widths
            final_label = labels

            while len(final_img) < EXTRA_CHANNEL:
                num_cp = EXTRA_CHANNEL - len(final_img)
                final_idx = final_idx + idxs[:num_cp]
                final_img = final_img + imgs[:num_cp]
                final_img_width = final_img_width + img_widths[:num_cp]
                final_label = final_label + labels[:num_cp]
            final_img = np.stack(final_img, axis=0)

        _id = np.random.randint(EXTRA_CHANNEL)
        img_xt = final_img[_id:_id + 1]
        if self.oov:
            label_xt = np.random.choice(text_corpus)
            label_xt = np.array(self.label_padding(label_xt, num_tokens))
            label_xt_swap = np.random.choice(text_corpus)
            label_xt_swap = np.array(self.label_padding(label_xt_swap, num_tokens))
        else:
            label_xt = final_label[_id]
            label_xt_swap = self.new_ed1(label_xt)

        final_idx = np.delete(final_idx, _id, axis=0)
        final_img = np.delete(final_img, _id, axis=0)
        final_img_width = np.delete(final_img_width, _id, axis=0)
        final_label = np.delete(final_label, _id, axis=0)

        return 'src', final_wid, final_idx, final_img, final_img_width, final_label, img_xt, label_xt, label_xt_swap

    def __len__(self):
        return len(self.data_dict)

    def read_image_single(self, file_name):
        url = os.path.join(img_base, file_name + '.png')
        if not os.path.exists(url):
            print(f"{url} doesn't seem to exist")
            exit(3)
        img = cv2.imread(url, 0)

        if img is None:
            # image is present but corrupted
            return np.zeros((IMG_HEIGHT, IMG_WIDTH)), 0

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        img = img / 255.  # 0-255 -> 0-1

        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]  # TODO: WTF is this shit?!
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal, img_width

    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll) + 2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = self.output_max_len - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num)  # replace PAD_TOKEN
        return ll


def get_dict(groundtruth, wid_mapping):
    with open(groundtruth, 'r') as f:
        lines = f.readlines()

    lines = [line.strip().split(' ') for line in lines]
    writer_id_dict = dict()
    for line in lines:
        writer_id = line[0].split(',')[0]
        if writer_id not in writer_id_dict.keys():
            writer_id_dict[writer_id] = [line]
        else:
            writer_id_dict[writer_id].append(line)

    normalised_writer_id_dict = dict()
    if CREATE_PAIRS:
        create_pairs(writer_id_dict)

    for writer_id, lines in writer_id_dict.items():
        # workaround for new writers so I don't have to edit this beast of a dict every time, I change the dataset
        if writer_id not in wid_mapping:
            wid_mapping[writer_id] = len(wid_mapping)

        # if there are generated samples add them to each writer id proportionally to number of actual samples
        additional_samples = []
        normalised_writer_id_dict[wid_mapping[writer_id]] = lines + additional_samples

    return normalised_writer_id_dict


def loadData(oov):
    groundtruth_train = src
    groundtruth_test = tar

    new_train_dict = get_dict(groundtruth_train, WID2LABEL_TR)
    new_test_dict = get_dict(groundtruth_test, WID2LABEL_TE)

    data_train = IAM_words(new_train_dict, oov)
    data_test = IAM_words(new_test_dict, oov)
    return data_train, data_test


def create_pairs(ddict):
    num = len(ddict.keys())
    label2wid = list(zip(range(num), ddict.keys()))
    print(label2wid)


if __name__ == '__main__':
    pass
