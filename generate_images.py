import random

import Levenshtein as Lev
import argparse
import cv2
import numpy as np
import os
import torch

from load_data import IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index, tokens, num_tokens, OUTPUT_MAX_LEN, index2letter
from modules_tro import normalize
from network_tro import ConTranModel


def read_image(file_name):
    url = file_name + ".png"
    if not os.path.exists(url):
        print("Url doesn't exist:", url)
    img = cv2.imread(url, 0)
    if img is None:
        print("Img is broken:", url)

    rate = float(IMG_HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT),
                     interpolation=cv2.INTER_CUBIC)  # INTER_AREA con error
    img = img / 255.  # 0-255 -> 0-1

    img = 1. - img
    img_width = img.shape[-1]

    if img_width > IMG_WIDTH:
        out_img = img[:, :IMG_WIDTH]
    else:
        out_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype="float32")
        out_img[:, :img_width] = img
    out_img = out_img.astype("float32")

    mean = 0.5
    std = 0.5
    out_img_final = (out_img - mean) / std
    return out_img_final


def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def prep_images(imgs):
    random.shuffle(imgs)
    final_imgs = imgs[:50]
    if len(final_imgs) < 50:
        while len(final_imgs) < 50:
            num_cp = 50 - len(final_imgs)
            final_imgs = final_imgs + imgs[:num_cp]

    imgs = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).cuda()  # 1,50,64,216
    return imgs


def test_writer(imgs, wid, label, model, out_dir):
    imgs = prep_images(imgs)

    with torch.no_grad():
        f_xs = model.gen.enc_image(imgs)
        label = label.unsqueeze(0)
        f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
        f_mix = model.gen.mix(f_xs, f_embed)
        xg = model.gen.decode(f_mix, f_xt)
        pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

        label = label.squeeze().cpu().numpy().tolist()
        pred = torch.topk(pred, 1, dim=-1)[1].squeeze()
        pred = pred.cpu().numpy().tolist()
        for j in range(num_tokens):
            label = list(filter(lambda x: x != j, label))
            pred = list(filter(lambda x: x != j, pred))
        label = "".join([index2letter[c - num_tokens] for c in label])
        pred = "".join([index2letter[c - num_tokens] for c in pred])

        # TODO: fid score for eval?

        ed_value = Lev.distance(pred, label)
        lev_thresh = 2
        if ed_value > lev_thresh:
            pred_success = "fail"
            print(f"{ed_value}: {label} {pred}")
            return False
        else:
            pred_success = "succ"

        # out_filename = f"{out_dir}/{label}_{wid}_{pred_success}.png"
        out_filename = f"{out_dir}/{label}_{wid}.png"
        if os.path.exists(out_filename):
            return False

        xg = xg.cpu().numpy().squeeze()
        xg = normalize(xg)
        xg = 255 - xg
        cv2.imwrite(out_filename, xg)

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-file", type=str,
                        default="final_model_weights/word_dates_nums_styled_ds_pretrained_3000.model",
                        help="path of the pretrained model")
    parser.add_argument("-i", "--img-dir", type=str, default="/home/hendrik/GANwriting/data/",
                        help="directory of the style images")
    parser.add_argument("-w", "--wid-file", type=str, default="eval_files/short_wid_list",
                        help="file that contains writer ids (format is the same as for train and test files)")
    parser.add_argument("-l", "--label-files", nargs="+", type= str, default= ["eval_files/mixed_words_dates_nums"],
                        help="files that contain possible labels for generation")
    parser.add_argument("-o", "--out-dir", type=str, default="eval_files/mixed_imgs_3000/",
                        help="dir where the generated images should be saved")
    parser.add_argument("-n", "--num-images", type=int, default=100,
                        help="number of images that should be generated")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    labels = []
    for filename in args.label_files:
        with open(filename, "r") as _f:
            texts = _f.read().split()
        labels.extend([np.array(label_padding(label, num_tokens)) for label in texts])
    labels = torch.from_numpy(np.array(labels)).cuda()

    # data preparation
    data_dict = dict()
    with open(args.wid_file, "r") as _f:
        data = _f.readlines()
        data = [i.split(" ")[0] for i in data]
        data = [i.split(",") for i in data]

    wids = set()
    for wid, index in data:
        wids.add(wid)
        if wid in data_dict.keys():
            data_dict[wid].append(index)
        else:
            data_dict[wid] = [index]
    wids = list(wids)

    model = ConTranModel(NUM_WRITERS, 0, True).cuda()
    print("Loading " + args.model_file)
    model.load_state_dict(torch.load(args.model_file))  # load
    model.eval()

    # TODO
    # labels = labels[:10]

    max_num_combinations = len(wids) * len(labels)
    generated_samples = 0
    if args.num_images > max_num_combinations:
        print(f"Only {str(max_num_combinations)} different combinations can be generated based on provided writer ids "
              f"and labels.")
        for wid in wids:
            imgs = [read_image(os.path.join(args.img_dir, filename)) for filename in data_dict[wid]]
            for label in labels:
                success = test_writer(imgs, wid, label, model, args.out_dir)
                if success:
                    generated_samples += 1
    else:
        for i in range(args.num_images):
            label = random.choice(labels)
            wid = random.choice(wids)
            imgs = [read_image(os.path.join(args.img_dir, filename)) for filename in data_dict[wid]]
            success = test_writer(imgs, wid, label, model, args.out_dir)
            if success:
                generated_samples += 1

    print(f"{generated_samples} images with adequate quality were able to be generated.")


if __name__ == "__main__":
    main()
