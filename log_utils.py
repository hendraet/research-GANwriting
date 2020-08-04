import math
import numpy as np
import os

import torch
from PIL import ImageDraw
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid

from load_data import num_tokens, index2letter
from modules_tro import fine


def decode_labels(labels):
    decoded_labels = []
    for i in range(len(labels)):
        decoded_label = labels[i].tolist()
        decoded_label = fine(decoded_label)
        for j in range(num_tokens):
            decoded_label = list(filter(lambda x: x != j, decoded_label))
        decoded_label = ''.join([index2letter[c - num_tokens] for c in decoded_label])
        decoded_labels.append(decoded_label)

    return decoded_labels


def decode_predictions(labels):
    labels = torch.topk(labels, 1, dim=-1)[1].squeeze(-1)
    return decode_labels(labels)


def get_predicted_wids(classifier, generated_images):
    softmax = nn.Softmax()
    classifier_output = classifier(generated_images)
    predicted_wid = softmax(classifier_output).argmax(1)
    return predicted_wid


def combine_and_add_text(in_img, out_img, texts_gen, pred_gen, texts_real, pred_real, writer_id,
                         predicted_writer_id_gen, predicted_writer_id_real):
    text_color = (255, 0, 0)
    to_pil = transforms.ToPILImage()

    denormalized_out_img = (out_img + 1) / 2
    denormalized_in_img = (in_img + 1) / 2
    inflated_in_img = denormalized_in_img.expand((3, -1, -1))
    inflated_out_img = denormalized_out_img[0].expand((3, -1, -1))
    combined_img = to_pil(torch.cat([inflated_in_img, inflated_out_img], dim=1).cpu())

    draw = ImageDraw.Draw(combined_img)
    # Text real
    draw.text((0, 0), f"{texts_real}: {pred_real}", text_color)
    draw.text((0, inflated_in_img.shape[1] - 10), f"{writer_id}: {predicted_writer_id_real.item()}", text_color)
    # Text generated
    draw.text((0, inflated_in_img.shape[1]), f"{texts_gen}: {pred_gen}", text_color)
    draw.text((0, combined_img.height - 10), f"{writer_id}: {predicted_writer_id_gen.item()}", text_color)

    return combined_img


def log_img_grid(recognizer, classifier, generated_images, labels, train_images, train_labels, train_writer_ids,
                 filename):
    all_imgs = []
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    img_width = train_images.shape[-1]
    batch_size = train_images.shape[0]
    with torch.no_grad():
        encoded_predictions_generated = recognizer(generated_images, labels,
                                                   torch.from_numpy(np.array([img_width] * batch_size)))
        encoded_predictions_real = recognizer(train_images, train_labels,
                                              torch.from_numpy(np.array([img_width] * batch_size)))
        gen_preds = decode_predictions(encoded_predictions_generated)
        train_preds = decode_predictions(encoded_predictions_real)

        wid_gen_preds = get_predicted_wids(classifier, generated_images)
        wid_real_preds = get_predicted_wids(classifier, train_images)

    decoded_labels = decode_labels(labels[:, 1:])
    decoded_train_labels = decode_labels(train_labels[:, 1:])
    for train_image, train_label, generated_image, label, gen_pred, train_pred, train_writer_id, wid_gen_pred, wid_real_pred \
            in zip(train_images, decoded_train_labels, generated_images, decoded_labels, gen_preds, train_preds,
                   train_writer_ids, wid_gen_preds,
                   wid_real_preds):
        combined_img = combine_and_add_text(train_image, generated_image, label, gen_pred, train_label, train_pred,
                                            train_writer_id, wid_gen_pred, wid_real_pred)

        all_imgs.append(to_tensor(combined_img))

    all_imgs = torch.stack(all_imgs)
    img_grid = make_grid(all_imgs, nrow=int(math.sqrt(len(all_imgs))), pad_value=255)

    with open(os.path.join("imgs/", filename + ".png"), "wb") as out_f:
        to_pil(img_grid).save(out_f)
