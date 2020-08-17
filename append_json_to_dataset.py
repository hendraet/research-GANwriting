import os

import json


def main():
    base_data_dir = "/home/hendrik/GANwriting/data"
    json_data_dir = "iamondb_generated_nums_dates_20k"
    in_json_filename = os.path.join(base_data_dir, json_data_dir, "iamondb_generated_nums_dates_20k.json")
    orig_data_dir = "iamdb_images_flat"
    in_orig_ds_filename = "Groundtruth/train_with_numbers"
    # in_orig_ds_filename = "Groundtruth/train_numbers_gen_numbers_dates_mixed_no_wid"
    out_ds_filename = "Groundtruth/train_gen_numbers_dates_only_with_wid"
    wid_generation_method = "from_path"  # choose from ["random", "from_path", "static"]
    line_limit = 20000

    with open(in_orig_ds_filename, "r") as orig_file:
        orig_lines = orig_file.readlines()

    highest_wid = int(max([line.split(",")[0] for line in orig_lines])) + 1

    # TODO --------------------------------------------------
    orig_lines = []
    highest_wid = 0

    with open(in_json_filename, "r") as in_json:
        json_ds = json.load(in_json)

    new_lines = []
    # TODO: decide how to handle wids: new for every sample, one for all, or mixed approach
    #   - new is not feasible because we also need blocks auf k images
    #   - all could be working because data is online data, so some style info is missing anyways
    #   - best would probably to map chars to actual writers and take one of the actual writers?
    for running_wid, sample in enumerate(json_ds):
        if wid_generation_method == "random":
            new_wid = str(highest_wid + running_wid // 50)
        elif wid_generation_method == "static":
            new_wid = "-1"
        elif wid_generation_method == "from_path":
            filename = os.path.splitext(os.path.basename(sample['path']))[0]
            new_wid = filename.split("_")[1]
        else:
            assert False, f"{wid_generation_method} is an unknown generation method"

        new_line = f"{new_wid},{os.path.join(json_data_dir, os.path.splitext(sample['path'])[0])} {sample['string']}"
        new_lines.append(new_line)
    new_lines = new_lines[:line_limit]

    merged_lines = new_lines
    for i, line in enumerate(orig_lines):
        wid, rest = line.split(",")
        split_rest = rest.split()
        if len(split_rest) > 2:  # Weird strings
            continue
        path, string = rest.split()
        merged_lines.append(f"{wid},{os.path.join(orig_data_dir, path)} {string}")

    with open(out_ds_filename, "w") as out_file:
        for line in merged_lines:
            out_file.write(line.rstrip() + "\n")


if __name__ == '__main__':
    main()
