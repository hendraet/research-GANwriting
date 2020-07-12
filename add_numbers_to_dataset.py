from collections import Counter

import copy


def main():
    num_test_filename = "../GANwriting/data/train" # TODO: maybe add test and val as well
    orig_gt_filename = "Groundtruth/gan.iam.tr_va.gt.filter27"
    orig_writer_map_filename = "../GANwriting/data/orig_file_writer_map.csv"
    out_filename = "Groundtruth/train_with_numbers"

    with open(orig_writer_map_filename, "r") as ofmf:
        orig_mapping = [line.rstrip().split(",") for line in ofmf.readlines()]
    map_dict = {pair[0].split(".")[0]: pair[1] for pair in orig_mapping}

    # Check if my writer id map equal theirs:
    # ----------------------- works ------------------
    with open(orig_gt_filename, "r") as ogt:
        orig_lines = ogt.readlines()
    wid_file_map = [line.split()[0].split(",") for line in orig_lines]
    shortend_wid_file_map = set([(pair[0], "-".join(pair[1].split("-")[:2])) for pair in wid_file_map])

    for wid, filename in shortend_wid_file_map:
        if filename in map_dict.keys():
            if map_dict[filename] != wid:
                print(f"Mismatch: {wid} {map_dict[filename]}")

    with open(num_test_filename, "r") as ntf:
        number_lines = [line for line in ntf.readlines() if any(char.isdigit() for char in line.split(",")[1])]
    cleansed_number_lines = [line.rstrip().split(",") for line in number_lines]

    MAX_CHARS=10
    cleansed_number_lines = [(a, b) for a, b in cleansed_number_lines if len(b) <= MAX_CHARS]

    # migrate to format [wid,filename word]
    new_lines = []
    new_map ={}
    all_num_wids = set()
    for line in cleansed_number_lines:
        writer_id = map_dict["-".join(line[0].split("-")[:2])]
        new_lines.append(f"{writer_id},{line[0]} {line[1]}\n")
        all_num_wids.add(writer_id)

        if writer_id in new_map.keys():
            new_map[writer_id] = new_map[writer_id] + [line[1]]
        else:
            new_map[writer_id] = [line[1]]

    # append and sort
    merged_lines = copy.copy(orig_lines)
    merged_lines.extend(new_lines)
    sorted(merged_lines)
    # TODO: remove writers with too few images?

    # Check if enough images for writers with numbers
    writer_id_count = Counter([line.split(",")[0] for line in merged_lines])
    image_count_for_num_writers = [(id, writer_id_count[id]) for id in all_num_wids]

    lost_writer_ids = [blah[0] for blah in image_count_for_num_writers if blah[1] < 15]
    num_lost_writers = len(lost_writer_ids)
    print(f"Writers lost because of too few samples: {num_lost_writers}")

    num_lost_images = sum([len(new_map[writer_id]) for writer_id in lost_writer_ids])
    print(f"Images lost because of too few samples: {num_lost_images}")

    with open(out_filename, "w") as outf:
        for line in merged_lines:
            outf.write(line)

if __name__ == '__main__':
    main()