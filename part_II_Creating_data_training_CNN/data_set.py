from os.path import join

from part_I_Traffic_lights_attention.run_attention import *

try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def read_data(data_file, label_file, index):
    fpo = np.memmap(data_file, dtype=np.uint8, offset=81*81*3*index, mode='r')
    fpo = fpo[:81*81*3]
    img = fpo.reshape(81, 81, 3)
    label = np.memmap(label_file, dtype=np.uint8, mode='r', shape=(1,), offset=index)

    return img, label


def correct_data(data_file, label_file, index):
    img, label = read_data(data_file, label_file, index)
    plt.imshow(img)
    plt.title("Traffic light" if label else "Not Traffic light")
    plt.show()


def change(data_file):
    with open(f"{data_file}/data_mirror.bin", "ab") as data:
        images = np.memmap(join(data_file + '/data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list((81, 81)) + [3])

        for img in images:
            img = img[..., ::-1, :]
            save_image(img, data)


def save_label(labels, label):
    labels.write(label.to_bytes(1, byteorder='big', signed=False))


def save_image(crop_image, data):
    np.array(crop_image, dtype=np.uint8).tofile(data)


def crop(image, labels, x_coord, y_coord, str, data):
    crop_image = image[x_coord:x_coord + 81, y_coord:y_coord + 81]
    save_image(crop_image, data)
    save_label(labels, str)


def coord_to_crop(image_label, image, x_coord, y_coord, first, last, labels, data):
    count1, count2, flag1, flag2, index_traffic, index_not_traffic = 0, 0, 0, 0, -1, -1

    if first > last:
        side = -1

    else:
        side = 1

    for i in range(first, last, side):
        if not flag1 or not flag2:
            if image_label[x_coord[i], y_coord[i]] == 19:
                if not flag1:
                    flag1 = 1
                    index_traffic = i
            else:
                if not flag2:
                    flag2 = 1
                    index_not_traffic = i

        else:
            break

    if flag1 and flag2:
        crop(image, labels, x_coord[index_traffic], y_coord[index_traffic], 1, data)
        crop(image, labels, x_coord[index_not_traffic], y_coord[index_not_traffic], 0, data)


def pad_with_zeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0


def get_coord(image):
    x_red, y_red, x_green, y_green = find_tfl_lights(image, some_threshold=42)

    return x_red+x_green, y_red+y_green


def set_data(name_dir):
    label_path = './data/gtFine'
    image_path = './data/leftImg8bit'

    for root, dirs, files in os.walk(image_path + f'/{name_dir}'):

        for dir in dirs:
            list_images = glob.glob(os.path.join(image_path + f'/{name_dir}/' + dir, '*_leftImg8bit.png'))
            list_labels = glob.glob(os.path.join(label_path + f'/{name_dir}/' + dir, '*_gtFine_labelIds.png'))

            for im_path, la_path in zip(list_images, list_labels):
                image = np.array(Image.open(im_path))
                label = np.array(Image.open(la_path))
                x_coord, y_coord = get_coord(image)

                if not len(x_coord):
                    continue

                image = np.pad(image, 40, pad_with_zeros)[:, :, 40:43]

                with open(f"./data/Data_dir/{name_dir}/data.bin", "ab") as data:
                    with open(f"./data/Data_dir/{name_dir}/labels.bin", "ab") as labels:
                        coord_to_crop(label, image, x_coord, y_coord, 0, len(x_coord) - 1, labels, data)
                        coord_to_crop(label, image, x_coord, y_coord, len(x_coord) - 1, 0, labels, data)


# set_data('train')
# set_data('val')
# change(f"./data/Data_dir/train")
# correct_data(f"./data/Data_dir/train/data.bin", f"./data/Data_dir/train/labels.bin", 0)
# correct_data(f"./data/Data_dir/val/data.bin", f"./data/Data_dir/val/labels.bin", 2)
