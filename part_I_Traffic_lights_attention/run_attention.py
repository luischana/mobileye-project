import PIL
from matplotlib import patches

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


# def print_picture(im_green, c_image, result, grad):
#     fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(12, 30))
#     ax_orig.imshow(im_green)
#     ax_orig.set_title('Original')
#     ax_orig.set_axis_off()
#     ax_mag.imshow(np.absolute(grad))
#     ax_mag.set_title('Gradient magnitude')
#     ax_mag.set_axis_off()
#     ax_ang.imshow(np.absolute(result))
#     ax_ang.set_title('Gradient orientation')
#     ax_ang.set_axis_off()
#     fig.show()
#     fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
#     max_mag.set_axis_off()
#     max_mag.imshow(np.absolute(c_image))
#     fig.show()


def non_max_suppression(dim, result, c_image, color):
    coord_x, coord_y = [], []

    image_height, image_width = result.shape[:2]

    for i in range(0, image_height - dim, dim):
        for j in range(0, image_width - dim, dim):
            max_coord = np.argmax(result[i:i + dim, j:j + dim])
            local_max = np.amax(result[i:i + dim, j:j + dim])

            if local_max > 80:
                coord_x.append(max_coord // dim + i)
                coord_y.append(max_coord % dim + j)
                c_image[max_coord // dim + i, max_coord % dim + j] = color

    return coord_x, coord_y


def green_picture(im_green, filter_kernel, c_image):
    grad = sg.convolve2d(im_green, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    x_green, y_green = non_max_suppression(18, result, c_image, [0, 255, 0])
    # print_picture(im_green, c_image, result, grad)

    return x_green, y_green


def red_picture(im_red, filter_kernel, c_image):
    grad = sg.convolve2d(im_red, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    x_red, y_red = non_max_suppression(14, result, c_image, [255, 0, 0])
    # print_picture(im_red, c_image, result, grad)

    return x_red, y_red


def find_tfl_lights(c_image: np.ndarray, **kwargs):

    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    filter_kernel = ([
        [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, 11/225, 11/225, 11/225, 11/225, 11/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, 11/225, 11/225, 11/225, 11/225, 11/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, 11/225, 11/225, 11/225, 11/225, 11/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, 11/225, 11/225, 11/225, 11/225, 11/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, 11/225, 11/225, 11/225, 11/225, 11/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -2/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
        [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225]])

    im_red = c_image[:, :, 0]
    im_green = c_image[:, :, 1]

    x_red, y_red = red_picture(im_red, filter_kernel, c_image)
    x_green, y_green = green_picture(im_green, filter_kernel, c_image)

    print(f"x_red  : {x_red}")
    print(f"y_red  : {y_red}")
    print(f"x_green: {x_green}")
    print(f"y_green: {y_green}")

    return x_red, y_red, x_green, y_green


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):

    """
    Run the attention code
    """

    image = np.array(Image.open(image_path))

    if json_path is None:
        objects = None

    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    x_red, y_red, x_green, y_green = find_tfl_lights(image, some_threshold=42)


def main(argv=None):

    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')

    args = parser.parse_args(argv)
    default_base = 'data'

    if args.dir is None:
        args.dir = default_base

    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None

        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")

    else:
        print("Bad configuration?? Didn't find any picture to show")

    plt.show(block=True)


if __name__ == '__main__':
    main()
