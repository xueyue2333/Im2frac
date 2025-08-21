import os
import argparse
import imageio.v3


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default=r'D:\A-Fkf\MVSNet_pytorch-master\new_data\keyboard',
                    help="the path contains the folder of images")

args = parser.parse_args()

base = "{}/dense/images".format(args.base_dir)

for i, name in enumerate(os.listdir(base)):
    os.rename("{}/{}".format(base, name), '{}/{:0>8}.jpg'.format(base, i))

height, weight = 10000, 10000
for i in os.listdir(base):
    h, w, _ = imageio.v3.imread("{}/{}".format(base, i)).shape
    height = min(height, h)
    weight = min(weight, w)

for i in os.listdir(base):
    img = imageio.v3.imread("{}/{}".format(base, i))
    h, w, _ = img.shape
    imageio.v3.imwrite("{}/{}".format(base, i), img[:height - height % 32, :weight - weight % 32, :])


