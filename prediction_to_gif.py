import imageio
import os
from natsort import natsorted, ns

image_dir = './log/test_images/'
images_list = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')])
images_list = natsorted(images_list, alg=ns.IGNORECASE)
images = []
for filename in images_list:
    images.append(imageio.imread(filename))

imageio.mimsave('./log/predict_segmentation.gif', images)
