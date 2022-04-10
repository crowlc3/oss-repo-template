# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image, ImageOps

print(tf.__version__)
# My Images

shirt_image = Image.open("./shirt.jpg")
shirt_img = ImageOps.grayscale(shirt_image).resize((28,28))
img_arr_shirt = np.array(shirt_img) / 255
shirt_img.save('gray_image_shirt.png')

pants_image = Image.open("./pants.jpg")
pants_img = ImageOps.grayscale(pants_image).resize((28,28))
img_arr_pant = np.array(pants_img) / 255
pants_img.save('gray_image_pants.png')

bag_image = Image.open("./bag.jpg")
bag_img = ImageOps.grayscale(bag_image).resize((28,28))
img_arr_bag = np.array(bag_img) / 255
bag_img.save('gray_image_bag.png')

print(img_arr_bag)