# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image, ImageOps

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  i = i + 9000
  plt.subplot(num_rows, 2*num_cols, 2*(i-9000)+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*(i-9000)+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# My Images

shirt_image = Image.open("./shirt.jpg")
shirt_img = ImageOps.grayscale(shirt_image).resize((28,28))
shirt_img = ImageOps.invert(shirt_img)
img_arr_shirt = np.array(shirt_img) / 255
img_arr_shirt = np.expand_dims(np.array(img_arr_shirt),0)
shirt_img.save('gray_image_shirt.png')

pants_image = Image.open("./pants.jpg")
pants_img = ImageOps.grayscale(pants_image).resize((28,28))
pants_img = ImageOps.invert(pants_img)
img_arr_pant = np.array(pants_img) / 255
img_arr_pant = np.expand_dims(np.array(img_arr_pant),0)
pants_img.save('gray_image_pants.png')

bag_image = Image.open("./bag.jpg")
bag_img = ImageOps.grayscale(bag_image).resize((28,28))
bag_img = ImageOps.invert(bag_img)
img_arr_bag = np.array(bag_img) / 255
img_arr_bag = np.expand_dims(np.array(img_arr_bag),0)
bag_img.save('gray_image_bag.png')

full_test_images = [img_arr_shirt,img_arr_pant,img_arr_bag]
predictions = [probability_model.predict(img) for img in full_test_images]

print("peepee")
print(predictions[0])

predictions[0]

np.argmax(predictions[0])

lab = np.array([6,1,8])

num_rows = 1
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i][0], lab, [shirt_img, pants_img, bag_img])
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i][0], lab)
plt.tight_layout()
plt.show()
plt.savefig("myguesses.jpg")