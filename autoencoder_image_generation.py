from tensorflow import keras
import sklearn
import matplotlib.pyplot as plt


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_reconstructions(model,images):
    n_images=len(images)
    reconstructions = model.predict(images)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    plt.show()


fashion_mnist = keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
X_train=X_train/255.0
X_test=X_test/255.0



model_img_reconstruction = keras.models.load_model("image_reconstruction_model.h5")

images=X_train[4:9]
show_reconstructions(model_img_reconstruction,images)

