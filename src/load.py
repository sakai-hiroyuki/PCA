from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_digits


def get_mnist():
    (_, _), (data, _) = mnist.load_data()

    data = data.reshape(-1,784)
    data = data.astype('float32')
    data /= 255 * 10

    return data

def get_digits():
    data, _ = load_digits(return_X_y=True)
    
    data = data.astype('float32')
    data /= 16 * 10

    return data
