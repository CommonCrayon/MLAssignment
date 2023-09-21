import numpy as np
import cv2

# Load the saved weights and biases from the binary file
loaded_network = np.load('neural_network_weights.npy', allow_pickle=True).item()

# Access the loaded weights and biases for the first and second sets
loaded_weights_1 = loaded_network['weights_1']
loaded_biases_1 = loaded_network['biases_1']

loaded_weights_2 = loaded_network['weights_2']
loaded_biases_2 = loaded_network['biases_2']

class_number_dict = loaded_network['class_number_dict']

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def predict(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return np.argmax(A2, 0)

def get_prediction(filename):
        image = cv2.imread(filename, 0)
        image = cv2.resize(image, (32, 32), interpolation= cv2.INTER_LINEAR)
        image = np.array(image)
        image = image.reshape(-1, 1024)

        X_test = image.T
        X_test = X_test / 255.

        prediction = predict(loaded_weights_1, loaded_biases_1, loaded_weights_2, loaded_biases_2, X_test)

        for x in class_number_dict.items():
            if (int(x[1]) == int(prediction)):
                return x[0]