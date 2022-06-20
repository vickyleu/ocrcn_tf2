import tensorflow as tf
import numpy as np
import cv2

from dataset.casia_hwdb import load_characters


def validation():
    # Load the TFLite model and allocate tensors. View details
    interpreter = tf.lite.Interpreter(model_path="./converted_model.tflite")
    # print(interpreter.get_input_details())
    # print(interpreter.get_output_details())
    # print(interpreter.get_tensor_details())
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and transform image
    img = cv2.imread('./samples/testing.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64), 1)
    img = np.asarray(img)/255
    img = np.reshape(img, newshape=(1, 64, 64, 1))

    # Use same image as Keras model
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    charactors = load_characters()  # 读取标签列表
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # 标签预测
    w = np.argmax(output_data)  # 值最大的位置
    print(charactors[w])


if __name__ == "__main__":
    validation()
