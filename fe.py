import tflite_runtime.interpreter as tflite
import numpy as np

from utils import load_img, img_to_array

class FeatureExtractor:
    def __init__(self):
        self.fe_interpreter = self._fe_model_init_()

    def _fe_model_init_(self):
        model_path = 'model.tflite'
        interpreter = tflite.Interpreter(model_path=model_path)
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        return interpreter

    def _load_img_(self, path):
        try:
            image = load_img(path, target_size=(224, 224))
            #image = tf.keras.utils.load_img(path, target_size=(224, 224))
            input_arr = img_to_array(image)
            input_arr = np.array([input_arr])
            input_arr = ((input_arr - 127.5) / 127.5)
        except FileNotFoundError:
            return

        return input_arr

    def feed_img(self, img):
        # input_details[0]['index']=396
        # output_details[1]['index'] = 417
        # output_details[0]['index'] = 397

        input_data = np.array(img, dtype=np.float32)
        self.fe_interpreter.set_tensor(396, input_data)
        self.fe_interpreter.invoke()
        output_data_slots = self.fe_interpreter.get_tensor(417)
        subslot_number = np.argmax(output_data_slots)
        output_data_features = self.fe_interpreter.get_tensor(397)

        return int(subslot_number), output_data_features[0]

    def feed_img_by_path(self, path):
        img = _load_img_(path)
        if not img:
            return

        return self.feed_img(img)
