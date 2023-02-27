import numpy as np
import pandas as pd
from PIL import Image
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import tensorflow as tf
from tqdm import tqdm


def load_img(path):
    try:
        image = tf.keras.utils.load_img(path, target_size=(224, 224))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])
        input_arr = ((input_arr - 127.5) / 127.5)
    except FileNotFoundError:
        return

    return input_arr


def model_init():
    model_path = 'model.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter


def feed_img(interpreter, img):
    #input_details[0]['index']=396
    #output_details[1]['index'] = 417
    #output_details[0]['index'] = 397

    input_shape = []
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(396, input_data)
    interpreter.invoke()
    output_data_slots = interpreter.get_tensor(417)
    slot_number = np.argmax(output_data_slots)
    output_data_features = interpreter.get_tensor(397)

    return int(slot_number), output_data_features[0].tolist()


def send_to_base(client, idx, category, color, ali_id, filename, slot, emb):

    operation_info = client.upsert(
        collection_name="products",
        wait=True,
        points=[
            PointStruct(id=idx,
                        vector={
                            "image_emb": emb,
                        },
                        payload={"category": category,
                                 "predicted_slot": slot,
                                 "ali_id": ali_id,
                                 "color": color,
                                 "filename": filename}),
        ]
    )



def get_img_path(row):
    path_to = '../clothes-dataset/imgs'
    path_to_id = os.path.join(os.path.join(path_to, str(row.id)))
    path_to_color = os.path.join(path_to_id, row.color)
    path_to_source = os.path.join(path_to_color, row.source)
    filename = row.url.split('/')[-1]
    path_to_source = os.path.join(path_to_source, filename)

    return path_to_source, filename, row.category


def dataframe_process(df_path):

    df = pd.read_csv(df_path)
    df = df[df.source == 'shop']
    df['id'] = df['id'].astype(int)

    interpreter = model_init()

    client = QdrantClient(host="localhost", port=6333)

    existing_ids = client.retrieve(collection_name="products",
                                    ids=df.index.tolist())
    existing_ids = set([item.id for item in existing_ids])

    print("{} objects are already in the database".format(len(existing_ids)))

    for idx, row in tqdm(df.iterrows(),total=df.shape[0]):
        if idx in existing_ids:
            continue

        img_path, filename, category = get_img_path(row)
        img = load_img(img_path)
        if img is None:
            continue

        slot, features = feed_img(interpreter, img)

        send_to_base(client, idx, row.category, row.color, row.id, filename, slot, features)


if __name__ == '__main__':
    df_path = '../clothes-dataset/clothes-dataset.csv'

    dataframe_process(df_path)
