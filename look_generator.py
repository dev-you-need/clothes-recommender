import tflite_runtime.interpreter as tflite
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from thing import Thing
from look import Look
from fe import FeatureExtractor

import random

class LookGenerator:
    def __init__(self):
        self.la_interpreter = self._la_model_init_()
        self.client = QdrantClient(
                host=os.environ.get('QDRANT_SERVER'),
                api_key=os.environ.get('QDRANT_API_KEY'),
            )
        self.fe = FeatureExtractor()

    def _la_model_init_(self):
        model_path = 'la.tflite'
        interpreter = tflite.Interpreter(model_path=model_path)
        #input_details = interpreter.get_input_details()
        #output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        return interpreter

    def _feed_data_(self, input_data):
        self.la_interpreter.set_tensor(10, np.expand_dims(input_data, axis=0))
        self.la_interpreter.invoke()
        output_data = self.la_interpreter.get_tensor(8)
        return output_data[0][1]

    def _get_things_for_slot_(self, slot, slot_offset):
        result = self.client.scroll(
            collection_name="products",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="slot",
                        match=MatchValue(value=slot)
                    ),
                ]
            ),
            limit=20,
            with_payload=True,
            with_vectors=True,
            offset=slot_offset
        )

        slot_offset=result[-1]
        records = result[0]

        things = []
        for record in records:
            things.append(Thing(record = record))

        return things, slot_offset

    def generate(self, main_thing, offset):
        look = Look(main_thing)
        look_features = look.get_features()
        look.pred_quality = self._feed_data_(look_features)

        # 4 BODY_SINGLE
        look, offset = self.generate_for_slot(look, 4, offset)

        # 5 LEGS
        # if main thing is not a dress
        if (main_thing is None) or (main_thing and main_thing.subslot != 2):
            look, offset = self.generate_for_slot(look, 5, offset)

        # 6 FEET
        look, offset = self.generate_for_slot(look, 6, offset)

        # 1 BODY_MULTI
        look, offset = self.generate_for_slot(look, 1, offset)

        # 0 NECK
        look, offset = self.generate_for_slot(look, 0, offset)

        # 3 HEAD
        look, offset = self.generate_for_slot(look, 3, offset)

        # 2 HAND
        look, offset = self.generate_for_slot(look, 2, offset)

        return look, offset

    def generate_for_slot(self, look, num_slot, offset):
        things, offset[num_slot] = self._get_things_for_slot_(num_slot, offset[num_slot])

        base_quality = look.pred_quality

        for thing in things:
            features = look.get_features_for_candidate(thing)
            pred = self._feed_data_(features)
            if pred>base_quality:
                look.add_thing(thing, pred)
            #if pred>0.9:
            #    break

        #print("look quality:", look.pred_quality)
        return look, offset


