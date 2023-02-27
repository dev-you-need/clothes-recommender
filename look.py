import numpy as np

class Look:
    def __init__(self, main_thing=None):
        self.things = [None]*7
        self.pred_quality = 0
        self.main_thing_slot = None
        if main_thing:
            self.things[main_thing.slot] = main_thing
            self.main_thing_slot = main_thing.slot

    def add_thing(self, new_thing, quality=None):
        self.things[new_thing.slot] = new_thing
        if quality:
            self.pred_quality = quality

    def get_features_for_candidate(self, new_thing):

        result = np.array([], dtype=np.float32)
        for num_slot, thing in enumerate(self.things):
            if num_slot == new_thing.slot:
                result = np.concatenate((result, new_thing.features), axis=0, dtype=np.float32)
            elif thing:
                result = np.concatenate((result, thing.features), axis=0, dtype=np.float32)
            else:
                result = np.concatenate((result, np.zeros(128, dtype=np.float32)), axis=0, dtype=np.float32)
        return result

    def get_features(self):

        result = np.array([], dtype=np.float32)
        for num_slot, thing in enumerate(self.things):
            if thing:
                result = np.concatenate((result, thing.features), axis=0, dtype=np.float32)
            else:
                result = np.concatenate((result, np.zeros(128, dtype=np.float32)), axis=0, dtype=np.float32)

        return result
