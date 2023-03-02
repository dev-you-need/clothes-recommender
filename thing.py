import os
import numpy as np

class Thing:
    def __init__(self, record = None, subslot=None, features=None):
        self.subslot = subslot
        self.slot = None
        self.features = features
        self.img_path = None

        if subslot:
            self.slot = self.subslot_to_slot(subslot)

        if record:
            self.subslot = record.payload['subslot']
            self.slot = record.payload['slot']
            self.features = np.array(record.vector['image_emb'], dtype=np.float32)
            self.img_path = record.payload['url'] #self.record_to_url(record)


    def category_to_subslot(self, subcat):
        if subcat in ['Cardigans']:
            return 0
        elif subcat in ['Jackets', 'Basic Jackets']:
            return 1
        elif subcat in ['Evening Dresses', 'Cocktail Dresses', 'Dresses']:
            return 2
        elif subcat in ['Hoodies & Sweatshirts', 'Skateboarding Hoodies']:
            return 3
        elif subcat in ['Tuxedo Shirts', 'Polo Shirts', 'Dress Shirts', 'Casual Shirts', 'Blouses & Shirts',
                        'Blouses &  Shirts']:
            return 4
        elif subcat in ['Pullovers']:
            return 5
        elif subcat in ['Tube Tops', 'Tank Tops']:
            return 6
        elif subcat in ['T-Shirts', 'Tees', 'Polo']:
            return 7
        elif subcat in ['Over-the-Knee Boots',
                        'Oxfords',
                        "Men's Casual Shoes",
                        'Basic Boots',
                        'Knee-High Boots',
                        "Men's Vulcanize Shoes",
                        'Formal Shoes',
                        'Boots']:
            return 8
        elif subcat in ['Mid-Calf Boots', 'Ankle Boots']:
            return 9
        elif subcat in ['Middle Heels',
                        'Low Heels',
                        'Flip Flops',
                        "Women's Sandals",
                        "Men's Sandals",
                        'High Heels',
                        'Slippers',
                        "Women's Flats",
                        "Women's Pumps"]:
            return 10
        elif subcat in []:
            return 11  # "Moccasins and slip-ons"
        elif subcat in ["Women's Vulcanize Shoes",
                        "Men's Vulcanize Shoes",
                        'Sneakers',
                        'Running Shoes',
                        'Basketball Shoes']:
            return 12
        elif subcat in []:
            return 13  # "Bags and Backpacks"
        elif subcat in ['Berets', 'Newsboy Caps', 'Skullies & Beanies', 'Baseball Caps']:
            return 14
        elif subcat in ['Jumpsuits', 'Overalls']:
            return 15
        elif subcat in ['Jeans']:
            return 16
        elif subcat in ['Casual Shorts', 'Running Shorts', 'Surfing & Beach Shorts', 'Shorts']:
            return 17
        elif subcat in ['Skirts']:
            return 18
        elif subcat in ['Cargo Pants',
                        'Casual Pants',
                        'Sweatpants',
                        'Skinny Pants',
                        'Harem Pants',
                        'Flare Pants',
                        'Leather Pants',
                        'Pants & Capris']:
            return 19
        elif subcat in ['Scarves', "Men's Scarves"]:
            return 20
        else:
            return -1

    def subslot_to_slot(self, subcat):
        if subcat in [0, 1]:
            return 1
        elif subcat in [2, 3, 4, 5, 6, 7]:
            return 4
        elif subcat in [8, 9, 10, 11, 12]:
            return 6
        elif subcat in [13]:
            return 2
        elif subcat in [14]:
            return 3
        elif subcat in [15, 16, 17, 18, 19]:
            return 5
        elif subcat in [20]:
            return 0
        else:
            print("wrong subcat:", subcat)
            return -1

    def record_to_url(self, record):
        payload = record.payload
        path_to = '../clothes-dataset/imgs/'
        path_to_id = os.path.join(os.path.join(path_to, str(payload['ali_id'])))
        path_to_color = os.path.join(path_to_id, payload['color'])
        path_to_source = os.path.join(path_to_color, 'shop')
        path = os.path.join(path_to_source, payload['filename'])

        return path
