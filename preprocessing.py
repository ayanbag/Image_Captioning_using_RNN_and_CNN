import tensorflow as tf
import collections
import random
import numpy as np
import json


from utils import load_image,standardize

def datalimit(limit_size, annotation_file, PATH):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    train_image_paths = image_paths[:limit_size]
    print(len(train_image_paths))


    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))
    
    return train_captions, img_name_vector
    
'''
def preprocessing(train_captions, img_name_vector):
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    max_length = 50
    vocabulary_size = 5000
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_size,standardize=standardize,output_sequence_length=max_length)

    tokenizer.adapt(caption_dataset)
    cap_vector = caption_dataset.map(lambda x: tokenizer(x))
    word_to_index = tf.keras.layers.StringLookup(mask_token="",vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(mask_token="",vocabulary=tokenizer.get_vocabulary(),invert=True)

    return word_to_index, index_to_word, cap_vector, image_features_extract_model
'''

def train_test_split(img_name_vector, cap_vector):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

    return img_name_train, cap_train, img_name_val, cap_val

