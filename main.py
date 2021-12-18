import tensorflow as tf
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import time

from evaluate import evaluate
from model import Attention, CNN_Encoder, RNN_Decoder
#from loss import loss_function
from preprocessing import datalimit, train_test_split
from utils import load_image, standardize, map_func, plot_attention
from data_download import data_download
from train import train_step

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
features_shape = 2048
attention_features_shape = 64
limit = 1000  # number of images to use for training


def main(start=True):
    annotation_file, PATH = data_download(data=True) # download data
    train_captions, img_name_vector = datalimit(limit, annotation_file, PATH) 
    
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet') # download model
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

    img_name_train, cap_train, img_name_val, cap_val = train_test_split(img_name_vector, cap_vector)

    num_steps = len(img_name_train) // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int64]),num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
    optimizer = tf.keras.optimizers.Adam()
    #loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0 # start from epoch 0 or last checkpoint epoch
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)

    EPOCHS = 20 # number of epochs to train the model
    loss_plot=[] # list to store the loss value for each epoch

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
        
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

    print("Testing the model")
    
    # Validation Dataset testing
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([tf.compat.as_text(index_to_word(i).numpy()) for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image,encoder,decoder,word_to_index,index_to_word,max_length, image_features_extract_model,attention_features_shape)
    print('Real Caption:', real_caption)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)

    # Unseen Data testing
    for i in range(1,3):
        image_path = 'data/unseen_image/'+i+'.jpg' # User path to image
        result, attention_plot = evaluate(image,encoder,decoder,word_to_index,index_to_word,max_length, image_features_extract_model,attention_features_shape)
        print('Prediction Caption:', ' '.join(result))
        plot_attention(image_path, result, attention_plot)
        # opening the image
        Image.open(image_path)

if __name__ == "__main__":
    start=True
    main(start)