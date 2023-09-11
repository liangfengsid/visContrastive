import yaml
from sklearn import svm
from joblib import dump
import time
import logging
import cebra
import numpy as np
import matplotlib.pyplot as plt
import extract_time

def perform_decoding_and_plot(models,time_train, time_test, emo_label_train, emo_label_test,
                   subject_label_train, subject_label_test, max_iter, embeddings, decoder_labels, embedding_dimensions,window):

    for model_name, offset in models:
        for d in embedding_dimensions:
            for embedding_type in embeddings:

                model_fullname = f'time_{model_name}_d{d}_i{max_iter*d}_label{use_label}_w{window}.model'
                cebra_model = cebra.CEBRA.load('models/'+model_fullname)

                print(f'transforming data for {model_fullname}')

                embeddings_test = cebra_model.transform(time_test)

                plt.figure()
                cebra.plot_embedding(embeddings_test, embedding_labels='time')
                plt.savefig(f'figures/{model_fullname}_embeddings_test.png')

                print('figs saved')

def main(config):

    models = config['models']
    max_iter = config['max_iter']
    save_path = config['save_path']
    embedding_dimensions = config['embedding_dimensions']
    embeddings = config['embeddings']
    decoder_labels = config['decoder_labels']
    windows = config['windows']

    for window in windows:
        time_train, time_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test = \
            extract_time.load_split_data(save_path+'/time_reduced_w{window}.npy')
        perform_decoding_and_plot(models, time_train, time_test, emo_label_train, emo_label_test,
                        subject_label_train, subject_label_test, max_iter, embeddings, decoder_labels, embedding_dimensions,window)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)