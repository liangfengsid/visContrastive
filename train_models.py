import os
import yaml
import time
import logging
import cebra
from cebra import CEBRA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import extract_time


def train_cebra_models(models, time_train, emo_label_train, subject_label_train,
                        batch_size, max_iter, embedding_dimensions,window):
    for model_name, offset in models:
        for d in embedding_dimensions:
            for use_label in ['emo', 'subject', 'none']:
                start_time = time.time()
                cebra_model = CEBRA(
                    model_architecture = model_name,
                    batch_size = batch_size,
                    temperature_mode="auto",
                    learning_rate = 0.001,
                    max_iterations = max_iter * d // 4,
                    time_offsets = offset,
                    output_dimension = d,
                    device = "cuda_if_available",
                    verbose = False
                )
                if use_label == 'emo':
                    cebra_model.fit(time_train, emo_label_train)
                elif use_label == 'subject':
                    cebra_model.fit(time_train, subject_label_train)
                else:
                    cebra_model.fit(time_train)

                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f'Training model {model_name} with d={d} and label={use_label} took {elapsed_time} seconds')
                print(f'Training model {model_name} with d={d} and label={use_label} took {elapsed_time} seconds')
                
                cebra_model.save(f'models/time_{model_name}_d{d}_i{max_iter*d}_label{use_label}_w{window}.model')
                plt.figure()
                cebra.plot_loss(cebra_model)
                plt.savefig(f'figures/loss_plot_{model_name}_d{d}_i{max_iter*d}_label{use_label}_w{window}.png')

def main(config):
    models = config['models']
    batch_size = config['batch_size']
    max_iter = config['max_iter']
    save_path = config['save_path']
    embedding_dimensions = config['embedding_dimensions']
    windows = config['windows']

    for window in windows:
        time_train, _, emo_label_train, _, subject_label_train, _ = extract_time.load_split_data(save_path+'/time_reduced_w{window}.npy')
        train_cebra_models(models, time_train, emo_label_train, subject_label_train,
                            batch_size, max_iter, embedding_dimensions,window)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
