# music_genre_classification
COMP 6321 project: music genre classification on GTZAN



## Requirements installation

Requirement can be installed following:

```
pip install -r requirements.txt
```



## Running

1. download the dataset from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
    unzip the file and paste it into folder **/original_data**
    
2. Before running the model,  it is required to **run dataset processor** first

   if trained on csv, run data_splitting_csv.ipynb
   if trained on waves, run music_wave_preprocessor.ipynb

   check the /processed_data there should be data inside (or run tests/test_data_loading)

3. Open the target model jupyter and modify switch to the root directory

    if run it locally, modify it into:

    ```
    os.chdir('..')
    ```

    if run from google colab, modify it into:

    ```
    from google.colab import drive
    drive.mount('/content/drive/')
    os.chdir(your_root_dir) // swtiching to your root directory of this project
    ```

    

## **Training and testing results**

|                                      | link                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| wave processing example              | [wave_feature_extraction_sample.ipynb](jupyter/wave_feature_extraction_sample.ipynb) |
| csv data preprocessing               | [data_splitting_csv.ipynb](jupyter/data_splitting_csv.ipynb) |
| wave data preprocessing              | [music_wave_preprocessor.ipynb](jupyter/music_wave_preprocessor.ipynb) |
| **logistic regression on csv**       |                                                              |
| logistic regression training on csv  | [logistic_regression_training_csv.ipynb](jupyter/logistic_regression_training_csv.ipynb) |
| logistic regression testing on csv   | [logistic_regression_testing_csv.ipynb](jupyter/logistic_regression_testing_csv.ipynb) |
| **MLP on csv**                       |                                                              |
| MLP training on csv                  | [MLP_training_csv.ipynb](jupyter/MLP_training_csv.ipynb)     |
| MLP testing on csv                   | [MLP_testing_csv.ipynb](jupyter/MLP_testing_csv.ipynb)       |
| **scikit-learn models on csv**       |                                                              |
| scikit-learn models on csv           | [scikit_learn_models_csv.ipynb](jupyter/scikit_learn_models_csv.ipynb) |
| **MLP on wave**                      |                                                              |
| MLP training on wave                 | [MLP_training_wave.ipynb](jupyter/MLP_training_wave.ipynb)   |
| MLP testing on wave                  | [MLP_testing_wave.ipynb](jupyter/MLP_testing_wave.ipynb)     |
| **1d CNN on wave**                   |                                                              |
| 1d CNN training on wave              | [CNN1d_training_wave.ipynb](jupyter/CNN1d_training_wave.ipynb) |
| 1d CNN testing on wave               | [CNN1d_testing_wave.ipynb](jupyter/CNN1d_testing_wave.ipynb) |
| **2d CNN (2 CNN block) on wave**     |                                                              |
| 2 CNN block 2dCNN training on wave   | [CNN2d_2CNNblock_training_wave.ipynb](jupyter/CNN2d_2CNNblock_training_wave.ipynb) |
| 2 CNN block 2dCNN testing on wave    | [CNN2d_2CNNBlock_testing_wave.ipynb](jupyter/CNN2d_2CNNBlock_testing_wave.ipynb) |
| **2d CNN (n CNN block) on wave**     |                                                              |
| 2dCNN training on wave               | [CNN2d_training_wave.ipynb](jupyter/CNN2d_training_wave.ipynb) |
| 2dCNN testing on wave                | [CNN2d_testing_wave.ipynb](jupyter/CNN2d_testing_wave.ipynb) |
| **2d CNN(Res block) on wave**        |                                                              |
| 2dCNN(Res block) training on wave    | [CNN2d_res_training_wave.ipynb](jupyter/CNN2d_res_training_wave.ipynb) |
| 2dCNN(Res block) testing on wave     | [CNN2d_res_testing_wave.ipynb](jupyter/CNN2d_res_testing_wave.ipynb) |
| **LSTM on wave**                     |                                                              |
| LSTM training on wave                | [LSTM_training_wave.ipynb](jupyter/LSTM_training_wave.ipynb) |
| LSTM testing on wave                 | [LSTM_testing_wave.ipynb](jupyter/LSTM_testing_wave.ipynb)   |
| LSTM bidirectional training on wave  | [LSTM_bidirectional_training_wave.ipynb](jupyter/LSTM_bidirectional_training_wave.ipynb) |
| LSTM bidirectional testing on wave   | [LSTM_bidirectional_testing_wave.ipynb](jupyter/LSTM_bidirectional_testing_wave.ipynb) |
| **CRDNN on wave**                    |                                                              |
| CRDNN training on wave               | [CRDNN_training_wave.ipynb](jupyter/CRDNN_training_wave.ipynb) |
| CRDNN testing on wave                | [CRDNN_testing_wave.ipynb](jupyter/CRDNN_testing_wave.ipynb) |
| CRDNN bidirectional training on wave | [CRDNN_bidirectional_training_wave.ipynb](jupyter/CRDNN_bidirectional_training_wave.ipynb) |
| CRDNN bidirectional testing on wave  | [CRDNN_bidirectional_testing_wave.ipynb](jupyter/CRDNN_bidirectional_testing_wave.ipynb) |

 



### templates

|                           | link                                                         |
| ------------------------- | ------------------------------------------------------------ |
| training on csv template  | [train_on_csv_template.ipynb](jupyter/train_on_csv_template.ipynb) |
| training on wave template | [train_on_wave_template.ipynb](jupyter/train_on_wave_template.ipynb) |

