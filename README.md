# DeepSAT
A Sentiment Analysis Tool (SAT) for feedback reviews in csv format using Deep Learning and Naïve Bayes.

### Installing Requirements
It is recommended to run DeepSAT in a `venv` environment. After creating your virtual environment called <env_name>, activate the environment, and install the requirements for the model you aim to run. For example, for the Deep Learning (LSTM) model, run:
```
python3 -m venv <env_name>
source <env_name>/bin/activate
pip install -r LSTM_requirements.txt
```

## Naïve Bayes (NB) Classifier 
To run the scripts for the NB Classifier, you need a config file. A default one will be generated the first time the training script is run. Edit the values in the config file to configure the parameters for training.

`input_data_file` - The csv file used for training. Must have at least one column with a header for the reviews.

`header_key` - The header of the column with the training data (i.e. reviews).

`language` - The language of the training data.

`class_options` - A list of the output label options of your training data.

`model_name` - A name for the model.

`ngrams` - When extracting features for training the data, words can be grouped together to form new features. For example, "very" and "good" are unigrams while "very good" is a bigram. This parameter allows you to set the maximum number of ngrams.

`test_split` - Fraction of how much of the training data to be used for testing. It is recommend to use a number greater than or equal to 0.2.

`verbosity` - Verbosity level of model and output in terminal.

### Training with `train_NB_model.py`
To start training your NB model, activate your virtual environment and run the training script.
```
source <env_name>/bin/activate
python train_NB_model.py
```
Training the NB Classfier takes little input to produce adequate results, but improves with the more trained data points it has. Follow the instructions in the terminal to train the model.

### Evaluating with `evaluate_NB_model.py`
To evaluate your model on a new dataset, activate your virtual environment and run the testing and evaluation script.
```
source <env_name>/bin/activate
python evaluate_NB_model.py
```
Be sure to change the `input_data_file` keyword in the config file before you run this if you want to test on a new dataset. This script will also produce a histogram of the distribution of sentiments in the dataset, calculated by multiplying the class (1 for Postive and -1 for Negative) by the confidence of the prediction. Lastly, it will generate a pie chart of the breakdown of all the classes.
