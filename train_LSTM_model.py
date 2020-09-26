import string
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize

from config_file import Config
# pylint: disable=no-member
################################################################################
if os.path.exists("eng_config.ini"):
    CONFIG_FILE_NAME = "eng_config.ini"
else:
    CONFIG_FILE_NAME = None
CONFIG = Config(CONFIG_FILE_NAME)
if CONFIG_FILE_NAME is None:
    CONFIG.write_defaults()
CONFIG.get_values_from_config_file()
################################################################################
def load_data(config_obj):
    csvfile = pd.read_csv(config_obj.input_data_file)

    sentences = []
    labels = []
    for sentence, label in zip(csvfile[config_obj.review_header], csvfile[config_obj.label_header]):
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

# cleaning function for english sentences
def clean_words_eng(words, stop_words=()):
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(words)
    words_tagged = []
    for token, tag in pos_tag(words):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        word = lemmatizer.lemmatize(token, pos)
        words_tagged.append(word)
    words_clean = []
    for word in words_tagged:
        word = stemmer.stem(word)
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

# cleaning function for spanish sentences
def clean_words_esp(words, nlp, stop_words=()):
    stemmer = SnowballStemmer('spanish')

    def lemmatizer(text, pos):
        text = " ".join(text)
        doc = nlp(text, pos)
        result = [word.lemma_ for word in doc]
        return result

    working_dir = os.getcwd()
    os.environ['CLASSPATH'] = working_dir+"/stanford-postagger-full-2020-08-06"
    model = "spanish-ud"
    lang_model = working_dir+"/stanford-postagger-full-2020-08-06/models/"+model+".tagger"
    postagger = StanfordPOSTagger(lang_model, encoding='utf8')

    tokens = []
    pos_list = []
    for token, tag in postagger.tag(words):
        if tag == "NOUN":
            pos = 'n'
        elif tag == "VERB":
            pos = 'v'
        else:
            pos = 'a'
        tokens.append(token)
        pos_list.append(pos)

    words_tagged = lemmatizer(tokens, pos_list)

    words_clean = []
    for word in words_tagged:
        word = stemmer.stem(word)
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

def clean_split_pad_data(sentences, labels, config_obj):
    """ Clean data, split into train/test groups, and pad sentences for training """
    print("Cleaning data (This may take a few minutes)...")
    if config_obj.language == "english":
        stopwords_list = set(stopwords.words('english'))
        important_words_english = ['above', 'below', 'off', 'over', 'under', 'more',
                                   'most', 'such', 'no', 'nor', 'not', 'only', 'so',
                                   'than', 'too', 'very', 'just', 'but']
        stopwords_list = stopwords_list - set(important_words_english)
        sentences_clean = [" ".join(clean_words_eng(sentence, stopwords_list)) for sentence in sentences]
    elif config_obj.language == "spanish":
        nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
        stopwords_list = set(stopwords.words('spanish'))
        important_words_spanish = ['encima', 'debajo', 'menos', 'abajo', 'más',
                                   'tal', 'no', 'ni', 'solamente', 'entonces',
                                   'que', 'demasiado', 'muy', 'también', 'sólo', 'pero']
        stopwords_list = stopwords_list - set(important_words_spanish)
        sentences_clean = [" ".join(clean_words_esp(sentence, nlp, stopwords_list)) for sentence in sentences]

    # Splitting the dataset into Train and Test
    training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(sentences_clean, labels, test_size=config_obj.test_split, random_state=42)

    # Fit the tokenizer on Training data
    tokenizer = Tokenizer(num_words=config_obj.vocab_size, oov_token=config_obj.oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    tokenizer_json = tokenizer.to_json()
    tokenizer_name = "tokenizer_"+config_obj.language[:2]+".json"
    with open(tokenizer_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # Creating padded sequences from train and test data
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=config_obj.max_len, padding=config_obj.pad_type, truncating=config_obj.trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=config_obj.max_len, padding=config_obj.pad_type, truncating=config_obj.trunc_type)


    # Converting the lists to numpy arrays
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)
    return training_padded, testing_padded, training_labels, testing_labels

def build_and_compile_model(config_obj):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(config_obj.vocab_size, config_obj.embedding_dim, input_length=config_obj.max_len, trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    if config_obj.verbosity == 1:
        model.summary()
    return model

def train_model(model, x_train, y_train, x_test, y_test, config_obj, plotting=True):
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=125, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=config_obj.num_epochs, validation_data=(x_test, y_test), callbacks=[earlystop_callback], verbose=config_obj.verbosity)

    scores = model.evaluate(x_test, y_test, verbose=config_obj.verbosity, use_multiprocessing=True)
    print("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))

    model.save(config_obj.model_name+".h5")

    if plotting:
        # Plotting results
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(train_accuracy))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_accuracy)
        plt.plot(epochs, val_accuracy)
        plt.legend(['train_acc', 'val_acc'])
        plt.title('Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_loss)
        plt.plot(epochs, val_loss)
        plt.legend(['train_loss', 'val_loss'])
        plt.title('Loss')
        plt.tight_layout()
        plt.savefig("model_history.png")

################################################################################
SENTENCES, LABELS = load_data(CONFIG)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = clean_split_pad_data(SENTENCES, LABELS, CONFIG)

MODEL = build_and_compile_model(CONFIG)

train_model(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, CONFIG, plotting=True)
