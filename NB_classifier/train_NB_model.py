from random import shuffle, choice
import os
from math import ceil
import string
import pickle
import spacy
import pandas as pd

from nltk import ngrams, classify, NaiveBayesClassifier
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger

from config_file_NB import Config

# pylint: disable=no-member
################################################################################
# function to clear screen after each training sample
def clear():
    os.system('clear')

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

# cleaning function for english sentences
def clean_words_eng(words, stop_words=()):
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

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

# feature extractor function for unigram
def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary

# feature extractor function for ngrams
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    return words_dictionary

# feature extractor function for all words
def bag_of_all_words(words, language, n=2, nlp=None):
    if language == "english":
        stopwords_list = set(stopwords.words('english'))
        important_words_english = ['above', 'below', 'off', 'over', 'under', 'more',
                                   'most', 'such', 'no', 'nor', 'not', 'only', 'so',
                                   'than', 'too', 'very', 'just', 'but']
        stopwords_ngrams_list = stopwords_list - set(important_words_english)
        words_clean = clean_words_eng(words, stopwords_list)
        words_clean_for_ngrams = clean_words_eng(words, stopwords_ngrams_list)

    elif language == "spanish":
        stopwords_list = set(stopwords.words('spanish'))
        important_words_spanish = ['encima', 'debajo', 'menos', 'abajo', 'más',
                                   'tal', 'no', 'ni', 'solamente', 'entonces',
                                   'que', 'demasiado', 'muy', 'también', 'sólo', 'pero']
        stopwords_ngrams_list = stopwords_list - set(important_words_spanish)
        words_clean = clean_words_esp(words, nlp, stopwords_list)
        words_clean_for_ngrams = clean_words_esp(words, nlp, stopwords_ngrams_list)

    unigram_features = bag_of_words(words_clean)
    ngram_features = bag_of_ngrams(words_clean_for_ngrams, n=n)

    all_features = unigram_features.copy()
    all_features.update(ngram_features)

    return all_features

# class to gather data and train model
class ReviewSet:
    def __init__(self, review_dict):
        self.classes = review_dict.keys()
        for key in self.classes:
            setattr(self, key.lower(), review_dict[key])

    def extract_features(self, language, n=2):
        for key in self.classes:
            key = key.lower()
            # load spacy tagger here so it's only loaded once per training set
            if language == 'spanish':
                nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
            else:
                nlp = None
            reviews_set = []
            for words in self.__dict__[key]:
                reviews_set.append((bag_of_all_words(words, language, n=n, nlp=nlp), key))
            setattr(self, key+"_reviews_set", reviews_set)

    def test_prep(self, language, n=2):
        for key in self.classes:
            key = key.lower()
            # load spacy tagger here so it's only loaded once per training set
            if language == 'spanish':
                nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
            else:
                nlp = None
            reviews_set = []
            for sample_review in self.__dict__[key]:
                sample_review_dict = {
                    "words": bag_of_all_words(sample_review["sample"], language, n=n, nlp=nlp),
                    "sentence": sample_review["sentence"]
                    }
                reviews_set.append(sample_review_dict)
            setattr(self, key+"_reviews_set", reviews_set)

    def train_test_split(self, test_split=0.2):
        train_set = []
        test_set = []
        total_samples = 0

        for key in self.classes:
            key = key.lower()
            total_set = self.__dict__[key+"_reviews_set"]
            total_samples += len(total_set)
            shuffle(total_set)

            split_idx = ceil(len(total_set)*test_split)
            train_set += total_set[:split_idx]
            test_set += total_set[split_idx:]
        setattr(self, "train_set", train_set)
        setattr(self, "test_set", test_set)
        return total_samples

    def train_model(self, model_name, test_split=0.2, verbose=0):
        total_samples = self.train_test_split(test_split=test_split)
        if len(self.train_set+self.test_set) < 10:
            if verbose == 1:
                print("Must have at least 10 samples to train. You have {}.".format(total_samples))
        else:
            classifier = NaiveBayesClassifier.train(self.train_set)

            print("Accuracy is: {:.1f} %".format(classify.accuracy(classifier, self.test_set)*100))

            with open(model_name + '.pickle', 'wb') as f:
                pickle.dump(classifier, f)

################################################################################
if __name__ == "__main__":
    if os.path.exists("eng_config.ini"):
        CONFIG_FILE_NAME = "eng_config.ini"
    else:
        CONFIG_FILE_NAME = None
    CONFIG = Config(CONFIG_FILE_NAME)
    if CONFIG_FILE_NAME is None:
        CONFIG.write_defaults()
    CONFIG.get_values_from_config_file()

    REVIEWS_LIST = pd.read_csv(CONFIG.input_data_file)

    TRAINED_SAMPLES = dict([output_class, []] for output_class in CONFIG.class_options)
    KEYBOARD_INPUT = None

    INSTRUCTIONS = "Choose a class by inputting the number on the keyboard and " \
                   "hitting enter. To exit the training, press 'Q' and hit enter."
    print("#"*os.get_terminal_size()[0])
    print("Welcome to the trainable Multinomial Naïve Bayes Classifier.")
    print("\n")
    print(INSTRUCTIONS)
    print("\n")
    print("#"*os.get_terminal_size()[0])

    REVIEWS_LIST.to_csv(CONFIG.input_data_file.split('.csv')[0]+"_trained.csv", index=False)

    while KEYBOARD_INPUT != "Q" and len(REVIEWS_LIST) > 0:
        TRAINING_SAMPLE_IDX = choice(range(len(REVIEWS_LIST)))
        TRAINING_SAMPLE = REVIEWS_LIST.iloc[TRAINING_SAMPLE_IDX][CONFIG.header_key]
        REVIEWS_LIST = REVIEWS_LIST.drop(REVIEWS_LIST.loc[REVIEWS_LIST[CONFIG.header_key] == TRAINING_SAMPLE].index)
        print("\n")
        print(TRAINING_SAMPLE)
        print("\n")
        for idx, option in enumerate(CONFIG.class_options):
            print(str(idx+1)+". "+option)
        KEYBOARD_INPUT = input("Choose a class: ")
        if KEYBOARD_INPUT in [str(i) for i in range(1, len(CONFIG.class_options)+1)]:
            SAMPLE_CLASS = CONFIG.class_options[int(KEYBOARD_INPUT)-1]
            TRAINED_SAMPLES[SAMPLE_CLASS].append(word_tokenize(TRAINING_SAMPLE))

            RS = ReviewSet(TRAINED_SAMPLES)
            RS.extract_features(CONFIG.language, n=CONFIG.ngrams)
            clear()
            RS.train_model(CONFIG.model_name, CONFIG.test_split, verbose=CONFIG.verbosity)

            CSVFILE = pd.read_csv(CONFIG.input_data_file.split('.csv')[0]+"_trained.csv")
            CSVFILE.loc[CSVFILE[CONFIG.header_key] == TRAINING_SAMPLE, "SENTIMENT"] = SAMPLE_CLASS
            CSVFILE.to_csv(CONFIG.input_data_file.split('.csv')[0]+"_trained.csv", index=False)

        if KEYBOARD_INPUT not in ["Q"]+[str(i) for i in range(1, len(CONFIG.class_options)+1)]:
            clear()
            print("Invalid keyboard input {}".format(KEYBOARD_INPUT))
