from nltk.tokenize import word_tokenize
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import csv
import os

from config_file_NB import Config
from train_NB_model import ReviewSet, clear

# pylint: disable=no-member
################################################################################
# Initialize config variables
if os.path.exists("eng_config.ini"):
    CONFIG_FILE_NAME = "eng_config.ini"
else:
    CONFIG_FILE_NAME = None
CONFIG = Config(CONFIG_FILE_NAME)
if CONFIG_FILE_NAME is None:
    CONFIG.write_defaults()
CONFIG.get_values_from_config_file()

OUTPUT_HIST = "test_set_hist_"+CONFIG.language[:2]
OUTPUT_PIE = "test_set_pie_"+CONFIG.language[:2]

TRAINED_CSV = CONFIG.input_data_file.split('.csv')[0]+"_trained.csv"
PREDICTED_CSV = CONFIG.input_data_file
################################################################################

# Load model
with open(CONFIG.model_name+'.pickle', 'rb') as f:
    CLASSIFIER = pickle.load(f)

# Load csv
REVIEWS_LIST = []
with open(CONFIG.input_data_file, newline='') as csvfile:
    READER = csv.DictReader(csvfile)
    for review in READER:
        REVIEWS_LIST.append((review["ID"], review[CONFIG.header_key]))

# Create ReviewSet object and clean and gleam data
CLASS_OPTIONS = ["test"]
TEST_SAMPLES = dict([output_class, []] for output_class in CLASS_OPTIONS)
for review in REVIEWS_LIST:
    review_item = {"ID": review[0], "sentence": review[1], "sample": word_tokenize(review[1])}
    TEST_SAMPLES["test"].append(review_item)

TS = ReviewSet(TEST_SAMPLES)
TS.test_prep(CONFIG.language, n=CONFIG.ngrams)

# Predict sentiment for each review in review set
OVERALL_SENTIMENT = []
SENTIMENT_COUNTS = {"pos": 0, "neut": 0, "neg": 0}

PRED_LIST = []
HUMAN_REVIEW = []
for test_review in TS.test_reviews_set:
    # predict outcome of review
    prob_result = CLASSIFIER.prob_classify(test_review["words"]).max()
    confidence = CLASSIFIER.prob_classify(test_review["words"]).prob(prob_result)

    # sort outcome data
    if confidence < 0.5:
        HUMAN_REVIEW.append(test_review)
    else:
        if prob_result in ("positive", "positivo"):
            OVERALL_SENTIMENT.append(1*confidence*100)
            SENTIMENT_COUNTS[prob_result[:3]] += 1
        if prob_result in ("negative", "negativo"):
            OVERALL_SENTIMENT.append(-1*confidence*100)
            SENTIMENT_COUNTS[prob_result[:3]] += 1
        if prob_result == "neutral":
            SENTIMENT_COUNTS[prob_result[:4]] += 1

        with open(TRAINED_CSV, 'r') as csvfile:
            ROWREADER = csv.reader(csvfile)

            for row in ROWREADER:
                if row[0] == str(test_review["ID"]) and len(row[2]) == 0:
                    row[2] = prob_result.capitalize()
                    PRED_LIST.append(row)

with open(PREDICTED_CSV, 'w') as output:
    HEADERS = ["ID", CONFIG.header_key, "SENTIMENT"]
    WRITER = csv.DictWriter(output, fieldnames=HEADERS)
    WRITER.writeheader()

    SORTED_PRED_LIST = sorted(PRED_LIST, key=lambda x: int(x[0]))
    for row in SORTED_PRED_LIST:
        row_dict = {
            "ID": row[0],
            CONFIG.header_key: row[1],
            "SENTIMENT": row[2]
            }
        WRITER.writerow(row_dict)

# Human Intervention - if predicted has low confidence or doesn't match
#                      existing label provide a new label
REVIEW_UNSURE = None
if len(HUMAN_REVIEW) > 0:
    print("There are {} reviews with questionable tags. Would you like to review those now?".format(len(HUMAN_REVIEW)))
    REVIEW_UNSURE = input("Yes or No (y/n): ")

if REVIEW_UNSURE == "y":
    for test_review in HUMAN_REVIEW:
        # predict outcome of review
        prob_result = CLASSIFIER.prob_classify(test_review["words"]).max()
        confidence = CLASSIFIER.prob_classify(test_review["words"]).prob(prob_result)

        with open(TRAINED_CSV, 'r') as csvfile:
            ROWREADER = csv.DictReader(csvfile)

            for row in ROWREADER:
                if row["ID"] == test_review["ID"]:
                    if row["SENTIMENT"] and row["SENTIMENT"] != prob_result.capitalize():
                        print("MISMATCH")
                    elif confidence < 0.5:
                        print("Predicted class confidence is low. Please verify or update class.")
                        print("\n")
                        print(test_review["sentence"])
                        print("class: ", prob_result)
                        print("\n")
                        for idx, option in enumerate(CONFIG.class_options):
                            print(str(idx+1)+". "+option)
                        KEYBOARD_INPUT = input("Choose a class: ")
                        row["SENTIMENT"] = CONFIG.class_options[int(KEYBOARD_INPUT)-1]

                        class_result = row["SENTIMENT"].lower()

                        if class_result in ("positive", "positivo"):
                            OVERALL_SENTIMENT.append(1*confidence*100)
                            SENTIMENT_COUNTS[class_result[:3]] += 1
                        if class_result in ("negative", "negativo"):
                            OVERALL_SENTIMENT.append(-1*confidence*100)
                            SENTIMENT_COUNTS[class_result[:3]] += 1
                        if class_result == "neutral":
                            SENTIMENT_COUNTS[class_result[:4]] += 1
                        clear()
                    else:
                        row["SENTIMENT"] = prob_result.capitalize()

                    row_list = [row["ID"], row[CONFIG.header_key], row["SENTIMENT"]]
                    PRED_LIST.append(row_list)

    with open(PREDICTED_CSV, 'w') as output:
        HEADERS = ["ID", CONFIG.header_key, "SENTIMENT"]
        WRITER = csv.DictWriter(output, fieldnames=HEADERS)
        WRITER.writeheader()

        SORTED_PRED_LIST = sorted(PRED_LIST, key=lambda x: int(x[0]))
        for row in SORTED_PRED_LIST:
            row_dict = {
                "ID": row[0],
                CONFIG.header_key: row[1],
                "SENTIMENT": row[2]
                }
            WRITER.writerow(row_dict)


print("Plotting results...")
# Plot sentiment distribution
plt.hist(OVERALL_SENTIMENT, bins=25, range=(-100, 100))
plt.title("Overall Sentiment for " + CONFIG.input_data_file)
plt.xlabel("Confidence x Sentiment")
plt.ylabel("Counts")
plt.savefig(OUTPUT_HIST + ".png", dpi=1250)
plt.cla()

plt.pie(SENTIMENT_COUNTS.values(), labels=CONFIG.class_options, autopct='%1.1f%%',
        startangle=90)
plt.axis('equal')
plt.title("Response Sentiments for " + CONFIG.input_data_file)
plt.savefig(OUTPUT_PIE + ".png", dpi=1250)
