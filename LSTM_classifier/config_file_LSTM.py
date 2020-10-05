import ast
from configparser import ConfigParser
from distutils import util
import numpy as np

# pylint: disable=no-member

class Config:
    def __init__(self, config_file=None):
        if config_file is not None:
            self.config_name = str(config_file)
            config_mode = bool(util.strtobool(input("Use configs in {}? (y/n) ".format(config_file))))
            if config_mode is True:
                pass
            else:
                self.config_name = input("Alternate config_file name: ")
        elif config_file is None:
            default_config_name = 'config.ini'
            config_name = input("Name of new config file ({}): ".format(default_config_name))
            if config_name != "":
                self.config_name = config_name
            else:
                self.config_name = default_config_name

    def get_values_from_config_file(self):
        """Read in values from a configuration file.
        Parameters
        ----------
        config_file : str
            Filepath to configuration file of GaussPy+.
        """
        config = ConfigParser()
        config.read(self.config_name)
        sections = list(config.keys())[1:]

        for section in sections:
            for key, value in config[section].items():
                try:
                    if section == "ADVANCED MODEL PARAMETERS" and key not in ("oov_tok", "trunc_type", "pad_type"):
                        setattr(self, key, ast.literal_eval(value))
                    else:
                        setattr(self, key, value)
                except ValueError:
                    raise

    def write_defaults(self):
        """ Write default values to config.ini file """
        config_object = ConfigParser()

        config_object["ESSENTIAL INPUTS"] = {
            "input_data_file": 'hotels_labeled.csv',
            "review_header": "REVIEW",
            "label_header": "LABEL",
            "language": "english",
            "model_name": "SENTIMENT_LSTM_en"
        }

        config_object["ADVANCED MODEL PARAMETERS"] = {
            "vocab_size": 5000,
            "oov_tok": "<oov>",
            "max_len": 50,
            "trunc_type": 'post',
            "pad_type": 'post',
            "test_split": 0.25,
            "embedding_dim": 64,
            "lstm_neurons": [256, 128],
            "dense_neurons": [24],
            "dropout": [0.5],
            "learning_rate": 1e-4,
            "num_epochs": 500,
            "verbosity": 1
        }

        #Write the above sections to config.ini file
        with open(self.config_name, 'w') as conf:
            config_object.write(conf)
