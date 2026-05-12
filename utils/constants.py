# Project-level constants, including API keys and directories
# Note: importing this file has the side effect of loading a configuration file
from pathlib import Path
import yaml

##############################
# API keys
##############################
OPENAI_API_KEY = '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'

##############################
# Model Configurations
##############################
model_name = 'llama2'
base_model = 'meta-llama/Llama-2-7b-hf'

######################
#   hyperparameter   #
######################
learning_rate = 2e-5
num_epochs = 5
batch_size = 1
seed = 42
threshold = 3
strategy = 'min'

##############################
# Perspective API
##############################
DISCOVERY_URL = "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
PERSPECTIVE_API_KEY = ""
PERSPECTIVE_API_LEN_LIMIT = 20480

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)


##############################
# non value datasets
##############################
non_value_datasets = [
    'alpaca',
    'grammar',
    'samsum',
    'vanilla',
    'dolly'
]
