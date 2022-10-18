from toxic_comment_collection import get_all_datasets, generate_statistics, get_dataset
import pandas as pd

#get_all_datasets(api_config_path='../toxic-comment-collection/src/toxic_comment_collection/api_config.json')

get_all_datasets(config_path='../toxic-comment-collection/src/toxic_comment_collection/config.json', skip_download=True, api_config_path='../toxic-comment-collection/src/toxic-comment-collection/api_config.json')

generate_statistics('./files')