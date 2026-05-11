import os
import logging

def setup_logger(name="logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(ch)
    return logger

def save_df(df, path):
    if path.endswith('.parquet'):
       df.to_parquet(path, index=False)
    elif path.endswith('.csv'):
       df.to_csv(path, index=False)
