import argparse
from scripts.data_fetcher import DataFetcher
from scripts.trainer import Trainer
from scripts.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="Sistema ML Cuantitativo Fútbol")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help="Modo: 'train' o 'predict'")
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer()
        trainer.run_training_pipeline()
    elif args.mode == 'predict':
        predictor = Predictor()
        predictor.generate_value_bets()

if __name__ == "__main__":
    main()
