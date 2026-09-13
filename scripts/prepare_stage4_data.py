"""Prepare fixed external corpora without running an encoder."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json
from foliorecall.benchmark_data import prepare_benchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage4-evaluation.json')
    parser.add_argument('--output', default='outputs/stage4/data')
    parser.add_argument('--training-data', default='data/vdr-stage2')
    parser.add_argument('--task')
    args = parser.parse_args()
    prepare_benchmark(read_json(args.config), args.output, args.training_data, args.task)
