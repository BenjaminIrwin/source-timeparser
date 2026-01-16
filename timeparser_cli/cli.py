import argparse
import logging

from .fasttext_manager import fasttext_downloader
from .utils import clear_cache


def entrance():
    timeparser_argparse = argparse.ArgumentParser(
        description="timeparser download manager."
    )
    timeparser_argparse.add_argument(
        "--fasttext",
        type=str,
        help='To download a fasttext language detection models. Supported models are "small" and "large"',
    )
    timeparser_argparse.add_argument(
        "--clear",
        "--clear-cache",
        help="To clear all cached models",
        action="store_true",
    )

    args = timeparser_argparse.parse_args()

    if args.clear:
        clear_cache()
        logging.info("timeparser-download: All cache deleted")

    if args.fasttext:
        fasttext_downloader(args.fasttext)

    if not (args.clear or args.fasttext):
        timeparser_argparse.error(
            "timeparser-download: You need to specify the command (i.e.: --fasttext or --clear)"
        )
