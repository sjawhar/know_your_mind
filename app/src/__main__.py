import argparse
import logging

logger = logging.getLogger(__package__)
logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser(prog="know_your_mind")
parser.add_argument("DATA_FILE")
parser.add_argument("RESULTS_DIR")
parser.add_argument("--debug", action="store_true")
parser.add_argument("-b", "--batch-size", type=int)
parser.add_argument("-d", "--dropout", type=float)
parser.add_argument("-r", "--reward-threshold", type=float)
parser.add_argument("-t", "--test-epoch-frequency", type=int)
parser.add_argument("-v", "--verbose", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        import ptvsd

        ptvsd.enable_attach(address=("0.0.0.0", 5678))
        ptvsd.wait_for_attach()
    del args.debug

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    del args.verbose

    from .runner import run

    args = vars(args)
    run(
        args.pop("DATA_FILE"),
        args.pop("RESULTS_DIR"),
        **{k: v for k, v in args.items() if v is not None},
    )
