import argparse
import logging

from icenet.cli import setup_logging


class TrainingArgParser(argparse.ArgumentParser):
    """An ArgumentParser specialised to support model training

    The 'allow_*' methods return self to permit method chaining.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("dataset", type=str)
        self.add_argument("run_name", type=str)
        self.add_argument("seed", type=int)

        self.add_argument("-o", "--output-path", type=str, default=None)
        self.add_argument("-v",
                          "--verbose",
                          action="store_true",
                          default=False)

        self.add_argument("-b", "--batch-size", type=int, default=4)
        self.add_argument("-ca",
                          "--checkpoint-mode",
                          default="max",
                          type=str)
        self.add_argument("-cm",
                          "--checkpoint-monitor",
                          default="val_icenetaccuracy",
                          type=str)
        self.add_argument("-ds",
                          "--additional-dataset",
                          dest="additional",
                          nargs="*",
                          default=[])
        self.add_argument("-e", "--epochs", type=int, default=4)
        self.add_argument("-p", "--preload", type=str)
        self.add_argument("-r", "--ratio", default=1.0, type=float)
        self.add_argument("--shuffle-train",
                          default=False,
                          action="store_true",
                          help="Shuffle the training set")
        self.add_argument("--lr", default=1e-4, type=float)
        # self.add_argument("--lr_10e_decay_fac",
        #                   default=1.0,
        #                   type=float,
        #                   help="Factor by which LR is multiplied by every 10 epochs "
        #                   "using exponential decay. E.g. 1 -> no decay (default)"
        #                   ", 0.5 -> halve every 10 epochs.")
        # self.add_argument('--lr_decay_start', default=10, type=int)
        # self.add_argument('--lr_decay_end', default=30, type=int)

    def add_unet(self):
        self.add_argument("-f", "--filter-size", type=int, default=3)
        self.add_argument("-n", "--n-filters-factor", type=float, default=1.)
        return self

    def parse_args(self, *args,
                   log_format="[%(asctime)-17s :%(levelname)-8s] - %(message)s",
                   **kwargs):
        args = super().parse_args(*args, **kwargs)

        logging.basicConfig(
            datefmt="%d-%m-%y %T",
            format=log_format,
            level=logging.DEBUG if args.verbose else logging.INFO)
        logging.getLogger("cdsapi").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return args


@setup_logging
def predict_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("network_name")
    ap.add_argument("output_name")
    ap.add_argument("seed", type=int, default=42)
    ap.add_argument("datefile", type=argparse.FileType("r"))

    ap.add_argument("-i",
                    "--train-identifier",
                    dest="ident",
                    help="Train dataset identifier",
                    type=str,
                    default=None)
    ap.add_argument("-n", "--n-filters-factor", type=float, default=1.)
    ap.add_argument("-l", "--legacy-rounding", action="store_true",
                    default=False, help="Ensure filter number rounding occurs last in channel number calculations")
    ap.add_argument("-t", "--testset", action="store_true", default=False)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    ap.add_argument("-s", "--save_args", action="store_true", default=False)

    return ap.parse_args()
