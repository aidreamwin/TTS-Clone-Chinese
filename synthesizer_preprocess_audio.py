from synthesizer.preprocess import preprocess_thchs30
from synthesizer.preprocess import preprocess_data_aishell
from synthesizer.preprocess import preprocess_aidatatang_200zh
from synthesizer.hparams import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=None, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")

    parser.add_argument("-d", "--dataset", type=str, default="", help=\
        "what dataset to process.")

    args = parser.parse_args()
    
    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)

    if not args.dataset:
        raise ValueError("please input dataset, support aidatatang_200zh | data_aishell | thchs30")
    if args.dataset == "thchs30":
        preprocess_thchs30(**vars(args))
    elif args.dataset == "aidatatang_200zh":
        preprocess_aidatatang_200zh(**vars(args)) 
    elif args.dataset == "data_aishell":
        preprocess_data_aishell(**vars(args)) 
