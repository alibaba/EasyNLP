import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["ViT-B/32"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name == "ViT-L/14":
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--train-img",
        type=str,
        default=None,
        help="Path to csv filewith training images npz file",
    )
    parser.add_argument(
        "--val-img",
        type=str,
        default=None,
        help="Path to csv file with validation images npz file",
    )
    parser.add_argument(
        "--img-key",
        type=str,
        default="filepath",
        help="For datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--caption-key",
        type=str,
        default="title",
        help="For datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "json"],
        default="json",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--model",
        choices=["ViT-B/32", "ViT-L/14"],
        default="ViT-L/14",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--openai-pretrained",
        default=False,
        action='store_true',
        help="Use the openai pretrained models.",
    )
    parser.add_argument(
        "--is-prompt",
        default=False,
        action='store_true',
        help="Use prompt loss.",
    )
    parser.add_argument(
        "--is-mask",
        default=False,
        action='store_true',
        help="Use mask loss.",
    )
    parser.add_argument(
        "--is-da-loss",
        default=False,
        action='store_true',
        help="Use VG DA loss.",
    )
    parser.add_argument(
        "--is-da-mask",
        default=False,
        action='store_true',
        help="Use VG DA mask loss.",
    )
    parser.add_argument(
        "--is-vg",
        default=False,
        action='store_true',
        help="Use VG mask loss.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--C", type=float, default=3.16, help="inverse regularizer for logistic reg."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--dp",
        default=False,
        action="store_true",
        help="Use DP instead of DDP."
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="In DP, which GPUs to use for multigpu training",
    )
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
