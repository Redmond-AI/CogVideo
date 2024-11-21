import argparse
from pathlib import Path
from typing import Optional
import os
import sys


def positive_int(value: str) -> int:
    """Custom type for positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")
    return ivalue


def float_range(min_val: float = 0.0, max_val: float = 1.0):
    """Factory for float range checker."""
    def check_range(value: str) -> float:
        fvalue = float(value)
        if fvalue < min_val or fvalue > max_val:
            raise argparse.ArgumentTypeError(f"{value} must be between {min_val} and {max_val}")
        return fvalue
    return check_range


class PathAction(argparse.Action):
    """Custom action to validate paths."""
    def __call__(self, parser, namespace, values, option_string=None):
        path = Path(values)
        if not path.exists():
            parser.error(f"Path does not exist: {path}")
        setattr(namespace, self.dest, str(path))


def validate_dataset_structure(dataset_root: str) -> None:
    """Validate that the dataset directory has the required structure."""
    root = Path(dataset_root)
    required_dirs = ["RGB_480", "MASK_480", "GT_480"]
    
    for subdir in required_dirs:
        path = root / subdir
        if not path.is_dir():
            raise ValueError(f"Required directory '{subdir}' not found in dataset root: {root}")
        
        # Check if directory is readable and contains files
        try:
            next(path.iterdir())
        except (PermissionError, StopIteration):
            raise ValueError(f"Directory '{subdir}' is empty or not readable: {path}")


def get_args():
    parser = argparse.ArgumentParser(description="Training script for CogVideoX inpainting with LoRA.")

    # Model information
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        action=PathAction,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    model_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    model_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        action=PathAction,
        help="Root directory containing the dataset folders with RGB_480, MASK_480, and GT_480 subdirectories.",
    )
    data_group.add_argument(
        "--max_train_samples",
        type=positive_int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    data_group.add_argument(
        "--frame_width",
        type=positive_int,
        default=480,
        help="Width of input frames.",
    )
    data_group.add_argument(
        "--frame_height",
        type=positive_int,
        default=640,
        help="Height of input frames.",
    )
    data_group.add_argument(
        "--sequence_length",
        type=positive_int,
        default=100,
        help="Number of frames in each sequence.",
    )
    data_group.add_argument(
        "--mask_threshold",
        type=float_range(0.0, 1.0),
        default=0.5,
        help="Threshold value for binarizing the mask (0-1).",
    )

    # Validation settings
    validation_group = parser.add_argument_group("Validation Configuration")
    validation_group.add_argument(
        "--validation_epochs",
        type=positive_int,
        default=5,
        help="Run validation every X epochs.",
    )
    validation_group.add_argument(
        "--num_validation_samples",
        type=positive_int,
        default=4,
        help="Number of validation samples to generate for visual inspection.",
    )
    validation_group.add_argument(
        "--validation_folder",
        type=str,
        default="validation_samples",
        help="Folder to save validation samples.",
    )

    # Training parameters
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )
    training_group.add_argument(
        "--rank",
        type=positive_int,
        default=128,
        help="The dimension of the LoRA update matrices.",
    )
    training_group.add_argument(
        "--lora_alpha",
        type=float_range(0.0, float('inf')),
        default=128,
        help="The scaling factor for LoRA weight updates. The actual scaling is lora_alpha/rank.",
    )
    training_group.add_argument(
        "--learning_rate",
        type=float_range(0.0, float('inf')),
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    training_group.add_argument(
        "--batch_size",
        type=positive_int,
        default=1,
        help="Batch size (per device) for training.",
    )
    training_group.add_argument(
        "--num_train_epochs",
        type=positive_int,
        default=100,
        help="Total number of training epochs to perform.",
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=positive_int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    training_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )

    # Output settings
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    output_group.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory for storing logs.",
    )
    output_group.add_argument(
        "--logging_steps",
        type=positive_int,
        default=10,
        help="Log every X updates steps.",
    )
    output_group.add_argument(
        "--save_steps",
        type=positive_int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    output_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    args = parser.parse_args()

    # Post-parsing validation
    try:
        # Validate dataset structure
        validate_dataset_structure(args.dataset_root)

        # Create output and logging directories if they don't exist
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
        os.makedirs(args.validation_folder, exist_ok=True)

        # Validate checkpoint path if provided
        if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint):
            parser.error(f"Checkpoint path does not exist: {args.resume_from_checkpoint}")

    except (ValueError, OSError) as e:
        parser.error(str(e))

    return args


if __name__ == "__main__":
    try:
        args = get_args()
        print("Arguments successfully validated:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
