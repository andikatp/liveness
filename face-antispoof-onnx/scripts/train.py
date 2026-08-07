"""Train MiniFASNet V2 SE for face anti-spoofing (2-class: Real, Spoof)."""

from src.minifasv2.config import TrainConfig
from src.minifasv2.main import Trainer
import argparse
import os

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Training Face-AntiSpoofing Model (2-class: Real, Spoof)"
    )
    p.add_argument(
        "--crop_dir", type=str, default="data", help="Subdir with cropped images"
    )
    p.add_argument(
        "--input_size",
        type=int,
        default=128,
        help="Input size of images passed to model",
    )
    p.add_argument(
        "--batch_size", type=int, default=256, help="Count of images in the batch"
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    p.add_argument(
        "--transfer_learning",
        action="store_true",
        help="Use transfer learning mode (load only model weights, reset optimizer/scheduler)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints and logs",
    )
    args = p.parse_args()

    spoof_categories = [[0], [1, 2, 3, 7, 8, 9]]

    config = TrainConfig(
        crop_dir=args.crop_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        spoof_categories=spoof_categories,
        output_dir=args.output_dir,
    )
    config.set_job("MINIFAS")
    print("Device:", config.device)

    resume_path = args.resume
    if resume_path is None:
        checkpoint_latest = os.path.join(config.model_path, "checkpoint_latest.pth")
        if os.path.exists(checkpoint_latest):
            resume_path = checkpoint_latest
            print(f"Found existing checkpoint: {checkpoint_latest}")
            print('Resuming training automatically. Use --resume "" to start fresh.')

    trainer = Trainer(
        config, resume_from=resume_path, transfer_learning=args.transfer_learning
    )
    trainer.train_model()
    print("Finished")
