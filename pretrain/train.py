import os

from dinov3.logging import setup_logging
from dinov3.train import main as train_main

if __name__ == "__main__":
    setup_logging()
    train_main()
