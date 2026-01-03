from .config import get_config
from .trainer import train

if __name__ == "__main__":

    conf = get_config()

    train(conf)
