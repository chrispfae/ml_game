import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import logging as absllog

absllog.set_verbosity(absllog.ERROR)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    filename = "log.log",
)
logger = logging.getLogger(__name__)

__all__ = ["aufgabe"]

from classification.aufgabe import Aufgabe

aufgabe = Aufgabe()
