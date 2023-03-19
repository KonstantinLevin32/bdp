import logging
import os

from habitat.core.logging import HabitatLogger

hrl_logger = HabitatLogger(
    name="hrl",
    level=logging.ERROR,
    format_str="%(message)s",
)
