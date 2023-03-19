import logging
import os

from habitat.core.logging import HabitatLogger

baselines_logger = HabitatLogger(
    name="bdp",
    level=int(os.environ.get("HABITAT_BASELINES_LOG", logging.ERROR)),
    format_str="[%(levelname)s,%(name)s] %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)
