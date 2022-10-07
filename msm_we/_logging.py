# This probably doesn't need to exist, but I'm not sure where else this could go. Maybe __init__

import logging
from rich.logging import RichHandler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(RichHandler())
log.propagate = False
