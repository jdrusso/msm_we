# This probably doesn't need to exist, but I'm not sure where else this could go. Maybe __init__

import logging
from rich.logging import RichHandler
from rich.progress import Progress

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(RichHandler())
log.propagate = False


class ProgressBar:

    def __init__(self, progress_bar: Progress = None):

        if progress_bar is None:
            self._progress_bar = Progress()
            self._used_existing = False
        else:
            self._progress_bar = progress_bar
            self._used_existing = True

    def __enter__(self):
        if not self._used_existing:
            self._progress_bar.start()
        return self._progress_bar

    def __exit__(self, exc_type, exc_val, exc_tb):

        self._progress_bar.refresh()
        if not self._used_existing:
            self._progress_bar.stop()
