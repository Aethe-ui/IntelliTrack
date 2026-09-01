"""No-prediction pass-through for the reactive PID baseline.

:class:`NoPrediction` exposes the same ``predict_next`` interface as the LSTM
and Transformer predictors so that the pipeline can swap prediction stages
without branching.  For the reactive baseline, the "predicted" position is
simply the most recent observed centroid — i.e., no prediction at all.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class NoPrediction:
    """Trivial prediction stage that returns the last observed centroid.

    This is the prediction component for ``TrackingMode.REACTIVE_PID``.  By
    returning the current measurement directly, the PID controller reacts only
    to the object's instantaneous position with no forward estimation.

    Args:
        None
    """

    def predict_next(
        self, history: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Return the most recent centroid as the "predicted" next position.

        Args:
            history: Ordered list of recent ``(x, y)`` centroid measurements,
                most recent last.

        Returns:
            The last element of ``history``, or ``(0.0, 0.0)`` if empty.
        """
        if not history:
            logger.warning("NoPrediction.predict_next(): history is empty, returning (0, 0).")
            return (0.0, 0.0)
        return history[-1]
