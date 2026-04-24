import math


class RunningStats:
    """Accumulates running statistics using Welford's online algorithm.

    Tracks count, min, max, mean, and standard deviation without
    storing individual observations. Numerically stable for variance
    computation.
    """

    __slots__ = ("count", "min", "max", "_mean", "_m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.min: float = math.inf
        self.max: float = -math.inf
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        """Incorporate a new observation."""
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean if self.count > 0 else math.nan

    @property
    def std(self) -> float:
        if self.count < 2:
            return math.nan
        return math.sqrt(self._m2 / (self.count - 1))
