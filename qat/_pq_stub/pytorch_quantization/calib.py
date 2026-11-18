class _Calibrator:
    def __init__(self, *a, **kw):
        self.amax = 1.0


class MaxCalibrator(_Calibrator):
    pass


class HistogramCalibrator(_Calibrator):
    pass
