class RunningAverage:
    def __init__(self):
        self.sum: float = 0.0
        self.count: int = 0
        self.average: float = 0

    def update(self, x: float, n: int = 1) -> None:
        self.sum += x
        self.count += n
        self.average = self.sum / self.count
