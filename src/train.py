from metrics import Metrics as SegMetrics

# Used to keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, problem,
          epochs=10, batch_size=8, learning_rate=1e-5,
          reports_per_epoch = 10):
    pass