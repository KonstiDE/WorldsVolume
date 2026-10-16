from ray.tune.stopper import Stopper


class PatienceStopper(Stopper):
    def __init__(self, metric, mode="min", patience=5):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.best_scores = {}
        self.counters = {}

    def __call__(self, trial_id, result):
        score = result.get(self.metric)
        if score is None:
            return False

        if trial_id not in self.best_scores:
            self.best_scores[trial_id] = score
            self.counters[trial_id] = 0
            return False

        if (self.mode == "min" and score < self.best_scores[trial_id]) or (self.mode == "max" and score > self.best_scores[trial_id]):
            self.best_scores[trial_id] = score
            self.counters[trial_id] = 0
        else:
            self.counters[trial_id] += 1

        return self.counters[trial_id] >= self.patience

    def stop_all(self):
        return False