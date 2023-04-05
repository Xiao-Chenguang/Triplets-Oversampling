from torchmetrics.classification import (
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryConfusionMatrix,
    BinaryAveragePrecision
)



class Metrics:
    def __init__(self, device):
        self.accuracy = BinaryAccuracy().to(device)
        self.conf_matrix = BinaryConfusionMatrix().to(device)
        self.recall = BinaryRecall().to(device)
        self.precision = BinaryPrecision().to(device)
        self.f1 = BinaryF1Score().to(device)
        self.auroc = BinaryAUROC().to(device)
        self.ap = BinaryAveragePrecision().to(device)
        self.conf_matrix = BinaryConfusionMatrix().to(device)
        # TODO: add G-mean and AUC-PR

    def __call__(self, y_hat, y):
        acc = self.accuracy(y_hat, y).item()
        recall = self.recall(y_hat, y).item()
        precision = self.precision(y_hat, y).item()
        f1 = self.f1(y_hat, y).item()
        (tn, fp), (fn, tp) = self.conf_matrix(y_hat, y)
        gmean = ((tp/(tp+fn) * tn/(tn+fp))**0.5).item()
        auc = self.auroc(y_hat, y).item()
        ap = self.ap(y_hat, y).item()
        return acc, recall, precision, f1, gmean, auc, ap

    # def compute(self, y_hat, y):
    #     auc = self.auroc(y_hat, y)
    #     f1 = self.f1(y_hat, y)
    #     recall = self.recall(y_hat, y)
    #     acc = self.accuracy(y_hat, y)
    #     return auc, f1, recall, acc
