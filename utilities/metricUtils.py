import os, csv
import numpy as np
import torch
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt

def get_class_weights(targets, nclasses):
    """
    Sample level weights fro balanced Loss statergy
    targets: assumed to be Long ints representing class from dataset
    """

    n_target = len(targets)
    count_per_class = np.zeros(nclasses, dtype=int)
    for c in targets:
        count_per_class[c] += 1

    count_per_class[count_per_class==0] = n_target

    weight_per_class = np.zeros(nclasses, dtype=float)
    for i in range(nclasses):
        weight_per_class[i] = float(n_target) / float(count_per_class[i])

    return weight_per_class



class MultiClassMetrics():
    def __init__(self, logpath, digits=4):
        self.tgt = []
        self.prd = []
        self.nnloss = []
        self.loss_info_list = []
        self.digits = digits
        self.logpath = logpath
        os.makedirs(self.logpath, exist_ok=True)

    def reset(self, save_results = False):
        if save_results: self._write_predictions()
        self.__init__(self.logpath, self.digits)

    def add_entry(self, prd, tgt, loss=0, loss_info={}):
        self.prd.extend(prd.cpu().detach().numpy())
        self.tgt.extend(tgt.cpu().detach().numpy())
        if loss: self.nnloss.append(loss.cpu().detach().numpy().round(decimals=self.digits+2))
        if loss_info: self.loss_info_list.append(loss_info)

    def get_loss(self):
        r = sum(self.nnloss) / len(self.nnloss)
        return round(r, self.digits)

    def get_accuracy(self):
        r = skmetrics.accuracy_score(self.tgt, self.prd)
        return round(r, self.digits)

    def get_balanced_accuracy(self):
        r = skmetrics.balanced_accuracy_score(self.tgt, self.prd)
        return round(r, self.digits)

    def get_f1score(self):
        r = skmetrics.f1_score(self.tgt, self.prd, average='macro')
        return round(r, self.digits)

    def get_class_report(self):
        return skmetrics.classification_report(self.tgt, self.prd,
                    output_dict= True,)

    def get_confusion_matrix(self, save_png = False, title="cls"):
        lbls = sorted(list(set(self.tgt)))
        cm = skmetrics.confusion_matrix(self.tgt, self.prd,
                                labels= lbls)
        if save_png:
            disp = skmetrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=lbls).plot()
            plt.savefig(self.logpath+f'/confusion-{title}.png', bbox_inches='tight')

        return cm

    def get_loss_info_aggregates(self):
        key_set = set()
        for lkv in self.loss_info_list: key_set.update(lkv.keys())
        info_agg = {}
        for k in key_set:
            info_agg[k] = 0.0 ; info_agg[str(k)+"_count"] = 0

        for lkv in self.loss_info_list:
            for k,v in lkv.items():
                info_agg[k] += v
                info_agg[str(k)+"_count"] += 1

        out_agg = {}
        for k in key_set:
            out_agg[k] = info_agg[k] / info_agg[str(k)+"_count"]

        return out_agg


    def _write_predictions(self, title="cls"):
        print(f"IN METRICS WRITE PRED {self.logpath}")
        with open(os.path.join(self.logpath, f"Predict-{title}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["target", "prediction"])
            writer.writerows(zip(self.tgt, self.prd))


if __name__ == "__main__":

    obj = MultiClassMetrics()
    obj.tgt = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
    obj.prd = [1,1,2,2,2,3,3,3,4,4,4,5,5,5,1]

    print(obj.get_class_report())