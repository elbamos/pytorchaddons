from torchnet.meter.meter import Meter
import numpy as np
from sortedcontainers import SortedList
from torch.autograd import Variable

class DCGMeter(Meter):
    def __init__(self, emphasize_recall = True):
        self.emphasize_recall = emphasize_recall
        self.stored = SortedList()

    def reset(self):
        self.stored.clear()

    def checkParams(self, scores, relevancies):
        if isinstance(scores, Variable):
            scores = scores.data
        if isinstance(relevancies, Variable):
            relevancies = relevancies.data
        scores = scores.squeeze()
        relevancies = relevancies.squeeze()
        assert scores.dim() == 1
        assert relevancies.dim() == 1
        return scores, relevancies

    def add(self, scores, relevancies):
        scores, relevancies = self.checkParams(scores, relevancies)
        if self.emphasize_recall:
            toadd = [(- score, (2 ** relevancy) - 1) for score, relevancy in zip(scores, relevancies)]
        else:
            toadd = [ (-score, relevancy) for score, relevancy in zip(scores, relevancies)]
        self.stored.update(toadd)

    def value(self):
        outlist = [relevancy / np.log2(i + 2) for i, (_, relevancy) in enumerate(self.stored)]
        outlist = np.cumsum(outlist)
        return outlist

class NDCGMeter(DCGMeter):
    def __init__(self, emphasize_recall = True):
        super(NDCGMeter, self).__init__(emphasize_recall)
        self.best = SortedList()

    def reset(self):
        super(NDCGMeter, self).reset()

    def add(self, scores, relevancies):
        scores, relevancies = self.checkParams(scores, relevancies)
        super(NDCGMeter, self).add(scores, relevancies)
        if self.emphasize_recall:
            toadd = [(- relevancy, (2 ** relevancy) - 1) for relevancy in relevancies]
        else:
            toadd = [(- relevancy, relevancy) for relevancy in relevancies]
        self.best.update(toadd)

    def value(self):
        dcgs = super(NDCGMeter, self).value()
        ideals = [relevancy / np.log2(i + 2) for i, (_, relevancy) in enumerate(self.best)]
        ideals = np.cumsum(ideals)
        outlist = [dcg / idcg for dcg, idcg in zip(dcgs, ideals)]
        return outlist
