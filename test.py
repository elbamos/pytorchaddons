import torch
from torch.autograd import Variable
import pytorchaddons
import torch.nn as nn


def testone(func, cls, nm):
    print("Testing %s" % nm)
    a = torch.randn(3, 2)
    b = torch.randn(4, 2)
    av = Variable(a, requires_grad = True)
    bv = Variable(b, requires_grad = True)
    out = func(av, bv)
    tester = torch.zeros(3, 4)
    tester[0, 0] = 1
    back = out.backward(tester)
    agrad = av.grad.data
    bgrad = bv.grad.data
    av = Variable(a, requires_grad = True)
    bv = Variable(b, requires_grad = True)
    mod = cls()
    out = mod(av, bv)
    back = out.backward(tester)
    print("Grad a")
    print(agrad)
    print(av.grad.data)
    print("Grad b")
    print(bgrad)
    print(bv.grad.data)
    av = Variable(a, requires_grad = True)
    bv = Variable(b, requires_grad = True)
    print(torch.autograd.gradcheck(func, (av, bv)))

class TestEuclidean(nn.Module):
    def forward(self, a, b):
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), a.size(1)) # [a, dim, b]
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        d = a2.sub(b2).pow(2).sum(2).pow(.5).squeeze(2)
        return d

class TestCosine(nn.Module):
    def forward(self, a, b):
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), a.size(1)) # [a, dim, b]
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        anorms = torch.norm(a, p = 2, dim = 1).squeeze(1)
        bnorms = torch.norm(b, p = 2, dim = 1).squeeze(1)
        denom = torch.ger(anorms, bnorms) # [a, b]
        numer = a2.mul(b2).sum(2).squeeze(2)
        d = numer.div(denom)
        return d

class TestPoincare(nn.Module):
    def forward(self, a, b):
        d = self.euclideandistance(a, b) # [a.size(0), b.size(0)]
        anorms = torch.norm(a, p = 2, dim = 1).squeeze(1) # [a]
        bnorms = torch.norm(b, p = 2, dim = 1).squeeze(1) # [b]
        a_sz = a.size(0)
        b_sz = b.size(0)
        apow = anorms.pow(2).neg_().add_(1).unsqueeze(1).expand(a_sz, b_sz)
        bpow = bnorms.pow(2).neg_().add_(1).unsqueeze(0).expand(a_sz, b_sz)
        denom = apow.mul(bpow) # [a.size(0), b.size(0)]
        inner = d.div(denom).mul_(2).add_(1)
        out = self.arcosh(inner) #[a.size(0), b.size(0)]
        return out

    @staticmethod
    def arcosh(x):
        return x.pow(2).sub_(1).pow(.5).add_(x).log()
    @staticmethod
    def euclideandistance(a, b):
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        d = a2.sub(b2).pow(2).sum(2).pow(.5)
        if d.dim() > 2:
            d = d.squeeze(2)
        return d

testone(pytorchaddons.euclideandistance, TestEuclidean, "Euclidean")
testone(pytorchaddons.cosinedistance, TestCosine, "Cosine")
testone(pytorchaddons.poincaredistance, TestPoincare, "Poincare")