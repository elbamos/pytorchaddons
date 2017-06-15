from torch.autograd import Function
import torch

def cosinedistance(x, param, similarity = True):
    return CosineDistance(similarity)(x, param)

class CosineDistance(Function):
    def __init__(self, similarity = True):
        super(CosineDistance, self).__init__()
        self.similarity = similarity

    def forward(self, a, b):
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), a.size(1)) # [a, dim, b]
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        anorms = torch.norm(a, p = 2, dim = 1).squeeze(1)
        bnorms = torch.norm(b, p = 2, dim = 1).squeeze(1)
        denom = torch.ger(anorms, bnorms) # [a, b]
        numer = a2.mul(b2).sum(2).squeeze(2)
        d = numer.div(denom)
        if not self.similarity:
            d = d.neg_().add_(2)
        self.save_for_backward(a, b, d)
        return d

    def half(self, a, b, ddivided, dmultiplier, anorm):
    # The partial derivative wrt a is (b / (anorm * bnorm)) - cossim(a, b) * (a / anorm^2)
        tmp1 = b.mul(ddivided) # ddivided = grad_d / (anorm * bnorm)
        tmp2 = dmultiplier.mul(a).div_(anorm.pow(2)) # dmultiplier = d.mul(grad_d)
        return tmp1.sub_(tmp2)

    def backward(self, grad_d):
        grad_a = grad_b = None
        a, b, d = self.saved_tensors
        dims = a.size(1)
        a_sz = a.size(0)
        b_sz = b.size(0)
        anorms = torch.norm(a, p = 2, dim = 1).squeeze(1)
        bnorms = torch.norm(b, p = 2, dim = 1).squeeze(1)
        denom = torch.ger(anorms, bnorms) # [a, b]
        if not self.similarity:
            d = d.neg()
        # Create some structures to simplify the calculation
        atmp = a.unsqueeze(1).expand(a.size(0), b.size(0), a.size(1))
        btmp = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1))
        dmultiplier = d.mul(grad_d).unsqueeze(2).expand(a_sz, b_sz, dims)
        ddivided = grad_d.div(denom).unsqueeze(2).expand(a_sz, b_sz, dims)
        if self.needs_input_grad[0]:
            anormpow = anorms.unsqueeze(1).unsqueeze(2).expand(a_sz, b_sz, dims)
            grad_a = self.half(atmp, btmp, ddivided, dmultiplier, anormpow).sum(1).squeeze(1)
        if self.needs_input_grad[1]:
            bnormpow = bnorms.unsqueeze(0).unsqueeze(2).expand(a_sz, b_sz, dims)
            grad_b = self.half(btmp, atmp, ddivided, dmultiplier, bnormpow).sum(0).squeeze(0)
        return grad_a, grad_b