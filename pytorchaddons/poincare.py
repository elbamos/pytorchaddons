from torch.autograd import Function
import torch
from .euclidean import EuclideanDistance

def poincaredistance(x, param, use_poincare_for_input = False):
    return PoincareDistance(use_poincare_for_input)(x, param)

class PoincareDistance(Function):
    def __init__(self, use_poincare_for_input = False):
        super(PoincareDistance, self).__init__()
        self.use_poincare_for_input = use_poincare_for_input

    def forward(self, a, b):
        d = self.euclideandistance(a, b) # [a.size(0), b.size(0)]
        anorms = torch.norm(a, p = 2, dim = 1).squeeze(1) # [a]
        bnorms = torch.norm(b, p = 2, dim = 1).squeeze(1) # [b]
        a_sz = a.size(0)
        b_sz = b.size(0)
        apow = anorms.pow(2).neg_().add_(1).unsqueeze(1).expand(a_sz, b_sz)
        bpow = bnorms.pow(2).neg_().add_(1).unsqueeze(0).expand(a_sz, b_sz)
        denom = apow.mul(bpow) # [a.size(0), b.size(0)]
        inner = d.pow(2).div(denom).mul_(2).add_(1)
        out = self.arcosh(inner) #[a.size(0), b.size(0)]
        self.save_for_backward(a, b, out)
        return out

    def backward(self, grad_out):
        x, theta, d = self.saved_tensors
        grad_x = grad_theta = None
        x_sz = x.size(0)
        theta_sz = theta.size(0)
        dim = x.size(1)
        # Some things we'll reuse
        xnormspow = torch.norm(x, p = 2, dim = 1).pow(2).squeeze()
        thetanormspow = torch.norm(theta, p = 2, dim = 1).pow(2).squeeze()
        # Equations 3
        alpha = thetanormspow.neg().add_(1).unsqueeze(0).expand(x_sz, theta_sz)
        beta = xnormspow.neg().add_(1).unsqueeze(1).expand(x_sz, theta_sz)
        euc_d = self.euclideandistance(x, theta)
        gamma = alpha.mul(beta).reciprocal_().mul_(euc_d.pow(2)).mul_(2).add_(1) # [a, b]
        # Create some things we'll reuse
        multiplier = grad_out.mul(4).div_(gamma.pow(2).sub_(1).pow(.5)).unsqueeze(2).expand(x_sz, theta_sz, dim)
        xt = x.unsqueeze(1).expand(x_sz, theta_sz, dim)
        thetat = theta.unsqueeze(0).expand(x_sz, theta_sz, dim)
        alphat = alpha.unsqueeze(2).expand(x_sz, theta_sz, dim)
        betat = beta.unsqueeze(2).expand(x_sz, theta_sz, dim)
        mult = torch.mm(x, theta.t()).mul_(-2).add_(1)
        if self.needs_input_grad[1]:
            # Equation 4 for theta
            grad_theta = xnormspow.unsqueeze(1).expand(x_sz, theta_sz).add(mult).div_(alpha.pow(2)).unsqueeze(2).expand(x_sz, theta_sz, dim).mul(thetat).sub_(xt.div(alphat)).mul_(multiplier).div_(betat).sum(0).squeeze(0)
            riemannian = thetanormspow.neg().add_(1).pow(2).div_(4).unsqueeze(1).expand(theta_sz, dim)
            grad_theta.mul_(riemannian)
        # Equation 4 for X
        if self.needs_input_grad[0]:
            grad_x = thetanormspow.unsqueeze(0).expand(x_sz, theta_sz).add(mult).div_(beta.pow(2)).unsqueeze(2).expand(x_sz, theta_sz, dim).mul(xt).sub_(thetat.div(betat)).mul_(multiplier).div_(alphat).sum(1).squeeze(1)
            if self.use_poincare_for_input:
                riemannian = xnormspow.neg().add_(1).pow(2).div_(4).unsqueeze(1).expand(x_sz, dim)
                grad_x.mul_(riemannian)
        return grad_x, grad_theta

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
