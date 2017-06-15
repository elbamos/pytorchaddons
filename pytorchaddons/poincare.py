from torch.autograd import Function
import torch
from .euclidean import EuclideanDistance


def poincaredistance(x, param):
    return PoincareDistance()(x, param)

class PoincareDistance(Function):
    def forward(self, a, b):
        d = self.euclideandistance(a, b).pow(2).mul_(2) # [a.size(0), b.size(0)]
        anorms = torch.norm(a, p = 2, dim = 1).squeeze(1) # [a]
        bnorms = torch.norm(b, p = 2, dim = 1).squeeze(1) # [b]
        apow = anorms.pow(2).neg_().add_(1)
        bpow = bnorms.pow(2).neg_().add_(1)
        denom = torch.ger(apow, bpow) # [a.size(0), b.size(0)]
        inner = d.div(denom).add_(1)
        gamma = inner.pow(2).sub_(1).sqrt() # This and the next line implement arcosh
        out = gamma.add(inner).log()
        self.save_for_backward(a, b)
        self.alphabeta = apow, bpow
        self.gamma = gamma
        return out

    def backward(self, grad_out):
        x, theta = self.saved_tensors
        grad_x = grad_theta = None
        # track the sizes
        x_sz = x.size(0)
        theta_sz = theta.size(0)
        dim = x.size(1)
        # Some things we'll reuse
        xnormspow = torch.norm(x, p = 2, dim = 1).pow(2).squeeze()
        thetanormspow = torch.norm(theta, p = 2, dim = 1).pow(2).squeeze()
        # Equations 3, the tensors are intermediate steps from forward
        beta, alpha = self.alphabeta
        gamma = self.gamma # [a, b]
        # Create some things we'll reuse
        multiplier = grad_out.mul(4).div_(gamma).unsqueeze(2).expand(x_sz, theta_sz, dim)
        mult = torch.mm(x, theta.t()).mul_(-2).add_(1)
        # Resize tensors for simplicty later
        alpha2 = alpha.unsqueeze(0).expand(x_sz, theta_sz)
        beta2 = beta.unsqueeze(1).expand(x_sz, theta_sz)
        x3 = x.unsqueeze(1).expand(x_sz, theta_sz, dim)
        theta3 = theta.unsqueeze(0).expand(x_sz, theta_sz, dim)
        alpha3 = alpha2.unsqueeze(2).expand(x_sz, theta_sz, dim)
        beta3 = beta2.unsqueeze(2).expand(x_sz, theta_sz, dim)
        # Equation 4
        if self.needs_input_grad[1]:
            # Equation 4 for theta
            grad_theta = xnormspow.unsqueeze(1).expand(x_sz, theta_sz).add(mult).div_(alpha2.pow(2)).unsqueeze(2).expand(x_sz, theta_sz, dim).mul(theta3).sub_(x3.div(alpha3)).mul_(multiplier).div_(beta3).sum(0).squeeze(0)
            riemannian = alpha.pow(2).div_(4).unsqueeze(1).expand(theta_sz, dim)
            grad_theta.mul_(riemannian)
        # Equation 4 for X
        if self.needs_input_grad[0]:
            grad_x = thetanormspow.unsqueeze(0).expand(x_sz, theta_sz).add(mult).div_(beta2.pow(2)).unsqueeze(2).expand(x_sz, theta_sz, dim).mul(x3).sub_(theta3.div(beta3)).mul_(multiplier).div_(alpha3).sum(1).squeeze(1)
            riemannian = beta.pow(2).div_(4).unsqueeze(1).expand(x_sz, dim)
            grad_x.mul_(riemannian)
        return grad_x, grad_theta

    @staticmethod
    def euclideandistance(a, b):
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), b.size(1))
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1))
        return a2.sub(b2).pow(2).sum(2).squeeze(2).sqrt()
    
    @staticmethod
    def confinetodisc(theta, eps = 1e-4):
        norms = torch.norm(theta, p = 2, dim = 1).clamp(min = 1.0 - eps).add(eps).expand_as(theta)
        theta.div_(norms)