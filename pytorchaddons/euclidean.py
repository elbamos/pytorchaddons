from torch.autograd import Function

def euclideandistance(x, param):
    return EuclideanDistance()(x, param)

class EuclideanDistance(Function):
    def forward(self, a, b):
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        d = a2.sub(b2).pow(2).sum(2).pow(.5)
        if d.dim() > 2:
            d = d.squeeze(2)
        self.save_for_backward(a, b, d)
        return d

    def backward(self, grad_d):
        # grad_d is [a, b]
        print("euc back")
        grad_a = grad_b = None
        a, b, d = self.saved_tensors
        a2 = a.unsqueeze(1).expand(a.size(0), b.size(0), a.size(1)) # [a, dim, b]
        b2 = b.unsqueeze(0).expand(a.size(0), b.size(0), b.size(1)) # [a, dim, b]
        grad_d = grad_d.unsqueeze(2).expand(a.size(0), b.size(0), b.size(1))
        d = d.unsqueeze(2).expand(d.size(0), d.size(1), a.size(1))
        if self.needs_input_grad[0]:
            grad_a = a2.sub(b2).mul_(grad_d).div_(d).sum(1).squeeze(1)
        if self.needs_input_grad[1]:
            grad_b = b2.sub(a2).mul_(grad_d).div_(d).sum(0).squeeze(0)
        return grad_a, grad_b