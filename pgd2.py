from attack import Attacker
import torch
import torch.nn.functional as F

#for MUFAC ; changed the loss function following the MUCAC setting
class PGD2(Attacker):
    def __init__(self, model, config, target=None):
        super(PGD2, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        x_adv = x.detach().clone()
        if self.config['random_init'] :
            x_adv = self._random_init(x_adv)
        for _ in range(self.config['attack_steps']):
            x_adv.requires_grad = True
            self.model.zero_grad()
            logits = self.model(x_adv) #f(T((x))
            if self.target is None:
                # sample-wise
                #loss = F.cross_entropy(logits, y,  reduction="sum")
                loss = F.binary_cross_entropy_with_logits(logits, y,  reduction="sum")
                loss.backward()                      
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv - self.config['attack_lr'] * grad

            else:
                # class-wise
                assert self.target.size() == y.size()           
                loss = F.cross_entropy(logits, self.target)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv - self.config['attack_lr'] * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.config['eps'], max=self.config['eps'])
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv