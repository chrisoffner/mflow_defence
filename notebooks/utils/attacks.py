import torch
import torch.nn as nn

def fgsm(
        model:    torch.nn.Module,
        x:        torch.Tensor,
        label:    int,
        eps:      float,
        targeted: bool = True,
        clip_min: None|float = None,
        clip_max: None|float = None
    ) -> torch.Tensor:
    """
    Internal functionality for all FGSM and PGD attacks.

    Parameters
    ----------
    model : torch.nn.Module
        A model trained on the dataset for which the adversarial sample is to
        be generated.
    x : torch.Tensor
        The data sample to be turned into an adversarial sample.
    label : int
        If `targeted == True` this is the index of the (incorrect) class that
        the adversarial sample should get classified to. If `targeted == False`,
        this is the index of the correct class of `x`.
    eps : float
        Defines how far `x` gets shifted in any dimension. Should be small.
    targeted : bool, optional
        Defines whether the attack should be targeted or not, by default True.
    clip_min : _type_, optional
        Lower clipping range (e.g. 0 for pixels in [0, 1]), by default None
    clip_max : _type_, optional
        Upper clipping range (e.g. 1 for pixels in [0, 1]), by default None

    Returns
    -------
    torch.Tensor
        The perturbed input (adversarial example).
    """
    
    # Copy the input and remove connection to the compute graph
    input = x.clone().detach_()
    
    # Make sure we compute the loss gradient w.r.t. the input
    input.requires_grad_()

    # Run a forward pass of the model and compute the loss
    model.zero_grad()
    logits = model(input)
    target = torch.LongTensor([label]).to(label.device)
    loss   = nn.CrossEntropyLoss()(logits, target)

    # Run backprop to get gradient w.r.t. the input we wish to modify
    loss.backward()
    
    # Perform either targeted or untargeted attack
    if targeted:
        x_adv = input - eps * input.grad.sign()
    else:
        x_adv = input + eps * input.grad.sign()
    
    # Optionally, clip the output back to a data domain
    # (useful for images in [0, 1]^n domain)
    if clip_min is not None or clip_max is not None:
        x_adv.clamp_(min=clip_min, max=clip_max)
    
    return x_adv


def pgd(
        model:    torch.nn.Module,
        x:        torch.Tensor,
        label:    int,
        k:        int,
        eps:      float,
        eps_step: float,
        targeted: bool,
        clip_min: None|float = None,
        clip_max: None|float = None
    ) -> torch.Tensor:
    """
    Performs a Projected Gradient Descent (PGD) attack on the input x.

    PGD is an iterative adversarial attack that attempts to find an adversarial
    example within an eps-ball around the input.

    Parameters
    ----------
    model : torch.nn.Module
        The model to attack.
    x : torch.Tensor
        The input tensor to perturb.
    label : int
        If `targeted == True` this is the index of the (incorrect) class that
        the adversarial sample should get classified to. If `targeted == False`,
        this is the index of the correct class of `x`.
    k : int
        The number of PGD iterations.
    eps : float
        Defines how far `x` gets shifted in any dimension. Should be small.
    eps_step : float
        The step size for each iteration.
    targeted : bool
       Defines whether the attack should be targeted or not, by default True.
    clip_min : _type_, optional
        Lower clipping range (e.g. 0 for pixels in [0, 1]), by default None
    clip_max : _type_, optional
        Upper clipping range (e.g. 1 for pixels in [0, 1]), by default None

    Returns
    -------
    torch.Tensor
        The perturbed input (adversarial example).
    """
    x_min = x - eps
    x_max = x + eps
    
    # Randomise the starting point x
    x_adv = x + eps * (2 * torch.rand_like(x) - 1)

    # Clamp starting point back to valid range
    if clip_min is not None or clip_max is not None:
        x_adv.clamp_(min=clip_min, max=clip_max)
    
    for _ in range(k):
        # FGSM step
        x_adv = fgsm(model, x_adv, label, eps_step, targeted)
        
        # Projection Step
        x_adv = torch.min(x_max, torch.max(x_min, x_adv))
        
    # Optionally, clip the output back to a data domain
    # (useful for images in [0, 1]^n domain)
    if clip_min is not None or clip_max is not None:
        x_adv.clamp_(min=clip_min, max=clip_max)

    return x_adv
