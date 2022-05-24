import torch


def pgd_defense(discriminator, images, args):

    images = images.detach()

    eps = args.defense_eps
    alpha = eps / 4
    iters = args.defense_step

    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True

        outputs = discriminator(images, images, args).sum()

        grad = torch.autograd.grad(outputs, images,
                                   retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()

        adv_images = images + alpha * grad_sign

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        del grad

    adv_images = images

    return adv_images
