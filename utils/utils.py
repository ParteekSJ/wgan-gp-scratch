import sys

sys.path.append("../")

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch import nn
import torch
import datetime
from pathlib import Path
from constants import *


def init_setting():
    timestr = str(datetime.datetime.now().strftime("%Y-%m%d_%H%M"))
    experiment_dir = Path(LOG_PATH)
    experiment_dir.mkdir(exist_ok=True)  # directory for saving experimental results
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)  # root directory of each experiment

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_dir = checkpoint_dir.joinpath(timestr)
    checkpoint_dir.mkdir(exist_ok=True)  # root directory of each checkpoint

    image_dir = Path(IMAGE_DIR)
    image_dir.mkdir(exist_ok=True)
    image_dir = image_dir.joinpath(timestr)
    image_dir.mkdir(exist_ok=True)  # root directory of each image (generated and real)

    # returns several directory paths
    return experiment_dir, checkpoint_dir, image_dir


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), show=False, plot_name=""):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2  # [-1, 1] -> [0, 1]
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

    plt.savefig(f"{plot_name}")
    plt.close(fig)


def make_grad_hook():
    gradients_list = []

    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            gradients_list.append(m.weight.grad)

        return gradients_list, grad_hook


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(tensor=m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(tensor=m.weight, mean=0.0, std=0.02)
        nn.init.constant_(tensor=m.bias, val=0.0)


def get_gen_loss(critic_fake_prediction):
    """
    REAL IMAGES: +ve, FAKE IMAGES: -ve scores.
    - Generator tries to maximize D(G(z)) [high pred for fake images],
      which is equivalent to minimizing - D(G(z)
    - Generator Loss = -[average critic score on fake/generated  images] = -(D(G(z))"""

    gen_loss = -1.0 * torch.mean(critic_fake_prediction)
    return gen_loss


# Critic Loss
def get_crit_loss(critic_fake_prediction, critic_real_prediction, gp, c_lambda):
    """
    Math for the Loss Function for the Critic and Generator is
    Critic Loss: D(X) - D(G(z)) <- MAXIMIZE THIS EXPRESSION, i.e., difference between real and fake images.

    So arithmetically, maximizing an expression is equivalent to minimizing the -ve of that expression.
    i.e., max D(X) - D(G(z)) = min -(D(X) - D(G(z))) = min D(G(z)) - D(X)
    """

    # Mean is for all the scores in the batch
    crit_loss = (
        torch.mean(critic_fake_prediction) - torch.mean(critic_real_prediction)
    ) + c_lambda * gp  # gradient penalty

    return crit_loss
