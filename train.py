import torch
from constants import *
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model.generator import Generator, get_noise
from model.critic import Critic
from utils.utils import weights_init, init_setting, show_tensor_images
import ipdb

"""
- Enforces a soft version of the Lipschitz constraint on the critic. 
- Crucial for the stability and effectiveness of WGAN training.
"""


def gradient_of_critic_score(critic, real, fake, epsilon):
    # print("\n INSIDE gradient_of_critic_score() \n")
    # ipdb.set_trace()

    # # Mix the images together: x_hat - [128, 1, 28, 28]
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images: c(x_hat) - [128, 1]
    mixed_scores = critic(interpolated_images)

    # Computes the gradient of mixed_scores with respect to interpolated_images: \nabla c(x_hat)
    # Magnitude of each image's gradient.
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),  # necessary because mixed_scores is not a scalar
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def test_gradient_of_critic_score(image_shape):
    print("\n INSIDE test_gradient_of_critic_score() \n")
    ipdb.set_trace()

    critic = Critic(im_chan=1, hidden_dim=HIDDEN_DIM).to(DEVICE)
    critic.apply(weights_init)
    real = torch.randn(*image_shape, device=DEVICE) + 1
    fake = torch.randn(*image_shape, device=DEVICE) - 1
    eps_shape = [1 for _ in image_shape]  # [1, 1, 1, 1]

    eps_shape[0] = image_shape[0]  # batch dimension
    eps = torch.rand(eps_shape, device=DEVICE).requires_grad_()  # values between 0-1

    gradient = gradient_of_critic_score(critic, real, fake, eps)
    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient


"""
- Ideally, we want the penalty to be zero, i.e., grad[c(x_hat)] approx. 1. 
- For a GOOD GRADIENT, the expression || nabla c(x_hat) - 1 || ^ 2 would be MINIMUM if nabla c(x_hat) is MAXIMUM, i.e., (1-1)^2 = 0
- For a BAD GRADIENT, the expression || nabla c(x_hat) - 1 || ^ 2 would be MAXIMUM if nabla c(x_hat) is MINIMUM, i.e., (0-1)^2 = 1
"""


# Calculate the Penalty on the Norm of Gradient
def gradient_penalty_l2_norm(gradient):
    # print("\nINSIDE gradient_penalty_l2_norm() \n")
    # ipdb.set_trace()

    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)  # [B, 1, 28, 28] -> [B, 784]

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(p=2, dim=1)  # [B, 784] -> [B]

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)  # regularization term
    return penalty


def test_gradient_penalty_l2_norm(image_shape):
    # When gradient = 0s, then REG TERM will be MAXIMIZED, i.e., (0-1)^2 = 1
    bad_gradient = torch.zeros(*image_shape)
    bad_gradient_penalty = gradient_penalty_l2_norm(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.0))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))
    print(f"{torch.sqrt(image_size)=}")

    # 0.357 - each grad val
    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
    """
    (norm(grad(score'x)) - 1)^ 2 = 0 
    (norm(grad(score'x)) - 1) = 0 | sqrt on both sides 
    norm(grad(score'x)) = 1 | move 1 to RHS
    sqrt(sum(grad_i^2)) = 1 | for i=1,..,n
    After more simplification, grad_i would be equal to 1/sqrt(image_size) 
    """
    good_gradient_penalty = gradient_penalty_l2_norm(good_gradient)
    assert torch.isclose(good_gradient_penalty, torch.tensor(0.0))

    random_gradient = test_gradient_of_critic_score(image_shape)
    random_gradient_penalty = gradient_penalty_l2_norm(random_gradient)
    assert torch.abs(random_gradient_penalty - 1) < 0.1


if __name__ == "__main__":
    exp_path, checkpoint_dir, image_dir = init_setting()
    dataset = MNIST(root="./data", transform=TRANSFORM, download=False, train=True)
    # mnist_subset = Subset(dataset, range(DSET_SUBSET_SIZE))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    ipdb.set_trace()
    # test_gradient_of_critic_score(image_shape=(128, 1, 28, 28))
    # test_gradient_penalty_l2_norm((128, 1, 28, 28))

    generator = Generator(im_chan=1, z_dim=Z_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    generator.apply(weights_init)
    critic = Critic(im_chan=1, hidden_dim=HIDDEN_DIM).to(DEVICE)
    critic.apply(weights_init)

    optimizerG = torch.optim.Adam(params=generator.parameters(), lr=LR, betas=(BETA_1, BETA_2))
    optimizerC = torch.optim.Adam(params=critic.parameters(), lr=LR, betas=(BETA_1, BETA_2))

    current_step = 0
    generator_losses = []
    critic_losses_across_critic_repeats = []
    for epoch in range(N_EPOCHS):
        for real, _ in train_loader:
            cur_batch_size = len(real)
            real = real.to(DEVICE)

            mean_critic_loss_for_this_iteration = 0
            for _ in range(CRIT_REPEATS):
                optimizerC.zero_grad()
                fake_noise = get_noise(n_samples=cur_batch_size, z_dim=Z_DIM).to(DEVICE)  # [B, Z]
                fake = generator(fake_noise)  # G(z) = [B, 1, 28, 28]
                critic_fake_pred = critic(fake.detach())  # C(G(z)) = [B, 1]
                critic_real_pred = critic(real)  # C(x) = [B, 1]

                eps = torch.rand(len(real), 1, 1, 1, device=DEVICE, requires_grad=True)  # [B, 1, 1, 1]
                gradient = gradient_of_critic_score(critic, real, fake, eps)  # [32, 1, 28, 28]
                gp = gradient_penalty_l2_norm(gradient)  # scalar

                # minG maxC [E(C(x)) - E(C(G(z))) + lambda * gp].
                # Add a "-" to MIN instead of MAX
                crit_loss = -1.0 * (torch.mean(critic_real_pred) - torch.mean(critic_fake_pred)) + C_LAMBDA * gp

                mean_critic_loss_for_this_iteration += crit_loss.item() / CRIT_REPEATS

                crit_loss.backward(retain_graph=True)  # Update Gradients
                optimizerC.step()

            critic_losses_across_critic_repeats += [mean_critic_loss_for_this_iteration]

            ## Train Generator
            optimizerG.zero_grad()
            fake_noise_2 = get_noise(n_samples=cur_batch_size, z_dim=Z_DIM).to(DEVICE)  # [B, Z]
            fake_2 = generator(fake_noise_2)  # [32, 1, 28, 28]
            critic_fake_pred = critic(fake_2)  # [32, 1]

            # Maximizing the critic's prediction on the generator's fake images
            gen_loss = -1.0 * torch.mean(critic_fake_pred)  # - [D(G(z))]
            gen_loss.backward()
            optimizerG.step()
            generator_losses += [gen_loss]  # keep track of generator losses.

            if current_step % DISPLAY_STEP == 0 and current_step > 0:
                gen_mean_loss = sum(generator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                crit_mean_loss = sum(critic_losses_across_critic_repeats[-DISPLAY_STEP:]) / DISPLAY_STEP
                print(f"EPOCH {epoch} | G_LOSS {gen_mean_loss} | C_LOSS {crit_mean_loss}")
            current_step += 1

        show_tensor_images(fake, show=False, plot_name=f"{image_dir}/epoch-{epoch}-fake.png")
        show_tensor_images(real, show=False, plot_name=f"{image_dir}/epoch-{epoch}-real.png")

        checkpoint = {
            "epoch": epoch,
            "gen_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "gen_optimizer": optimizerG.state_dict(),
            "critic_optimizer": optimizerC.state_dict(),
        }  # save state dictionary

        torch.save(checkpoint, f"{checkpoint_dir}/model.pth")
