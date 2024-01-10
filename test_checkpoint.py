import copy
import sys

import torch
from torchdyn.core import NeuralODE
from absl import app, flags

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

from torchcfm.models.unet.unet import UNetModelWrapper

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "NSGF", help="flow matching model type")
flags.DEFINE_integer("step", 2000, help="training steps")
flags.DEFINE_integer("multi_samples", 20, help="multi flow samples training")
FLAGS(sys.argv)

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=128,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


PATH = f"{FLAGS.input_dir}/{FLAGS.model}/blur2.0_scaling0.8_steps100_size0.03_50/"
print("path: ", PATH)

checkpoint = torch.load(PATH+f'cifar10_weights_step_{FLAGS.step}.pt')
state_dict = checkpoint["ema_model"]

new_net.load_state_dict(state_dict)

new_net.eval()

steps = 5
stepsize = 0.2
x0 = torch.randn(64, 3, 32, 32).to(device)
# for i in range(steps):
#     t = i*stepsize * torch.ones(x0.shape[0],).to(device)
#     vt = new_net(t, x0)
#     x0 = x0 + vt * stepsize
node_ = NeuralODE(new_net, solver="euler", sensitivity="adjoint")
with torch.no_grad():
    traj = node_.trajectory(
        torch.randn(64, 3, 32, 32).to(device),
        t_span=torch.linspace(0, -3, 100).to(device),
    )
    traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
    traj = traj / 2 + 0.5
save_image(traj, PATH + f"ema_generated_FM_images_step_{FLAGS.step}.png", nrow=8)
