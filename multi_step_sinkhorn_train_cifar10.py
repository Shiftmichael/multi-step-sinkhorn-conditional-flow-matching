import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from algorithm.utils import ema, generate_samples, infiniteloop

from algorithm.sinkhorn_flow_matching import multistepSinkhornflowFlowMatching
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "msfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_integer("seed", 20, help="seed")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 600001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 64, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_bool("multi_samples", True, help="multi flow samples training")

# Sinkhorn flow
flags.DEFINE_float("blur", 2, help="entropy regulized parameter")
flags.DEFINE_float("scaling", 0.80, help="scaling")
flags.DEFINE_float("SD_stepsize", 0.2, help="Sinkhorn flow step size")
flags.DEFINE_integer("SD_timestep", 10, help="Sinkhorn flow steps")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(argv):
    print(
        "lr, total_steps, ema decay, save_step, blur, scaling, stepsize, timestep, seed:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
        FLAGS.blur,
        FLAGS.scaling,
        FLAGS.SD_stepsize,
        FLAGS.SD_timestep,
        FLAGS.seed,
    )
    
    setup_seed(FLAGS.seed)

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    FM = multistepSinkhornflowFlowMatching(FLAGS.blur, FLAGS.scaling, FLAGS.SD_timestep, FLAGS.SD_stepsize)

    savedir = FLAGS.output_dir + FLAGS.model + "/" + f'blur{FLAGS.blur}_scaling{FLAGS.scaling}_steps{FLAGS.SD_timestep}_size{FLAGS.SD_stepsize}' + "/"
    os.makedirs(savedir, exist_ok=True)
    time = FLAGS.SD_timestep * FLAGS.SD_stepsize
    
    if FLAGS.multi_samples == False:
        total_steps = FLAGS.total_steps
    else:
        multi = 10
        total_steps = FLAGS.total_steps // multi + 1
    
    
    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).view(FLAGS.batch_size, -1).to(device)
            x0 = torch.randn_like(x1)
            if FLAGS.multi_samples == False:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                xt = xt.view(-1, 3, 32, 32)
                ut = ut.view(-1, 3, 32, 32)
                vt = net_model(t, xt)
                loss = torch.mean((vt - ut) ** 2)
                # loss = F.mse_loss(vt, ut, reduction='none')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
                optim.step()
                sched.step()
                ema(net_model, ema_model, FLAGS.ema_decay)  # new

                # sample and Saving the weights
                if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                    generate_samples(net_model, FLAGS.parallel, savedir, step, time, net_="normal")
                    generate_samples(ema_model, FLAGS.parallel, savedir, step, time, net_="ema")
                    torch.save(
                        {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "step": step,
                        },
                        savedir + f"cifar10_weights_step_{step}.pt",
                    )
            else:
                t_multi, xt_multi, ut_multi = FM.multi_sample_location_and_conditional_flow(x0, x1, multi_samples=multi)
                for i in range(multi):
                    t = t_multi[i]
                    xt = xt_multi[i].view(-1, 3, 32, 32)
                    ut = ut_multi[i].view(-1, 3, 32, 32)
                    vt = net_model(t, xt)
                    loss = torch.mean((vt - ut) ** 2)
                    # loss = F.mse_loss(vt, ut, reduction='none')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
                    optim.step()
                    sched.step()
                    ema(net_model, ema_model, FLAGS.ema_decay)  # new

                # sample and Saving the weights
                if FLAGS.save_step > 0 and step % (FLAGS.save_step / multi) == 0:
                    generate_samples(net_model, FLAGS.parallel, savedir, step, time, net_="normal")
                    generate_samples(ema_model, FLAGS.parallel, savedir, step, time, net_="ema")
                    torch.save(
                        {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "step": step,
                        },
                        savedir + f"cifar10_weights_step_{step}.pt",
                    )


if __name__ == "__main__":
    app.run(train)
