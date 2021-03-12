# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import pathlib
import re
import shutil
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--prefix', help='Add prefix to image name', required=False, default=None)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        projected_w: Optional[str],
        prefix: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if prefix is None:
            fname = f'{outdir}/seed{seed:04d}.png'
        else:
            fname = f'{outdir}/{prefix}seed{seed:04d}.png'
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(fname)


def prepare_train_set(trv_ns_path, dbp_ns_path, gan_seeds, truncation_psi, outdir, class_idx=None, dlatents_npz=None,
                      baseline_path=None):

    assert os.path.exists(outdir), f"{outdir} path does not exists."
    if not isinstance(gan_seeds, dict):
        gan_seeds = {"trv": num_range(gan_seeds), "dbp": num_range(gan_seeds)}

    if "trv" in gan_seeds:
        print(f"\tSeeds for trv: {gan_seeds['trv']}")
        generate_images(trv_ns_path, num_range(gan_seeds["trv"]), truncation_psi, outdir, class_idx, dlatents_npz, "trv_")

    if "dbp" in gan_seeds:
        print(f"\tSeeds for dbp: {gan_seeds['dbp']}")
        generate_images(dbp_ns_path, num_range(gan_seeds["dbp"]), truncation_psi, outdir, class_idx, dlatents_npz, "dbp_")

    if baseline_path:
        print("copying baseline images")
        size = copy_images(baseline_path / "images", pathlib.Path(outdir))
        print(f"copying baseline images done. {size} files copied")


def copy_images(src_path, dst_path):
    files = [img.name for img in get_files(src_path)]
    for img in files:
        shutil.copy(src_path / img, dst_path / img)

    return len(files)


def get_files(path, ext=".png"):
    path = pathlib.Path(path)
    files = [f for f in path.ls() if f.name.endswith(ext)]
    return files


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
