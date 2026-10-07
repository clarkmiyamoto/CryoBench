"""
PyTorch dataset of paired (ground-truth volume, particle image) samples from CryoBench IgG-1D

Each sample i is (x_i, y_i), where y_i is the i-th particle image and x_i is the ground-truth
density map of the conformation that image was simulated from (100 conformations, 1000 images
each). Volumes and images are downsampled together by Fourier cropping from the released
128 px / 3.0 A/px, so the physical box (384 A) is unchanged and the pixel size becomes
384 / resolution A/px.

Up to a global intensity scale, the pair follows the CryoBench image formation model
    y = -CTF * shift_t( project_R(x) ) + noise
where R, t, and the CTF parameters are returned with `return_meta=True`. The CTF is negative at
low frequency, so the released images (bright particles, the usual convention for extracted
particle stacks) are sign-flipped relative to CTF * projection; `invert=True` multiplies them by
-1 (dark particles) to match it, as cryoDRGN does by default.

With `volume_frame="image"`, x_i is instead the ground-truth volume rotated by the image's pose
R_i and shifted by t_i, so that summing it along z gives the clean projection underlying y_i:
    y = -CTF * x.sum(0) + noise
Every image then has its own randomly oriented target. The rotation is applied to the released
128 px volume (trilinear) before Fourier cropping to `resolution`. `phase_flip=True` removes the
CTF's sign changes from y (multiplies its Fourier transform by sign(-CTF)), which makes the image
look more like that projection.

Example usage
-------------
    from torch.utils.data import DataLoader
    from cryobench_data.igg1d import IgG1DDataset

    ds = IgG1DDataset("/mnt/ceph/users/cmiyamoto/IgG-1D", resolution=64)
    x, y = ds[0]  # x: (64, 64, 64) volume, y: (64, 64) image
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
"""
import os
import pickle
import struct

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MRC_HEADER_BYTES = 1024
ORIG_D = 128  # box size (px) of the released images and GT volumes
ORIG_APIX = 3.0  # pixel size (A/px) of the released images and GT volumes


def rotate_volume(vol, rot, shift=None):
    """Rotate volume(s) about the voxel at index D/2 the way CryoBench's projector does.

    Returns out(p) = vol(R^T (p - s)), where p = (x, y, z) voxel coordinates relative to D/2, so that
    `out.sum(-3)` is the projection of `vol` at pose R shifted in-plane by s.

    Args:
        vol: (..., D, D, D) volume(s) in (z, y, x) order; all are rotated the same way
        rot: (3, 3) rotation matrix (cryoDRGN convention)
        shift: optional (2,) in-plane (x, y) shift in pixels; with cryoDRGN's `trans` (fraction of
            the box, the shift that re-centers the image) this is `-trans * D`
    """
    D = vol.shape[-1]
    lead = vol.shape[:-3]
    idx = torch.arange(D, dtype=torch.float32, device=vol.device) - D // 2
    z, y, x = torch.meshgrid(idx, idx, idx, indexing="ij")
    p = torch.stack([x, y, z], dim=-1)
    if shift is not None:
        s = torch.zeros(3, dtype=torch.float32, device=vol.device)
        s[:2] = torch.as_tensor(shift, dtype=torch.float32, device=vol.device)
        p = p - s
    q = p @ torch.as_tensor(rot, dtype=torch.float32, device=vol.device)  # row vectors: R^T p
    grid = (q + D // 2) * (2 / (D - 1)) - 1
    v = vol.reshape(-1, 1, D, D, D).float()
    grid = grid[None].expand(len(v), -1, -1, -1, -1)
    out = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(*lead, D, D, D)


def compute_ctf(D, apix, dfu, dfv, dfang, volt, cs, w, phase_shift=0.0):
    """cryoDRGN's CTF on a D x D grid in FFT (unshifted) order; dfang, phase_shift in degrees."""
    f = torch.fft.fftfreq(D, d=apix, dtype=torch.float64)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    volt = volt * 1000
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2) ** 0.5
    ang = torch.atan2(fy, fx)
    s2 = fx**2 + fy**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - np.deg2rad(dfang))))
    gamma = 2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * 1e7 * lam**3 * s2**2)
    gamma = gamma - np.deg2rad(phase_shift)
    return ((1 - w**2) ** 0.5 * torch.sin(gamma) - w * torch.cos(gamma)).float()


def read_mrc(path, mmap=False):
    """Read a float32 MRC/MRCS file as a (nz, ny, nx) array, memory-mapped if requested."""
    with open(path, "rb") as f:
        header = f.read(MRC_HEADER_BYTES)
    nx, ny, nz, mode = struct.unpack("<4i", header[:16])
    nsymbt = struct.unpack("<i", header[92:96])[0]
    assert mode == 2, f"{path}: expected float32 data (MRC mode 2), got mode {mode}"
    offset = MRC_HEADER_BYTES + nsymbt
    if mmap:
        return np.memmap(path, np.float32, "r", offset=offset, shape=(nz, ny, nx))
    return np.fromfile(path, np.float32, offset=offset).reshape(nz, ny, nx)


def read_mrc_count(path):
    """Number of images (nz) in an MRC stack, from its header."""
    with open(path, "rb") as f:
        return struct.unpack("<i", f.read(12)[8:12])[0]


def fourier_downsample(x, size, ndim):
    """Downsample the last `ndim` (square, even-sized) dims of `x` to `size` by Fourier cropping.

    Mean intensity is preserved, i.e. the output samples the band-limited input on a coarser
    grid; index 0 stays fixed, so the box center (index D/2) maps to size/2.
    """
    n = x.shape[-1]
    assert size <= n and size % 2 == 0, f"cannot downsample {n} -> {size}"
    if size == n:
        return x
    dims = tuple(range(-ndim, 0))
    ft = torch.fft.fftshift(torch.fft.fftn(x, dim=dims), dim=dims)
    start = n // 2 - size // 2
    ft = ft[(...,) + (slice(start, start + size),) * ndim]
    out = torch.fft.ifftn(torch.fft.ifftshift(ft, dim=dims), dim=dims).real
    return out * (size / n) ** ndim


class IgG1DDataset(Dataset):
    """Paired (volume, image) samples from CryoBench IgG-1D.

    Args:
        root: path to the unzipped `IgG-1D/` directory
        resolution: box size D of the returned volumes (D^3) and images (D^2); even, <= 128
        snr: noise level subdirectory to read, i.e. `images/snr{snr}/`
        return_meta: also return a dict with the image's conformation index, pose, and CTF
        invert: multiply images by -1 (dark particles), matching a +CTF forward model
        volume_frame: "canonical" returns each conformation's volume as released; "image" returns
            it rotated and shifted by the image's pose (see module docstring)
        phase_flip: multiply each image's Fourier transform by sign(-CTF) (before `invert`)

    Returns (x, y) or (x, y, meta), where x is a (D, D, D) float32 volume, y a (D, D) float32
    image, and meta has keys
        conf  (int)       conformation index; x is `volumes[conf]` (rotated, for the image frame),
                          at angle 3.6 * conf deg
        rot   (3, 3)      rotation matrix (cryoDRGN convention)
        trans (2,)        in-plane shift as a fraction of the box (cryoDRGN convention)
        ctf   (9,)        cryoDRGN CTF params [D, Apix, dfU, dfV, dfang, kV, Cs, w, phase_shift],
                          with D and Apix set to this dataset's resolution
    """

    def __init__(self, root, resolution=64, snr=0.01, return_meta=False, invert=False,
                 volume_frame="canonical", phase_flip=False):
        assert resolution % 2 == 0 and resolution <= ORIG_D
        assert volume_frame in ("canonical", "image"), volume_frame
        self.root = root
        self.resolution = resolution
        self.apix = ORIG_APIX * ORIG_D / resolution
        self.return_meta = return_meta
        self.invert = invert
        self.volume_frame = volume_frame
        self.phase_flip = phase_flip

        # Per-image labels, poses, and CTFs, all in the same (sorted) order as the images
        self.labels = torch.from_numpy(self._load_pkl("gt_latents.pkl")).long()
        rots, trans = self._load_pkl("combined_poses.pkl")
        self.rotations = torch.from_numpy(np.asarray(rots, dtype=np.float32))
        self.translations = torch.from_numpy(np.asarray(trans, dtype=np.float32))
        ctf = np.array(self._load_pkl("combined_ctfs.pkl"), dtype=np.float32)
        ctf[:, 0], ctf[:, 1] = resolution, self.apix  # pkl stores the 256 px / 1.5 A/px sim grid
        self.ctf = torch.from_numpy(ctf)

        # Image stacks: 100 files of 1000 images; memory-mapped lazily in __getitem__
        img_dir = os.path.join(root, "images", f"snr{snr}")
        with open(os.path.join(img_dir, f"sorted_particles.{ORIG_D}.txt")) as f:
            names = [line.strip() for line in f if line.strip()]
        self.stack_paths = [os.path.join(img_dir, name) for name in names]
        counts = [read_mrc_count(p) for p in self.stack_paths]
        self.stack_offsets = np.concatenate([[0], np.cumsum(counts)])
        self._stacks = {}
        n = int(self.stack_offsets[-1])
        assert n == len(self.labels) == len(self.rotations) == len(self.ctf), (
            f"{n} images but {len(self.labels)} labels, {len(self.rotations)} poses, "
            f"{len(self.ctf)} CTFs"
        )

        # Ground-truth volumes, one per conformation, downsampled once up front. The image frame
        # also keeps the released 128 px volumes, which are rotated per image before downsampling.
        n_conf = int(self.labels.max()) + 1
        vols, orig_vols = [], []
        for c in range(n_conf):
            vol = read_mrc(os.path.join(root, "vols", f"{ORIG_D}_org", f"{c:03d}.mrc"))
            assert vol.shape == (ORIG_D,) * 3, f"volume {c:03d} has shape {vol.shape}"
            vol = torch.from_numpy(vol)
            vols.append(fourier_downsample(vol, resolution, ndim=3))
            if volume_frame == "image":
                orig_vols.append(vol)
        self.volumes = torch.stack(vols).float()  # (n_conf, D, D, D)
        self.orig_volumes = torch.stack(orig_vols) if orig_vols else None  # (n_conf, 128, 128, 128)

    def _load_pkl(self, name):
        with open(os.path.join(self.root, name), "rb") as f:
            return pickle.load(f)

    def _image(self, i):
        k = int(np.searchsorted(self.stack_offsets, i, side="right")) - 1
        if k not in self._stacks:
            self._stacks[k] = read_mrc(self.stack_paths[k], mmap=True)
        return torch.from_numpy(np.array(self._stacks[k][i - self.stack_offsets[k]]))

    def __len__(self):
        return len(self.labels)

    def image_shift(self, i):
        """In-plane (x, y) shift of image i in pixels at this resolution (see `rotate_volume`)."""
        return -self.translations[i] * self.resolution

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        conf = int(self.labels[i])
        if self.volume_frame == "image":
            vol = rotate_volume(self.orig_volumes[conf], self.rotations[i], -self.translations[i] * ORIG_D)
            x = fourier_downsample(vol, self.resolution, ndim=3)
        else:
            x = self.volumes[conf]
        y = fourier_downsample(self._image(i), self.resolution, ndim=2)
        if self.phase_flip:
            ctf = compute_ctf(self.resolution, *self.ctf[i, 1:].tolist())
            y = torch.fft.ifft2(torch.fft.fft2(y) * torch.sign(-ctf)).real
        if self.invert:
            y = -y
        if not self.return_meta:
            return x, y
        meta = dict(
            conf=conf, rot=self.rotations[i], trans=self.translations[i], ctf=self.ctf[i]
        )
        return x, y, meta

    def __getstate__(self):
        # Don't ship open memmaps to DataLoader workers; each worker reopens its own
        state = self.__dict__.copy()
        state["_stacks"] = {}
        return state

