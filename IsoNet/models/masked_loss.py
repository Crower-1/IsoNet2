import torch
from torch import fft
import torch.nn as nn
import torch.nn.functional as F


_SHELL_INDEX_CACHE = {}


def tukey_window_3d(tensor, alpha=0.15):
    """Return a broadcastable 3-D Tukey window for a BCHWD tensor."""
    if alpha <= 0:
        return torch.ones((1, 1, *tensor.shape[-3:]), device=tensor.device, dtype=tensor.dtype)

    def tukey_1d(size):
        if size <= 1:
            return torch.ones(size, device=tensor.device, dtype=tensor.dtype)
        positions = torch.arange(size, device=tensor.device, dtype=tensor.dtype)
        if alpha >= 1:
            return 0.5 * (1.0 - torch.cos(2.0 * torch.pi * positions / (size - 1)))
        edge = alpha * (size - 1) / 2.0
        window = torch.ones_like(positions)
        left = positions < edge
        right = positions > (size - 1 - edge)
        window[left] = 0.5 * (
            1.0 + torch.cos(torch.pi * (2.0 * positions[left] / (alpha * (size - 1)) - 1.0))
        )
        distance_from_right = size - 1 - positions[right]
        window[right] = 0.5 * (
            1.0 + torch.cos(torch.pi * (2.0 * distance_from_right / (alpha * (size - 1)) - 1.0))
        )
        return window

    z_window = tukey_1d(tensor.shape[-3])[:, None, None]
    y_window = tukey_1d(tensor.shape[-2])[None, :, None]
    x_window = tukey_1d(tensor.shape[-1])[None, None, :]
    return (z_window * y_window * x_window)[None, None]


def _shell_indices(shape, device):
    key = (tuple(shape), str(device))
    if key not in _SHELL_INDEX_CACHE:
        z = torch.arange(shape[0], device=device) - shape[0] // 2
        y = torch.arange(shape[1], device=device) - shape[1] // 2
        x = torch.arange(shape[2], device=device) - shape[2] // 2
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        _SHELL_INDEX_CACHE[key] = torch.sqrt(
            zz.float().square() + yy.float().square() + xx.float().square()
        ).floor().long().reshape(-1)
    return _SHELL_INDEX_CACHE[key]


def _prepare_fourier_terms(prediction, target, mask, window_alpha):
    prediction = prediction.to(torch.float32)
    target = target.to(torch.float32)
    if window_alpha > 0:
        window = tukey_window_3d(prediction, alpha=window_alpha)
        prediction = prediction * window
        target = target * window
    prediction_f = fft_3d(prediction)
    target_f = fft_3d(target)
    mask = mask.to(device=prediction.device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask[None, None]
    elif mask.ndim == 4:
        mask = mask[:, None]
    mask = torch.broadcast_to(mask, prediction_f.shape)
    return prediction_f, target_f, mask


def normalized_complex_fourier_loss(
    prediction,
    target,
    mask,
    eps=1e-8,
    shell_balanced=False,
    min_shell=3,
    max_nyquist=0.8,
    window_alpha=0.15,
):
    """Phase-sensitive Fourier MSE normalized by supervised coefficients.

    With ``shell_balanced=True`` every valid radial shell contributes equally,
    preventing the much stronger low-frequency power from dominating missing-
    wedge recovery.
    """
    prediction_f, target_f, mask = _prepare_fourier_terms(
        prediction, target, mask, window_alpha
    )
    error = (prediction_f - target_f).abs().square()
    if not shell_balanced:
        numerator = (error * mask).sum(dim=(-3, -2, -1))
        denominator = mask.sum(dim=(-3, -2, -1)).clamp_min(eps)
        return (numerator / denominator).mean()

    shape = prediction.shape[-3:]
    shell_index = _shell_indices(shape, prediction.device)
    max_shell = max(int(min(shape) * 0.5 * float(max_nyquist)), int(min_shell))
    valid_points = shell_index <= max_shell
    shell_index = shell_index[valid_points]

    error = error.reshape(-1, error.shape[-3] * error.shape[-2] * error.shape[-1])[:, valid_points]
    mask = mask.reshape(-1, mask.shape[-3] * mask.shape[-2] * mask.shape[-1])[:, valid_points]
    shell_index = shell_index[None].expand(error.shape[0], -1)
    shell_error = torch.zeros(
        (error.shape[0], max_shell + 1), device=prediction.device, dtype=torch.float32
    )
    shell_weight = torch.zeros_like(shell_error)
    shell_error.scatter_add_(1, shell_index, error * mask)
    shell_weight.scatter_add_(1, shell_index, mask)
    shell_mean = shell_error / shell_weight.clamp_min(eps)
    valid_shells = shell_weight > eps
    valid_shells[:, :min_shell] = False
    per_sample = (shell_mean * valid_shells).sum(dim=1) / valid_shells.sum(dim=1).clamp_min(1)
    return per_sample.mean()


def masked_fourier_shell_correlation_loss(
    prediction,
    target,
    mask,
    eps=1e-6,
    min_shell=3,
    max_nyquist=0.8,
    window_alpha=0.15,
):
    """One minus mean phase-sensitive shell correlation in ``mask``."""
    prediction_f, target_f, mask = _prepare_fourier_terms(
        prediction, target, mask, window_alpha
    )
    shape = prediction.shape[-3:]
    shell_index = _shell_indices(shape, prediction.device)
    max_shell = max(int(min(shape) * 0.5 * float(max_nyquist)), int(min_shell))
    valid_points = shell_index <= max_shell
    shell_index = shell_index[valid_points]

    prediction_f = prediction_f.reshape(-1, prediction_f[0, 0].numel())[:, valid_points]
    target_f = target_f.reshape(-1, target_f[0, 0].numel())[:, valid_points]
    mask = mask.reshape(-1, mask[0, 0].numel())[:, valid_points]
    shell_index = shell_index[None].expand(prediction_f.shape[0], -1)
    shape_out = (prediction_f.shape[0], max_shell + 1)
    cross = torch.zeros(shape_out, device=prediction.device, dtype=torch.float32)
    prediction_power = torch.zeros_like(cross)
    target_power = torch.zeros_like(cross)
    shell_weight = torch.zeros_like(cross)
    cross.scatter_add_(1, shell_index, (prediction_f * target_f.conj()).real * mask)
    prediction_power.scatter_add_(1, shell_index, prediction_f.abs().square() * mask)
    target_power.scatter_add_(1, shell_index, target_f.abs().square() * mask)
    shell_weight.scatter_add_(1, shell_index, mask)
    correlation = cross / torch.sqrt(
        (prediction_power + eps) * (target_power + eps)
    )
    correlation = correlation.clamp(-1.0, 1.0)
    valid_shells = shell_weight > eps
    valid_shells[:, :min_shell] = False
    mean_correlation = (correlation * valid_shells).sum(dim=1) / valid_shells.sum(dim=1).clamp_min(1)
    return (1.0 - mean_correlation).mean()


def masked_fourier_shell_power_loss(
    prediction,
    target,
    mask,
    eps=1e-8,
    relative_floor=1e-6,
    huber_beta=0.5,
    min_shell=3,
    max_nyquist=0.8,
    window_alpha=0.15,
    return_amplitude_ratio=False,
):
    """Match mean Fourier power per supervised radial shell.

    Complex MSE can reduce its error by shrinking predictions whose phase is
    uncertain, while shell correlation is invariant to a positive global
    rescaling.  This loss closes that gap by applying a Huber penalty to the
    log power ratio in each valid shell.  ``mask`` may be a soft confidence
    mask; shells without supervised coefficients are ignored.

    When requested, the second return value is the geometric-mean shell
    amplitude ratio.  It is a diagnostic metric: 1 means matched power, values
    below 1 indicate amplitude collapse.
    """
    if huber_beta <= 0:
        raise ValueError("huber_beta must be positive")
    if relative_floor < 0:
        raise ValueError("relative_floor must be non-negative")

    prediction_f, target_f, mask = _prepare_fourier_terms(
        prediction, target, mask, window_alpha
    )
    shape = prediction.shape[-3:]
    shell_index = _shell_indices(shape, prediction.device)
    max_shell = max(int(min(shape) * 0.5 * float(max_nyquist)), int(min_shell))
    valid_points = shell_index <= max_shell
    shell_index = shell_index[valid_points]

    prediction_f = prediction_f.reshape(-1, prediction_f[0, 0].numel())[:, valid_points]
    target_f = target_f.reshape(-1, target_f[0, 0].numel())[:, valid_points]
    mask = mask.reshape(-1, mask[0, 0].numel())[:, valid_points]
    shell_index = shell_index[None].expand(prediction_f.shape[0], -1)
    shape_out = (prediction_f.shape[0], max_shell + 1)

    prediction_power = torch.zeros(
        shape_out, device=prediction.device, dtype=torch.float32
    )
    target_power = torch.zeros_like(prediction_power)
    shell_weight = torch.zeros_like(prediction_power)
    prediction_power.scatter_add_(
        1, shell_index, prediction_f.abs().square() * mask
    )
    target_power.scatter_add_(1, shell_index, target_f.abs().square() * mask)
    shell_weight.scatter_add_(1, shell_index, mask)

    prediction_power = prediction_power / shell_weight.clamp_min(eps)
    target_power = target_power / shell_weight.clamp_min(eps)
    valid_shells = shell_weight > eps
    valid_shells[:, :min_shell] = False
    valid_shell_count = valid_shells.sum(dim=1)

    target_scale = (target_power * valid_shells).sum(dim=1, keepdim=True)
    target_scale = target_scale / valid_shell_count[:, None].clamp_min(1)
    power_floor = (target_scale.detach() * relative_floor).clamp_min(eps)
    log_power_ratio = torch.log(
        (prediction_power + power_floor) / (target_power + power_floor)
    )

    shell_loss = F.smooth_l1_loss(
        log_power_ratio,
        torch.zeros_like(log_power_ratio),
        reduction="none",
        beta=huber_beta,
    )
    per_sample_loss = (shell_loss * valid_shells).sum(dim=1)
    per_sample_loss = per_sample_loss / valid_shell_count.clamp_min(1)
    loss = per_sample_loss.mean()

    if not return_amplitude_ratio:
        return loss

    mean_log_amplitude_ratio = 0.5 * (
        log_power_ratio * valid_shells
    ).sum(dim=1) / valid_shell_count.clamp_min(1)
    amplitude_ratio = torch.exp(mean_log_amplitude_ratio)
    amplitude_ratio = torch.where(
        valid_shell_count > 0,
        amplitude_ratio,
        torch.zeros_like(amplitude_ratio),
    ).mean()
    return loss, amplitude_ratio


class FSCLoss(nn.Module):
    def __init__(self, eps=1e-6, min_shell=1):
        super().__init__()
        self.eps = eps
        self.min_shell = min_shell
        self._shell_cache = {}

    def _get_shell_index(self, shape, device):
        key = (tuple(shape), str(device))
        if key not in self._shell_cache:
            z = torch.arange(shape[0], device=device) - shape[0] // 2
            y = torch.arange(shape[1], device=device) - shape[1] // 2
            x = torch.arange(shape[2], device=device) - shape[2] // 2
            Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
            shell_index = torch.sqrt(Z.float() ** 2 + Y.float() ** 2 + X.float() ** 2).long()
            self._shell_cache[key] = shell_index.reshape(-1)
        return self._shell_cache[key]

    def forward(self, model_output, target):
        model_output = model_output.to(torch.float32)
        target = target.to(torch.float32)

        output_ft = fft_3d(model_output)
        target_ft = fft_3d(target)

        shell_index = self._get_shell_index(model_output.shape[-3:], model_output.device)
        max_shell = min(model_output.shape[-3:]) // 2
        valid_points = shell_index <= max_shell
        shell_index = shell_index[valid_points]

        output_ft = output_ft.reshape(output_ft.shape[0] * output_ft.shape[1], -1)
        target_ft = target_ft.reshape(target_ft.shape[0] * target_ft.shape[1], -1)

        losses = []
        for i in range(output_ft.shape[0]):
            output_shell = output_ft[i][valid_points]
            target_shell = target_ft[i][valid_points]

            numerator = torch.zeros(max_shell + 1, device=model_output.device, dtype=torch.float32)
            output_power = torch.zeros_like(numerator)
            target_power = torch.zeros_like(numerator)
            counts = torch.zeros_like(numerator)

            numerator.scatter_add_(0, shell_index, (output_shell * target_shell.conj()).real)
            output_power.scatter_add_(0, shell_index, output_shell.abs().pow(2))
            target_power.scatter_add_(0, shell_index, target_shell.abs().pow(2))
            counts.scatter_add_(0, shell_index, torch.ones_like(output_shell.real))

            fsc_curve = numerator / torch.sqrt(output_power * target_power + self.eps)
            fsc_curve = torch.clamp(fsc_curve, -1.0, 1.0)

            valid_shells = counts > 0
            valid_shells[:self.min_shell] = False
            if valid_shells.any():
                losses.append(1.0 - fsc_curve[valid_shells].mean())
            else:
                losses.append(torch.tensor(1.0, device=model_output.device, dtype=torch.float32))

        return torch.stack(losses).mean()

def ssim_loss(x, y, window_size=11, size_average=True):
    # Gaussian kernel for SSIM computation
    def gaussian_window(window_size, sigma):
        #gauss = torch.Tensor([torch.exp(-(z - window_size // 2) ** 2 / (2 * sigma ** 2)) for z in range(window_size)])
        z = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-(z - window_size // 2) ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        return gauss / gauss.sum()

    # Create 3D Gaussian window
    channels = x.size(1)
    window = gaussian_window(window_size, 1.5).unsqueeze(1).repeat(1, channels, 1, 1, 1).to(x.device)
    mu_x = F.conv3d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv3d(y, window, padding=window_size // 2, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv3d(x * x, window, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y = F.conv3d(y * y, window, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv3d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim_map.mean() if size_average else 1 - ssim_map

def simple_loss(model_output, target, rot_mw_mask,loss_func='L2'):
    filtered_model_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=rot_mw_mask, output="real")
    filtered_target = apply_fourier_mask_to_tomo(tomo=target, mask=rot_mw_mask, output="real")
    if loss_func == "L2":
        loss = nn.MSELoss()
        return loss(filtered_model_output, filtered_target)
    elif loss_func == "smoothL1":
        loss = nn.SmoothL1Loss()
        return loss(filtered_model_output, filtered_target)
    elif loss_func == "smoothL1-SSIM":
        loss = nn.SmoothL1Loss()
        return loss(filtered_model_output, filtered_target) + ssim_loss(filtered_model_output, target)
    elif loss_func == "FSC":
        return FSCLoss()(filtered_model_output, filtered_target)
    else:
        print("loss name is not correct")

def masked_loss2(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0, loss_func=None):
    """
    The self-supervised per-sample loss function for denoising and missing wedge reconstruction.
    """
    outside_mw_mask = rot_mw_mask * mw_mask
    outside_mw_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=outside_mw_mask, output="real")
    outside_mw_target = apply_fourier_mask_to_tomo(tomo=target, mask=outside_mw_mask, output="real")

    inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    inside_mw_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=inside_mw_mask, output="real")
    inside_mw_target = apply_fourier_mask_to_tomo(tomo=target, mask=inside_mw_mask, output="real")

    if loss_func is None:
        loss_func = nn.MSELoss()

    outside_mw_loss = loss_func(outside_mw_output, outside_mw_target)
    inside_mw_loss = loss_func(inside_mw_output, inside_mw_target)
    #loss = outside_mw_loss + mw_weight * inside_mw_loss
    #return loss
    return outside_mw_loss,inside_mw_loss

def masked_loss(model_output, target, rot_mw_mask, mw_mask, loss_func=None):

    inside_mask = rot_mw_mask * mw_mask
    inside_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=inside_mask, output="real")
    inside_target = apply_fourier_mask_to_tomo(tomo=target, mask=inside_mask, output="real")

    outside_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    outside_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=outside_mask, output="real")
    outside_target = apply_fourier_mask_to_tomo(tomo=target, mask=outside_mask, output="real")

    if loss_func == None:
        print("loss name is not correct")
    else:
        return [loss_func(outside_output, outside_target), loss_func(inside_output, inside_target)]

# def masked_loss(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0, loss_func='L2'):
#     # This is essence of deepdewedge
#     # inside_mw_loss is for IsoNet
#     # outside_mw_loss is for noise2noise
#     outside_mw_mask = rot_mw_mask * mw_mask
#     outside_mw_loss = (
#         apply_fourier_mask_to_tomo(
#             tomo=target - model_output, mask=outside_mw_mask, output="real"
#         )
#         .abs()
#         .pow(2)
#         .mean()
#     )
#     inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
#     inside_mw_loss = (
#         apply_fourier_mask_to_tomo(
#             tomo=target - model_output, mask=inside_mw_mask, output="real"
#         )
#         .abs()
#         .pow(2)
#         .mean()
#     )
#     # loss = outside_mw_loss + mw_weight * inside_mw_loss
#     return outside_mw_loss, inside_mw_loss

def fft_3d(tomo, norm="ortho"):
    fft_dim = (-1, -2, -3)
    return fft.fftshift(fft.fftn(tomo, dim=fft_dim, norm=norm), dim=fft_dim)


def ifft_3d(tomo, norm="ortho"):
    fft_dim = (-1, -2, -3)
    return fft.ifftn(fft.ifftshift(tomo, dim=fft_dim), dim=fft_dim, norm=norm)


def apply_fourier_mask_to_tomo(tomo, mask, output="real"):
    tomo_ft = fft_3d(tomo)
    tomo_ft_masked = tomo_ft * mask
    vol_filt = ifft_3d(tomo_ft_masked)
    if output == "real":
        return vol_filt.real
    elif output == "complex":
        return vol_filt
