import random

import numpy as np
import torch

from IsoNet.models.masked_loss import (
    fft_3d,
    ifft_3d,
    masked_fourier_shell_correlation_loss,
    masked_fourier_shell_power_loss,
    normalized_complex_fourier_loss,
)
from IsoNet.models.train import prepare_isonet2_batch
from IsoNet.utils.missing_wedge import mw3D
from IsoNet.utils.processing import fourier_data_consistency, robust_mean_std
from IsoNet.utils.rotations import cube_rotations_24, rotate_cube_24


def test_cube_rotations_are_24_unique_exact_support_transforms():
    assert len(cube_rotations_24) == 24
    assert len(set(cube_rotations_24)) == 24

    size = 16
    mask = torch.from_numpy(
        mw3D(size, missingAngle=[30, 30], spherical=False)
    )[None, None]
    spectrum = torch.randn(1, 1, size, size, size, dtype=torch.complex64) * mask
    measured_volume = ifft_3d(spectrum).real

    for rotation in cube_rotations_24:
        rotated_volume = rotate_cube_24(measured_volume, rotation, fourier=False)
        rotated_mask = rotate_cube_24(mask, rotation, fourier=True)
        rotated_spectrum = fft_3d(rotated_volume)
        outside_fraction = (
            rotated_spectrum.abs().square() * (1.0 - rotated_mask)
        ).sum() / rotated_spectrum.abs().square().sum().clamp_min(1e-8)
        assert outside_fraction < 1e-6


def test_soft_missing_wedge_supports_axis_angle_and_rectangular_shapes():
    mask = mw3D(
        (16, 18, 20),
        missingAngle=[30, 40],
        spherical=False,
        tilt_axis_angle=17.0,
        taper_deg=4.0,
    )
    assert mask.shape == (16, 18, 20)
    assert mask.min() == 0.0
    assert mask.max() == 1.0
    assert np.any((mask > 0.0) & (mask < 1.0))
    assert np.all(mask[16 // 2] == 1.0)


def test_single_map_holdout_has_identity_baseline_and_restore_gradient():
    random.seed(7)
    torch.manual_seed(7)
    size = 20
    source = torch.randn(2, 1, size, size, size)
    measured_mask = torch.from_numpy(
        mw3D(size, missingAngle=[30, 30], taper_deg=3.0)
    )[None, None].expand(2, -1, -1, -1, -1)
    (
        net_input,
        target,
        valid_mask,
        _,
        restore_mask,
        visible_mask,
        coverage,
    ) = prepare_isonet2_batch(
        source,
        measured_mask,
        random_rot_weight=0.0,
        min_restore_coverage=0.10,
    )

    assert coverage >= 0.10
    torch.testing.assert_close(restore_mask + visible_mask, valid_mask)
    restore_identity = normalized_complex_fourier_loss(
        net_input, target, restore_mask, shell_balanced=True, window_alpha=0.0
    )
    visible_identity = normalized_complex_fourier_loss(
        net_input, target, visible_mask, shell_balanced=True, window_alpha=0.0
    )
    assert restore_identity > 0.05
    assert visible_identity < restore_identity * 0.02

    prediction = torch.nn.Parameter(torch.zeros_like(target))
    loss = normalized_complex_fourier_loss(
        prediction, target, restore_mask, shell_balanced=True, window_alpha=0.0
    ) + 0.1 * masked_fourier_shell_correlation_loss(
        prediction, target, restore_mask, window_alpha=0.0
    )
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_complex_fourier_loss_is_phase_sensitive():
    target = torch.randn(1, 1, 12, 12, 12)
    mask = torch.ones_like(target)
    matching = normalized_complex_fourier_loss(
        target, target, mask, min_shell=0, max_nyquist=1.0, window_alpha=0.0
    )
    phase_reversed = normalized_complex_fourier_loss(
        -target, target, mask, min_shell=0, max_nyquist=1.0, window_alpha=0.0
    )
    assert matching == 0
    assert phase_reversed > 0


def test_shell_power_loss_penalizes_amplitude_collapse():
    torch.manual_seed(11)
    target = torch.randn(2, 1, 16, 16, 16)
    mask = torch.ones_like(target)

    matching, matching_ratio = masked_fourier_shell_power_loss(
        target,
        target,
        mask,
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
        return_amplitude_ratio=True,
    )
    half, half_ratio = masked_fourier_shell_power_loss(
        0.5 * target,
        target,
        mask,
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
        return_amplitude_ratio=True,
    )
    collapsed = masked_fourier_shell_power_loss(
        0.2 * target,
        target,
        mask,
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
    )

    torch.testing.assert_close(matching, torch.zeros_like(matching))
    torch.testing.assert_close(matching_ratio, torch.ones_like(matching_ratio))
    torch.testing.assert_close(half_ratio, torch.full_like(half_ratio, 0.5))
    assert matching < half < collapsed

    matching_corr = masked_fourier_shell_correlation_loss(
        target, target, mask, min_shell=1, max_nyquist=1.0, window_alpha=0.0
    )
    collapsed_corr = masked_fourier_shell_correlation_loss(
        0.2 * target,
        target,
        mask,
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
    )
    torch.testing.assert_close(collapsed_corr, matching_corr, atol=1e-6, rtol=1e-6)


def test_shell_power_loss_is_scale_stable_and_has_finite_gradients():
    torch.manual_seed(13)
    target = torch.randn(1, 1, 14, 14, 14)
    soft_mask = torch.rand_like(target)
    prediction = torch.nn.Parameter(0.3 * target.clone())

    loss = masked_fourier_shell_power_loss(
        prediction,
        target,
        soft_mask,
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
    )
    scaled_loss = masked_fourier_shell_power_loss(
        7.0 * prediction,
        7.0 * target,
        soft_mask,
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
    )
    torch.testing.assert_close(loss, scaled_loss, atol=1e-6, rtol=1e-6)

    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0

    empty_loss, empty_ratio = masked_fourier_shell_power_loss(
        prediction.detach(),
        target,
        torch.zeros_like(target),
        min_shell=1,
        max_nyquist=1.0,
        window_alpha=0.0,
        return_amplitude_ratio=True,
    )
    torch.testing.assert_close(empty_loss, torch.zeros_like(empty_loss))
    torch.testing.assert_close(empty_ratio, torch.zeros_like(empty_ratio))


def test_full_volume_data_consistency_preserves_measured_coefficients():
    rng = np.random.default_rng(3)
    shape = (12, 14, 16)
    input_volume = rng.normal(size=shape).astype(np.float32)
    prediction = rng.normal(size=shape).astype(np.float32)
    measured_mask = mw3D(
        shape, missingAngle=[30, 30], spherical=False, taper_deg=0.0
    )
    result = fourier_data_consistency(input_volume, prediction, measured_mask)
    result_f = np.fft.fftshift(np.fft.fftn(result))
    input_f = np.fft.fftshift(np.fft.fftn(input_volume))
    prediction_f = np.fft.fftshift(np.fft.fftn(prediction))
    np.testing.assert_allclose(
        result_f[measured_mask == 1], input_f[measured_mask == 1], atol=5e-5
    )
    np.testing.assert_allclose(
        result_f[measured_mask == 0], prediction_f[measured_mask == 0], atol=5e-5
    )


def test_robust_statistics_ignore_extreme_outlier_and_respect_mask():
    volume = np.empty((8, 8, 8), dtype=np.float32)
    volume[:, :, :4] = np.linspace(0.8, 1.2, 8 * 8 * 4).reshape(8, 8, 4)
    volume[:, :, 4:] = 3.0
    volume[0, 0, 0] = 1e9
    mask = np.zeros_like(volume)
    mask[:, :, :4] = 1
    mean, std = robust_mean_std(volume, mask=mask)
    assert abs(mean - 1.0) < 0.005
    assert 0.05 < std < 0.2
