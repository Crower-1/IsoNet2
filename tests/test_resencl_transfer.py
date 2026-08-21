from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from IsoNet.models.nnssl_encoder_import import (
    EXPECTED_ENCODER_PARAMETERS,
    _canonical_alias_key,
    extract_encoder_state_dict,
    load_encoder_pretrained,
)
from IsoNet.models.resencl import ResEncLEncoder3D, ResEncLRestorationNet
from IsoNet.models.train import apply_F_filter_torch, build_optimizer, build_scheduler
from IsoNet.utils.missing_wedge import mw3D


NNSSL_CHECKPOINT = Path(
    os.environ.get(
        "NNSSL_RESENCL_CHECKPOINT",
        "/share/data/CryoET_Data/liushuo/dataset/nnssl_dataset/nnssl_results/"
        "Dataset001_2015/EffSparkMAETrainer_BS8_1000ep_MW__nnsslPlans__noresample/"
        "fold_all/checkpoint_latest.pth",
    )
)
NNSSL_SOURCE = Path(os.environ.get("NNSSL_SOURCE", "/home/liushuo/Documents/code/nnssl"))
NNSSL_RAW = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnssl_dataset/nnssl_raw/"
    "Dataset001_2015/images/pp3414-bin4-wbp.rec"
)
NNSSL_PREPROCESSED = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnssl_dataset/nnssl_preprocessed/"
    "Dataset001_2015/nnsslPlans_noresample/Dataset001_2015/wbp/pp3414-bin4/"
    "ses-DEFAULT/pp3414-bin4-wbp.b2nd"
)


def test_encoder_configuration_and_parameter_count():
    encoder = ResEncLEncoder3D()
    assert encoder.channels == (32, 64, 128, 256, 320, 320)
    assert encoder.blocks_per_stage == (1, 3, 4, 6, 6, 6)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == EXPECTED_ENCODER_PARAMETERS
    assert len(encoder.state_dict()) == 224
    assert not any("all_modules" in key for key in encoder.state_dict())


@pytest.mark.skipif(not NNSSL_CHECKPOINT.exists(), reason="nnSSL validation checkpoint unavailable")
def test_real_nnssl_checkpoint_loads_strictly_and_preserves_canonical_tensors():
    encoder = ResEncLEncoder3D()
    report = load_encoder_pretrained(encoder, NNSSL_CHECKPOINT)
    assert report["parameter_count"] == EXPECTED_ENCODER_PARAMETERS
    assert report["stem_loaded"] is True
    assert report["stages_loaded"] == list(range(6))
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["shape_mismatches"] == []
    assert report["source_epoch"] == 222
    assert report["normalization"] == "zscore"
    assert report["pretrain_spacing"] == pytest.approx([17.1428566] * 3)

    checkpoint = torch.load(str(NNSSL_CHECKPOINT), map_location="cpu", mmap=True, weights_only=False)
    converted, _ = extract_encoder_state_dict(checkpoint, encoder.state_dict())
    for key, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, converted[key]), key


@pytest.mark.skipif(
    not (NNSSL_CHECKPOINT.exists() and NNSSL_SOURCE.exists()),
    reason="nnSSL source/checkpoint unavailable",
)
def test_dense_encoder_forward_exactly_matches_nnssl(monkeypatch):
    monkeypatch.syspath_prepend(str(NNSSL_SOURCE / "src"))
    from nnssl.architectures.architecture_registry import get_res_enc_l

    checkpoint = torch.load(str(NNSSL_CHECKPOINT), map_location="cpu", mmap=True, weights_only=False)
    isonet_encoder = ResEncLEncoder3D()
    canonical, _ = extract_encoder_state_dict(checkpoint, isonet_encoder.state_dict())
    isonet_encoder.load_state_dict(canonical, strict=True)

    nnssl_encoder = get_res_enc_l(1, 1).encoder
    nnssl_state = {}
    for key in nnssl_encoder.state_dict():
        alias_target = _canonical_alias_key(key)
        nnssl_state[key] = canonical[alias_target if alias_target is not None else key]
    nnssl_encoder.load_state_dict(nnssl_state, strict=True)

    x = torch.randn(1, 1, 32, 32, 64)
    isonet_encoder.eval()
    nnssl_encoder.eval()
    with torch.inference_mode():
        isonet_features = isonet_encoder(x)
        nnssl_features = nnssl_encoder(x)
    assert len(isonet_features) == len(nnssl_features) == 6
    for ours, source in zip(isonet_features, nnssl_features):
        assert torch.equal(ours, source)


def test_restoration_shape_identity_and_decoder_backward():
    # 32x32x64 is the smallest inexpensive shape with >1 voxel at the deepest
    # InstanceNorm level. Freeze the large encoder to keep the test lightweight.
    model = ResEncLRestorationNet(encoder_input_sign=-1)
    model.set_encoder_trainable("frozen")
    x = torch.randn(1, 1, 32, 32, 64)
    output = model(x)
    assert output.shape == x.shape
    assert torch.equal(output, x)  # zero-initialized residual head

    loss = output.square().mean()
    loss.backward()
    assert model.output_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_input_shape_must_be_divisible_by_32():
    model = ResEncLRestorationNet()
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(1, 1, 32, 48, 64))


def test_encoder_trainability_and_differential_learning_rates():
    model = ResEncLRestorationNet()
    optimizer = build_optimizer(
        model,
        {
            "learning_rate": 3e-4,
            "encoder_trainable": "deep",
            "encoder_learning_rate": 1e-5,
            "decoder_learning_rate": 1e-4,
        },
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["encoder"]["lr"] == pytest.approx(1e-5)
    assert groups["restoration"]["lr"] == pytest.approx(1e-4)
    assert not any(parameter.requires_grad for parameter in model.encoder.stem.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.stages[2].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.stages[3].parameters())

    scheduler = build_scheduler(
        optimizer,
        {"learning_rate": 3e-4, "learning_rate_min": 3e-5, "T_max": 10},
    )
    initial_ratio = groups["restoration"]["lr"] / groups["encoder"]["lr"]
    optimizer.step()
    scheduler.step()
    assert groups["restoration"]["lr"] / groups["encoder"]["lr"] == pytest.approx(initial_ratio)


@pytest.mark.skipif(not NNSSL_SOURCE.exists(), reason="nnSSL source unavailable")
def test_missing_wedge_axis_and_shift_match_nnssl_on_central_tilt_plane(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(NNSSL_SOURCE / "src"))
    from nnssl.ssl_data.data_augmentation.missing_wedge import (
        apply_missing_wedge,
        create_missing_wedge_mask,
    )

    size = 64
    nnssl_unshifted = create_missing_wedge_mask(
        size, size, 60.0, device=torch.device("cpu"), dtype=torch.float32
    )
    nnssl_shifted = torch.fft.fftshift(nnssl_unshifted, dim=(0, 1)).numpy()
    isonet = mw3D(size, missingAngle=[30, 30], spherical=True)
    assert np.array_equal(isonet[:, size // 2, :], nnssl_shifted)
    # Both use Z/X as the wedge plane and Y as tilt axis. IsoNet2 additionally
    # applies a 3-D spherical aperture, while nnSSL repeats its circular Z/X
    # aperture along Y, so the complete 3-D masks intentionally differ.
    nnssl_3d = np.repeat(nnssl_shifted[:, None, :], size, axis=1)
    assert not np.array_equal(isonet, nnssl_3d)

    # A deliberately asymmetric phantom confirms the volume order and FFT shift,
    # not just mask appearance. Once both inputs share IsoNet2's spherical
    # aperture, the two implementations produce the same filtered volume.
    phantom = torch.zeros(1, 1, size, size, size)
    phantom[0, 0, 7, 19, 41] = 1.0
    phantom[0, 0, 43, 11, 13] = -0.35
    phantom[0, 0, 27:30, 48, 5] = torch.tensor([0.2, 0.5, -0.1])

    coords = np.arange(size) - size // 2
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    sphere = ((zz * zz + yy * yy + xx * xx) <= (size // 2) ** 2).astype(np.float32)
    sphere_tensor = torch.from_numpy(sphere)[None, None]
    band_limited = apply_F_filter_torch(phantom, sphere_tensor)
    isonet_filtered = apply_F_filter_torch(
        band_limited, torch.from_numpy(isonet)[None, None]
    )
    nnssl_filtered = apply_missing_wedge(band_limited, 60.0)
    assert torch.allclose(isonet_filtered, nnssl_filtered, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(
    not (NNSSL_RAW.exists() and NNSSL_PREPROCESSED.exists()),
    reason="nnSSL raw/preprocessed validation pair unavailable",
)
def test_real_data_polarity_requires_encoder_input_flip():
    blosc2 = pytest.importorskip("blosc2")
    mrcfile = pytest.importorskip("mrcfile")

    preprocessed = blosc2.open(urlpath=str(NNSSL_PREPROCESSED), mode="r", mmap_mode="r")
    with mrcfile.mmap(NNSSL_RAW, permissive=True) as mrc:
        z, y, x = (size // 2 for size in mrc.data.shape)
        raw = np.asarray(mrc.data[z - 32:z + 32, y - 32:y + 32, x - 32:x + 32], dtype=np.float32)
    nnssl_patch = np.asarray(preprocessed[0, z - 32:z + 32, y - 32:y + 32, x - 32:x + 32])
    raw_zscore = (raw - raw.mean()) / raw.std()
    isonet_inverted_zscore = -raw_zscore

    assert np.corrcoef(raw_zscore.ravel(), nnssl_patch.ravel())[0, 1] > 0.999
    assert np.corrcoef(isonet_inverted_zscore.ravel(), nnssl_patch.ravel())[0, 1] < -0.999
