import pandas as pd

from IsoNet.bin import isonet as isonet_module


def _run_deconv(monkeypatch, tmp_path, row, *, ignore_defocus):
    captured = {}

    def fake_deconv_one(*args, **kwargs):
        captured.update(kwargs)

    def fake_process_tomograms(star_file, output_dir, tomo_idx, desc, row_processor):
        output_star = pd.DataFrame(index=[0])
        row_processor(0, pd.Series(row), output_star)

    monkeypatch.setattr(isonet_module, "deconv_one", fake_deconv_one)
    monkeypatch.setattr(isonet_module, "process_tomograms", fake_process_tomograms)

    isonet_module.ISONET().deconv(
        "tomograms.star",
        output_dir=str(tmp_path),
        ignore_defocus=ignore_defocus,
    )
    return captured


def test_deconv_can_ignore_missing_star_defocus(monkeypatch, tmp_path):
    captured = _run_deconv(
        monkeypatch,
        tmp_path,
        {
            "rlnTomoName": "input.mrc",
            "rlnVoltage": 300.0,
            "rlnSphericalAberration": 2.7,
            "rlnPixelSize": 10.0,
        },
        ignore_defocus=True,
    )

    assert captured["defocus"] == 0.0


def test_deconv_uses_star_defocus_by_default(monkeypatch, tmp_path):
    captured = _run_deconv(
        monkeypatch,
        tmp_path,
        {
            "rlnTomoName": "input.mrc",
            "rlnVoltage": 300.0,
            "rlnSphericalAberration": 2.7,
            "rlnPixelSize": 10.0,
            "rlnDefocus": 25000.0,
        },
        ignore_defocus=False,
    )

    assert captured["defocus"] == 2.5
