import json

from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE
from src.deeprl_liquidity_provision_uniswapv3.experiments.sensitivity import render


def write_rows(root, schedule="event_driven", shaping="none", paper=False, omit=None):
    for pool in CORE:
        for step in range(24):
            if omit == (pool, step):
                continue
            cfg = {"algo": "ppo", "schedule": schedule, "shaping": shaping,
                   "paper_extractor": paper}
            row = {"unit": {"pool": pool, "step": step},
                   "arms": {"PPO": {"test": float(step)}},
                   "provenance": {"config": cfg}}
            tag = f"{schedule}-{shaping}-{paper}"
            (root / f"{pool}-{step}-{tag}.json").write_text(json.dumps(row))


def test_complete_sensitivity_collection_writes_report(tmp_path):
    primary, variants, report = tmp_path / "p", tmp_path / "v", tmp_path / "r"
    primary.mkdir(); variants.mkdir()
    write_rows(primary)
    for schedule in ("hourly", "daily", "weekly"):
        write_rows(variants, schedule=schedule)
    for shaping in ("shadow", "lvr"):
        write_rows(variants, shaping=shaping)
    write_rows(variants, paper=True)
    text = render(primary, variants, report, 1, 100, 4)
    assert text.startswith("COMPLETE — FINAL COLLECTION")
    assert (report / "comparisons.csv").exists()


def test_partial_variant_refuses_final_artifacts(tmp_path):
    primary, variants, report = tmp_path / "p", tmp_path / "v", tmp_path / "r"
    primary.mkdir(); variants.mkdir()
    write_rows(primary)
    write_rows(variants, schedule="daily", omit=(CORE[0], 0))
    text = render(primary, variants, report, 1, 100, 4)
    assert text.startswith("PARTIAL — NOT A FINAL RESULT")
    assert not report.exists()


def test_missing_entire_variant_refuses_final_artifacts(tmp_path):
    primary, variants, report = tmp_path / "p", tmp_path / "v", tmp_path / "r"
    primary.mkdir(); variants.mkdir()
    write_rows(primary)
    for schedule in ("hourly", "daily", "weekly"):
        write_rows(variants, schedule=schedule)
    for shaping in ("shadow", "lvr"):
        write_rows(variants, shaping=shaping)
    text = render(primary, variants, report, 1, 100, 4)
    assert "missing variants: paper_extractor" in text
    assert not report.exists()
