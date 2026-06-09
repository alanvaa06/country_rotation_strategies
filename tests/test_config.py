from country_rotation.config import PlatformConfig, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.json")
    assert isinstance(cfg, PlatformConfig)
    assert cfg.data.inputs_folder == "Inputs"
    assert cfg.data.ffill_limit == 63
    assert cfg.data.publication_lag_days == {}
    assert cfg.backtest.periodicity == 63
    assert cfg.backtest.transaction_cost_bps == 2.0
    assert cfg.backtest.mode in ("blend", "active")


def test_config_is_frozen():
    cfg = load_config("configs/default.json")
    import pytest
    with pytest.raises(Exception):
        cfg.backtest.periodicity = 5
