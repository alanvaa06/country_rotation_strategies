from country_rotation.factors.catalog import CATALOG, by_category, get_spec


def test_no_raw_levels_in_catalog():
    names = {f.name for f in CATALOG}
    for lvl in ("Price", "GDP", "M2", "Market_Cap", "EV", "Revenue", "Debt", "Equity"):
        assert lvl not in names


def test_roe_is_profitability():
    assert get_spec("ROE").category == "Profitability"
    assert get_spec("ROE").direction == 1


def test_momentum_has_price_momentum():
    moms = {f.name for f in by_category("Momentum")}
    assert {"Momentum_12_1", "Momentum_6_1"}.issubset(moms)


def test_directions_sane():
    assert get_spec("PE").direction == -1
    assert get_spec("EarningsYieldTTM").direction == 1
    assert get_spec("RollingVol").direction == -1
    assert all(f.category in {"Valuation", "Quality", "Profitability", "Momentum"} for f in CATALOG)
