def test_package_imports():
    import country_rotation
    from country_rotation import data, factors, signals, backtest, validation, reporting, selection
    assert country_rotation.__version__ == "0.2.0"
