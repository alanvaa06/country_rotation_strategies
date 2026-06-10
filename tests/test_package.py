def test_package_imports():
    import country_rotation
    from country_rotation import data, factors, signals, backtest, validation, reporting, selection
    assert country_rotation.__version__ == "1.0.0"
