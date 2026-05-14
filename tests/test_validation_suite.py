def test_validation_suite_module_importable():
    import src.analysis.run_validation_suite as m

    assert callable(m.main)
