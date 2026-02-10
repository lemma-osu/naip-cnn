def pytest_sessionstart(session):
    import ee

    ee.Initialize()
