"""Explicit opt-in to a locally supplied backend; no weights loaded by default."""
import importlib
import pytest


def pytest_addoption(parser):
    group = parser.getgroup('contract-conformance')
    group.addoption('--td-backend-factory', help='Local module:callable returning a predict(state, questions) backend')
    group.addoption('--td-max-options', type=int, default=255, help='Declared backend Choice capacity')
    group.addoption('--td-independence-atol', type=float, default=1e-5, help='Probability tolerance for ID/order/batch isolation')
    group.addoption('--td-generates-tokens', action='store_true', help='Allow nonzero reported output tokens (e.g. llama-server n_predict=1)')


@pytest.fixture
def selected_backend(pytestconfig):
    name = pytestconfig.getoption('--td-backend-factory')
    if not name:
        pytest.skip('No backend requested; use --td-backend-factory for explicit integration')
    module, separator, factory = name.partition(':')
    if not separator or not module or not factory:
        raise pytest.UsageError('Expected --td-backend-factory=module:callable')
    backend = getattr(importlib.import_module(module), factory)()
    try:
        yield backend
    finally:
        close = getattr(backend, 'close', None)
        if close:
            close()
