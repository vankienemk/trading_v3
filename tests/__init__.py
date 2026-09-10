"""trading_v3 test suite.

Making ``tests`` a package lets every pattern plugin's test suite reuse the
shared no-lookahead template via ``from tests.no_lookahead_base import ...``
(works from the repository root, e.g. ``trading_v3/tests/test_*.py``).
"""