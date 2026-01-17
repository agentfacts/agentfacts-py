import json

import pytest

from agentfacts.crypto.canonicalization import canonicalize_to_string

GOLDEN_VECTORS = [
    (
        '{"b":1,"a":2}',
        '{"a":2,"b":1}',
    ),
    (
        '{"a":{"b":1,"a":2}}',
        '{"a":{"a":2,"b":1}}',
    ),
    (
        '{"arr":[3,2,1]}',
        '{"arr":[3,2,1]}',
    ),
    (
        '{"str":"Hello\\nWorld"}',
        '{"str":"Hello\\nWorld"}',
    ),
    (
        '{"quote":"\\""}',
        '{"quote":"\\""}',
    ),
    (
        '{"control":"\\u0001"}',
        '{"control":"\\u0001"}',
    ),
    (
        '{"unicode":"\\u20ac"}',
        '{"unicode":"\u20ac"}',
    ),
    (
        '{"n":-0}',
        '{"n":0}',
    ),
    (
        '{"big":1e21,"small":1e-7,"threshold":1e-6}',
        '{"big":1e+21,"small":1e-7,"threshold":0.000001}',
    ),
    (
        '{"a\\ud834\\udd1e":2,"a\\uffff":1}',
        '{"a\U0001d11e":2,"a\uffff":1}',
    ),
]


@pytest.mark.parametrize("input_json,expected", GOLDEN_VECTORS)
def test_canonicalization_vectors(input_json: str, expected: str) -> None:
    data = json.loads(input_json)
    result = canonicalize_to_string(data)
    assert result == expected
