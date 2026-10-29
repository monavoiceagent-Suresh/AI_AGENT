from types import SimpleNamespace

import pytest

from app import calls
from app.models import Agent
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio


class FakeExotel:
    def __init__(self) -> None:
        self.dialled: list[str] = []

    async def connect_to_flow(self, number: str, **_: object) -> str:
        self.dialled.append(number)
        return "CAtest1"


class FakePool:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[tuple] = []

    async def fetchval(self, _query: str, *args: object) -> int:
        if self.fail:
            raise OSError("database is down")
        self.rows.append(args)
        return 7


def services(pool: FakePool) -> SimpleNamespace:
    settings = make_settings(enable_outbound_calls=True, max_concurrent_calls=2, inbound_reserved_calls=0)
    return SimpleNamespace(settings=settings, exotel=FakeExotel(), pool=pool, active_calls=0, ringing={})


AGENT = Agent.model_construct(id=3)


@pytest.mark.parametrize(
    ("given", "dialled"),
    [
        ("+919876543210", "+919876543210"),
        ("919876543210", "+919876543210"),
        ("9876543210", "+919876543210"),
        ("09876543210", "+919876543210"),
    ],
)
async def test_outbound_numbers_reach_exotel_with_the_country_code(given, dialled):
    svc = services(FakePool())

    await calls.place_outbound_call(svc, AGENT, given, None)

    assert svc.exotel.dialled == [dialled]
    assert svc.pool.rows[0][3] == dialled  # the recorded to_number is the number that was dialled


async def test_a_call_placed_but_not_recorded_is_logged_with_its_sid(caplog):
    svc = services(FakePool(fail=True))

    with pytest.raises(OSError):
        await calls.place_outbound_call(svc, AGENT, "+919876543210", None)

    assert svc.exotel.dialled == ["+919876543210"]
    assert any("CAtest1" in r.getMessage() and "could not be recorded" in r.getMessage() for r in caplog.records)
