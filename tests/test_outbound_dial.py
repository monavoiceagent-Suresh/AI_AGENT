from types import SimpleNamespace

import pytest

from app import calls
from app.calls import CallRejected
from app.models import Agent
from app.providers.exotel import ExotelError
from tests.fakes import FakeAllocator, make_settings

pytestmark = pytest.mark.anyio


class FakeExotel:
    def __init__(self) -> None:
        self.dialled: list[str] = []
        self.caller_ids: list[str | None] = []

    async def connect_to_flow(self, number: str, *, caller_id: str | None = None, **_: object) -> str:
        self.dialled.append(number)
        self.caller_ids.append(caller_id)
        return "CAtest1"


class RejectingExotel:
    """Simulates Exotel's connect call itself failing, e.g. a 502 from the provider."""

    def __init__(self, error: ExotelError) -> None:
        self.error = error

    async def connect_to_flow(self, *_args: object, **_kwargs: object) -> str:
        raise self.error


class FakePool:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[tuple] = []

    async def fetchval(self, _query: str, *args: object) -> int:
        if self.fail:
            raise OSError("database is down")
        self.rows.append(args)
        return 7


def services(pool: FakePool, exotel: object | None = None) -> SimpleNamespace:
    settings = make_settings(enable_outbound_calls=True, max_concurrent_calls=2, inbound_reserved_calls=0)
    allocator = FakeAllocator(max_concurrent_calls=2)
    return SimpleNamespace(settings=settings, exotel=exotel or FakeExotel(), pool=pool, allocator=allocator)


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


async def test_a_resolved_caller_id_reaches_exotel_and_the_saved_call():
    svc = services(FakePool())

    await calls.place_outbound_call(svc, AGENT, "+919876543210", None, caller_id="+918000000001")

    assert svc.exotel.caller_ids == ["+918000000001"]
    assert svc.pool.rows[0][2] == "+918000000001"  # the recorded from_number is the resolved caller id


async def test_no_caller_id_falls_back_to_the_engine_wide_setting():
    svc = services(FakePool())

    await calls.place_outbound_call(svc, AGENT, "+919876543210", None)

    assert svc.exotel.caller_ids == [svc.settings.exotel_caller_id]
    assert svc.pool.rows[0][2] == svc.settings.exotel_caller_id


async def test_a_call_placed_but_not_recorded_is_logged_with_its_sid(caplog):
    svc = services(FakePool(fail=True))

    with pytest.raises(OSError):
        await calls.place_outbound_call(svc, AGENT, "+919876543210", None)

    assert svc.exotel.dialled == ["+919876543210"]
    assert any("CAtest1" in r.getMessage() and "could not be recorded" in r.getMessage() for r in caplog.records)


# ---- reservation lifecycle on a failed Exotel connection -----------------------------------


async def test_a_definite_4xx_releases_the_reservation_immediately():
    """A 4xx from Exotel's connect call is a definitive answer: nothing was dialled, so the
    reservation is released right away and the slot is free for the next call."""
    error = ExotelError("Exotel connect failed (401): bad credentials", placement_unknown=False)
    svc = services(FakePool(), RejectingExotel(error))

    with pytest.raises(ExotelError):
        await calls.place_outbound_call(svc, AGENT, "+919876543210", None)

    assert svc.allocator.allocations == {}


async def test_a_502_or_other_5xx_keeps_the_reservation_as_placement_unknown():
    """A 5xx (e.g. the reported HTTP 502) may be hiding a call that was actually placed, so
    place_outbound_call deliberately does not release it here — only a confirmed terminal state
    (app/allocator.py ChannelAllocator.reconcile_on_startup) may release it later."""
    error = ExotelError("Exotel connect failed (502): bad gateway", placement_unknown=True)
    svc = services(FakePool(), RejectingExotel(error))

    with pytest.raises(ExotelError):
        await calls.place_outbound_call(svc, AGENT, "+919876543210", None)

    assert len(svc.allocator.allocations) == 1


async def test_capacity_exhaustion_is_reported_as_call_rejected_without_touching_exotel():
    """A full account must refuse before ever calling the provider, and must say why."""
    svc = services(FakePool())
    svc.allocator.allocations["out-existing-1"] = "outbound"
    svc.allocator.allocations["out-existing-2"] = "outbound"  # fills the fixture's 2-call limit

    with pytest.raises(CallRejected, match="Concurrent call limit reached"):
        await calls.place_outbound_call(svc, AGENT, "+919876543210", None)

    assert svc.exotel.dialled == []
