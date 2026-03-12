# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quak.time_conversions — all input type branches and error paths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from quak.time_conversions import from_epoch_seconds, to_duration_seconds, to_epoch_seconds

# ── to_epoch_seconds ─────────────────────────────────────────────────────


class TestToEpochSeconds:
    """Cover float, int, datetime, np.datetime64, and TypeError branches."""

    def test_float_passthrough(self) -> None:
        assert to_epoch_seconds(100.5) == 100.5

    def test_int_passthrough(self) -> None:
        assert to_epoch_seconds(42) == 42.0
        assert isinstance(to_epoch_seconds(42), float)

    def test_datetime_utc(self) -> None:
        dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        epoch = to_epoch_seconds(dt)
        assert isinstance(epoch, float)
        assert epoch == dt.timestamp()

    def test_datetime_round_trip(self) -> None:
        dt = datetime(2026, 3, 7, 12, 30, 45, 123456, tzinfo=timezone.utc)
        epoch = to_epoch_seconds(dt)
        restored = datetime.fromtimestamp(epoch, tz=timezone.utc)
        # microsecond precision
        assert abs((restored - dt).total_seconds()) < 1e-6

    def test_numpy_datetime64(self) -> None:
        ndt = np.datetime64("2026-01-01T00:00:00", "ns")
        epoch = to_epoch_seconds(ndt)
        assert isinstance(epoch, float)
        assert epoch > 0

    def test_numpy_datetime64_round_trip(self) -> None:
        ndt = np.datetime64("2026-03-07T12:30:45.123456789", "ns")
        epoch = to_epoch_seconds(ndt)
        restored = from_epoch_seconds(epoch)
        # float64 can represent ~15 significant digits; at epoch ~1.77e9
        # that yields ~100ns precision, so 1µs tolerance is appropriate
        delta_ns = abs(int(ndt) - int(restored))
        assert delta_ns < 1000  # within 1 microsecond tolerance

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported time type"):
            to_epoch_seconds("2026-01-01")  # type: ignore[arg-type]

    def test_unsupported_type_list_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported time type"):
            to_epoch_seconds([1.0, 2.0])  # type: ignore[arg-type]


# ── from_epoch_seconds ───────────────────────────────────────────────────


class TestFromEpochSeconds:
    """Cover from_epoch_seconds output type and precision."""

    def test_returns_datetime64(self) -> None:
        result = from_epoch_seconds(0.0)
        assert isinstance(result, np.datetime64)

    def test_epoch_zero(self) -> None:
        result = from_epoch_seconds(0.0)
        assert result == np.datetime64(0, "ns")

    def test_positive_epoch(self) -> None:
        result = from_epoch_seconds(1_000_000_000.0)
        # 2001-09-09T01:46:40 UTC
        expected = np.datetime64("2001-09-09T01:46:40", "ns")
        assert result == expected

    def test_fractional_seconds(self) -> None:
        result = from_epoch_seconds(1.5)
        expected = np.datetime64(1_500_000_000, "ns")
        assert result == expected


# ── to_duration_seconds ──────────────────────────────────────────────────


class TestToDurationSeconds:
    """Cover float, int, timedelta, np.timedelta64, and TypeError branches."""

    def test_float_passthrough(self) -> None:
        assert to_duration_seconds(0.1) == 0.1

    def test_int_passthrough(self) -> None:
        assert to_duration_seconds(5) == 5.0
        assert isinstance(to_duration_seconds(5), float)

    def test_timedelta(self) -> None:
        td = timedelta(seconds=2, milliseconds=500)
        assert to_duration_seconds(td) == pytest.approx(2.5)

    def test_timedelta_negative(self) -> None:
        td = timedelta(seconds=-1)
        assert to_duration_seconds(td) == pytest.approx(-1.0)

    def test_numpy_timedelta64_seconds(self) -> None:
        ntd = np.timedelta64(3, "s")
        assert to_duration_seconds(ntd) == pytest.approx(3.0)

    def test_numpy_timedelta64_milliseconds(self) -> None:
        ntd = np.timedelta64(500, "ms")
        assert to_duration_seconds(ntd) == pytest.approx(0.5)

    def test_numpy_timedelta64_nanoseconds(self) -> None:
        ntd = np.timedelta64(1_000_000_000, "ns")
        assert to_duration_seconds(ntd) == pytest.approx(1.0)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported duration type"):
            to_duration_seconds("1s")  # type: ignore[arg-type]
