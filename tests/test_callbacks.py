from typing import Any

import mlx.core as mx
import pytest

from mlx_timeseries_workbench.callbacks import Callback, CallbackList, MetricTracker


def test_metric_tracker_accumulation() -> None:
    tracker = MetricTracker()
    tracker.on_epoch_begin(1)

    # Batch 1: size 256, loss 1.0 (as mx.array)
    tracker.on_train_batch_end(0, {"loss": mx.array(1.0), "size": 256})
    # Batch 2: size 100, loss 0.5 (as mx.array)
    tracker.on_train_batch_end(1, {"loss": mx.array(0.5), "size": 100})

    logs: dict[str, Any] = {}
    tracker.on_epoch_end(1, logs)

    expected_loss = (1.0 * 256 + 0.5 * 100) / (256 + 100)
    assert "train" in logs
    assert abs(logs["train"] - expected_loss) < 1e-6


def test_metric_tracker_resets_on_epoch_begin() -> None:
    tracker = MetricTracker()
    tracker.on_epoch_begin(1)
    tracker.on_train_batch_end(0, {"loss": 2.0, "size": 10})

    logs_ep1: dict[str, Any] = {}
    tracker.on_epoch_end(1, logs_ep1)
    assert logs_ep1["train"] == 2.0

    # Start epoch 2 -> should reset accumulators
    tracker.on_epoch_begin(2)
    tracker.on_train_batch_end(0, {"loss": 0.5, "size": 20})

    logs_ep2: dict[str, Any] = {}
    tracker.on_epoch_end(2, logs_ep2)
    assert logs_ep2["train"] == 0.5


def test_metric_tracker_in_callback_list() -> None:
    tracker = MetricTracker()
    callback_list = CallbackList([tracker])

    callback_list.on_epoch_begin(1)
    callback_list.on_train_batch_end(0, {"loss": 0.8, "size": 50})

    logs: dict[str, Any] = {"eval": 0.9}
    callback_list.on_epoch_end(1, logs)

    assert "train" in logs
    assert abs(logs["train"] - 0.8) < 1e-6
    assert logs["eval"] == 0.9


@pytest.mark.parametrize(
    "other, expected_added_count",
    [
        ([MetricTracker()], 1),
        ([MetricTracker(), MetricTracker()], 2),
        (MetricTracker(), 1),
        (CallbackList([MetricTracker(), MetricTracker()]), 2),
        (None, 0),
    ],
    ids=["list_one", "list_multiple", "single_callback", "callback_list", "none"],
)
def test_callback_list_add(
    other: CallbackList | list[Callback] | Callback | None,
    expected_added_count: int,
) -> None:
    t1 = MetricTracker()
    initial = CallbackList(t1)
    result = initial + other

    # Verify a new instance is returned
    assert result is not initial
    # Verify the original list was not modified
    assert len(initial.callbacks) == 1
    assert initial.callbacks[0] is t1
    # Verify the combined length
    assert len(result.callbacks) == 1 + expected_added_count
    assert result.callbacks[0] is t1


@pytest.mark.parametrize(
    "other, expected_added_count",
    [
        ([MetricTracker()], 1),
        ([MetricTracker(), MetricTracker()], 2),
        (MetricTracker(), 1),
        (CallbackList([MetricTracker(), MetricTracker()]), 2),
        (None, 0),
    ],
    ids=["list_one", "list_multiple", "single_callback", "callback_list", "none"],
)
def test_callback_list_iadd(
    other: CallbackList | list[Callback] | Callback | None,
    expected_added_count: int,
) -> None:
    t1 = MetricTracker()
    target = CallbackList(t1)
    target += other

    # Verify in-place modification
    assert len(target.callbacks) == 1 + expected_added_count
    assert target.callbacks[0] is t1



