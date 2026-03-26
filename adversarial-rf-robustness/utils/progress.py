"""Progress bars and JSON status files for long-running evaluation scripts."""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def iter_batches(
    loader: Iterable[T],
    desc: str,
    show_progress: bool = True,
) -> Iterator[T]:
    """Wrap a DataLoader with tqdm, or pass through when disabled."""
    if show_progress:
        from tqdm import tqdm

        return iter(
            tqdm(
                loader,
                desc=desc,
                leave=False,
                mininterval=0.5,
                dynamic_ncols=True,
            )
        )
    return iter(loader)


def write_status(path: Optional[str], **kwargs: Any) -> None:
    """Write a small JSON snapshot for monitoring (e.g. tail from another terminal or Colab)."""
    if not path:
        return
    payload = {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload.update(kwargs)
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass
