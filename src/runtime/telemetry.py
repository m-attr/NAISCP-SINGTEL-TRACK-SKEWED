from __future__ import annotations

import sys


def peak_rss_bytes() -> tuple[int | None, str]:
    """Return the best available process peak-memory measurement."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return (
            value if sys.platform == "darwin" else value * 1024
        ), "resource.ru_maxrss"
    except Exception:
        try:
            import psutil  # type: ignore

            return (
                int(psutil.Process().memory_info().rss),
                "psutil.current_rss_fallback",
            )
        except Exception:
            return None, "unavailable"
