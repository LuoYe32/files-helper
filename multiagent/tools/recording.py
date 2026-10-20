import functools

from multiagent.guardrails.facts import record


def recorded(func):
    """Feed the tool's output to the fact checker, so numbers from tools count as verified."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        record(result)
        return result

    return wrapper
