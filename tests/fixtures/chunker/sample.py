"""Module docstring."""

import os


# Leading comment for top_level_function
def top_level_function(x):
    """Docstring."""
    return x + 1


class Example:
    """Class docstring."""

    def method_one(self):
        return 1

    @staticmethod
    def method_two():
        return 2


@decorator
def decorated_function():
    return None
