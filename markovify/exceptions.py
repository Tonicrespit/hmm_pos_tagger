"""
This file includes all custom exceptions and error classes of the project.
"""
__all__ = ['NotFittedError']


class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if the model is used before fitting.

    Inherits from both ValueError and AttributeError to help with exception handling and backward compatibility.
    """