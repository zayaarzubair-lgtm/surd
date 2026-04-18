"""Custom exceptions so errors are easy to understand."""


class SURDviewError(Exception):
    """Base error for anything that goes wrong in SURDview."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DataValidationError(SURDviewError):
    """Raised when the uploaded data has problems."""
    pass


class AnalysisError(SURDviewError):
    """Raised when the SURD computation fails."""
    pass
