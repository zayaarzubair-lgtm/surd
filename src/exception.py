"""Custom exceptons so errors are easy to understand."""


class SURDviewError(Exception):
    """ Main error to be used for now """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DataValidationError(SURDviewError):
    """error for uploading"""
    pass


class AnalysisError(SURDviewError):
    """error for actually seeing the file"""
    pass
