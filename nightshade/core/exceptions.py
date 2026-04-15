class NightShadeException(Exception):
    pass


class ConfigError(NightShadeException):
    pass


class ModelError(NightShadeException):
    pass


class DataError(NightShadeException):
    pass


class TrainingError(NightShadeException):
    pass


class CheckpointError(NightShadeException):
    pass
