import logging, re, sys

class CustomLogFormatter(logging.Formatter):
    # derived from https://stackoverflow.com/a/71336115
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_cache = {}
        self.threadname_re = re.compile(r"^.*\((.*)\)$")

    def format(self, record):
        saved_name = record.name  # save and restore for other formatters if desired
        abbrev = self.name_cache.get(saved_name, None)
        if abbrev is None:
            parts = saved_name.split('.')
            if len(parts) > 1:
                parts[1:-1] = [p[0] for p in parts[1:-1]]
                abbrev = '.'.join(parts)
            else:
                abbrev = saved_name
            abbrev = abbrev[:20]
            self.name_cache[saved_name] = abbrev
            # print(saved_name, abbrev)
        record.name = abbrev

        saved_threadname = record.threadName
        m = self.threadname_re.match(record.threadName)
        if m:
            record.threadName = m.group(1)
        record.threadName = record.threadName[:20]

        result = super().format(record)

        record.name = saved_name
        record.threadName = saved_threadname
        return result

h = logging.StreamHandler(stream=sys.stderr)
logging_formatter = CustomLogFormatter('%(asctime)s %(levelname)-8s %(threadName)-20s %(module)-12s %(funcName)-12s %(message)s')
h.setFormatter(logging_formatter)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(h)
root_logger.setLevel(logging.INFO)


class ResetLevelLogFilter:
    '''
    this filter calls suppress_function with the message text. if the function
    returns true, then the message has it's level reset to DEBUG, and the message's
    logger is checked to see if it should be logged. not sure how this will work
    as a filter for a handler.
    '''
    def __init__(self, suppress_function = None):
        self._suppress_function = suppress_function
        self._my_logger = None

    def filter(self, r: logging.LogRecord):
        m = r.getMessage()
        do_suppress = self._suppress_function(m)
        if do_suppress:
            r.levelno = logging.DEBUG
            r.levelname = logging.getLevelName(r.levelno)

            if self._my_logger is None:
                self._my_logger = logging.getLogger(r.name)
            if not self._my_logger.isEnabledFor(r.levelno):
                return False
        return True


class SuppressLogFilter:
    '''
    this filter calls suppress_function with the message text. if the function
    returns true, then the message is suppressed.
    '''
    def __init__(self, suppress_function = None):
        self._suppress_function = suppress_function

    def filter(self, r: logging.LogRecord):
        m = r.getMessage()
        do_suppress = self._suppress_function(m)
        return not do_suppress


if __name__ == '__main__':
    class BrokenResetLevelLogFilter:
        '''
        this one tries to change the log level of the message if it's undesirable,
        but does not work; the logger has already looked at the level of
        the message and has determined this message should be logged.
        '''
        def filter(self, r: logging.LogRecord):
            m = r.getMessage()
            do_suppress = 'suppress' in m
            if do_suppress:
                r.levelno = logging.DEBUG
                r.levelname = logging.getLevelName(r.levelno)
            return True

    root_logger.setLevel(logging.INFO)
    for level in (logging.DEBUG, logging.INFO):
        for log_filter in (
                BrokenResetLevelLogFilter(),
                ResetLevelLogFilter(lambda m: 'suppress' in m),
                SuppressLogFilter(lambda m: 'suppress' in m)
        ):
            name = logging.getLevelName(level) + str(type(log_filter))
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addFilter(log_filter)

            logger.warning("Warning")
            logger.warning("suppress Warning")
            logger.info("Info")
            logger.info("suppress Info")
            logger.debug("Debug")
            logger.debug("suppress Debug")
