import logging
import time
import threading


try:
    from greenlet import getcurrent as get_ident
    logging.info("imported greenlet")
except ImportError:
    try:
        from thread import get_ident
        logging.info("imported thread")
    except ImportError:
        from _thread import get_ident
        logging.info("imported _thread")


ev_logger = logging.getLogger("distributor.events")
ev_logger.setLevel(logging.INFO)
logger = logging.getLogger("distributor")
logger.setLevel(logging.INFO)


class DistributorEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        ev_logger.debug("thread %s is going to wait", ident)
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class Receiver:
    def __init__(self, distributor):
        logger.info("creating %s", self)
        self.distributor = distributor

        distributor.start_background_thread()

    def get_last_result(self):
        """Return the current camera frame."""
        self.distributor.last_access = time.time()

        # wait for a signal from the camera thread
        ev_logger.debug("waiting for Distributor.event")
        self.distributor.event.wait()
        ev_logger.debug("got Distributor.event")
        self.distributor.event.clear()

        rv = self.distributor.last_result
        logging.info("returning %s %s", type(rv), rv)
        return rv


class Distributor:
    def __init__(self, source=None, background_timeout: int = None):
        logger.info("creating %s", self)
        self.provider = source
        self.background_timeout = background_timeout

        self.thread = None  # background thread that reads input
        self.last_result = None  # current frame is stored here by background thread
        self.last_access = 0  # time of last client access to the source
        self.event = DistributorEvent()

        self.start_background_thread()

    def start_background_thread(self):
        """Start the background thread if it isn't running yet."""
        if self.thread is None:
            self.last_access = time.time()

            # start background frame thread
            self.thread = threading.Thread(target=self._thread)
            self.thread.start()

            # wait until first frame is available
            ev_logger.debug("waiting for first event")
            self.event.wait()
            ev_logger.debug("got first event")
        else:
            logging.info("background thread is already running")

    def get_receiver(self) -> Receiver:
        return Receiver(self)

    def _thread(self):
        """Camera background thread."""
        logging.info('Starting background thread: provider = %s', self.provider)
        input_iterator = self.provider
        for result in input_iterator():
            ev_logger.info("got %s, setting Distributor.event", type(result))
            self.last_result = result
            self.event.set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            if self.background_timeout is not None and (time.time() - self.last_access > self.background_timeout):
                input_iterator.close()  # need to fix this
                logging.info('Stopping background thread due to inactivity.')
                break
        self.thread = None
        logging.info('Background thread is exiting.')
