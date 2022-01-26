# vim: expandtab:ts=4:sw=4

#import asyncio
from distutils import command
import zlib
import requests
#import asyncio
import ujson
import numpy as np
import io
from threading import Thread
from queue import Queue
#from .queue import Queue


class Command:
    """
    Command for communication function. `FETCH` will intract with server to receive the global ID.
    `Update` will send a new feature to the server to update its feature pool.
    """

    FETCH = 1
    UPDATE = 2
    STOP = 3

    def __init__(self, cmd_type, feature) -> None:
        self.cmd_type = cmd_type
        self.feature = feature
        self.cmp_feature = None
        self.response = None

    def __str__(self) -> str:
        cmd_type = 'Update' 
        if self.cmd_type == 2:
            cmd_type = 'FETCH'
        else:
            cmd_type = 'STOP'
        return 'Type: {}'.format(cmd_type)


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        feature=None,
        cls_id=None,
        client_cfg=None
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.global_id = -1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.cls_id = cls_id

        self.client_cfg = client_cfg

        self._queue_comm = Queue(self.client_cfg.Budget)
        self._queue_comm.put_nowait(Command(Command.FETCH, feature))

        self._run_thread = True
        self._thread_comm = Thread(target=self.communicate)
        self._thread_comm.start()

        #self.event_loop = event_loop

        # asyncio.run_coroutine_threadsafe(self.communicate(
        #    ), event_loop)

        # self._cmd_queue.put_nowait(
        #    Command(Command.FETCH, feature)
        # )

    def _set_track_id(self, track_id):
        self.global_id = track_id

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)
        try:
            if self.is_confirmed():
                # I know it sounds crazy, but this the way to comm between two worlds of
                # async and sync worlds
                if (self.hits % self.client_cfg.Budget) == 0:
                    self._queue_comm.put_nowait(
                        Command(Command.UPDATE, detection.feature))
        except Exception as e:
            print(str(e))

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def _stop_comm_thread(self):
        print('x> Stopping thread for track {}'.format(self.track_id))
        self._run_thread = False
        self._queue_comm.put_nowait(Command(Command.STOP, None))
        '''
         while self._queue_comm.not_empty:
             print("x> Skipped command `{}` for Track: {}".format(
                 self.track_id, self._queue_comm.get_nowait()))
        '''
        self._queue_comm.task_done()
        self._queue_comm.join()
        self._thread_comm.join()

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            self._stop_comm_thread()
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            self._stop_comm_thread()
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def get_cls_id(self):
        """Returns the class id of this track"""
        return self.cls_id

    def compress_nparr(self, cmd):  # nparr):
        """
        Returns the given numpy array as compressed bytestring,
        the uncompressed and the compressed byte size.
        """
        bytestream = io.BytesIO()
        nparr = cmd.feature
        np.save(bytestream, nparr)
        uncompressed = bytestream.getvalue()
        cmd.cmp_feature = zlib.compress(uncompressed)
        return cmd

        # return compressed, len(uncompressed), len(compressed)

    def post_nparr(self, cmd):
        print('Sending the data for track: {}'.format(self.track_id))
        server_address = "http://" + self.client_cfg.Server_IP + \
            ":" + str(self.client_cfg.Server_Socket)
        if cmd.cmd_type == Command.UPDATE:
            url = "{}/{}?cam_id={}&track_id={}&global_id={}".format(server_address,
                                                                    self.client_cfg.Update_URL,
                                                                    self.client_cfg.Camera_ID,
                                                                    self.track_id, self.global_id)
        elif cmd.cmd_type == Command.FETCH:
            url = "{}/{}?cam_id={}&track_id={}".format(server_address,
                                                       self.client_cfg.Fetch_URL,
                                                       self.client_cfg.Camera_ID, self.track_id)

        with requests.Session() as session:  # json_serialize=ujson.dumps
            try:
                with session.post(
                    url,
                    data=cmd.cmp_feature,
                    headers={"Content-Type": "application/octet-stream"},
                ) as response:
                    # TODO: Fix the error handling and logging.
                    cmd.response = response.json()
            except Exception as e:
                print("The error is : {}".format(str(e)))

        return cmd

    def process_response(self, cmd):
        if cmd.response is not None:
            if cmd.cmd_type == Command.FETCH or cmd.cmd_type == Command.UPDATE:
                self._set_track_id(cmd.response['global_id'])

    def communicate(self):
        if self.client_cfg is not None:
            while True and self._run_thread:
                cmd = self._queue_comm.get(block=True)
                if cmd.cmd_type == Command.STOP:
                    break
                cmd = self.compress_nparr(cmd)
                cmd = self.post_nparr(cmd)
                self.process_response(cmd)
                self._queue_comm.task_done()
        else:
            print("!> Client config is not set. System is based on local ReID for track:{}".format(
                self.track_id))
