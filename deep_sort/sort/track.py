# vim: expandtab:ts=4:sw=4

#import asyncio
#from distutils import command
from time import sleep
import zlib
import requests
import numpy as np
import io
from threading import Thread, Event
from queue import Empty, Queue
from api_utils.interfaces import Command
from threading import Semaphore

#from .queue import Queue


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
        logger=None,
        client_cfg=None,
        shut_down_event: Event = None,
        # Only these classes will be sent to server for global reID.
        filtered_classes=[0]
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.global_id = -1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.logger = logger
        self.state_lock = Semaphore()
        self.comm_thread_status_lock = Semaphore()

        self.set_state(TrackState.Tentative)

        self.features = []
        self.last_features = []
        if feature is not None:
            self.features.append(feature)

        '''
            This variable will be used at SmartCity `run` function to get the feature from the current frame.
            The feature list variable will be erased after each update function.
            Try to understand why though!
        '''
        self.last_features = feature

        self._n_init = n_init
        self._max_age = max_age
        self.cls_id = cls_id

        self.client_cfg = client_cfg

        self.filtered_classes = filtered_classes

        self.should_be_joined_forcefully = False
        self.shut_down_event = shut_down_event
        self.queue_comm = Queue()

        self._thread_comm = Thread(
            target=self.communicate, name="Track_COMM:{}".format(self.track_id))
        self._thread_comm.setDaemon(False)
        # self._thread_comm.start()

        self.global_id_history = {}
        self.global_id_hit = {}

        # Define the arbiter callback function
        self.arbiter = self.arbiter_history_hit  # arbiter_immediate_update

        #self.event_loop = event_loop

        # asyncio.run_coroutine_threadsafe(self.communicate(
        #    ), event_loop)

        # self._cmd_queue.put_nowait(
        #    Command(Command.FETCH, feature)
        # )

    def get_state(self):
        self.state_lock.acquire()
        state = self.state
        self.state_lock.release()
        return state

    def set_state(self, state):
        self.state_lock.acquire()
        self.state = state
        self.state_lock.release()

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
        self.last_features = self.features

        self.hits += 1
        self.time_since_update = 0
        if self.get_state() == TrackState.Tentative and self.hits >= self._n_init:
            self.set_state(TrackState.Confirmed)

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.get_state() == TrackState.Tentative:
            self.set_state(TrackState.Deleted)
            # self.queue_comm.put_nowait(None)
            self.logger.debug(
                "Track#{} is marked as deleted.".format(self.track_id))
        elif self.time_since_update > self._max_age:
            self.set_state(TrackState.Deleted)
            # self.queue_comm.put_nowait(None)
            self.logger.debug(
                "Track#{} is marked as deleted.".format(self.track_id))
        # self._stop_comm_thread()

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.get_state() == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.get_state() == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.get_state() == TrackState.Deleted

    def get_cls_id(self):
        """Returns the class id of this track"""
        return int(self.cls_id)

    def compress_nparr(self, cmd: Command):  # nparr):
        """
        Returns the given numpy array as compressed bytestring,
        the uncompressed and the compressed byte size.
        """
        bytestream = io.BytesIO()
        np.save(bytestream, cmd.feature)
        uncompressed = bytestream.getvalue()
        cmd.cmp_feature = zlib.compress(uncompressed)
        if cmd.img_bbox is not None:
            bytestream_img = io.BytesIO()
            np.save(bytestream_img, cmd.img_bbox)
            uncompressed_img = bytestream_img.getvalue()
            cmd.cmp_image = zlib.compress(uncompressed_img)

        return cmd

        # return compressed, len(uncompressed), len(compressed)

    def post_nparr(self, cmd: Command):
        self.logger.info(
            'Sending the data for track#{}'.format(self.track_id))
        server_address = "http://" + self.client_cfg.Server_IP + \
            ":" + str(self.client_cfg.Server_Socket)
        bbox_tlwh = "{:.0f},{:.0f},{:.0f},{:.0f}".format(
            *(self.to_tlwh().tolist()))
        cam_id = self.client_cfg.Additional_Camera_ID if self.client_cfg.Camera_ID == - \
            1 else self.client_cfg.Camera_ID
        if cmd.cmd_type == Command.UPDATE:
            url_args = "?cam_id={}&track_id={}&global_id={}&class_id={}&bbox_tlwh={}".format(cam_id,
                                                                                             self.track_id, self.global_id,
                                                                                             int(
                                                                                                 self.cls_id),
                                                                                             bbox_tlwh)
            id_url = '{}/{}{}'.format(server_address,
                                      self.client_cfg.Update_URL,
                                      url_args)
        elif cmd.cmd_type == Command.FETCH:
            url_args = "?cam_id={}&track_id={}&class_id={}&bbox_tlwh={}".format(
                cam_id, self.track_id,
                int(self.cls_id),
                bbox_tlwh)

            id_url = '{}/{}{}'.format(server_address,
                                      self.client_cfg.Fetch_URL,
                                      url_args)

        with requests.Session() as session:
            '''
                NOTE: You may ask why I am getting and sending back
                the `last_inserted_key_val` when we have all the information here. You can use it 
                and decouple the POST procedure of sending image in a seperated thread to improve the
                performance. ;)
            '''
            try:
                with session.post(
                    id_url,
                    data=cmd.cmp_feature,
                    headers={"Content-Type": "application/octet-stream"},
                ) as response:
                    cmd.response = response.json()
            except Exception as e:
                self.logger.debug(
                    "While sending the payload for track#{}, the error is caught: {}".format(str(e)))

        # Check if need also to send bbox img or not.
        if cmd.cmp_image is not None and cmd.response is not None:

            last_inserted_key_val = cmd.response['last_inserted_key_val']
            if last_inserted_key_val > -1:
                img_url = '{}/{}?last_inserted_key_val={}'.format(server_address,
                                                                  self.client_cfg.BBOx_IMG_URL,
                                                                  last_inserted_key_val)

                with requests.Session() as session:
                    try:
                        with session.post(
                            img_url,
                            data=cmd.cmp_image,
                            headers={
                                "Content-Type": "application/octet-stream"},
                        ) as response:
                            cmd.img_response = response.json()
                    except Exception as e:
                        self.logger.debug(
                            "While sending the image for track#{}, the error is caught: {}".format(str(e)))
            else:
                self.logger.debug(
                    "There was an error on server side to save the data.")

        return cmd

    def arbiter_immediate_update(self, recv_global_id):
        return recv_global_id

    def arbiter_history_hit(self, recv_global_id):
        most_recent_before_update = [
            k for k, v in self.global_id_history.items() if v == 1][0]
        most_hitted_value_before_update = max(
            self.global_id_hit, key=self.global_id_hit.get)

        for key in self.global_id_history.keys():
            if key != recv_global_id:
                # They are not any more most recent values.
                self.global_id_history[key] = 0
        self.global_id_history[recv_global_id] = 1

        if recv_global_id in self.global_id_hit.keys():
            self.global_id_hit[recv_global_id] += 1
        else:
            self.global_id_hit[recv_global_id] = 1

        most_hitted_value_after_update = max(
            self.global_id_hit, key=self.global_id_hit.get)
#        most_recent_after_update = [k for k, v in self.global_id_history.items() if v == 0][0]

        if most_hitted_value_after_update >= most_hitted_value_before_update:
            return most_hitted_value_after_update
        else:
            return most_recent_before_update

    def process_response(self, cmd):
        if cmd.response is not None:
            recv_global_id = cmd.response['global_id']
            if cmd.cmd_type == Command.FETCH:
                self.global_id_history[recv_global_id] = 1
                self.global_id_hit[recv_global_id] = 1
                self._set_track_id(recv_global_id)
            elif cmd.cmd_type == Command.UPDATE:
                if self.global_id == -1:
                    self.global_id_history[recv_global_id] = 1
                    self.global_id_hit[recv_global_id] = 1
                    self._set_track_id(recv_global_id)
                # We have multiple global IDs, let's ask arbiter to judge!
                else:
                    final_global_id = self.arbiter(recv_global_id)
                    self._set_track_id(final_global_id)

    def set_to_joined_forcefully(self, state: bool):
        self.comm_thread_status_lock.acquire()
        self.should_be_joined_forcefully = state
        self.comm_thread_status_lock.release()

    def get_to_joined_forcefully(self):
        self.comm_thread_status_lock.acquire()
        state = self.should_be_joined_forcefully
        self.comm_thread_status_lock.release()
        return state

    def communicate(self):
        if self.client_cfg is not None:
            self.logger.debug(
                'The COMM thread of track#{} is started...'.format(self.track_id))
            while not self.shut_down_event.is_set():  # self._run_thread:
                try:
                    cmd = self.queue_comm.get(timeout=1)
                except Empty:
                    continue

                #if cmd is None:
                #    self.queue_comm.task_done()
                #    self.logger.debug(
                #        'None is captured for track#{}'.format(self.track_id))
                #    break
                '''
                if cmd.cmd_type == Command.STOP:
                    # Flush the queue if it is not empty.
                    while not self.queue_comm.empty():
                        try:
                            cmd = self.queue_comm.get(block=True)
                            self.logger.debug("Skipped command `{}` for Track: {}".format(
                                cmd.cmd_type ,self.track_id))
                            self.queue_comm.task_done()
                        except Empty:
                            break
                    self.queue_comm.task_done()
                    break
                '''
                # Let's filter out the objects we want to the server
                try:
                    if self.get_cls_id() in self.filtered_classes:
                        self.logger.debug(
                            "Compressing payload for track#{}.".format(self.track_id))
                        cmd = self.compress_nparr(cmd)
                        self.logger.debug(
                            "Posting payload for track#{}.".format(self.track_id))
                        cmd = self.post_nparr(cmd)
                        self.logger.debug(
                            "Processing the request for track#{}.".format(self.track_id))
                        self.process_response(cmd)
                        self.queue_comm.task_done()
                        self.logger.debug(
                            "Task is done for track#{}.".format(self.track_id))

                except Exception as error:
                    self.logger.debug(
                        "Received exception {} for track#{}.".format(str(error), self.track_id))
                    # Task is not actually done, but I have to call it, otherwise the Q join function will be hung up.
                    self.queue_comm.task_done()
                    continue

                # self.queue_comm.task_done()

                # if not self.get_thread_status():
                #    self.logger.debug(
                #        "Thread COMM of track#{} has stopped.".format(self.track_id))
                #    break

        else:
            self.queue_comm.task_done()
            print("!> Client config is not set. System is based on local ReID for track#{}".format(
                self.track_id))
