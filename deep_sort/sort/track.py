# vim: expandtab:ts=4:sw=4

#import asyncio
#from distutils import command
import cv2
import zlib
import requests
import aiohttp
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

    def __init__(self, cmd_type, feature, img_bbox=None) -> None:
        self.cmd_type = cmd_type
        self.feature = feature
        self.response = None
        self.img_response = None

        if img_bbox is not None:
            self.img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR)

        self.cmp_feature = None
        self.cmp_image = None

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
        client_cfg=None,
        org_img=None,
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

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.cls_id = cls_id

        self.client_cfg = client_cfg

        self.filtered_classes = filtered_classes

        self._queue_comm = Queue(self.client_cfg.Budget)
        img_bbox = None
        if self.client_cfg.DeepSORT.Send_BBOX_IMG and org_img is not None:
            img_bbox = self._get_bbox_img(org_img)

        self._queue_comm.put_nowait(Command(Command.FETCH, feature, img_bbox))

        self._run_thread = True
        self._thread_comm = Thread(target=self.communicate)
        self._thread_comm.start()

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

    def _get_bbox_img(self, org_img):
        height, width = org_img.shape[:2]
        x, y, w, h = self.to_tlwh()
        x1 = max(int(x), 0)
        x2 = min(int(x+w), width-1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), height-1)
        return org_img[y1:y2, x1:x2]

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

    def update(self, kf, detection, org_img):
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
                    img_bbox = None
                    if self.client_cfg.DeepSORT.Send_BBOX_IMG and org_img is not None:
                        img_bbox = self._get_bbox_img(org_img)

                    self._queue_comm.put_nowait(
                        Command(Command.UPDATE, detection.feature, img_bbox))

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
        print('Sending the data for track: {}'.format(self.track_id))
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
                    # TODO: Fix the error handling and logging.
                    cmd.response = response.json()
            except Exception as e:
                print("The error is : {}".format(str(e)))

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
                            # TODO: Fix the error handling and logging.
                            cmd.img_response = response.json()
                    except Exception as e:
                        print("The error is : {}".format(str(e)))
            else:
                print("There was an error on server side to save the data.")

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

    def communicate(self):
        if self.client_cfg is not None:
            while True and self._run_thread:
                cmd = self._queue_comm.get(block=True)
                if cmd.cmd_type == Command.STOP:
                    break

                # Let's filter out the objects we want to the server
                if self.get_cls_id() in self.filtered_classes:
                    cmd = self.compress_nparr(cmd)
                    cmd = self.post_nparr(cmd)
                    self.process_response(cmd)

                self._queue_comm.task_done()
        else:
            print("!> Client config is not set. System is based on local ReID for track:{}".format(
                self.track_id))
