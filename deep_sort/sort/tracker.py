# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from queue import Empty
from threading import Semaphore
import numpy as np
from threading import Thread
from time import sleep
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, logger=None, client_cfg=None):
        self.metric = metric
        # print("tracker.metric:=-=-=--=",self.metric)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        '''
        self.tracks gradually will contain only confirmed tracks. We need
        to keep track of all `tracks` so we can join the COMM thread and Q.
        '''
        self.all_tracks = []
        self._next_id = 1

        #self.event_loop = asyncio.new_event_loop()
        #self._thread_event_loop = Thread(target=self._run_comm_async)
        # self._thread_event_loop.start()

        self.client_cfg = client_cfg

        self.logger = logger

        self.trackers_lock = Semaphore()
        self.read_trackers_status_lock = Semaphore()
        self.read_all_tracks = False
        self._run_thread_collector = True
        self._thread_comm_collector = Thread(
            target=self.deleted_track_collector, name="Track_Thread_Collector")
        self._thread_comm_collector.start()

    # def _run_comm_async(self):
    #    asyncio.set_event_loop(self.event_loop)
    #    self.event_loop.run_forever()
    #    self._thread_event_loop.join()

    def should_read_all_tracks(self):
        self.read_trackers_status_lock.acquire()
        status = self.read_all_tracks
        self.read_trackers_status_lock.release()
        return status

    def update_read_all_tracks_status(self, status):
        self.read_trackers_status_lock.acquire()
        self.read_all_tracks = status
        self.read_trackers_status_lock.release()

    def shuting_down_the_tracker(self):
        self.logger.info('Shouting down tracker...')

        # self.trackers_lock.acquire()
        # for track in self.all_tracks:
        # self.logger.debug(
        #    'Inserting None into Q of track#{}'.format(track.track_id))
        # track.queue_comm.put(None)
        # self.trackers_lock.release()

        self.update_read_all_tracks_status(True)

        live_process = []
        self.trackers_lock.acquire()
        for track in self.all_tracks:
            live_process.append(track._thread_comm)
        self.trackers_lock.release()

        while live_process:
            live_process = [p for p in live_process if p.is_alive()]
            # Let's give time to queue to see None

        self.logger.info('Joining the tracker thread...')
        while self._thread_comm_collector.is_alive():
            self._thread_comm_collector.join(0.1)

    def update_all_track_list(self, val):
        self.trackers_lock.acquire()
        if isinstance(val, list):
            self.all_tracks = val
        else:
            self.all_tracks.append(val)
        self.trackers_lock.release()

    def deleted_track_collector(self):
        while self._run_thread_collector:
            tmp_track_list = []
            should_be_added = True
            self.trackers_lock.acquire()
            current_track = self.all_tracks
            self.trackers_lock.release()

            for track in current_track:
                if track.is_deleted() or self.should_read_all_tracks():
                    self.logger.debug(
                        'Stopping COMM thread for track#{}'.format(track.track_id))
                    track.logger.debug('Queue for track#{} has {} members.'.format(
                        track.track_id, track.queue_comm.qsize()))
                    
                    if self.should_read_all_tracks():
                        if not track.get_to_joined_forcefully():
                            track.set_to_joined_forcefully(True)
                            self.logger.debug(
                                'Inserting None into Q of track#{}'.format(track.track_id))
                            track.queue_comm.put(None, block=True)

                    if track.queue_comm.empty():
                        track.logger.debug(
                            'Trying to join the queue of track#{}.'.format(track.track_id))
                        track.queue_comm.join()
                        # I've added these sleep hear to make sure there is not race between `get` and `join` function of Q to get its thread lock.
                        sleep(1)
                        track.logger.debug(
                            'Trying to join the COMM thread of track#{}.'.format(track.track_id))
                        track._thread_comm.join(0.1)
                        should_be_added = False
                        track.logger.debug(
                            'The COMM thread of track#{} is officially dead!'.format(track.track_id))
                    else:
                        # There are some cases that thread is full but `get` is blocking.
                        track.logger.debug(
                            "Couldn't stop queue of track#{} as it was not empty. It will be tried later".format(track.track_id))
                        #while not track.queue_comm.empty():
                        #    try:
                        #        _ = track.queue_comm.get_nowait()
                        #        track.queue_comm.task_done()
                        #    except Empty:
                        #        break
                if should_be_added:
                    tmp_track_list.append(track)

            if self.should_read_all_tracks() and len(tmp_track_list) == 0:
                self._run_thread_collector = False

            # if len(tmp_track_list) > 0:
            #    self.update_all_track_list(tmp_track_list)
            sleep(0.1)

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # print('tracker.py, predict, predict:', type(self.tracks), len(self.tracks))
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # print(type(detections))
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        # print("tracker.matches:=-=-=--=",matches)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # print("tracker.gated metric.dets",type(track_indices))
            features = np.array([dets[i].feature for i in detection_indices])
            # print(features)
            targets = np.array([tracks[i].track_id for i in track_indices])
            # print("-------------=================")
            # print(targets)
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())

        track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature, detection.cls_id, self.logger, self.client_cfg)

        self.tracks.append(track)
        self.update_all_track_list(track)
        self._next_id += 1
