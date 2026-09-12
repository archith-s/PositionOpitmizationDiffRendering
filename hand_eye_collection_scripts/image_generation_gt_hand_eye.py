#!/usr/bin/env python3

# Captures synced left/right image pairs (same approach as
# charuco_image_generation.py) together with the dVRK gripper pose at
# capture time, using an existing hand-eye calibration (as loaded by
# dvrk_camera_registration's vis_gripper_pose.py) to also express that
# pose in the camera frame. Produces a dataset of image pairs + a JSON
# file of ground-truth gripper poses for evaluating hand-eye calibration.

import argparse
import json
import os
import sys
import time

import cv2
import cv_bridge
import message_filters
import numpy
import sensor_msgs.msg
from scipy.spatial.transform import Rotation

import crtk
import dvrk_camera_registration

RIG = os.environ.get('RIG', 'jhu_dVRK')  # falls back if $RIG isn't set


def load_hand_eye_calibration(json_file: str) -> numpy.ndarray:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return numpy.array(data['base-frame']['transform']).reshape(4, 4)


def measured_cp_to_matrix(m_cp) -> numpy.ndarray:
    rotation = Rotation.from_quat(m_cp.M.GetQuaternion())
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = numpy.float64(rotation.as_matrix())
    matrix[0, 3] = m_cp.p[0]
    matrix[1, 3] = m_cp.p[1]
    matrix[2, 3] = m_cp.p[2]
    return matrix


def draw_axis(img, camera_matrix, dist_coeffs, pose, size=0.01):
    # Mirrors dvrk_camera_registration's vis_gripper_pose.py draw_axis, so the
    # overlay reflects the exact same projection math that script uses.
    thickness = 2
    rotation, translation = pose[:3, :3], pose[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    points = numpy.float32([[size, 0, 0], [0, size, 0], [0, 0, size], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rvec, translation, camera_matrix, dist_coeffs)
    axis_points = axis_points.astype(int)

    origin = tuple(int(v) for v in axis_points[3].ravel())
    img = cv2.line(img, origin, tuple(axis_points[0].ravel()), (255, 0, 0), thickness)
    img = cv2.line(img, origin, tuple(axis_points[1].ravel()), (0, 255, 0), thickness)
    img = cv2.line(img, origin, tuple(axis_points[2].ravel()), (0, 0, 255), thickness)
    return img, origin


class GroundTruthCapture:
    def __init__(self, ral, arm_handle, cam_T_base: numpy.ndarray, out_dir: str, interval: float,
                 startup_delay: float = 5.0):
        self.arm_handle = arm_handle
        self.cam_T_base = cam_T_base
        self.out_dir = out_dir
        self.interval = interval
        self.startup_delay = startup_delay
        self.start_time = time.time()
        os.makedirs(self.out_dir, exist_ok=True)

        self.gt_json_path = os.path.join(self.out_dir, 'gripper_ground_truth.json')
        self.records = self._load_existing_records()
        self.count = len(self.records)
        self.last_save = 0.0

        self.bridge = cv_bridge.CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = numpy.zeros((5, 1))

        left_topic = f'/{RIG}/left/image_raw'
        right_topic = f'/{RIG}/right/image_raw'
        # The hand-eye calibration (and vis_gripper_pose.py, which this script's
        # projection math mirrors) is defined relative to the RECTIFIED right
        # camera frame, projected with its P matrix and zero distortion. Raw
        # image_raw frames are still distorted, so projecting onto them with
        # that same math would be wrong (and could land points off-screen) --
        # we additionally grab the rectified right image + camera_info just to
        # draw a debug overlay that is correctly aligned.
        right_rect_topic = f'/{RIG}/right/image_rect'
        right_info_topic = f'/{RIG}/right/camera_info'
        print(f'Subscribing to {left_topic}, {right_topic}, {right_rect_topic}, {right_info_topic}')
        print(f'Saving pairs + gripper ground truth to ./{self.out_dir}/')

        # crtk.ral has no public accessor for its underlying rclpy node,
        # but message_filters needs a real node to subscribe on.
        node = ral._node
        node.create_subscription(
            sensor_msgs.msg.CameraInfo, right_info_topic, self._info_callback, 10
        )
        left_sub = message_filters.Subscriber(node, sensor_msgs.msg.Image, left_topic)
        right_sub = message_filters.Subscriber(node, sensor_msgs.msg.Image, right_topic)
        right_rect_sub = message_filters.Subscriber(node, sensor_msgs.msg.Image, right_rect_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, right_rect_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

    def _info_callback(self, info_msg):
        projection_matrix = numpy.array(info_msg.p).reshape((3, 4))
        self.camera_matrix = projection_matrix[0:3, 0:3]

    def _load_existing_records(self):
        if os.path.exists(self.gt_json_path):
            with open(self.gt_json_path, 'r') as f:
                return json.load(f)
        return []

    def _save_records(self):
        with open(self.gt_json_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def callback(self, left_msg, right_msg, right_rect_msg):
        now = time.time()
        if now - self.start_time < self.startup_delay:
            return
        if now - self.last_save < self.interval:
            return
        if self.camera_matrix is None:
            print('Skipping frame, no camera_info received yet for right/image_rect')
            return

        # Read the arm pose before doing any file I/O: this callback runs on
        # crtk's single-threaded executor, so time spent here delays the
        # executor from servicing the measured_cp subscription, which can
        # make crtk consider the state stale and raise TimeoutError.
        try:
            m_cp, _ = self.arm_handle.local.measured_cp()
        except TimeoutError as e:
            print(f'Skipping frame, gripper pose not available: {e}')
            return

        self.last_save = now

        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        right_rect_img = self.bridge.imgmsg_to_cv2(right_rect_msg, desired_encoding='bgr8')

        lfname = f'left-{self.count:04d}.png'
        rfname = f'right-{self.count:04d}.png'
        rect_fname = f'right_rect-{self.count:04d}.png'
        overlay_fname = f'right_rect_overlay-{self.count:04d}.png'
        cv2.imwrite(os.path.join(self.out_dir, lfname), left_img)
        cv2.imwrite(os.path.join(self.out_dir, rfname), right_img)
        cv2.imwrite(os.path.join(self.out_dir, rect_fname), right_rect_img)

        base_T_gripper = measured_cp_to_matrix(m_cp)
        cam_T_gripper = self.cam_T_base @ base_T_gripper

        overlay_img, gripper_pixel = draw_axis(
            right_rect_img.copy(), self.camera_matrix, self.dist_coeffs, cam_T_gripper
        )
        overlay_img = cv2.circle(overlay_img, gripper_pixel, 6, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.out_dir, overlay_fname), overlay_img)

        self.records.append({
            'frame_index': self.count,
            'timestamp': now,
            'left_image': lfname,
            'right_image': rfname,
            'right_rect_image': rect_fname,
            'right_rect_overlay_image': overlay_fname,
            'base_T_gripper': base_T_gripper.tolist(),
            'cam_T_gripper': cam_T_gripper.tolist(),
            'gripper_pixel_right_rect': list(gripper_pixel),
        })
        self._save_records()

        print(f'Saved frame {self.count}: {lfname}, {rfname}, {rect_fname} (gripper px {gripper_pixel})')
        self.count += 1


def main():
    argv = crtk.ral.parse_argv(sys.argv[1:])  # skip argv[0], script name

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--psm-name', type=str, required=True, choices=['PSM1', 'PSM2', 'PSM3'],
        help='PSM name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace',
    )
    parser.add_argument(
        '-H', '--hand-eye-json', type=str, required=True,
        help='hand-eye calibration matrix in JSON format using OpenCV coordinate system',
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='gt_hand_eye_dataset',
        help='directory to save image pairs and gripper_ground_truth.json in',
    )
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='minimum time in seconds between captures',
    )
    parser.add_argument(
        '--startup-delay', type=float, default=5.0,
        help='seconds to wait after connecting before saving the first frame',
    )
    args = parser.parse_args(argv)

    ral = crtk.ral('image_generation_gt_hand_eye')
    # expected_interval is crtk's staleness timeout for measured_cp/measured_js.
    # The default (0.1s) is tuned for tight control loops; it's too strict here
    # since this node's single-threaded executor also does image file I/O.
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name=args.psm_name, expected_interval=0.5)
    ral.spin()
    ral.check_connections()

    cam_T_base = load_hand_eye_calibration(args.hand_eye_json)

    GroundTruthCapture(ral, arm_handle, cam_T_base, args.output_dir, args.interval, args.startup_delay)

    print(f'Waiting for synced image pairs... (capture starts in {args.startup_delay:.0f}s)')
    try:
        while not ral.is_shutdown():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    ral.shutdown()


if __name__ == '__main__':
    main()
