import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import os
import time

RIG = os.environ.get('RIG', 'jhu_dVRK')  # falls back if $RIG isn't set

class SyncedCapture(Node):
    def __init__(self):
        super().__init__('synced_capture')
        self.bridge = CvBridge()
        self.out_dir = 'charuco_dataset_new'
        os.makedirs(self.out_dir, exist_ok=True)
        self.count = 0
        self.last_save = 0.0
        self.interval = 2.0  # seconds between captures

        left_topic = f'/{RIG}/left/image_raw'
        right_topic = f'/{RIG}/right/image_raw'
        self.get_logger().info(f'Subscribing to {left_topic} and {right_topic}')
        self.get_logger().info(f'Saving pairs to ./{self.out_dir}/ as left-NNNN.png / right-NNNN.png')

        left_sub = message_filters.Subscriber(self, Image, left_topic)
        right_sub = message_filters.Subscriber(self, Image, right_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info('Waiting for synced image pairs...')

    def callback(self, left_msg, right_msg):
        now = time.time()
        if now - self.last_save < self.interval:
            return
        self.last_save = now

        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        lfname = f'{self.out_dir}/left-{self.count:04d}.png'
        rfname = f'{self.out_dir}/right-{self.count:04d}.png'
        cv2.imwrite(lfname, left_img)
        cv2.imwrite(rfname, right_img)
        self.get_logger().info(f'Saved pair {self.count}: {lfname}, {rfname}')
        self.count += 1

def main():
    rclpy.init()
    node = SyncedCapture()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
