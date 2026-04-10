"""Conveyor Stop Signal ROS2 Node.

Subscribes to a flag_event and sends a Modbus TCP stop command to a PLC
when the flag matches the configured trigger value.

Uses pymodbus for PLC communication. If pymodbus is unavailable (dev mode),
the node logs the intended command and continues without crashing.
"""

import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    from pymodbus.client import ModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False


class ConveyorStopNode(Node):
    def __init__(self):
        super().__init__('conveyor_stop')

        self.trigger_on_true = os.environ.get('PARAM_TRIGGER_ON_TRUE', 'true').lower() == 'true'
        plc_address = os.environ.get('PARAM_PLC_ADDRESS', '192.168.1.100:502')
        self.register_address = int(os.environ.get('PARAM_REGISTER_ADDRESS', '0'))
        self.cooldown_s = int(os.environ.get('PARAM_COOLDOWN_MS', '2000')) / 1000.0

        host, port = plc_address.rsplit(':', 1) if ':' in plc_address else (plc_address, '502')
        self._plc_host = host
        self._plc_port = int(port)
        self._last_trigger_time = 0.0

        if not MODBUS_AVAILABLE:
            self.get_logger().warn(
                'pymodbus not installed — PLC commands will be logged but NOT sent. '
                'Install pymodbus to enable real PLC control.'
            )

        self.sub = self.create_subscription(Bool, 'flag_in', self._on_flag, 10)

        self.get_logger().info(
            f'Conveyor Stop ready. plc={self._plc_host}:{self._plc_port} '
            f'register={self.register_address} trigger_on_true={self.trigger_on_true}'
        )

    def _on_flag(self, msg: Bool):
        if msg.data != self.trigger_on_true:
            return

        now = time.monotonic()
        if now - self._last_trigger_time < self.cooldown_s:
            self.get_logger().debug('Stop suppressed by cooldown.')
            return

        self._last_trigger_time = now
        self._send_stop()

    def _send_stop(self):
        if not MODBUS_AVAILABLE:
            self.get_logger().info(
                f'[DRY RUN] Would write coil=True to {self._plc_host}:{self._plc_port} '
                f'register={self.register_address}'
            )
            return

        try:
            client = ModbusTcpClient(self._plc_host, port=self._plc_port)
            if client.connect():
                client.write_coil(self.register_address, True)
                client.close()
                self.get_logger().info('Stop signal sent to PLC.')
            else:
                self.get_logger().error(
                    f'Could not connect to PLC at {self._plc_host}:{self._plc_port}'
                )
        except Exception as e:
            self.get_logger().error(f'PLC communication error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ConveyorStopNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
