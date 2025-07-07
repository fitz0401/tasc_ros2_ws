#!/usr/bin/env python3
"""
WebSocket Server for AANP ROS2 Node
Provides communication interface with external AANP client running in conda environment
"""

import asyncio
import websockets
import json
import threading
import base64
import numpy as np
from typing import Set, Optional, Dict, Any, Callable
import logging
import time
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation

# ROS2 imports
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped


class AANPWebSocketServer:
    """WebSocket server class for AANP client communication"""
    
    def __init__(self, host="0.0.0.0", port=8765, logger=None, max_points=10000, max_clients=10):
        """
        Initialize WebSocket server
        
        Args:
            host: Server listening address
            port: Server port
            logger: ROS2 logger object
            max_points: Maximum number of points to send (for performance)
            max_clients: Maximum number of concurrent clients
        """
        self.host = host
        self.port = port
        self.logger = logger or self._setup_default_logger()
        self.max_points = max_points
        self.max_clients = max_clients
        
        # WebSocket server state
        self.server = None
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server_thread = None
        self.running = False
        self.loop = None
        
        # Data processing tools
        self.cv_bridge = CvBridge()
        
        # Current data cache
        self.current_eef_pose = np.eye(4)
        self.current_twist = np.zeros(6)
        self.current_gripper_action = 0
        self.pointcloud_data = None
        self.image_data = None
        self.sensor_data_available = False
        
        # Callback functions
        self.assist_action_callback: Optional[Callable] = None
        
    def _setup_default_logger(self):
        """Setup default logger"""
        logger = logging.getLogger("AANPWebSocketServer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def start_server(self):
        """Start WebSocket server"""
        if self.running:
            self.logger.warn("WebSocket server is already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        self.logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
    
    def stop_server(self):
        """Stop WebSocket server"""
        self.running = False
        if self.loop and self.server:
            asyncio.run_coroutine_threadsafe(self._stop_server_async(), self.loop)
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        
        self.logger.info("WebSocket server stopped")
    
    def _run_server(self):
        """Run WebSocket server in separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._start_server_async())
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
        finally:
            self.loop.close()
    
    async def _start_server_async(self):
        """Start server asynchronously"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB limit for large messages
            )
            self.logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
            await self.server.wait_closed()
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _stop_server_async(self):
        """Stop server asynchronously"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def handle_client(self, websocket):
        """Handle client connections"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        # Check client limit
        if len(self.connected_clients) >= self.max_clients:
            self.logger.warn(f"Maximum clients ({self.max_clients}) reached, rejecting {client_addr}")
            await websocket.close(code=1013, reason="Server overloaded")
            return
        
        self.connected_clients.add(websocket)
        self.logger.info(f"WebSocket client connected from {client_addr} ({len(self.connected_clients)}/{self.max_clients})")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_client_message(websocket, data)
                except json.JSONDecodeError:
                    self.logger.warn(f"Invalid JSON received from {client_addr}")
                    await self.send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_addr}: {e}")
                    await self.send_error(websocket, f"Processing error: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket client {client_addr} disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket client error from {client_addr}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def process_client_message(self, websocket, data):
        """Process client messages"""
        message_type = data.get("type")
        
        if message_type == "get_pointcloud_and_image":
            await self.send_pointcloud_and_image(websocket)
        elif message_type == "assist_action":
            await self.handle_assist_action(data)
        elif message_type == "test":
            await self.send_test_response(websocket, data)
        else:
            self.logger.warn(f"Unknown message type: {message_type}")
            await self.send_error(websocket, f"Unknown message type: {message_type}")
    
    async def send_pointcloud_and_image(self, websocket):
        """Send point cloud and image data"""
        if not self.sensor_data_available:
            await self.send_error(websocket, "No sensor data available")
            return
        
        try:
            # Check data validity
            if self.image_data is None or self.pointcloud_data is None:
                await self.send_error(websocket, "Invalid sensor data")
                return
            
            # Encode data as base64
            rgb_bytes = self.image_data.tobytes()
            points_bytes = self.pointcloud_data.tobytes()
            
            rgb_b64 = base64.b64encode(rgb_bytes).decode('utf-8')
            points_b64 = base64.b64encode(points_bytes).decode('utf-8')
            
            message = {
                "type": "pointcloud_and_image",
                "rgb": rgb_b64,
                "rgb_shape": list(self.image_data.shape),
                "points": points_b64,
                "points_shape": list(self.pointcloud_data.shape),
                "timestamp": self._get_timestamp()
            }
            
            await websocket.send(json.dumps(message))
            self.logger.info(f"Sent pointcloud and image data: RGB {self.image_data.shape}, Points {self.pointcloud_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error sending sensor data: {e}")
            await self.send_error(websocket, f"Failed to send sensor data: {str(e)}")
    
    async def handle_assist_action(self, data):
        """Handle assist action from client"""
        try:
            assist_action = np.array(data.get("assist_action", [0.0]*6), dtype=np.float64)
            gripper_action = data.get("gripper_action")
            
            # Validate assist action data
            if len(assist_action) != 6:
                self.logger.warn(f"Invalid assist action length: {len(assist_action)}, expected 6")
                return
            
            self.logger.info(f"Received assist action: {assist_action}, gripper: {gripper_action}")
            
            # Call callback function if available
            if self.assist_action_callback:
                self.assist_action_callback(assist_action, gripper_action)
            else:
                self.logger.warn("No assist action callback registered")
                
        except Exception as e:
            self.logger.error(f"Error handling assist action: {e}")
    
    async def send_test_response(self, websocket, data):
        """Send test response"""
        try:
            response = {
                "type": "test_response",
                "message": "Hello from ROS2 AANP WebSocket Server",
                "received": data.get("message", ""),
                "timestamp": self._get_timestamp()
            }
            await websocket.send(json.dumps(response))
            self.logger.info(f"Sent test response")
        except Exception as e:
            self.logger.error(f"Error sending test response: {e}")
            await self.send_error(websocket, f"Failed to send test response: {str(e)}")
    
    async def send_error(self, websocket, error_message):
        """Send error message"""
        try:
            error_msg = {
                "type": "error",
                "message": error_message,
                "timestamp": self._get_timestamp()
            }
            await websocket.send(json.dumps(error_msg))
        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    # ==== Data Update Methods ====
    
    def update_eef_pose(self, pose_stamped: PoseStamped):
        """Update end effector pose"""
        self.current_eef_pose = self._pose_to_matrix(pose_stamped)
    
    def update_twist(self, linear: np.ndarray, angular: np.ndarray):
        """Update twist data"""
        self.current_twist = np.concatenate([linear, angular])
    
    def update_gripper_action(self, action: int):
        """Update gripper action"""
        self.current_gripper_action = action
    
    def update_sensor_data(self, pointcloud_msg: PointCloud2, image_msg: Image):
        """Update sensor data (pointcloud and image)"""
        try:
            # Convert image
            self.image_data = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # Convert pointcloud with optimization for large data
            points_list = []
            point_count = 0
            
            for point in pc2.read_points(pointcloud_msg, 
                                       field_names=("x", "y", "z"), 
                                       skip_nans=True):
                if point_count >= self.max_points:
                    break
                points_list.append([point[0], point[1], point[2]])
                point_count += 1
            
            if len(points_list) > 0:
                self.pointcloud_data = np.array(points_list, dtype=np.float32)
                self.sensor_data_available = True
                self.logger.info(f"Updated sensor data: RGB {self.image_data.shape}, Points {self.pointcloud_data.shape}")
                
                if point_count >= self.max_points:
                    self.logger.info(f"Point cloud truncated to {self.max_points} points for performance")
            else:
                self.logger.warn("No valid points in point cloud")
                
        except Exception as e:
            self.logger.error(f"Error updating sensor data: {e}")
            self.sensor_data_available = False
    
    def _pose_to_matrix(self, pose_stamped: PoseStamped) -> np.ndarray:
        """Convert PoseStamped to 4x4 transformation matrix"""
        try:
            pos = pose_stamped.pose.position
            quat = pose_stamped.pose.orientation
            
            # Use scipy to convert quaternion to rotation matrix
            r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
            rotation_matrix = r.as_matrix()
            
            # Build 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [pos.x, pos.y, pos.z]
            
            return transform_matrix
        except Exception as e:
            self.logger.error(f"Error converting pose to matrix: {e}")
            return np.eye(4)
    
    # ==== Data Publishing Methods ====
    
    def publish_eef_pose(self):
        """Publish EEF pose to all connected clients"""
        if not self.connected_clients:
            return
        
        message = {
            "type": "eef_pose",
            "pose": self.current_eef_pose.tolist(),
            "timestamp": self._get_timestamp()
        }
        
        self._broadcast_message(message)
    
    def publish_twist(self):
        """Publish twist data to all connected clients"""
        if not self.connected_clients:
            return
        
        message = {
            "type": "twist",
            "twist": self.current_twist.tolist(),
            "timestamp": self._get_timestamp()
        }
        
        self._broadcast_message(message)
    
    def publish_gripper_action(self):
        """Publish gripper action to all connected clients"""
        if not self.connected_clients:
            return
        
        message = {
            "type": "gripper_action",
            "gripper_action": self.current_gripper_action,
            "timestamp": self._get_timestamp()
        }
        
        self._broadcast_message(message)
    
    def publish_all_data(self):
        """Publish all real-time data"""
        self.publish_eef_pose()
        self.publish_twist()
        self.publish_gripper_action()
    
    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all clients"""
        if not self.loop or not self.connected_clients:
            return
        
        # Execute broadcast in server event loop
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(message),
            self.loop
        )
    
    async def _broadcast_async(self, message: Dict[str, Any]):
        """Async broadcast message"""
        disconnected_clients = set()
        
        for client in self.connected_clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Clean up disconnected connections
        self.connected_clients -= disconnected_clients
        
        if disconnected_clients:
            self.logger.info(f"Removed {len(disconnected_clients)} disconnected clients")
    
    # ==== Callback Function Settings ====
    
    def set_assist_action_callback(self, callback):
        """Set assist action callback function"""
        self.assist_action_callback = callback
    
    # ==== Status Query Methods ====
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.running and self.server_thread and self.server_thread.is_alive()
    
    def get_connected_clients_count(self) -> int:
        """Get number of connected clients"""
        return len(self.connected_clients)
    
    def has_sensor_data(self) -> bool:
        """Check if sensor data is available"""
        return self.sensor_data_available
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "running": self.is_running(),
            "host": self.host,
            "port": self.port,
            "connected_clients": self.get_connected_clients_count(),
            "has_sensor_data": self.has_sensor_data(),
            "server_thread_alive": self.server_thread.is_alive() if self.server_thread else False
        }

# Test and demo code
if __name__ == "__main__":
    import time
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create server instance
    server = AANPWebSocketServer(host="localhost", port=8765)
    
    # Mock assist action callback
    def assist_action_callback(assist_action, gripper_action):
        print(f"Received assist action: {assist_action}, gripper: {gripper_action}")
    
    server.set_assist_action_callback(assist_action_callback)
    
    try:
        # Start server
        server.start_server()
        
        # Simulate data updates
        for i in range(10):
            # Simulate EEF pose update
            server.current_eef_pose = np.eye(4)
            server.current_eef_pose[0, 3] = i * 0.1  # Move x position
            
            # Simulate twist data
            server.current_twist = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1])
            
            # Simulate gripper action
            server.current_gripper_action = i % 2
            
            # Publish data
            server.publish_all_data()
            
            print(f"Published data iteration {i}, clients: {server.get_connected_clients_count()}")
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop_server()
        print("Server stopped")
