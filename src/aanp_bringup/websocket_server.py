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
from typing import Set, Optional, Dict, Any, Callable, Tuple
import logging
import time
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

# ROS2 imports
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf2_ros import TransformException
from rclpy.duration import Duration
import tf2_ros as tf2


class AANPWebSocketServer:
    """WebSocket server class for AANP client communication"""
    
    def __init__(self, host="0.0.0.0", port=8765, logger=None, max_clients=10, 
                 tf_buffer=None, target_frame="fr3_link0", 
                 camera_frame_id="camera_depth_optical_frame"):
        """
        Initialize WebSocket server
        
        Args:
            host: Server listening address
            port: Server port
            logger: ROS2 logger object
            max_clients: Maximum number of concurrent clients
            tf_buffer: TF2 buffer for coordinate transformations
            target_frame: Target frame for point cloud transformation
            camera_frame_id: Frame ID for the camera depth frame
        """
        self.host = host
        self.port = port
        self.logger = logger or self._setup_default_logger()
        self.max_clients = max_clients
        self.tf_buffer = tf_buffer
        self.target_frame = target_frame
        self.camera_frame_id = camera_frame_id
        
        # Camera parameters for depth to point cloud conversion
        # Initialize with defaults, will be updated from camera_info topic
        self.setup_camera_parameters()
        
        # WebSocket server state
        self.server = None
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server_thread = None
        self.running = False
        self.loop = None
        
        # Data processing tools
        self.cv_bridge = CvBridge()
        
        # TF transformation cache for performance
        self.cached_transform = None
        self.cached_transform_time = 0
        self.transform_cache_duration = 5.0  # Cache transform for 5 seconds
        
        # Current data cache
        self.current_eef_pose = np.eye(4)
        self.current_twist = np.zeros(6)
        self.current_gripper_action = 0
        self.pointcloud_data = None
        self.image_data = None
        self.sensor_data_available = False
        
        # Store latest raw sensor messages for on-demand processing
        self.latest_depth_msg = None
        self.latest_image_msg = None
        self.raw_sensor_data_available = False
        
        # Callback functions
        self.assist_action_callback: Optional[Callable] = None
        
    def setup_camera_parameters(self):
        """Setup camera parameters for depth to point cloud conversion with defaults"""
        # Default camera intrinsics (RealSense L515 typical values)
        self.camera_intrinsics = np.array([
            [602.9156, 0.0, 328.1451],
            [0.0, 603.0374, 244.0271],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Default camera extrinsics (identity - no transformation)
        # TF system handles coordinate transformations
        self.camera_extrinsics = np.eye(4, dtype=np.float32)
        
        # Extract focal lengths and principal points for depth to point conversion
        self.fx = self.camera_intrinsics[0, 0]
        self.fy = self.camera_intrinsics[1, 1]
        self.cx = self.camera_intrinsics[0, 2]
        self.cy = self.camera_intrinsics[1, 2]
        
        self.logger.info(f"Initialized default camera parameters: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        self.logger.debug(f"Camera intrinsics:\n{self.camera_intrinsics}")
        self.logger.debug(f"Camera extrinsics:\n{self.camera_extrinsics}")
    
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
                    self.logger.warn(f"Invalid JSON received from {client_addr}, received message: {message}")
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
        elif message_type == "heartbeat":
            await self.handle_heartbeat(websocket, data)
        else:
            self.logger.warn(f"Unknown message type: {message_type}")
            await self.send_error(websocket, f"Unknown message type: {message_type}")
    
    async def send_pointcloud_and_image(self, websocket):
        """Send point cloud and image data - process on demand"""
        if not self.raw_sensor_data_available:
            await self.send_error(websocket, "No sensor data available")
            return
        
        try:
            # Process sensor data on demand when client requests it
            self.update_sensor_data(self.latest_depth_msg, self.latest_image_msg)
            
            # Check data validity after processing
            if self.image_data is None or self.pointcloud_data is None:
                await self.send_error(websocket, "Failed to process sensor data")
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
            
            self.logger.debug(f"Received assist action: {assist_action}, gripper: {gripper_action}")
            
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
    
    async def handle_heartbeat(self, websocket, data):
        """Handle heartbeat from client"""
        try:
            client_timestamp = data.get("timestamp", 0)
            response = {
                "type": "heartbeat_response",
                "server_timestamp": self._get_timestamp(),
                "client_timestamp": client_timestamp,
                "message": "pong"
            }
            await websocket.send(json.dumps(response))
            self.logger.debug(f"Responded to heartbeat from client")
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {e}")
            # Don't send error response for heartbeat failures to avoid noise
    
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
    
    def store_raw_sensor_data(self, depth_msg: Image, image_msg: Image):
        """Store raw sensor messages for on-demand processing"""
        self.latest_depth_msg = depth_msg
        self.latest_image_msg = image_msg
        self.raw_sensor_data_available = True
    
    def update_sensor_data(self, depth_msg: Image, image_msg: Image):
        """Update sensor data (depth and RGB image) - generate ordered H×W×3 point cloud from depth"""
        try:
            # Convert RGB image
            self.image_data = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_height, image_width = self.image_data.shape[:2]
            
            # Convert depth image
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            # Log image information for debugging
            self.logger.debug(f"RGB image: {self.image_data.shape}, dtype: {self.image_data.dtype}")
            self.logger.debug(f"Depth image: {depth_image.shape}, dtype: {depth_image.dtype}")
            self.logger.debug(f"Depth range: {np.min(depth_image)} - {np.max(depth_image)}")
            
            # Check if images are from the same timestamp (within tolerance)
            time_diff = abs(depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9 - 
                           image_msg.header.stamp.sec - image_msg.header.stamp.nanosec * 1e-9)
            if time_diff > 0.1:  # 100ms tolerance
                self.logger.warn(f"Large timestamp difference between depth and RGB: {time_diff:.3f}s")
            
            # Ensure depth image matches RGB image dimensions
            if depth_image.shape[:2] != (image_height, image_width):
                self.logger.warn(f"Depth and RGB image size mismatch: depth {depth_image.shape[:2]}, rgb {(image_height, image_width)}")
                # Resize depth to match RGB if needed
                import cv2
                depth_image = cv2.resize(depth_image, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                self.logger.info(f"Resized depth image to match RGB: {depth_image.shape}")
            
            # Generate ordered point cloud from depth image
            self.pointcloud_data = self._depth_to_pointcloud(depth_image, depth_msg.header)
            
            if self.pointcloud_data is not None:
                self.sensor_data_available = True
                self.logger.debug(f"Updated sensor data: RGB {self.image_data.shape}, Points {self.pointcloud_data.shape}")
                
                # Additional validation: check if point cloud and image have same spatial dimensions
                if self.pointcloud_data.shape[:2] != self.image_data.shape[:2]:
                    self.logger.error(f"Point cloud and image spatial dimensions mismatch: "
                                    f"points {self.pointcloud_data.shape[:2]}, image {self.image_data.shape[:2]}")
            else:
                self.logger.warn("Failed to generate point cloud from depth image")
                self.sensor_data_available = False
                
        except Exception as e:
            self.logger.error(f"Error updating sensor data: {e}")
            self.sensor_data_available = False
    
    def _depth_to_pointcloud(self, depth_image: np.ndarray, header) -> Optional[np.ndarray]:
        """
        Convert depth image to ordered point cloud using camera intrinsics
        
        Args:
            depth_image: Depth image as numpy array (H, W)
            header: Header from depth image message for frame info
            
        Returns:
            Point cloud as numpy array of shape (H, W, 3) in camera coordinates or target frame
        """
        try:
            # Check if camera intrinsics are valid (not just defaults)
            # We assume valid intrinsics have been updated from camera_info topic
            if self.camera_intrinsics is None or self.fx <= 0 or self.fy <= 0:
                self.logger.warn("Invalid camera intrinsics - point cloud generation skipped")
                return None
            
            height, width = depth_image.shape[:2]
            
            # Create coordinate grids
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert depth values to meters if needed (assuming depth is in millimeters)
            # This depends on your camera configuration - adjust as needed
            if depth_image.dtype == np.uint16:
                # Common for RealSense: depth in millimeters, convert to meters
                depth_m = depth_image.astype(np.float32) / 1000.0
            else:
                # Assume already in meters
                depth_m = depth_image.astype(np.float32)
            
            # Create boolean mask for valid depth values
            valid_mask = (depth_m > 0) & (depth_m < 10.0)  # Filter out invalid depths
            
            # Convert pixel coordinates to 3D points using camera intrinsics
            # Camera coordinate system: X right, Y down, Z forward
            x = (u - self.cx) * depth_m / self.fx
            y = (v - self.cy) * depth_m / self.fy
            z = depth_m
            
            # Stack coordinates to create point cloud
            points = np.stack([x, y, z], axis=-1)  # Shape: (H, W, 3)
            
            # Set invalid points to NaN
            points[~valid_mask] = np.nan
            
            # Debug: Log point cloud statistics in camera frame
            valid_points_flat = points[valid_mask]
            if len(valid_points_flat) > 0:
                self.logger.debug(f"Generated {len(valid_points_flat)} valid points in camera frame")
                sample_points = valid_points_flat[:5] if len(valid_points_flat) >= 5 else valid_points_flat
                self.logger.debug(f"Sample points (camera frame): {sample_points}")
            
            # Apply camera extrinsics transformation if provided
            if not np.allclose(self.camera_extrinsics, np.eye(4)):
                points = self._apply_extrinsics_transformation(points)
                self.logger.debug("Applied camera extrinsics transformation")
            
            # Transform to target frame if needed
            if self.tf_buffer is not None and self.target_frame != self.camera_frame_id:
                transformed_points = self._transform_point_cloud_structured(points, header, self.target_frame)
                if transformed_points is not None:
                    points = transformed_points
                    self.logger.debug(f"Point cloud transformed from {self.camera_frame_id} to {self.target_frame}")
                    
                    # Debug: Log statistics after transformation
                    valid_transformed = points[~np.isnan(points).any(axis=2)]
                    if len(valid_transformed) > 0:
                        mean_z = np.nanmean(valid_transformed[:, 2])
                        self.logger.debug(f"Mean Z coordinate after transformation: {mean_z:.3f}m")
                else:
                    self.logger.warn(f"TF transformation failed, using points in {self.camera_frame_id}")
            
            return points.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error converting depth to point cloud: {e}")
            return None
    
    def _apply_extrinsics_transformation(self, points: np.ndarray) -> np.ndarray:
        """
        Apply camera extrinsics transformation to point cloud
        
        Args:
            points: Point cloud of shape (H, W, 3)
            
        Returns:
            Transformed point cloud of shape (H, W, 3)
        """
        try:
            original_shape = points.shape
            H, W, _ = original_shape
            
            # Reshape to (H*W, 3) for transformation
            points_flat = points.reshape(-1, 3)
            
            # Find valid points (not NaN)
            valid_mask = ~np.isnan(points_flat).any(axis=1)
            valid_points = points_flat[valid_mask]
            
            if len(valid_points) > 0:
                # Convert to homogeneous coordinates
                valid_points_hom = np.column_stack([valid_points, np.ones(len(valid_points))])
                
                # Apply extrinsics transformation
                transformed_points_hom = (self.camera_extrinsics @ valid_points_hom.T).T
                transformed_points = transformed_points_hom[:, :3]
                
                # Put transformed points back
                points_flat[valid_mask] = transformed_points.astype(np.float32)
            
            # Reshape back to original format
            return points_flat.reshape(original_shape)
            
        except Exception as e:
            self.logger.error(f"Error applying extrinsics transformation: {e}")
            return points
    
    def _transform_point_cloud_structured(self, points: np.ndarray, header, target_frame: str) -> Optional[np.ndarray]:
        """
        Transform structured point cloud (H×W×3) from source frame to target frame
        
        Args:
            points: Input point cloud as numpy array of shape (H, W, 3)
            header: Header from original message with frame_id and timestamp
            target_frame: Target frame for transformation
            
        Returns:
            Transformed point cloud as numpy array of shape (H, W, 3), or None if transformation failed
        """
        try:
            current_time = time.time()
            source_frame = self.camera_frame_id  # Use camera frame ID instead of header frame_id
            
            # Check if we can use cached transform (for static transforms)
            if (self.cached_transform is not None and 
                current_time - self.cached_transform_time < self.transform_cache_duration):
                
                translation = self.cached_transform['translation']
                rotation_matrix = self.cached_transform['rotation_matrix']
                self.logger.debug("Using cached transform for structured point cloud transformation")
            else:
                # Get fresh transformation from TF
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    header.stamp,
                    timeout=Duration(seconds=0.1)
                )
                
                # Extract translation and rotation
                translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                
                quat = transform.transform.rotation
                rotation = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
                rotation_matrix = rotation.as_matrix()
                
                # Cache the transform for future use
                self.cached_transform = {
                    'translation': translation,
                    'rotation_matrix': rotation_matrix
                }
                self.cached_transform_time = current_time
                self.logger.debug("Cached new transform for structured point cloud transformation")
            
            # Get original shape
            original_shape = points.shape
            H, W, _ = original_shape
            
            # Reshape to (H*W, 3) for transformation, but keep track of valid points
            points_flat = points.reshape(-1, 3)
            
            # Find valid points (not NaN)
            valid_mask = ~np.isnan(points_flat).any(axis=1)
            valid_points = points_flat[valid_mask]
            
            if len(valid_points) > 0:
                # Apply transformation to valid points only
                transformed_valid = (rotation_matrix @ valid_points.T).T + translation
                
                # Put transformed points back into the flat array
                points_flat[valid_mask] = transformed_valid.astype(np.float32)
            
            # Reshape back to original H×W×3 format
            transformed_points = points_flat.reshape(original_shape)
            
            return transformed_points
            
        except (TransformException, tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            self.logger.warn(f"Failed to transform structured point cloud from {source_frame} to {target_frame}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in structured point cloud transformation: {e}")
            return None
    
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
            self.logger.debug(f"Removed {len(disconnected_clients)} disconnected clients")

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
    
    def update_camera_intrinsics_from_camera_info(self, camera_info_msg):
        """
        Update camera intrinsics from CameraInfo message
        
        Args:
            camera_info_msg: sensor_msgs/CameraInfo message
        """
        try:
            # Extract intrinsic matrix from CameraInfo
            K = camera_info_msg.k  # 3x3 intrinsic matrix as flat array
            self.camera_intrinsics = np.array([
                [K[0], K[1], K[2]],
                [K[3], K[4], K[5]], 
                [K[6], K[7], K[8]]
            ], dtype=np.float32)
            
            # Update focal lengths and principal points
            self.fx = self.camera_intrinsics[0, 0]
            self.fy = self.camera_intrinsics[1, 1]
            self.cx = self.camera_intrinsics[0, 2]
            self.cy = self.camera_intrinsics[1, 2]
            
            self.logger.info(f"Updated camera intrinsics from CameraInfo: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            self.logger.info(f"Image size: {camera_info_msg.width}x{camera_info_msg.height}")
            
        except Exception as e:
            self.logger.error(f"Failed to update camera intrinsics from CameraInfo: {e}")

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
