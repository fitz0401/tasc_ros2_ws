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
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf2_ros import TransformException
from rclpy.duration import Duration


class AANPWebSocketServer:
    """WebSocket server class for AANP client communication"""
    
    def __init__(self, host="0.0.0.0", port=8765, logger=None, max_points=10000, max_clients=10, 
                 sampling_method="random", voxel_size=0.01, tf_buffer=None, target_frame="fr3_link0",
                 workspace_bounds=None):
        """
        Initialize WebSocket server
        
        Args:
            host: Server listening address
            port: Server port
            logger: ROS2 logger object
            max_points: Maximum number of points to send (for performance)
            max_clients: Maximum number of concurrent clients
            sampling_method: "random" for uniform random sampling, "voxel" for voxel grid sampling
            voxel_size: Size of voxel grid for voxel sampling (in meters)
            tf_buffer: TF2 buffer for coordinate transformations
            target_frame: Target frame for point cloud transformation
            workspace_bounds: Dict with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max' for filtering
        """
        self.host = host
        self.port = port
        self.logger = logger or self._setup_default_logger()
        self.max_points = max_points
        self.max_clients = max_clients
        self.sampling_method = sampling_method
        self.voxel_size = voxel_size
        self.tf_buffer = tf_buffer
        self.target_frame = target_frame
        
        # Default workspace bounds (in target frame coordinates)
        if workspace_bounds is None:
            self.workspace_bounds = {
                'x_min': -0.2, 'x_max': 0.8,
                'y_min': -0.2, 'y_max': 0.6,
                'z_min': -0.1, 'z_max': 0.8
            }
        else:
            self.workspace_bounds = workspace_bounds
        
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
        self.latest_pointcloud_msg = None
        self.latest_image_msg = None
        self.raw_sensor_data_available = False
        
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
        """Send point cloud and image data - process on demand"""
        if not self.raw_sensor_data_available:
            await self.send_error(websocket, "No sensor data available")
            return
        
        try:
            # Process sensor data on demand when client requests it
            self.update_sensor_data(self.latest_pointcloud_msg, self.latest_image_msg)
            
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
    
    def store_raw_sensor_data(self, pointcloud_msg: PointCloud2, image_msg: Image):
        """Store raw sensor messages for on-demand processing"""
        self.latest_pointcloud_msg = pointcloud_msg
        self.latest_image_msg = image_msg
        self.raw_sensor_data_available = True
    
    def update_sensor_data(self, pointcloud_msg: PointCloud2, image_msg: Image):
        """Update sensor data (pointcloud and image)"""
        try:
            # Convert image
            self.image_data = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # Step 1: Collect all valid points
            all_points = []
            for point in pc2.read_points(pointcloud_msg, 
                                       field_names=("x", "y", "z"), 
                                       skip_nans=True):
                all_points.append([point[0], point[1], point[2]])
            
            if len(all_points) > 0:
                all_points = np.array(all_points, dtype=np.float32)
                self.logger.debug(f"Collected {len(all_points)} initial points")
                
                # Step 2: Apply coordinate transformation first (if needed)
                if self.tf_buffer is not None and self.target_frame != pointcloud_msg.header.frame_id:
                    transformed_points = self._transform_point_cloud(all_points, pointcloud_msg.header, self.target_frame)
                    if transformed_points is not None:
                        all_points = transformed_points
                        self.logger.debug(f"Point cloud transformed from {pointcloud_msg.header.frame_id} to {self.target_frame}")
                
                # Step 3: Apply workspace filtering early to reduce point count
                workspace_filtered_points = self._filter_workspace_points(all_points)
                if len(workspace_filtered_points) < len(all_points):
                    self.logger.debug(f"Workspace filter: {len(all_points)} -> {len(workspace_filtered_points)} points")
                
                # Step 4: Apply downsampling strategy
                if len(workspace_filtered_points) > self.max_points:
                    if self.sampling_method == "random":
                        # Strategy 1: Uniform random sampling
                        indices = np.random.choice(len(workspace_filtered_points), self.max_points, replace=False)
                        final_points = workspace_filtered_points[indices]
                        self.logger.debug(f"Random sampling: {len(workspace_filtered_points)} -> {len(final_points)} points")
                    elif self.sampling_method == "voxel":
                        # Strategy 2: Voxel grid sampling
                        final_points = self._voxel_grid_downsample(workspace_filtered_points, self.voxel_size)
                        if len(final_points) > self.max_points:
                            indices = np.random.choice(len(final_points), self.max_points, replace=False)
                            final_points = final_points[indices]
                        self.logger.debug(f"Voxel sampling: {len(workspace_filtered_points)} -> {len(final_points)} points")
                    else:
                        # Fallback to random sampling
                        indices = np.random.choice(len(workspace_filtered_points), self.max_points, replace=False)
                        final_points = workspace_filtered_points[indices]
                        self.logger.debug(f"Fallback random sampling: {len(workspace_filtered_points)} -> {len(final_points)} points")
                else:
                    final_points = workspace_filtered_points
                    self.logger.debug(f"No downsampling needed: {len(final_points)} points")
                
                self.pointcloud_data = final_points
                self.sensor_data_available = True
                self.logger.debug(f"Updated sensor data: RGB {self.image_data.shape}, Points {self.pointcloud_data.shape}")
            else:
                self.logger.warn("No valid points in point cloud")
                
        except Exception as e:
            self.logger.error(f"Error updating sensor data: {e}")
            self.sensor_data_available = False
    
    def _voxel_grid_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Downsample point cloud using voxel grid
        
        Args:
            points: Input point cloud as numpy array of shape (N, 3)
            voxel_size: Size of each voxel in meters
            
        Returns:
            Downsampled point cloud as numpy array
        """
        try:
            # Create voxel grid indices
            voxel_indices = np.floor(points / voxel_size).astype(np.int32)
            
            # Create unique voxel identifiers
            unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
            
            # For each unique voxel, keep the centroid of all points in that voxel
            downsampled_points = []
            for i in range(len(unique_voxels)):
                voxel_points = points[inverse_indices == i]
                # Use centroid of points in this voxel
                centroid = np.mean(voxel_points, axis=0)
                downsampled_points.append(centroid)
            
            return np.array(downsampled_points, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in voxel grid downsampling: {e}")
            # Fallback to random sampling
            if len(points) > self.max_points:
                indices = np.random.choice(len(points), self.max_points, replace=False)
                return points[indices]
            return points
    
    def _transform_point_cloud(self, points: np.ndarray, header, target_frame: str) -> Optional[np.ndarray]:
        """
        Transform point cloud from source frame to target frame (with caching for performance)
        
        Args:
            points: Input point cloud as numpy array of shape (N, 3)
            header: Header from original PointCloud2 message
            target_frame: Target frame for transformation
            
        Returns:
            Transformed point cloud as numpy array, or None if transformation failed
        """
        try:
            current_time = time.time()
            
            # Check if we can use cached transform (for static transforms)
            if (self.cached_transform is not None and 
                current_time - self.cached_transform_time < self.transform_cache_duration):
                
                translation = self.cached_transform['translation']
                rotation_matrix = self.cached_transform['rotation_matrix']
                self.logger.debug("Using cached transform for point cloud transformation")
            else:
                # Get fresh transformation from TF
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    header.frame_id,
                    header.stamp,
                    timeout=Duration(seconds=0.1)  # Reduced timeout for faster processing
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
                self.logger.debug("Cached new transform for point cloud transformation")
            
            # Apply transformation to all points
            # Use efficient matrix operations with broadcasting
            transformed_points = (rotation_matrix @ points.T).T + translation
            
            return transformed_points.astype(np.float32)
            
        except (TransformException, tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            self.logger.warn(f"Failed to transform point cloud from {header.frame_id} to {target_frame}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in point cloud transformation: {e}")
            return None
    
    def _filter_workspace_points(self, points: np.ndarray) -> np.ndarray:
        """
        Filter point cloud to keep only points within workspace bounds
        
        Args:
            points: Input point cloud as numpy array of shape (N, 3)
            
        Returns:
            Filtered point cloud as numpy array
        """
        try:
            if len(points) == 0:
                return points
                
            # Apply workspace bounds filtering
            mask = (
                (points[:, 0] >= self.workspace_bounds['x_min']) &
                (points[:, 0] <= self.workspace_bounds['x_max']) &
                (points[:, 1] >= self.workspace_bounds['y_min']) &
                (points[:, 1] <= self.workspace_bounds['y_max']) &
                (points[:, 2] >= self.workspace_bounds['z_min']) &
                (points[:, 2] <= self.workspace_bounds['z_max'])
            )
            
            filtered_points = points[mask]
            return filtered_points
            
        except Exception as e:
            self.logger.error(f"Error in workspace filtering: {e}")
            return points
    
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
