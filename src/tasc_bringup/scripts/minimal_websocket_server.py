#!/usr/bin/env python3
"""
Minimal WebSocket Server for Basic Connectivity Testing
This script is independent of ROS2 and can be run directly with Python.
Used for testing WebSocket connectivity with TASC client before full ROS2 integration.
"""

import asyncio
import websockets
import json
import time
import numpy as np
import base64
import logging
from typing import Set, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MinimalWebSocketServer:
    """
    Minimal WebSocket server for testing connectivity with TASC client
    """
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        
        # Mock data for testing
        self.current_eef_pose = np.eye(4)
        self.current_twist = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1])
        self.current_gripper_action = 0
        
        # Mock sensor data
        self.mock_rgb_image = self._create_mock_rgb_image()
        self.mock_pointcloud = self._create_mock_pointcloud()
        
        logger.info(f"Minimal WebSocket server initialized on {host}:{port}")
    
    def _create_mock_rgb_image(self) -> np.ndarray:
        """Create a mock RGB image for testing"""
        # Create a simple test pattern (640x480x3)
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some colorful patterns
        image[:, :width//3] = [255, 0, 0]  # Red
        image[:, width//3:2*width//3] = [0, 255, 0]  # Green  
        image[:, 2*width//3:] = [0, 0, 255]  # Blue
        
        # Add some diagonal lines
        for i in range(0, min(height, width), 20):
            if i < height and i < width:
                image[i, i] = [255, 255, 255]  # White diagonal
        
        return image
    
    def _create_mock_pointcloud(self) -> np.ndarray:
        """Create a mock point cloud for testing"""
        # Create a simple point cloud (1000 points)
        n_points = 1000
        
        # Create a sphere of points
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(0.5, 1.0, n_points)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
        return points
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        ):
            logger.info("WebSocket server is running... Press Ctrl+C to stop")
            
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("Stopping server...")
                self.running = False
                raise
    
    async def handle_client(self, websocket):
        """Handle client connections and messages"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.connected_clients.add(websocket)
        logger.info(f"Client connected from {client_addr} (Total clients: {len(self.connected_clients)})")
        
        # Start sending realtime data to this client
        realtime_task = asyncio.create_task(self.send_realtime_data(websocket))
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_addr}")
                    await self.send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error processing message from {client_addr}: {e}")
                    await self.send_error(websocket, f"Processing error: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_addr} disconnected")
        except Exception as e:
            logger.error(f"Client error from {client_addr}: {e}")
        finally:
            # Cancel the realtime data task
            realtime_task.cancel()
            try:
                await realtime_task
            except asyncio.CancelledError:
                pass
            
            self.connected_clients.discard(websocket)
            logger.info(f"Client {client_addr} removed (Remaining clients: {len(self.connected_clients)})")
    
    async def process_message(self, websocket, data):
        """Process incoming messages from clients"""
        message_type = data.get("type")
        
        if message_type == "get_pointcloud_and_image":
            await self.send_pointcloud_and_image(websocket)
        elif message_type == "assist_action":
            await self.handle_assist_action(data)
        elif message_type == "test":
            await self.send_test_response(websocket, data)
        else:
            logger.warning(f"Unknown message type: {message_type}")
            await self.send_error(websocket, f"Unknown message type: {message_type}")
    
    async def send_pointcloud_and_image(self, websocket):
        """Send mock point cloud and image data"""
        try:
            # Encode data as base64
            rgb_bytes = self.mock_rgb_image.tobytes()
            points_bytes = self.mock_pointcloud.tobytes()
            
            rgb_b64 = base64.b64encode(rgb_bytes).decode('utf-8')
            points_b64 = base64.b64encode(points_bytes).decode('utf-8')
            
            message = {
                "type": "pointcloud_and_image",
                "rgb": rgb_b64,
                "rgb_shape": list(self.mock_rgb_image.shape),
                "points": points_b64,
                "points_shape": list(self.mock_pointcloud.shape),
                "timestamp": time.time()
            }
            
            await websocket.send(json.dumps(message))
            logger.info(f"Sent pointcloud and image data: RGB {self.mock_rgb_image.shape}, Points {self.mock_pointcloud.shape}")
            
        except Exception as e:
            logger.error(f"Error sending sensor data: {e}")
            await self.send_error(websocket, f"Failed to send sensor data: {str(e)}")
    
    async def handle_assist_action(self, data):
        """Handle assist action from client"""
        try:
            assist_action = data.get("assist_action", [0.0]*6)
            gripper_action = data.get("gripper_action")
            
            logger.info(f"Received assist action: {assist_action}")
            if gripper_action is not None:
                logger.info(f"Received gripper action: {gripper_action}")
            
            # In a real implementation, this would control the robot
            # For now, just log the received action
            
        except Exception as e:
            logger.error(f"Error handling assist action: {e}")
    
    async def send_realtime_data(self, websocket):
        """Send realtime data to a specific client"""
        try:
            while websocket in self.connected_clients:
                # Update mock data
                self.update_mock_data()
                
                # Send EEF pose
                eef_message = {
                    "type": "eef_pose",
                    "pose": self.current_eef_pose.tolist(),
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(eef_message))
                
                # Send twist
                twist_message = {
                    "type": "twist",
                    "twist": self.current_twist.tolist(),
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(twist_message))
                
                # Send gripper action
                gripper_message = {
                    "type": "gripper_action",
                    "gripper_action": self.current_gripper_action,
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(gripper_message))
                
                # Wait before next update
                await asyncio.sleep(0.1)  # 10 Hz
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error sending realtime data: {e}")

    async def send_test_response(self, websocket, data):
        """Send test response"""
        try:
            response = {
                "type": "test_response",
                "message": "Hello from Minimal WebSocket Server",
                "received": data.get("message", ""),
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(response))
            logger.info(f"Sent test response")
        except Exception as e:
            logger.error(f"Error sending test response: {e}")
            await self.send_error(websocket, f"Failed to send test response: {str(e)}")
    
    async def send_error(self, websocket, error_message):
        """Send error message"""
        try:
            error_msg = {
                "type": "error",
                "message": error_message,
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(error_msg))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    def update_mock_data(self):
        """Update mock data with time-based changes"""
        t = time.time()
        
        # Update EEF pose (circular motion)
        self.current_eef_pose = np.eye(4)
        self.current_eef_pose[0, 3] = 0.5 * np.cos(t * 0.5)  # X position
        self.current_eef_pose[1, 3] = 0.5 * np.sin(t * 0.5)  # Y position
        self.current_eef_pose[2, 3] = 0.8 + 0.1 * np.sin(t)  # Z position
        
        # Update twist (sinusoidal motion)
        self.current_twist = np.array([
            0.1 * np.sin(t * 0.5),  # vx
            0.1 * np.cos(t * 0.5),  # vy
            0.05 * np.sin(t * 2),   # vz
            0.05 * np.cos(t),       # wx
            0.05 * np.sin(t),       # wy
            0.02 * np.cos(t * 0.5)  # wz
        ])
        
        # Update gripper action (toggle every 2 seconds)
        self.current_gripper_action = int(t) % 2

def main():
    """Main function to run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal WebSocket Server for TASC Testing")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port number (default: 8765)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = MinimalWebSocketServer(host=args.host, port=args.port)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()
