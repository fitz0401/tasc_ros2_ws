import time
import numpy as np
import asyncio
import websockets
import json
import threading
from queue import Queue
import base64
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


class WebSocketClient:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.data_queue = Queue()
        self.running = False
        self.thread = None
        self.rgb_data = None
        self.points_data = None
        self.twist_data = np.zeros(6)
        self.eef_pose_data = np.eye(4)
        self.gripper_action_data = 0  # 0 or 1
        self.data_received = False
        
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri, max_size=10 * 1024 * 1024)
            print(f"Connected to WebSocket at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def request_pointcloud_and_image(self):
        """Request point cloud and RGB image from ROS2"""
        if self.websocket:
            request = {"type": "get_pointcloud_and_image"}
            await self.websocket.send(json.dumps(request))
    
    async def listen(self):
        """Listen for incoming messages"""
        try:
            while self.running and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "pointcloud_and_image":
                    # Decode base64 encoded data
                    rgb_bytes = base64.b64decode(data["rgb"])
                    points_bytes = base64.b64decode(data["points"])
                    
                    # Convert to numpy arrays
                    self.rgb_data = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(data["rgb_shape"])
                    self.points_data = np.frombuffer(points_bytes, dtype=np.float32).reshape(data["points_shape"])
                    self.data_received = True
                    print(f"Received RGB shape: {self.rgb_data.shape}, Points shape: {self.points_data.shape}")
                    
                elif data["type"] == "twist":
                    self.twist_data = np.array(data["twist"])
                    
                elif data["type"] == "eef_pose":
                    self.eef_pose_data = np.array(data["pose"]).reshape(4, 4)
                    
                elif data["type"] == "gripper_action":
                    self.gripper_action_data = int(data["gripper_action"])
                    
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error in WebSocket listener: {e}")
        finally:
            self.running = False
    
    def start(self):
        """Start WebSocket client in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_async)
        self.thread.start()
    
    def _run_async(self):
        """Run async event loop in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_main())
    
    async def _async_main(self):
        """Main async function"""
        if await self.connect():
            await self.listen()
    
    def stop(self):
        """Stop WebSocket client"""
        self.running = False
        if self.websocket:
            # Use asyncio to close the websocket properly
            try:
                # Check if websocket is still open before trying to close
                should_close = True
                try:
                    should_close = self.websocket.open
                except AttributeError:
                    # For different websockets versions, assume we should try to close
                    should_close = True
                
                if should_close:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.websocket.close())
                    else:
                        loop.run_until_complete(self.websocket.close())
            except Exception:
                pass
        if self.thread:
            self.thread.join()
    
    def get_twist(self):
        """Get latest twist data"""
        return self.twist_data.copy()
    
    def get_eef_pose(self):
        """Get latest EEF pose"""
        return self.eef_pose_data.copy()
    
    def get_gripper_action(self):
        """Get latest gripper action (0 or 1)"""
        return self.gripper_action_data
    
    async def send_assist_action(self, assist_action, gripper_action=None):
        """Send assist action back to ROS2"""
        if self.websocket:
            try:
                # Check if websocket is open
                is_open = True
                try:
                    is_open = self.websocket.open
                except AttributeError:
                    # Assume it's open if we can't check
                    is_open = True
                
                if is_open:
                    request = {
                        "type": "assist_action",
                        "assist_action": assist_action.tolist() if isinstance(assist_action, np.ndarray) else assist_action
                    }
                    if gripper_action is not None:
                        request["gripper_action"] = float(gripper_action)
                    await self.websocket.send(json.dumps(request))
            except Exception as e:
                print(f"Error sending assist action: {e}")
    
    async def request_data_once(self):
        """Request point cloud and image once"""
        if await self.connect():
            await self.request_pointcloud_and_image()
            # Wait for response
            timeout = 10  # seconds
            start_time = time.time()
            while not self.data_received and time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    if data["type"] == "pointcloud_and_image":
                        rgb_bytes = base64.b64decode(data["rgb"])
                        points_bytes = base64.b64decode(data["points"])
                        
                        self.rgb_data = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(data["rgb_shape"])
                        self.points_data = np.frombuffer(points_bytes, dtype=np.float32).reshape(data["points_shape"])
                        self.data_received = True
                        print(f"Received data once - RGB: {self.rgb_data.shape}, Points: {self.points_data.shape}")
                        break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error receiving data: {e}")
                    break
            
            if not self.data_received:
                print("Timeout waiting for pointcloud and image data")
            
            await self.websocket.close()
        
        return self.rgb_data, self.points_data
    
    def is_connected(self):
        """Check if WebSocket is connected and running"""
        try:
            return self.running and self.websocket and self.websocket.open
        except AttributeError:
            # Fallback for different websockets versions
            return self.running and self.websocket is not None
    
    def get_connection_status(self):
        """Get detailed connection status"""
        websocket_open = False
        try:
            websocket_open = self.websocket.open if self.websocket else False
        except AttributeError:
            websocket_open = self.websocket is not None
            
        return {
            "running": self.running,
            "websocket_exists": self.websocket is not None,
            "websocket_open": websocket_open,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "data_received": self.data_received
        }

# Test function to verify the WebSocket client
async def test_websocket_connection(uri="ws://localhost:8765"):
    """Test function to verify WebSocket connection"""
    print(f"Testing WebSocket connection to {uri}")
    
    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as websocket:
            print("✓ Successfully connected to WebSocket server")
            
            # Test sending a request
            test_request = {"type": "test", "message": "hello"}
            await websocket.send(json.dumps(test_request))
            print("✓ Successfully sent test message")
            
            # Try to receive a response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"✓ Received response: {data}")
            except asyncio.TimeoutError:
                print("⚠ No response received (timeout)")
            
            return True
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def visualize_rgb_and_points(rgb, points):
    """Visualize RGB image and point cloud"""
    try:
        import cv2
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Display RGB image
        if rgb is not None:
            print(f"RGB Image shape: {rgb.shape}")
            cv2.imshow("RGB Image", rgb)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()
        else:
            print("No RGB image data received")
        
        # Display point cloud
        if points is not None and len(points) > 0:
            print(f"Point cloud shape: {points.shape}")
            
            # Sample points if too many (for faster visualization)
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points_sample = points[indices]
            else:
                points_sample = points
            
            # 3D scatter plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color by Z-coordinate for better visualization
            colors = points_sample[:, 2]
            scatter = ax.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2], 
                               c=colors, cmap='viridis', s=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Point Cloud Visualization')
            plt.colorbar(scatter)
            plt.show()
        else:
            print("No point cloud data received")
            
    except ImportError as e:
        print(f"Visualization requires opencv-python and matplotlib: {e}")
    except Exception as e:
        print(f"Error in visualization: {e}")

async def test_rgb_and_points(uri="ws://localhost:8765"):
    """Test RGB and point cloud data reception and visualization"""
    print("\n=== Testing RGB and Point Cloud Data ===")
    
    client = WebSocketClient(uri)
    try:
        # Get data once
        rgb, points = await client.request_data_once()
        
        if rgb is not None and points is not None:
            print("✓ Successfully received RGB and point cloud data")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  Points shape: {points.shape}")
            
            # Visualize the data
            visualize_rgb_and_points(rgb, points)
            return True
        else:
            print("✗ Failed to receive RGB and point cloud data")
            return False
            
    except Exception as e:
        print(f"✗ Error testing RGB and points: {e}")
        return False

async def test_assist_action(uri="ws://localhost:8765"):
    """Test sending assist action to robot"""
    print("\n=== Testing Assist Action Sending ===")
    
    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as websocket:
            print("✓ Connected for assist action test")
            
            # Send a small, safe assistance action
            assist_action = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            request = {
                "type": "assist_action",
                "assist_action": assist_action,
                "gripper_action": None
            }
            
            await websocket.send(json.dumps(request))
            print(f"✓ Sent safe assist action: {assist_action}")
            
            # Wait a moment for the action to be processed
            await asyncio.sleep(1.0)
            
            # Send a neutral action to stop any movement
            neutral_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            neutral_request = {
                "type": "assist_action", 
                "assist_action": neutral_action
            }
            await websocket.send(json.dumps(neutral_request))
            print("✓ Sent neutral action to stop movement")
            
            return True
            
    except Exception as e:
        print(f"✗ Error testing assist action: {e}")
        return False

async def test_continuous_data(uri="ws://localhost:8765", duration=10):
    """Test continuous EEF pose and gripper action reception"""
    print(f"\n=== Testing Continuous Data Reception for {duration} seconds ===")
    
    client = WebSocketClient(uri)
    client.start()
    
    # Wait for connection
    await asyncio.sleep(1)
    
    if not client.is_connected():
        print("✗ Failed to establish continuous connection")
        return False
    
    print("✓ Continuous connection established")
    print("Monitoring EEF pose and gripper action (press Ctrl+C to stop)...")
    
    start_time = time.time()
    last_print_time = 0
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Print data every 0.5 seconds
            if current_time - last_print_time >= 0.5:
                eef_pose = client.get_eef_pose()
                gripper_action = client.get_gripper_action()
                
                # Print EEF position (first 3 elements of pose matrix)
                eef_pos = eef_pose[:3, 3]
                print(f"Time: {current_time - start_time:.1f}s | "
                      f"EEF Pos: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] | "
                      f"Gripper: {gripper_action}")
                
                last_print_time = current_time
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error during continuous monitoring: {e}")
    finally:
        client.stop()
        print("✓ Continuous monitoring stopped")
    
    return True

async def run_all_tests(server_uri="ws://localhost:8765"):
    """Run all tests sequentially"""
    print("🧪 WebSocket Client Test Suite")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    print("Test 1: Basic Connectivity")
    connectivity_result = await test_websocket_connection(server_uri)
    
    if not connectivity_result:
        print("\n❌ Basic connectivity failed. Skipping other tests.")
        return
    
    # Test 2: RGB and point cloud visualization
    print("\nTest 2: RGB and Point Cloud Data")
    rgb_points_result = await test_rgb_and_points(server_uri)
    
    # Test 3: Assist action sending (optional for safety)
    assist_result = True  # Default to pass if not tested
    print("\nTest 3: Assist Action Sending")
    print("⚠️  WARNING: This will send a small movement command to the real robot!")
    user_confirm = input("Do you want to proceed? (y/N): ").lower().strip()
    if user_confirm == 'y' or user_confirm == 'yes':
        assist_result = await test_assist_action(server_uri)
    else:
        print("❌ Assist action test skipped by user")
        assist_result = False
    
    # Test 4: Continuous data monitoring
    print("\nTest 4: Continuous Data Monitoring")
    continuous_result = await test_continuous_data(server_uri, duration=5)  # 5 seconds for demo
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Connectivity: {'✓ PASS' if connectivity_result else '✗ FAIL'}")
    print(f"  RGB & Points: {'✓ PASS' if rgb_points_result else '✗ FAIL'}")
    print(f"  Assist Action: {'✓ PASS' if assist_result else '✗ FAIL'}")
    print(f"  Continuous:   {'✓ PASS' if continuous_result else '✗ FAIL'}")
    
    if all([connectivity_result, rgb_points_result, continuous_result, assist_result]):
        print("\n🎉 All tests passed! WebSocket client is ready for use!")
    else:
        print("\n⚠️  Some tests failed. Please check your ROS2 WebSocket server.")

def main():
    parser = argparse.ArgumentParser(description="AANP WebSocket Client")
    parser.add_argument("--server_host", default="localhost", 
                        help="WebSocket server host (default: localhost)")
    parser.add_argument("--server_port", type=int, default=8765,
                        help="WebSocket server port (default: 8765)")
    args = parser.parse_args()
    server_uri = f"ws://{args.server_host}:{args.server_port}"
    print(f"🌐 Server: {server_uri}")
    asyncio.run(run_all_tests(server_uri))

if __name__ == "__main__":
    main()