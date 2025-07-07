#!/usr/bin/env python3
"""
Simple WebSocket Test Script
用于快速测试WebSocket连接的最简脚本
"""

import asyncio
import websockets
import json

async def test_websocket_connection(host="localhost", port=8765):
    """测试WebSocket连接"""
    uri = f"ws://{host}:{port}"
    
    try:
        print(f"🔗 Connecting to {uri}...")
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as websocket:
            print("✅ Connected successfully!")
            
            # 发送测试消息
            test_message = {
                "type": "test",
                "message": "Hello from test client!"
            }
            
            await websocket.send(json.dumps(test_message))
            print("📤 Sent test message")
            
            # 接收几条消息
            for i in range(10):
                try:
                    # Request pointcloud_and_image on the first iteration
                    if i == 0:
                        rgb_request = {
                            "type": "get_pointcloud_and_image"
                        }
                        await websocket.send(json.dumps(rgb_request))
                        print("📤 Sent pointcloud_and_image request")
                    
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    print(f"📥 Received [{i+1}]: {data.get('type', 'unknown')}")
                    
                    # Send assist_action
                    assist_msg = {
                        "type": "assist_action",
                        "assist_action": [0.1, 0.0, 0.0, 0.0, 0.0, 0.1],
                        "gripper_action": 1
                    }
                    await websocket.send(json.dumps(assist_msg))
                    print("📤 Sent assist_action")
                    
                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for message")
                    break
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
    
    except ConnectionRefusedError:
        print(f"❌ Connection refused to {uri}")
        print("Make sure the server is running!")
    except Exception as e:
        print(f"❌ Connection error: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple WebSocket Test Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    
    args = parser.parse_args()
    
    print("🧪 WebSocket Connection Test")
    print("=" * 40)
    print(f"Target: ws://{args.host}:{args.port}")
    print()
    
    asyncio.run(test_websocket_connection(args.host, args.port))

if __name__ == "__main__":
    main()
