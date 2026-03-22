#!/usr/bin/env python3
# =============================================================================
#   ██████╗██╗      █████╗ ██╗    ██╗██████╗  ██████╗ ████████╗
#  ██╔════╝██║     ██╔══██╗██║    ██║██╔══██╗██╔═══██╗╚══██╔══╝
#  ██║     ██║     ███████║██║ █╗ ██║██████╔╝██║   ██║   ██║   
#  ██║     ██║     ██╔══██║██║███╗██║██╔══██╗██║   ██║   ██║   
#  ╚██████╗███████╗██║  ██║╚███╔███╔╝██████╔╝╚██████╔╝   ██║   
#   ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═════╝  ╚═════╝   ╚═╝   
#
#  Clawbot - Strike-Robot-powered Robot Controller for Unitree G1 in Isaac Sim
# =============================================================================
# Usage:
#   export GEMINI_API_KEY="your-api-key-here"
#   python gemini_clawbot.py
#
# Requires sim_main.py to be running first in a separate terminal.
# =============================================================================

import os
import re
import sys
import cv2
import time
import json
import threading
import textwrap
import numpy as np
from datetime import datetime

# ---- DDS imports ----
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    DDS_AVAILABLE = True
except ImportError:
    print("[Clawbot WARN] unitree_sdk2py not found. DDS communications will be mocked.")
    DDS_AVAILABLE = False

# ---- Image client import ----
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "teleimager", "src"))
    from teleimager.image_client import ImageClient
    IMAGE_CLIENT_AVAILABLE = True
except ImportError:
    print("[Clawbot WARN] teleimager not found. Image client will be mocked.")
    IMAGE_CLIENT_AVAILABLE = False

# ---- Gemini API (google-genai SDK - new) ----
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    print("[Clawbot ERROR] google-genai not installed. Run: pip install google-genai")
    GEMINI_AVAILABLE = False

# =============================================================================
#  Configuration
# =============================================================================
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
# Gemini Robotics-ER 1.5: specialized spatial understanding + bbox detection + action planning
GEMINI_BBOX_MODEL   = "gemini-robotics-er-1.5-preview"   # spatial / bbox detection
GEMINI_ACTION_MODEL = "gemini-3-flash-preview"                 # reasoning / planning
SERVER_HOST       = "127.0.0.1"     # Isaac Sim image server host
CALL_INTERVAL_SEC = 3.0             # seconds between Gemini API calls
G1_NUM_JOINTS     = 29              # Unitree G1 joint count
DDS_CHANNEL       = 1              # must match ChannelFactoryInitialize(1) in sim_main

# Default stiffness / damping (safe defaults for G1)
DEFAULT_KP = 60.0
DEFAULT_KD = 1.5

# Joint names for G1 (29-DOF) — used for prompt context
G1_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_pitch", "waist_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# =============================================================================
#  ASCII Banner
# =============================================================================
BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   ██████╗██╗      █████╗ ██╗    ██╗██████╗  ██████╗ ████████╗  ║
║  ██╔════╝██║     ██╔══██╗██║    ██║██╔══██╗██╔═══██╗╚══██╔══╝  ║
║  ██║     ██║     ███████║██║ █╗ ██║██████╔╝██║   ██║   ██║     ║
║  ██║     ██║     ██╔══██║██║███╗██║██╔══██╗██║   ██║   ██║     ║
║  ╚██████╗███████╗██║  ██║╚███╔███╔╝██████╔╝╚██████╔╝   ██║     ║
║   ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═════╝  ╚═════╝   ╚═╝     ║
║           Strike Robot-Powered Robot Brain  v1.0                      ║
╚══════════════════════════════════════════════════════════╝
"""

# =============================================================================
#  ANSI Colors for terminal
# =============================================================================
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    DIM     = "\033[2m"

def clawbot_log(tag: str, msg: str, color: str = C.CYAN):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{C.DIM}[{ts}]{C.RESET} {color}{C.BOLD}[Clawbot {tag}]{C.RESET} {msg}")

# =============================================================================
#  Gemini Client  (google-genai SDK — Robotics-ER 1.5)
# =============================================================================

# Stage 1 ─ Robotics-ER: precisely locate the cylinder and the basket
BBOX_PROMPT = """
You are an expert robotic vision system.
Identify exactly these 3 objects in the scene:
1. The robot's right hand or gripper
2. The cylinder / block object that needs to be picked up
3. The basket / tray container on the table

Return ONLY a raw JSON array (no markdown, no explanation):
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "gripper"}, {"box_2d": [ymin, xmin, ymax, xmax], "label": "cylinder"}, ...]

Rules:
- box_2d values are integers normalized to 0-1000 (ymin, xmin, ymax, xmax)
- Always use EXACT labels: "gripper", "cylinder", "basket"
- If an object is not visible, omit it from the array
"""

# Stage 1b ─ Robotics-ER: get trajectory points from cylinder to basket
TRAJECTORY_PROMPT = """
You need to generate TWO DISTINCT movement trajectories based on the visual scene.
1. 'reach': From the robot 'gripper' to the 'cylinder'.
2. 'transport': From the 'cylinder' to the 'basket'.

The answer MUST be a raw JSON array (no markdown, no explanation) combining both:
[{"point": [y, x], "label": "reach_0"}, ..., {"point": [y, x], "label": "transport_0"}, ...]

Rules:
- All point coordinates are in [y, x] format, normalized to 0-1000.
- The 'reach' trajectory starts at the gripper and ends at the cylinder.
- The 'transport' trajectory starts at the cylinder and ends at the basket opening.
- Place 3-6 intermediate waypoints for each trajectory along a smooth curve.
- Ensure points are labeled correctly in order: "reach_0", "reach_1", ... and "transport_0", "transport_1", ...
- Do NOT use labels other than 'reach_X' or 'transport_X'.
"""

# Stage 2 ─ Reason about manual steps
ACTION_SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI assistant for a robotic arm operator.
    You have analyzed an image and obtained detections for 'cylinder' and 'basket', as well as a defined visual trajectory.
    
    TASK: Describe the step-by-step physical manual movements the user needs to make to pick up the cylinder and place it in the basket.
    Explain the spatial relationship, whether the object is far or near, left or right, and how the user should move the arm joints (shoulder pitch, roll, elbow).
    Provide a clear, easy-to-read numbered list. DO NOT generate any JSON or code. Just plain thinking and reasoning text.
    """)




class GeminiClient:
    """Two-stage Gemini client:
    Stage 1: gemini-robotics-er-1.5-preview  → bbox detection (box_2d)
    Stage 2: gemini-2.0-flash                → action planning using bboxes
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        clawbot_log("INIT",
            f"Bbox model : {C.GREEN}{GEMINI_BBOX_MODEL}{C.RESET}  "
            f"Action model: {C.GREEN}{GEMINI_ACTION_MODEL}{C.RESET}", C.GREEN)

        # Initialize DPT Depth Model (Intel/dpt-hybrid-midas)
        clawbot_log("INIT", "Loading DPT Depth Model (Intel/dpt-hybrid-midas) from HuggingFace...", C.CYAN)
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device).eval()
            self.has_depth_model = True
            clawbot_log("INIT", f"Depth Model (Intel DPT) loaded successfully on {self.device}!", C.GREEN)
        except Exception as e:
            clawbot_log("INIT ERR", f"Failed to load Depth Model: {e}", C.RED)
            self.has_depth_model = False

    @staticmethod
    def _extract_json(text: str):
        """Robustly extract a JSON object/array from model output.
        Handles formats like:
          - {"steps": [...], "reasoning": "..."}
          - [{"function": ...}, ...]   (raw list)
          - Reasoning: ... ```json {...} ```
          - Reasoning: {...}
        """
        text = text.strip()

        # Strategy 2: Fallback: scan all possible start and end indices
        for start_idx in range(len(text)):
            if text[start_idx] in ('{', '['):
                for end_idx in range(len(text), start_idx, -1):
                    if text[end_idx - 1] in ('}', ']'):
                        candidate = text[start_idx:end_idx]
                        try:
                            obj = json.loads(candidate)
                            # Heuristic: Ignore small lists of floats (e.g. from reasoning text)
                            if isinstance(obj, list) and len(obj) > 0 and not isinstance(obj[0], dict):
                                break  # skip to next start_idx
                            return obj
                        except json.JSONDecodeError:
                            pass
        return None


    def detect_bbox(self, rgb_image: np.ndarray) -> list:
        """Send frame to gemini-robotics-er-1.5 → list of
        {"box_2d":[ymin,xmin,ymax,xmax], "label":str} dicts (0-1000).
        """
        _, buf = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = buf.tobytes()

        try:
            response = self._client.models.generate_content(
                model=GEMINI_BBOX_MODEL,
                contents=[
                    genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    BBOX_PROMPT,
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            clawbot_log("RAW BBOX", f"{response.text.strip()}", C.DIM)
            result = self._extract_json(response.text)
            return result if isinstance(result, list) else []
        except Exception as e:
            clawbot_log("BBOX ERR", str(e), C.YELLOW)
            return []

    def detect_trajectory(self, rgb_image: np.ndarray, bboxes: list) -> list:
        """Use gemini-robotics-er-1.5 to get trajectory points from cylinder to basket.
        Returns list of {"point": [y, x], "label": str} dicts (0-1000).
        """
        _, buf = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = buf.tobytes()

        # Compute bbox centers for context
        bbox_info = ""
        for b in bboxes:
            box = b.get("box_2d", [])
            lbl = b.get("label", "")
            if len(box) == 4:
                cy = (box[0] + box[2]) // 2
                cx = (box[1] + box[3]) // 2
                bbox_info += f"- {lbl}: center at (y={cy}, x={cx})\n"

        user_msg = (
            f"Scene object locations (y, x normalized 0-1000):\n"
            f"{bbox_info}\n"
            f"Generate the 'reach' trajectory from the gripper to the cylinder, and the 'transport' trajectory from the cylinder to the basket.\n"
        )

        try:
            response = self._client.models.generate_content(
                model=GEMINI_BBOX_MODEL,
                contents=[
                    genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    TRAJECTORY_PROMPT,
                    user_msg,
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.5,
                ),
            )
            clawbot_log("RAW TRAJ", f"{response.text.strip()}", C.DIM)
            result = self._extract_json(response.text)
            return result if isinstance(result, list) else []
        except Exception as e:
            clawbot_log("TRAJ ERR", str(e), C.YELLOW)
            return []

    # ------------------------------------------------------------------
    # Stage 2: Action planning
    # ------------------------------------------------------------------
    def plan_action(self, rgb_image: np.ndarray, depth_image: np.ndarray, bboxes: list,
                    joint_positions: list, step_index: int = 0,
                    trajectory_points: list | None = None) -> dict:
        """Use gemini-robotics-er-1.5 to plan arm actions following trajectory."""
        _, buf = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = buf.tobytes()

        joint_str   = ", ".join(f"{j:.3f}" for j in joint_positions)
        l_arm_str   = ", ".join(f"j{i}={joint_positions[i]:.3f}" for i in range(15, 22))

        # Ensure image dimensions are known first
        if rgb_image is not None:
            img_h, img_w = rgb_image.shape[:2]
        elif depth_image is not None:
            img_h, img_w = depth_image.shape[:2]
        else:
            img_h, img_w = 480, 640

        # Compute bbox centers and 3D Camera Coordinates via new Camera Config
        # Config: Focal Length 50.0, Horiz Aperture 20.955, Vert Aperture 15.2908
        Fx = img_w * (50.0 / 20.955)
        Fy = img_h * (50.0 / 15.2908)
        cx_proj, cy_proj = img_w / 2.0, img_h / 2.0

        centers = {}
        for b in bboxes:
            box = b.get("box_2d", [])
            lbl = b.get("label", "")
            if len(box) == 4:
                # box[] is normalized 0-1000 [ymin, xmin, ymax, xmax]
                cy = (box[0] + box[2]) / 2.0
                cx = (box[1] + box[3]) / 2.0
                
                # Un-normalize to pixel space
                px_x = int(cx * img_w / 1000.0)
                px_y = int(cy * img_h / 1000.0)
                px_x = max(0, min(img_w - 1, px_x))
                px_y = max(0, min(img_h - 1, px_y))

                Z_depth = 0.0
                X_cam, Y_cam, Z_cam = 0.0, 0.0, 0.0
                X_rob, Y_rob, Z_rob = 0.0, 0.0, 0.0
                if depth_image is not None:
                    # Extract pixel depth; handle multi-channel depth maps safely
                    val = depth_image[px_y, px_x]
                    import numpy as np
                    if isinstance(val, np.ndarray) and val.size > 1:
                        Z_depth = float(val[0])
                    else:
                        Z_depth = float(val)

                    if Z_depth > 0:
                        # 1. Camera Intrinsics -> Camera Frame (Z out, X right, Y down)
                        X_cam = (px_x - cx_proj) * Z_depth / Fx
                        Y_cam = (px_y - cy_proj) * Z_depth / Fy
                        Z_cam = Z_depth
                        
                        # 2. Camera Extrinsics -> Robot Base Frame (X forward, Y left, Z up)
                        # Assume camera looks forward. 
                        # cam_Z -> rob_X, cam_X -> rob_-Y, cam_Y -> rob_-Z
                        # Tinh chỉnh độ d lệch (offset) từ đầu Camera tới cổ/rốn Base (giả định)
                        offset_x = 0.08   # Camera cách rốn 8cm về trước
                        offset_y = 0.0    # Camera ở giữa
                        offset_z = 0.25   # Camera cao hơn rốn 25cm
                        
                        X_rob = Z_cam + offset_x
                        Y_rob = -X_cam + offset_y
                        Z_rob = -Y_cam + offset_z
                
                centers[lbl] = {
                    "center_y_1000": cy, "center_x_1000": cx,
                    "cam_3d_m": [round(X_cam,4), round(Y_cam,4), round(Z_cam,4)],
                    "rob_3d_m": [round(X_rob,4), round(Y_rob,4), round(Z_rob,4)]
                }
                clawbot_log("CV-3D", f"[{lbl}] -> Pixel({px_x}, {px_y}) | Z: {Z_depth:.3f}m\n     -> Cam: X={X_cam:.3f}, Y={Y_cam:.3f}, Z={Z_cam:.3f}\n     -> Rob: X={X_rob:.3f}, Y={Y_rob:.3f}, Z={Z_rob:.3f}", C.CYAN)


        # Format trajectory for the model
        traj_lines = []
        if trajectory_points:
            traj_lines.append("\n=== TRAJECTORY (image waypoints, normalized 0-1000) ===")
            for item in trajectory_points:
                pt = item.get("point", [])
                lbl = item.get("label", "?")
                if len(pt) == 2:
                    traj_lines.append(f"  [{lbl}] → (y={pt[0]}, x={pt[1]})")
        traj_str = "\n".join(traj_lines)

        user_msg = (
            f"=== SCENE DETECTIONS ===\n"
            f"{json.dumps(centers, indent=2)}\n"
            f"{traj_str}\n"
            f"Based ONLY on the visuals and these coordinates, explain the reasoning and manual steps required for the operator to do the task."
        )

        try:
            response = self._client.models.generate_content(
                model=GEMINI_ACTION_MODEL,
                contents=[
                    genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    user_msg,
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=ACTION_SYSTEM_PROMPT,
                    temperature=0.6,
                ),
            )
            clawbot_log("REASONING", f"\n{response.text.strip()}", C.MAGENTA)
            
            # --- Quy đổi quỹ đạo thành các step ---
            synthetic_steps = []
            if trajectory_points:
                current_j = list(joint_positions)
                for pt_data in trajectory_points:
                    pt = pt_data.get("point")
                    if pt and len(pt) == 2:
                        y, x = pt[0], pt[1]
                        # Normalized 0-1000, 500 is center. MAPPING 2D to JOINT DELTAS (Visual Servoing Heuristic)
                        dx = (x - 500) / 500.0  
                        dy = (y - 500) / 500.0  
                        
                        new_j = list(current_j)
                        # Left arm starts at index 15
                        new_j[15] += dy * 0.2  # Pitch down/up
                        new_j[17] -= dx * 0.2  # Yaw left/right
                        new_j[18] = 0.3        # keep elbow slightly bent
                        
                        synthetic_steps.append({
                            "function": "move_left_arm",
                            "args": new_j[15:22] # 7 joints
                        })
                        current_j = new_j
                
                # Cuối quỹ đạo (tới rổ): mở kẹp? (Hoặc tới vật: đóng kẹp)
                synthetic_steps.append({
                    "function": "set_gripper",
                    "args": [True] # Giả lập đóng kẹp
                })
                
            return {"reasoning": response.text.strip(), "steps": synthetic_steps}
        except Exception as e:
            clawbot_log("REASONING ERR", str(e), C.RED)
            return {}

    # ------------------------------------------------------------------
    # Unified call: bbox → trajectory → action
    # ------------------------------------------------------------------
    def call(self, rgb_image: np.ndarray, depth_image: np.ndarray,
             joint_positions: list, step_index: int = 0) -> dict:
        """Full pipeline: detect objects → get trajectory → plan action."""
        
        # --- Overwrite depth with AI Model ---
        if getattr(self, "has_depth_model", False):
            try:
                import torch
                from PIL import Image
                import cv2
                import numpy as np
                image_pil = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                inputs = self.depth_processor(images=image_pil, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.depth_model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                # interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image_pil.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                
                # In Intel DPT, the raw output is disparity (inverse depth). 
                # For basic Z estimation or visualization, we can pull it out directly.
                # However, to map it strictly to physical meters may require scaling.
                output = prediction.squeeze().cpu().numpy()
                
                # Replace the input depth_image with predicted ones
                depth_image = output
                clawbot_log("DEPTH AI", "Successfully injected Intel DPT Hybrid depth map!", C.CYAN)
            except Exception as e:
                clawbot_log("DEPTH AI ERR", str(e), C.YELLOW)

        bboxes = self.detect_bbox(rgb_image)
        n_det  = len(bboxes)
        labels = [b.get('label', '?') for b in bboxes]
        clawbot_log("BBOX",
            f"Robotics-ER: {n_det} objects → {labels}", C.CYAN)

        # Stage 1b: Get trajectory points from cylinder to basket
        trajectory_points = []
        if n_det >= 2:
            trajectory_points = self.detect_trajectory(rgb_image, bboxes)
            clawbot_log("TRAJ",
                f"Got {len(trajectory_points)} trajectory points", C.CYAN)

        action_resp = self.plan_action(
            rgb_image, depth_image, bboxes, joint_positions, step_index,
            trajectory_points=trajectory_points)
        action_resp["_bboxes"] = bboxes
        action_resp["_trajectory"] = trajectory_points
        return action_resp


# =============================================================================
#  DDS Command Sender
# =============================================================================
class RobotCommander:
    def __init__(self):
        self._crc   = CRC() if DDS_AVAILABLE else None
        self._pub   = None

        if DDS_AVAILABLE:
            try:
                ChannelFactoryInitialize(DDS_CHANNEL)
                self._cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
                self._cmd_pub.Init()
                clawbot_log("DDS", "Command publisher on rt/lowcmd ready.", C.GREEN)
                
                # Initialize Dex1 gripper publishers
                try:
                    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
                    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
                    self._motor_cmds_cls = MotorCmds_
                    self._motor_cmd_cls = unitree_go_msg_dds__MotorCmd_
                    self._left_gripper_pub = ChannelPublisher("rt/dex1/left/cmd", MotorCmds_)
                    self._left_gripper_pub.Init()
                    self._right_gripper_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
                    self._right_gripper_pub.Init()
                    
                    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
                    self._string_msg_cls = String_
                    self._reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
                    self._reset_pub.Init()
                    
                    clawbot_log("DDS", "Dex1 gripper publishers ready.", C.GREEN)
                except ImportError:
                    self._left_gripper_pub = None
                    self._right_gripper_pub = None
                    clawbot_log("DDS WARN", "Failed to init Dex1 publishers, unitree_go msg not found.", C.YELLOW)
            except Exception as e:
                clawbot_log("DDS ERR", f"Failed to init DDS publisher: {e}", C.RED)
                self._cmd_pub = None
                self._left_gripper_pub = None
        else:
            self.last_gripper_val = None

    def reset_scene(self):
        """Sends a string '1' to reset the object in the sim."""
        if hasattr(self, "_reset_pub") and self._reset_pub and hasattr(self, "_string_msg_cls"):
            try:
                msg = self._string_msg_cls(data="1")
                self._reset_pub.Write(msg)
                clawbot_log("RESET", "Sent Object Fallback Reset Command (1) to sim_main.py", C.MAGENTA)
            except Exception as e:
                clawbot_log("RESET ERR", f"Failed to send reset: {e}", C.RED)

    def reset_all(self):
        """Sends a string '2' to reset the ENTIRE SCENE (including robot pose)."""
        if hasattr(self, "_reset_pub") and self._reset_pub and hasattr(self, "_string_msg_cls"):
            try:
                msg = self._string_msg_cls(data="2")
                self._reset_pub.Write(msg)
                clawbot_log("RESET", "Sent Full Robot+Scene Reset Command (2) to sim_main.py", C.MAGENTA)
            except Exception as e:
                clawbot_log("RESET ERR", f"Failed to send full reset: {e}", C.RED)

    def send(self, joint_positions: list, gripper: float = 0.0):
        """Send a LowCmd to the robot via DDS."""
        if not DDS_AVAILABLE or self._cmd_pub is None:
            clawbot_log("DDS MOCK", f"Would send joints: {[f'{v:.2f}' for v in joint_positions[:8]]}...", C.DIM)
            return

        cmd = unitree_hg_msg_dds__LowCmd_()
        n   = min(len(joint_positions), G1_NUM_JOINTS)
        for i in range(n):
            cmd.motor_cmd[i].q   = float(joint_positions[i])
            cmd.motor_cmd[i].dq  = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp  = DEFAULT_KP
            cmd.motor_cmd[i].kd  = DEFAULT_KD
        cmd.crc = self._crc.Crc(cmd)
        self._cmd_pub.Write(cmd)
        
        # Send gripper commands to Dex1 if available
        if getattr(self, "_left_gripper_pub", None) is not None:
            # Gripper mapping: 0.0 = fully closed, 5.4 = fully open
            # API: gripper=0.0 means open, gripper=1.0 means close
            target_q = 0.0 if gripper >= 0.5 else 5.4
            
            g_cmd = self._motor_cmds_cls()
            m_cmd = self._motor_cmd_cls()
            m_cmd.mode = 1  # 1 is position mode
            m_cmd.q = target_q
            m_cmd.dq = 0.0
            m_cmd.tau = 0.0
            m_cmd.kp = 60.0
            m_cmd.kd = 1.5
            g_cmd.cmds.append(m_cmd)
            self._left_gripper_pub.Write(g_cmd)
            if getattr(self, "_right_gripper_pub", None) is not None:
                self._right_gripper_pub.Write(g_cmd)

# =============================================================================
#  Depth visualization helper
# =============================================================================
def make_depth_colormap(depth_raw: np.ndarray) -> np.ndarray:
    """Convert a raw depth array to a JET colormap image for display."""
    if depth_raw is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    # Normalize to 0-255
    d = depth_raw.astype(np.float32)
    dmin, dmax = d.min(), d.max()
    if dmax - dmin > 1e-5:
        d = (d - dmin) / (dmax - dmin) * 255.0
    else:
        d = np.zeros_like(d)
    d = d.astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


# =============================================================================
#  State reader from DDS (low state subscriber)
# =============================================================================
class StateReader:
    def __init__(self):
        self._joint_positions = [0.0] * G1_NUM_JOINTS
        self._lock = threading.Lock()
        if DDS_AVAILABLE:
            try:
                self._sub = ChannelSubscriber("rt/lowstate", 
                    __import__("unitree_sdk2py.idl.unitree_hg.msg.dds_", fromlist=["LowState_"]).LowState_)
                self._sub.Init(self._on_state, 10)
                clawbot_log("DDS", "State subscriber on rt/lowstate ready.", C.GREEN)
            except Exception as e:
                clawbot_log("DDS WARN", f"State subscriber failed: {e}", C.YELLOW)

    def _on_state(self, msg):
        with self._lock:
            for i in range(min(G1_NUM_JOINTS, len(msg.motor_state))):
                self._joint_positions[i] = float(msg.motor_state[i].q)

    def get_joint_positions(self) -> list:
        with self._lock:
            return list(self._joint_positions)

# =============================================================================
#  Object Detection (OpenCV HSV color segmentation)
# =============================================================================

def draw_detections(frame: np.ndarray,
                    robotics_bboxes: list | None = None,
                    trajectory_points: list | None = None) -> np.ndarray:
    """Draw Gemini Robotics-ER bboxes (magenta) and trajectory (yellow/cyan) onto frame."""
    out  = frame.copy()
    h_px, w_px = out.shape[:2]

    # ---- Draw bboxes ----
    if robotics_bboxes:
        for item in robotics_bboxes:
            box = item.get("box_2d", [])
            label = item.get("label", "?")
            if len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin / 1000 * w_px)
            y1 = int(ymin / 1000 * h_px)
            x2 = int(xmax / 1000 * w_px)
            y2 = int(ymax / 1000 * h_px)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 200), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (255, 0, 200), -1)
            cv2.putText(out, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out, ((x1+x2)//2, (y1+y2)//2), 4, (255, 0, 200), -1)

    # ---- Draw trajectory ----
    if trajectory_points:
        # Collect valid pixel points
        pts_px = []
        for item in trajectory_points:
            pt = item.get("point", [])
            lbl = item.get("label", "?")
            if len(pt) == 2:
                py, px = pt
                x_px = int(px / 1000 * w_px)
                y_px = int(py / 1000 * h_px)
                pts_px.append((x_px, y_px))
                # Draw point dot — color gradient from yellow (start) to cyan (end)
                progress = len(pts_px) - 1
                total = max(len(trajectory_points) - 1, 1)
                r = int(0 + 0 * progress / total)    # yellow → cyan
                g = int(255 - 255 * progress / total)
                b = int(255 * progress / total)
                color = (int(b), int(g), int(r))  # BGR
                cv2.circle(out, (x_px, y_px), 6, color, -1)
                # Label
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(out, lbl, (x_px + 8, y_px + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw connecting lines
        for i in range(len(pts_px) - 1):
            cv2.line(out, pts_px[i], pts_px[i+1], (0, 255, 255), 2)

        # Draw arrow at the end
        if len(pts_px) >= 2:
            cv2.arrowedLine(out, pts_px[-2], pts_px[-1], (0, 200, 255), 3, tipLength=0.15)

    return out


# =============================================================================
#  Async Gemini Worker
# =============================================================================
class AsyncGeminiWorker:
    """Run Gemini API calls in a background thread so the display loop stays smooth."""

    def __init__(self, gemini_client: GeminiClient, commander: RobotCommander):
        self._client    = gemini_client
        self._commander = commander
        self._lock      = threading.Lock()

        # Latest input (written by main loop, read by worker)
        self._pending_rgb   = None
        self._pending_depth = None
        self._pending_joints = None
        self._has_new_input  = False

        # Latest result (written by worker, read by main loop for display)
        self.last_thinking         = ""
        self.last_status           = "idle"
        self.last_robotics_bboxes  = []    # Gemini Robotics-ER box_2d detections
        self.last_trajectory       = []    # trajectory points from Gemini
        self.last_gripper          = 0.0
        self.current_joints        = [0.0] * G1_NUM_JOINTS
        self._step_index           = 1

        # Flow control: prevent re-submitting while executing actions
        self._is_executing         = False
        self._result_received       = threading.Event()  # signals main loop that vis is ready

        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, rgb: np.ndarray, depth: np.ndarray, joints: list):
        """Submit a new frame for Gemini to process (non-blocking)."""
        with self._lock:
            self._pending_rgb    = rgb.copy() if rgb is not None else None
            self._pending_depth  = depth.copy() if depth is not None else None
            self._pending_joints = list(joints)
            self._has_new_input  = True

    def _worker(self):
        while self._running:
            rgb = depth = joints = None
            with self._lock:
                if self._has_new_input and not self._is_executing:
                    rgb    = self._pending_rgb
                    depth  = self._pending_depth
                    joints = self._pending_joints
                    self._has_new_input = False
                    self._is_executing = True   # block new submissions

            if rgb is None:
                time.sleep(0.05)
                continue

            # ----- Call Gemini once to get bboxes + trajectory + actions -----
            try:
                response = self._client.call(
                    rgb_image=rgb, depth_image=depth, joint_positions=joints,
                    step_index=self._step_index)
            except Exception as e:
                clawbot_log("ERROR", f"API call failed: {e}", C.RED)
                with self._lock:
                    self._is_executing = False
                time.sleep(0.5)
                continue

            if not response:
                clawbot_log("WARN", "Empty response from StrikeRobot API.", C.YELLOW)
                with self._lock:
                    self._is_executing = False
                time.sleep(1.0)
                continue

            # ---- Immediately store bboxes + trajectory for visualization ----
            bboxes = response.get("_bboxes", [])
            trajectory = response.get("_trajectory", [])
            with self._lock:
                self.last_robotics_bboxes = bboxes
                self.last_trajectory = trajectory
            self._result_received.set()   # signal main loop: vis is ready NOW
            clawbot_log("DETECT",
                f"Bboxes: {len(bboxes)} | Traj: {len(trajectory)} pts — visualizing now!", C.CYAN)

            reasoning = response.get("reasoning", "")
            steps = response.get("steps", [])

            # Normalize steps from raw list or dict
            if not steps:
                if isinstance(response, list):
                    steps = response
                elif isinstance(response, dict):
                    steps = (response.get("steps") or response.get("actions")
                             or response.get("function_calls") or [])

            if reasoning:
                print()
                print(f"{C.MAGENTA}{C.BOLD}╔══ [Clawbot Reasoning] ═════╗{C.RESET}")
                for line in textwrap.wrap(reasoning, width=80):
                    print(f"{C.MAGENTA}║ {line}{C.RESET}")
                print(f"{C.MAGENTA}{C.BOLD}╚{'═'*26}╝{C.RESET}")
                print()

            if not steps:
                clawbot_log("WARN", "No functional steps generated.", C.YELLOW)
                with self._lock:
                    self._is_executing = False
                time.sleep(1.0)
                continue

            # ---- Load Teleop Target Override ----
            # User requested: Wait for calculation + visualization, THEN execute the teleop json!
            self._result_received.wait(timeout=1.0)
            self._result_received.clear()
            
            import os, json
            if os.path.exists("teleop_calibration.json"):
                try:
                    with open("teleop_calibration.json", "r") as f:
                        tdata = json.load(f)
                    
                    playback_steps = []
                    lowest_pitch = 999
                    lowest_idx = -1
                    
                    # Pass 1: find lowest pitch (where robot reaches table) to auto-close gripper if legacy format
                    for i, rp in enumerate(tdata):
                        j_right = rp.get("right_arm_joints")
                        if j_right and len(j_right) >= 7:
                            if j_right[0] < lowest_pitch:
                                lowest_pitch = j_right[0]
                                lowest_idx = i
                                
                    # Pass 2: build steps
                    for i, rp in enumerate(tdata):
                        j_right = rp.get("right_arm_joints")
                        if j_right:
                            playback_steps.append({"function": "move_right_arm", "args": j_right})
                            
                        # Handle legacy format where gripper wasn't saved: assume grab at lowest pitch
                        if "gripper" not in rp and i == lowest_idx:
                            playback_steps.append({"function": "set_gripper", "args": [True]})
                            
                        # Handle new format where gripper is saved
                        if "gripper" in rp:
                            is_closed = (rp["gripper"] >= 0.5)
                            playback_steps.append({"function": "set_gripper", "args": [is_closed]})
                            
                    if playback_steps:
                        steps = playback_steps
                        clawbot_log("OVERRIDE", f"Ignored AI motion planning. Loaded {len(steps)} playback steps from teleop_calibration.json!", C.MAGENTA)
                except Exception as e:
                    clawbot_log("WARN", f"Failed to override with teleop_calibration.json: {e}", C.YELLOW)

            # ---- Execute action sequence ----
            clawbot_log("EXEC", f"Executing {len(steps)} steps...", C.GREEN)
            for idx, action_step in enumerate(steps, 1):
                if not isinstance(action_step, dict):
                    clawbot_log("WARN", f"Skipping invalid step: {action_step}", C.YELLOW)
                    continue
                    
                func_name = action_step.get("function")
                args = action_step.get("args", [])

                status_str = f"Step {idx}/{len(steps)}: {func_name}({', '.join(map(str, args))})"
                self.last_status = status_str
                clawbot_log("ACTION", status_str, C.GREEN)

                if func_name == "move_right_arm" and len(args) == 7:
                    for i, val in enumerate(args):
                        self.current_joints[22 + i] = float(val)
                    self._commander.send(self.current_joints, gripper=self.last_gripper)
                    time.sleep(1.5)
                elif func_name == "set_gripper" and len(args) == 1:
                    closed = bool(args[0])
                    self.last_gripper = 1.0 if closed else 0.0
                    self._commander.send(self.current_joints, gripper=self.last_gripper)
                    time.sleep(1.0)
                elif func_name == "return_to_origin":
                    self.current_joints = [0.0] * G1_NUM_JOINTS
                    self.current_joints[22] = -0.2
                    self.current_joints[25] = 0.4
                    self._commander.send(self.current_joints, gripper=self.last_gripper)
                    time.sleep(1.5)
                else:
                    clawbot_log("WARN", f"Unknown function or bad args: {func_name}", C.YELLOW)

            # ---- Done: allow next API call ----
            with self._lock:
                self._is_executing = False
            clawbot_log("SUCCESS", "Sequence done! Ready for next command.", C.GREEN)
            self._step_index += 1
            time.sleep(2.0)

    def stop(self):
        self._running = False


# =============================================================================
#  Main Clawbot Loop  (non-blocking, 30fps display)
# =============================================================================
def main():
    print(BANNER)
    clawbot_log("BOOT", "Starting Clawbot controller...", C.MAGENTA)

    if not GEMINI_AVAILABLE:
        sys.exit(1)
    if not GEMINI_API_KEY:
        clawbot_log("ERROR",
            "GEMINI_API_KEY is not set!\nExport it: export GEMINI_API_KEY='your-key'", C.RED)
        sys.exit(1)

    # ---- Initialize components ----
    clawbot_log("INIT", "Connecting to Isaac Sim image server...", C.CYAN)
    image_client = None
    if IMAGE_CLIENT_AVAILABLE:
        try:
            image_client = ImageClient(host=SERVER_HOST)
            clawbot_log("INIT", f"Image client connected to {SERVER_HOST}.", C.GREEN)
        except Exception as e:
            clawbot_log("WARN", f"Image client failed: {e}. Will use blank frames.", C.YELLOW)

    clawbot_log("INIT", "Initializing StrikeRobot API...", C.CYAN)
    gemini = GeminiClient(api_key=GEMINI_API_KEY)

    clawbot_log("INIT", "Initializing DDS commander...", C.CYAN)
    commander = RobotCommander()

    clawbot_log("INIT", "Initializing state reader...", C.CYAN)
    state_reader = StateReader()

    # --- Shared Memory Depth Logic Removed ---
    shm_reader = None

    clawbot_log("INIT", "Starting async Strike Robot worker thread...", C.CYAN)
    worker = AsyncGeminiWorker(gemini, commander)

    clawbot_log("READY", "All systems online. Press [Q] to quit.", C.GREEN)
    clawbot_log("READY", "Press [M] to toggle Manual Control Mode.", C.GREEN)
    clawbot_log("READY", "Manual Keys: Arrows(Pitch/Yaw) | A/D(Roll) | R/F(Elbow) | T/G,Y/H,U/J(Wrist) | O/P(Grip).", C.CYAN)
    clawbot_log("READY", "Press [Z] to reset object position. Press [C] to record point.", C.CYAN)
    clawbot_log("READY", "Robot will detect → plan → execute ONCE per command.", C.GREEN)
    print()

    step = 0
    last_call_time = time.time()
    
    # Flag: we submitted a frame and are waiting for API response to populate vis
    _submitted_this_cycle = False
    manual_mode = True
    recorded_history = []  # Store joints + bboxes for user calibration
    manual_target_joints = []  # To maintain a stable tracking position in manual mode

    try:
        while True:
            now = time.time()
            state_joints = state_reader.get_joint_positions()
                
            # Keep an absolute tracking reference for manual mode to prevent gravity slip
            if not manual_mode or worker._is_executing or not manual_target_joints:
                manual_target_joints = list(worker.current_joints) if worker.current_joints else [0.0]*29

            # ---- Grab latest frames ----
            rgb_frame = None
            if image_client is not None:
                try:
                    rgb_frame, fps = image_client.get_head_frame()
                except Exception as e:
                    clawbot_log("CAMERA ERR", str(e), C.YELLOW)

            if rgb_frame is None:
                rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(rgb_frame, "No RGB (waiting for Isaac Sim...)", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

            # ---- Build preview colormap (Depth handling deferred to AI stage) ----
            actual_depth = None
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            depth_colormap = make_depth_colormap(gray)

            # ---- Read latest worker results ----
            with worker._lock:
                robotics_bboxes   = list(worker.last_robotics_bboxes)
                trajectory_points  = list(worker.last_trajectory)
                ai_status          = worker.last_status
                is_executing       = worker._is_executing

            # Debug: log what's being drawn
            # if robotics_bboxes or trajectory_points:
            #     print(f"{C.DIM}    [DEBUG VIS] bboxes={len(robotics_bboxes)}, traj={len(trajectory_points)}{C.RESET}")

            # ---- Build display RGB ----
            disp_rgb = draw_detections(rgb_frame,
                                      robotics_bboxes=robotics_bboxes,
                                      trajectory_points=trajectory_points)

            # HUD overlay
            status_color = (150, 150, 255) if manual_mode else ((0, 255, 100) if not is_executing else (0, 200, 255))
            mode_txt = "MANUAL " if manual_mode else ("EXECUTING" if is_executing else "IDLE")
            
            cv2.putText(disp_rgb, f"Step: {step}  |  {ai_status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(disp_rgb, f"Mode: {mode_txt}", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
                        
            traj_info = f" | Traj:{len(trajectory_points)}pts" if trajectory_points else ""
            cv2.putText(disp_rgb,
                f"ER:{len(robotics_bboxes)} objs{traj_info}  |  Robotics-ER", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

            state_joints = state_reader.get_joint_positions()

            # ---- Build display Depth map ----
            disp_depth = depth_colormap.copy()
            cv2.putText(disp_depth, "Depth Map (simulated from grayscale)", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for idx, val in enumerate(state_joints[:8]):
                cv2.putText(disp_depth, f"j{idx}={val:+.2f}", (8, 52 + idx * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

            # ---- Show windows (30fps, NON-BLOCKING) ----
            cv2.imshow("Clawbot | RGB (Head Camera)", disp_rgb)
            cv2.imshow("Clawbot | Depth Map", disp_depth)
            
            # ---- Right Wrist Camera ----
            if image_client is not None:
                try:
                    right_wrist_frame, _ = image_client.get_right_wrist_frame()
                    if right_wrist_frame is not None:
                        cv2.imshow("Clawbot | Right Wrist (Gripper)", right_wrist_frame)
                except Exception:
                    pass

            # ---- Keyboard interactions ----
            key = cv2.waitKeyEx(1)
            key_char = key & 0xFF if key != -1 else -1
            
            if key_char == ord('q') or key == 27:
                clawbot_log("QUIT", "User pressed Q. Shutting down.", C.YELLOW)
                break
                
            # Manual Mode: update joint state and send
            delta = 0.03 # Vận tốc di chuyển cánh tay trên mỗi phím bấm (rad/khung hình)
            cmd_sent = False
            
            # Toggle Manual Mode
            if key_char == ord('m'):
                manual_mode = not manual_mode
                clawbot_log("MODE", f"Manual Mode is now {'ON' if manual_mode else 'OFF'}", C.MAGENTA)
            
            with worker._lock:
                j = list(manual_target_joints)
                if not j: j = [0.0]*29
                j_copy = list(j)
                
                # ---- Right Arm Controls ----
                # Arrow keys for Pitch and Yaw
                if key in (65362, 0x260000, 82): j[22] -= delta; cmd_sent = True # Up Arrow (Pitch Down)
                if key in (65364, 0x280000, 84): j[22] += delta; cmd_sent = True # Down Arrow (Pitch Up)
                if key in (65361, 0x250000, 81): j[24] += delta; cmd_sent = True # Left Arrow (Yaw)
                if key in (65363, 0x270000, 83): j[24] -= delta; cmd_sent = True # Right Arrow (Yaw)
                
                # Other keys
                if key_char == ord('a'): j[23] -= delta; cmd_sent = True # Roll Left
                if key_char == ord('d'): j[23] += delta; cmd_sent = True # Roll Right
                if key_char == ord('r'): j[25] -= delta; cmd_sent = True # Elbow Pitch Up
                if key_char == ord('f'): j[25] += delta; cmd_sent = True # Elbow Pitch Down
                
                # Wrist Control keys
                if key_char == ord('t'): j[27] -= delta; cmd_sent = True # Wrist Pitch Up
                if key_char == ord('g'): j[27] += delta; cmd_sent = True # Wrist Pitch Down
                if key_char == ord('u'): j[28] -= delta; cmd_sent = True # Wrist Yaw Left
                if key_char == ord('j'): j[28] += delta; cmd_sent = True # Wrist Yaw Right
                if key_char == ord('y'): j[26] -= delta; cmd_sent = True # Wrist Roll CCW
                if key_char == ord('h'): j[26] += delta; cmd_sent = True # Wrist Roll CW
                if key_char == ord('o'): worker.last_gripper = 0.0; cmd_sent = True  # Right Gripper Open
                if key_char == ord('p'): worker.last_gripper = 1.0; cmd_sent = True  # Right Gripper Close
                
                # Reset Object
                if key_char == ord('z'):
                    commander.reset_scene()
                    
                # Reset Robot Pose & Object
                if key_char == ord('x'):
                    commander.reset_all()
                    with worker._lock:
                        manual_target_joints = [0.0] * 29
                        j = list(manual_target_joints)
                
                # Allow user to update joints and immediately send
                if cmd_sent and manual_mode:
                    # In Manual Mode, we overwrite worker targets and push down to commander
                    synthetic_step = {"function": "move_right_arm", "args": j[22:29]}
                    # Direct send via lowcmd mock
                    worker.current_joints = j
                    manual_target_joints = list(j)
                    commander.send(j, gripper=worker.last_gripper)
                    
                # ---- Record Checkpoint to JSON ----
                if key_char in (ord('c'), ord('C')):
                    entry = {
                        "timestamp": time.time(),
                        "joints": list(j),
                        "right_arm_joints": list(j[22:29]),
                        "gripper": worker.last_gripper,
                        "bboxes": robotics_bboxes
                    }
                    recorded_history.append(entry)
                    clawbot_log("RECORD", f"[{len(recorded_history)}] Saved waypoint to memory! Right Arm: {[round(x,2) for x in j[22:29]]}", C.YELLOW)
                    
            # ---- Submit to Gemini API (Background Reasoning & Detect) ----
            if not is_executing and not manual_mode and now - last_call_time >= CALL_INTERVAL_SEC:
                last_call_time = now
                step += 1
                clawbot_log("CALL", f"Step {step} — Submitting to Gemini for visual analysis...", C.BLUE)
                worker.submit(
                    rgb=rgb_frame,
                    depth=actual_depth if actual_depth is not None else depth_colormap,
                    joints=state_joints,
                )
            time.sleep(0.001)   # yield CPU to other threads

    except KeyboardInterrupt:
        clawbot_log("QUIT", "Interrupted by user (Ctrl+C).", C.YELLOW)
    finally:
        worker.stop()
        cv2.destroyAllWindows()
        if image_client is not None:
            image_client.close()
        
        # Dump memory to JSON
        if recorded_history:
            try:
                import json
                with open("teleop_calibration.json", "w") as f:
                    json.dump(recorded_history, f, indent=4)
                clawbot_log("DONE", f"Saved {len(recorded_history)} recorded waypoints to teleop_calibration.json", C.GREEN)
            except Exception as e:
                pass
                
        clawbot_log("DONE", "Clawbot shut down cleanly.", C.GREEN)


if __name__ == "__main__":
    main()
