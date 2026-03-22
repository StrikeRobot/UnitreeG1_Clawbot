import torch
import math
from action_provider.action_base import ActionProvider
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

class AutoGraspActionProvider(ActionProvider):
    def __init__(self, env, args_cli):
        self.name = "AutoGraspActionProvider"
        self.env = env
        self.robot = env.scene["robot"]
        self.object = env.scene["object"]
        
        # Configure IK
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        
        # Find arm joints and end effector index
        self.ee_idx = -1
        for i, name in enumerate(self.robot.data.body_names):
            if "right_wrist_roll_link" in name or "right_wrist_yaw_link" in name or ("right" in name and "hand" in name):
                self.ee_idx = i
                # Keep looking but tentatively assume this one
        
        if self.ee_idx == -1:
            print("[AutoGrasp] WARNING: Could not confidently find right wrist body link. Using the end of body list.")
            self.ee_idx = len(self.robot.data.body_names) - 1
        else:
            print(f"[AutoGrasp] Selected end-effector index {self.ee_idx} ({self.robot.data.body_names[self.ee_idx]})")
            
        self.state = "APPROACH"
        self.step_counter = 0
        
    def start(self):
        print("[AutoGraspActionProvider] Started.")
        
    def stop(self):
        print("[AutoGraspActionProvider] Stopped.")
        
    def cleanup(self):
        pass
        
    def get_action(self, env):
        self.step_counter += 1
        
        # Current data
        obj_pos = self.object.data.root_pos_w.clone()
        robot_joints = self.robot.data.joint_pos.clone()
        jacobian = self.robot.root_physx_view.get_jacobians()
        
        # Isaac Lab Jacobian shape varies, usually (num_envs, num_bodies - 1, 6, num_dofs)
        jac_idx = self.ee_idx
        if jacobian.shape[1] == len(self.robot.data.body_names) - 1:
            jac_idx = self.ee_idx - 1
        elif jacobian.shape[1] <= self.ee_idx:
            jac_idx = jacobian.shape[1] - 1

        if jacobian.dim() == 4:
            ee_jacobian = jacobian[:, jac_idx, :, :]
        else:
            # Fallback if different structure
            ee_jacobian = jacobian
            
        ee_pos = self.robot.data.body_pos_w[:, self.ee_idx]
        ee_quat = self.robot.data.body_quat_w[:, self.ee_idx]
        
        # Target Command (Pose: [x,y,z, w,x,y,z])
        ik_cmd = torch.zeros((env.num_envs, 7), device=env.device)
        
        obj_z = obj_pos[0, 2].item()
        obj_y = obj_pos[0, 1].item()
        
        # Simple State Machine
        if self.state == "APPROACH":
            # Go above the object
            target_pos = obj_pos.clone()
            target_pos[:, 2] += 0.20 # 20cm above
            # Fixed orientation for grasp
            target_quat = torch.tensor([[0.0, 0.707, 0.0, 0.707]], device=env.device) # pitch down 90 deg approx
            
            dist = torch.norm(target_pos - ee_pos, dim=-1).item()
            if dist < 0.05 and self.step_counter > 50:
                print("[AutoGrasp] State -> LOWER")
                self.state = "LOWER"
                self.step_counter = 0
                
        elif self.state == "LOWER":
            # Go down to object
            target_pos = obj_pos.clone()
            target_pos[:, 2] += 0.05 # slightly above
            target_quat = torch.tensor([[0.0, 0.707, 0.0, 0.707]], device=env.device)
            
            dist = torch.norm(target_pos - ee_pos, dim=-1).item()
            if dist < 0.02 and self.step_counter > 50:
                print("[AutoGrasp] State -> WAIT_CLOSE")
                self.state = "WAIT_CLOSE"
                self.step_counter = 0
                
        elif self.state == "WAIT_CLOSE":
            target_pos = ee_pos.clone()
            target_quat = ee_quat.clone()
            # Close gripper (joint 28 on G1)
            robot_joints[:, 28] = 1.0 
            
            if self.step_counter > 100:
                print("[AutoGrasp] State -> LIFT")
                self.state = "LIFT"
                self.step_counter = 0
                
        elif self.state == "LIFT":
            target_pos = ee_pos.clone()
            target_pos[:, 2] += 0.3 # lift up 30cm
            target_quat = ee_quat.clone()
            robot_joints[:, 28] = 1.0 # keep closed
            
            if self.step_counter > 100:
                print("[AutoGrasp] State -> DROP")
                self.state = "DROP"
                self.step_counter = 0
                
        else: # DROP
            target_pos = torch.tensor([[0.15, 0.45, 0.825 + 0.2]], device=env.device) # Basket pos + 20cm
            target_quat = ee_quat.clone()
            dist = torch.norm(target_pos - ee_pos, dim=-1).item()
            
            if dist < 0.05:
                robot_joints[:, 28] = 0.0 # open gripper
                
        # Differential IK 
        ik_cmd[..., :3] = target_pos
        ik_cmd[..., 3:] = target_quat # [w, x, y, z] is required by DifferentialIKController
        
        self.diff_ik.set_command(ik_cmd)
        joint_delta = self.diff_ik.compute(ee_pos, ee_quat, ee_jacobian, robot_joints)
        
        action = robot_joints + joint_delta
        return action
