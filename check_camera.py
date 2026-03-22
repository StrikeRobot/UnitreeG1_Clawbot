from omni.isaac.sensor import Camera
import numpy as np

camera = Camera(prim_path="/World/Unitree_G1/head_camera") # Đường dẫn tới camera trên G1
camera.initialize()

# Lấy ma trận Intrinsic và Extrinsic
view_matrix = camera.get_view_matrix() # Thế giới -> Camera
intrinsics = camera.get_intrinsics_matrix()
print(view_matrix)
print(intrinsics)