from spark_robot import RobotKinematics
from spark_utils import Geometry, compute_pairwise_dist
import numpy as np

from spark_safe.safe_algo.value_based.base.basic_collision_safety_index import BasicCollisionSafetyIndex

class BasicCollisionSafetyIndex2ndOrder(BasicCollisionSafetyIndex):
    
    def __init__(self,
                 robot_kinematics : RobotKinematics,
                 **kwargs
                 ):
        
        super().__init__(robot_kinematics, **kwargs)

        self.n = kwargs["phi_n"]
        self.k = kwargs["phi_k"]
            
        # General implementation of second order safety index; normal direction is the distance decreasing direction
        self.my_phi = lambda d, v, normal: self.dmin**self.n - np.sign(np.sum(d * normal, axis=1)) * np.abs(np.sum(d * normal, axis=1))**self.n + self.k * np.sum(v * normal, axis=1)
        
        # General implementation of the time derivative of second order safety index in the Cartesian space
        self.my_phi_dot = lambda d, v, a, normal, curv: self.n*np.abs(np.sum(d * normal, axis=1))**(self.n-1)*np.sum(v * normal, axis=1) + self.k*np.sum(a * normal, axis=1) - self.k*np.sum(v * (curv@v[..., None]).squeeze(-1), axis=1)

        self.Cartesian_Lg = lambda normal: self.k*normal
        self.Cartesian_Lf = lambda d, v, normal, curv: self.n*np.abs(np.sum(d * normal, axis=1))**(self.n-1)*np.sum(v * normal, axis=1) - self.k*np.sum(v * (curv@v[..., None]).squeeze(-1), axis=1)
    
    def phi(self,
            x        : np.ndarray,
            task_info: dict):
        '''
            [num_constraint,] Safety index function.

            Args:
                x (np.ndarray): [num_state,] robot state.
        '''
        
        robot_collision_vol, obstalce_collision_vol = self.get_vol_info(x, task_info)
        d, v, normal, curv = self.compute_pairwise_vol_info(robot_collision_vol, obstalce_collision_vol)
        
        phi = self.my_phi(d, v, normal)
        Cartesian_Lg = self.Cartesian_Lg(normal)
        Cartesian_Lf = self.Cartesian_Lf(d, v, normal, curv)
        
        q = self.robot_cfg.decompose_state_to_dof_pos(x)
        dq = self.robot_cfg.decompose_state_to_dof_vel(x)
        
        jacobian_full = self.jacobian_full(q)
        jacobian_dot_full = self.jacobian_dot_full(q, dq)
        
        Lg = np.einsum('ij,ijk->ik', Cartesian_Lg, jacobian_full)
        Lf = Cartesian_Lf + np.einsum('ij,ijk->ik', Cartesian_Lg, jacobian_dot_full) @ dq
        return phi, Lg, Lf
        
    def jacobian_full(self, q):
        jacobian = np.zeros((len(self.robot_cfg.Frames), 3, len(self.robot_cfg.DoFs)))
        
        for i, frame in enumerate(self.robot_cfg.Frames):
            frame_id = self.robot_kinematics.frame_dict[frame.name]
            jacobian[i] = self.robot_kinematics.compute_frame_jacobian(q, frame_id)[:3, :]
        
        jacobian_full_env = np.zeros((self.num_robot_vol, self.num_obstacle_vol, 3, len(self.robot_cfg.DoFs)))
        jacobian_full_env[:, :, :, :] = jacobian[:, np.newaxis, :, :]
        jacobian_full = np.zeros((self.num_constraint, 3, len(self.robot_cfg.DoFs)))
        jacobian_full[:self.num_constraint_env, :, :] = jacobian_full_env.reshape(-1, 3, len(self.robot_cfg.DoFs))
        
        if self.num_constraint_self > 0:
            jacobian_full_self = np.zeros((self.num_robot_vol, self.num_robot_vol, 3, len(self.robot_cfg.DoFs)))
            jacobian_full_self[:, :, :, :] = jacobian[:, np.newaxis, :, :]
            jacobian_full[self.num_constraint_env:, :, :] = jacobian_full_self[np.triu_indices(self.num_robot_vol, k=1)].reshape(-1, 3, len(self.robot_cfg.DoFs))
        
        return jacobian_full
        
    def jacobian_dot_full(self, q, dq):
        jacobian_dot = np.zeros((len(self.robot_cfg.Frames), 3, len(self.robot_cfg.DoFs)))
        
        for i, frame in enumerate(self.robot_cfg.Frames):
            frame_id = self.robot_kinematics.frame_dict[frame.name]
            jacobian_dot[i] = self.robot_kinematics.compute_frame_jacobian_dot(q, dq, frame_id)[:3, :]
            
        jacobian_dot_full_env = np.zeros((self.num_robot_vol, self.num_obstacle_vol, 3, len(self.robot_cfg.DoFs)))
        jacobian_dot_full_env[:, :, :, :] = jacobian_dot[:, np.newaxis, :, :]
        jacobian_dot_full = np.zeros((self.num_constraint, 3, len(self.robot_cfg.DoFs)))
        jacobian_dot_full[:self.num_constraint_env, :, :] = jacobian_dot_full_env.reshape(-1, 3, len(self.robot_cfg.DoFs))
        
        if self.num_constraint_self > 0:
            jacobian_dot_full_self = np.zeros((self.num_robot_vol, self.num_robot_vol, 3, len(self.robot_cfg.DoFs)))
            jacobian_dot_full_self[:, :, :, :] = jacobian_dot[:, np.newaxis, :, :]
            jacobian_dot_full[self.num_constraint_env:, :, :] = jacobian_dot_full_self[np.triu_indices(self.num_robot_vol, k=1)].reshape(-1, 3, len(self.robot_cfg.DoFs))
        
        return jacobian_dot_full