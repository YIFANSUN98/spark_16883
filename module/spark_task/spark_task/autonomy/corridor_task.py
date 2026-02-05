from spark_task.autonomy.benchmark_task import BenchmarkTask, TaskObject3D
from spark_utils import Geometry, VizColor
import numpy as np

class CorridorTask(BenchmarkTask):
    def __init__(self, robot_cfg, robot_kinematics, agent, **kwargs):
        super().__init__(robot_cfg, robot_kinematics, agent, **kwargs)

    def _init_obstacle(self):
        super()._init_obstacle()
        wall_1 = TaskObject3D(velocity=0.0, 
                                    keep_direction_step = self.obstacle_keep_direction_step,
                                    bound=self.obstacle_range,
                                    direction=self.obstacle_direction,
                                    smooth_weight = self.obstacle_smooth_weight,
                                    _seed = self._seed + len(self.obstacle_task),
                                    dt = self.dt)
        wall_1.frame[:3,3] = np.array([0.0, -1.0, 0.8])
        obstacle_geom = Geometry(type="box", length=10.0, width=0.1, height=1.6, color=VizColor.obstacle_task)
        self.obstacle_task.append(wall_1)
        self.obstacle_task_geom.append(obstacle_geom)
        
        wall_2 = TaskObject3D(velocity=0.0, 
                                    keep_direction_step = self.obstacle_keep_direction_step,
                                    bound=self.obstacle_range,
                                    direction=self.obstacle_direction,
                                    smooth_weight = self.obstacle_smooth_weight,
                                    _seed = self._seed + len(self.obstacle_task),
                                    dt = self.dt)
        wall_2.frame[:3,3] = np.array([0.0, 1.0, 0.8])
        obstacle_geom = Geometry(type="box", length=10.0, width=0.1, height=1.6, color=VizColor.obstacle_task)
        self.obstacle_task.append(wall_2)
        self.obstacle_task_geom.append(obstacle_geom)
    