from spark_pipeline import G1BenchmarkPipeline, G1BenchmarkPipelineConfig
from spark_utils import update_class_attributes

import numpy as np

cfg = G1BenchmarkPipelineConfig()
cfg.env.task.num_obstacle = 1
cfg.max_num_steps = 2000
cfg.env.task.max_episode_length = 2000
cfg.env.task.obstacle_direction = np.array([0.0, 1.0, 0.0])
cfg.env.task.obstacle_init = np.array([0.2, -0.5, 0.9])
cfg.env.task.obstacle_velocity = 0.01
cfg.env.task.goal_left_velocity = 0.0
cfg.env.task.goal_right_velocity = 0.0
cfg.env.task.mode = "Velocity"
cfg.robot.cfg.class_name = "G1BasicDynamic2Config"
cfg.algo.policy.class_name = "G1TeleopPIDPolicy2"
cfg.algo.safe_controller.safety_index.class_name = "BasicCollisionSafetyIndex2ndOrder"
cfg.env.task.enable_ros = False # if ROS is needed for receiving task info
cfg.algo.safe_controller.safety_index.min_distance['environment'] = 0.1
cfg.env.agent.obstacle_debug["manual_movement_step_size"] = 0.02 # Tune step size for keyboard controlled obstacles
    
safe_algo = "msc"

algo_params = {
        'bypass': dict(
                    class_name = "ByPassSafeControl"
                ),
        'msc': dict(
                    class_name = "MySafeControlAlgorithm",
                    eta = 0.1,
                    control_weight = [
                        1.0, 1.0, 1.0, # waist
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # left arm
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # right arm
                    ]
                ),
    }

safety_index_params = dict(
                            class_name = "BasicCollisionSafetyIndex2ndOrder",
                            phi_n = 1.0,
                            phi_k = 10.0,
                        )
update_class_attributes(cfg.algo.safe_controller.safety_index, safety_index_params)
update_class_attributes(cfg.algo.safe_controller.safe_algo, algo_params[safe_algo])

pipeline = G1BenchmarkPipeline(cfg)
pipeline.run()

