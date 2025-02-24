from spark_pipeline import G1BenchmarkPipeline, G1BenchmarkPipelineConfig
from spark_utils import update_class_attributes

import numpy as np

# ------------------------------- new pipeline ------------------------------- #

cfg = G1BenchmarkPipelineConfig()

# ------------------------------- robot config ------------------------------- #

cfg.robot.cfg.class_name = "G1BasicDynamic2Config"

# -------------------------------- task config ------------------------------- #

cfg.max_num_steps = 2000
cfg.env.task.num_obstacle = 1
cfg.env.task.max_episode_length = 2000
cfg.env.task.obstacle_direction = np.array([0.0, 1.0, 0.0])
cfg.env.task.obstacle_init = np.array([0.2, -1.0, 0.9])
cfg.env.task.obstacle_velocity = 0.1
cfg.env.task.goal_left_velocity = 0.0
cfg.env.task.goal_right_velocity = 0.0
cfg.env.task.mode = "Velocity"

# ------------------------------ nominal policy ------------------------------ #

cfg.algo.policy.class_name = "G1TeleopPIDPolicy2"

# ------------------------------- safe control ------------------------------- #
    
safe_algo = "msc"

safety_index_params = dict(
                            class_name = "BasicCollisionSafetyIndex2ndOrder",
                            phi_n = 1.0,
                            phi_k = 1.0,
                        )

algo_params = {
        'bypass': dict(
                    class_name = "ByPassSafeControl"
                ),
        'msc': dict(
                    class_name = "MySafeControlAlgorithm",
                    eta = 0.5,
                    control_weight = [
                        1.0, 1.0, 1.0, # waist
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # left arm
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # right arm
                    ]
                ),
    }

update_class_attributes(cfg.algo.safe_controller.safety_index, safety_index_params)
update_class_attributes(cfg.algo.safe_controller.safe_algo, algo_params[safe_algo])

# ------------------------------- run pipeline ------------------------------- #

pipeline = G1BenchmarkPipeline(cfg)
pipeline.run()

