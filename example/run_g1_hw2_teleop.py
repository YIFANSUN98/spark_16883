from spark_pipeline import G1BenchmarkPipeline, G1BenchmarkPipelineConfig
from spark_pipeline import G1SafeTeleopSimPipeline, G1SafeTeleopSimPipelineConfig
from spark_utils import update_class_attributes


cfg = G1SafeTeleopSimPipelineConfig()
cfg.env.task.num_obstacle = 1
cfg.max_num_steps = 2000
cfg.env.task.max_episode_length = 2000
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
                            phi_k = 0.1,
                        )
update_class_attributes(cfg.algo.safe_controller.safety_index, safety_index_params)
update_class_attributes(cfg.algo.safe_controller.safe_algo, algo_params[safe_algo])

pipeline = G1SafeTeleopSimPipeline(cfg)
pipeline.run()

