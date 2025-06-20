import argparse
import copy
import time

import d3rlpy
print("d3rlpy version:", d3rlpy.__version__)

from build_dataset import load_trajectories

def train_step_wise():
    base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/"
    dataset = load_trajectories(base_dir)

    print(f"Training using dataset with {dataset.size()} episodes and {dataset.transition_count} steps.")

    small_mlp_encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[32])

    # cql = d3rlpy.algos.CQLConfig(
    #     actor_learning_rate=1e-4,
    #     critic_learning_rate=3e-4,
    #     temp_learning_rate=1e-4,
    #     batch_size=32,
    #     n_action_samples=10,
    #     alpha_learning_rate=0.0,
    #     conservative_weight=10.0,
    #     observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
    #     action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
    #     reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
    #     actor_encoder_factory=small_mlp_encoder,
    #     critic_encoder_factory=small_mlp_encoder,
    # ).create(device=0)

    bc = d3rlpy.algos.BCConfig(
    batch_size=256,
    learning_rate=1e-4,
    policy_type="deterministic",  # "stochastic" can be used if you want diversity in output
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
    action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
    encoder_factory=small_mlp_encoder,
    ).create(device=0)

    logger_adapter = d3rlpy.logging.WanDBAdapterFactory(project="offlineRL")

    # get time 
    exp_time = time.time()

    # # pretraining
    # cql.fit(
    #     dataset,
    #     n_steps=2500,
    #     n_steps_per_epoch=1000,
    #     save_interval=2,
    #     # evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
    #     logger_adapter=logger_adapter,
    #     experiment_name=f"Clip_CQL_pretraining_{exp_time}",
    # )

    bc.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        # logger_adapter=logger_adapter,
        experiment_name=f"Clip_BC_pretraining_{exp_time}",
    )


if __name__ == "__main__":
    train_step_wise()
