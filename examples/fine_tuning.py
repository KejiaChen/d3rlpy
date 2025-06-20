import argparse
import copy

import d3rlpy
print("d3rlpy version:", d3rlpy.__version__)

def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--dataset", type=str, default="mujoco/hopper/medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    dataset, env = d3rlpy.datasets.get_minari(args.dataset)
    print(f"Training using dataset with {dataset.size()} episodes and {dataset.transition_count} steps.")
    
    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    small_mlp_encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[128, 128])

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        batch_size=128, # 256
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=10.0,
        actor_encoder_factory=small_mlp_encoder,
        critic_encoder_factory=small_mlp_encoder,
    ).create(device=args.gpu)

    '''Traning '''
    # print("Training CQL with wandb logger...")
    logger_adapter = d3rlpy.logging.WanDBAdapterFactory(project="offlineRL")

    # # pretraining
    cql.fit(
        dataset,
        n_steps=50000, # 100000,
        n_steps_per_epoch=1000,
        save_interval=2,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env),
                    "soft_opc": d3rlpy.metrics.SoftOPCEvaluator(return_threshold=2800),
                    },
        logger_adapter=logger_adapter,
        experiment_name=f"CQL_pretraining_{args.dataset}_{args.seed}",
    )

    # print(f"CQL training with n_steps*batch_size {10000*256} steps completed.")

    '''Evaluation '''
    # load pretrained models
    # logger = logger_adapter.create(
    #     algo=cql,
    #     experiment_name=f"CQL_pretraining_{args.dataset}_{args.seed}",
    #     n_steps_per_epoch=1000,
    # )
    # pretrained_model_path = logger.load_model(
    #     # f"CQL_pretraining_{args.dataset}_{args.seed}"
    #     "kejia-chen-tum-tu-munich/offlineRL/model-370000:v0"
    # )
    pretrained_model_path = "/home/tp2/Documents/kejia/d3rlpy/d3rlpy_logs/CQL_pretraining_mujoco/hopper/medium-v0_1_20250617175602/model_50000.pt"

    # option 1: online evaluation
    eval_env = copy.deepcopy(env)
    d3rlpy.envs.seed_env(eval_env, args.seed)
    cql.build_with_env(eval_env)
    cql.load_model(pretrained_model_path)
    # exp_return = d3rlpy.metrics.evaluate_qlearning_with_environment(cql, eval_env, n_trials=30)
    # print(f"Average return of CQL on {args.dataset} with seed {args.seed}: {exp_return}")

    # option 2: off-policy evaluation algorithm
    cql = d3rlpy.load_learnable("model.d3")
    fqe = d3rlpy.ope.FQE(algo=cql, config=d3rlpy.ope.FQEConfig())
    # train estimators to evaluate the trained policy
    fqe.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        evaluators={
            'init_value': d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
            'soft_opc': d3rlpy.metrics.SoftOPCEvaluator(return_threshold=2800),
        },
    )

    # sac = d3rlpy.algos.SACConfig(
    #     actor_learning_rate=3e-4,
    #     critic_learning_rate=3e-4,
    #     temp_learning_rate=3e-4,
    #     batch_size=256,
    # ).create(device=args.gpu)

    # # copy pretrained models to SAC
    # sac.build_with_env(env)
    # # sac.copy_policy_from(cql)  # type: ignore
    # # sac.copy_q_function_from(cql)  # type: ignore
    # # sac.load_model(pretrained_model_path)

    # # prepare FIFO buffer filled with dataset episodes
    # buffer = d3rlpy.dataset.create_fifo_replay_buffer(
    #     limit=100000,
    #     episodes=dataset.episodes,
    # )

    # # finetuning
    # eval_env = copy.deepcopy(env)
    # d3rlpy.envs.seed_env(eval_env, args.seed)
    # sac.fit_online(
    #     env,
    #     buffer=buffer,
    #     eval_env=eval_env,
    #     experiment_name=f"SAC_finetuning_{args.dataset}_{args.seed}",
    #     n_steps=100000,
    #     n_steps_per_epoch=1000,
    #     save_interval=10,
    #     # logger_adapter=logger_adapter
    # )


if __name__ == "__main__":
    main()
