CUDA_VISIBLE_DEVICES=0 python train/train_marine.py \
                            --algorithm_name hetgat_mappo \
                            --ppo_epoch 5 \
                            --entropy_coef 0.01 \
                            --lr 5e-4 \
                            --gamma 0.97 \
                            --hidden_size 128 \
                            --use_recurrent_policy \
                            --use_LSTM \
                            --n_rollout_threads 1 \
                            --env_name Marine \
                            --n_types 2 \
                            --num_P 2 \
                            --num_A 1 \
                            --dim 5 \
                            --vision 0 \
                            --num_env_steps 5000000 \
                            --experiment_name training_marine \
                            --episode_length 100 \
                            --save_interval 1000 \
                            --use_eval \
                            --eval_episodes 50 \
                            --eval_interval 1000 \
                            --seed 0 \
                            --logis_init_position_over_the_space \
                            --schedule_rewards \
                            --tensor_obs \
                            --binarized_obs_comm \
                            --cuda \

