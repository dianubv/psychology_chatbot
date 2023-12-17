import time

out_dir = 'out-psychology'
eval_interval = 250
eval_iters = 200
wandb_log = True # feel free to turn on
wandb_project = 'psychology'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'psychology'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 64
gradient_accumulation_steps = 1
max_iters = 5000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
