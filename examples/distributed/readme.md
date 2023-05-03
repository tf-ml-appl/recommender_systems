gcloud_ps_strategy runs distributed training on gcloud cluster (or instance groups). Example usage of demonstrating ps-strategy training on a local cluster:

gcloud_ps_strategy.run_local(task_type='master')
# wait for the cluser set up.
time.sleep(120)

for i in range(NUM_PS):
    gcloud_ps_strategy.run_local(task_type='ps', taks_id=i)

for i in range(NUM_WORKER):
    gcloud_ps_strategy.run_local(task_type='worker', task_id=i)

sample outputs:

Finished epoch 0, accuracy is 0.269120.
Finished epoch 1, accuracy is 0.322392.
Finished epoch 2, accuracy is 0.338255.
Finished epoch 3, accuracy is 0.341649.