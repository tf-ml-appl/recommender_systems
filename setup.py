from examples import plain_deep_main
from examples.distributed import local_ps_strategy
from examples.distributed import gcloud_ps_strategy

import argparse
import logging
import os

NUM_WORKER = 2
NUM_PS = 2

def parse_args():
    parser = argparse.ArgumentParser(description='process program level arguments.')
    parser.add_argument('filename', default='main.py')

    args = parser.parse_args()
    return args

# TODO: find path file and add the root path to python path.
if __name__=="__main__":
    os.environ["GRPC_FAIL_FAST"] = "use_caller"
    os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
    logging.getLogger().setLevel(logging.WARNING)
    # the root path is added to python path by default;or call add_root_path().
        
    # plain_deep_main.run()
    # local_ps_strategy.run()
    
    gcloud_ps_strategy.run_local(task_type='master')
    # wait for the cluser set up.
    time.sleep(120)
    
    for i in range(NUM_PS):
        gcloud_ps_strategy.run_local(task_type='ps', taks_id=i)
    
    for i in range(NUM_WORKER):
        gcloud_ps_strategy.run_local(task_type='worker', task_id=i)
    
