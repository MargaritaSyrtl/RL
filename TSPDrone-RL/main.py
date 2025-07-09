import numpy as np 
import os 
import torch 
import random
from utils.options import ParseParams
from utils.env import Env, DataGenerator
from model.nnets import Actor, Critic 
from utils.agent import A2CAgent 
import time

if __name__ == '__main__':
    start = time.time()

    args = ParseParams()
    #random_seed = args['random_seed']
    #print(f"random seed: {random_seed}")
    #if random_seed is not None and random_seed > 0:
    #    print("# Set random seed to %d" % random_seed)
    #np.random.seed(random_seed)
    #random.seed(random_seed)
    #torch.manual_seed(random_seed)

    max_epochs = args['n_train']
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    save_path = args['save_path']
    n_nodes = args['n_nodes']
    dataGen = DataGenerator(args)
    data = dataGen.get_test_all()
    env = Env(args, data)
    actor = Actor(args['hidden_dim'])
    critic = Critic(args['hidden_dim'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        path = save_path + 'n' + str(n_nodes) + '/best_model_actor_truck_params.pkl'
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, map_location='cpu'))
            path = save_path + 'n' + str(n_nodes) + '/best_model_critic_params.pkl'
            critic.load_state_dict(torch.load(path, map_location='cpu'))
            print("Succesfully loaded keys")
    
    agent = A2CAgent(actor, critic, args, env, dataGen)
    if args['train']:
        data = dataGen.get_train_next()
        random_seed = args['random_seed']
        print(f"random seed: {random_seed}")
        if random_seed is not None and random_seed > 0:
            print("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        agent.train()
    else:
        if args['sampling']:
            best_R = agent.sampling_batch(args['n_samples'])
        else:
            R = agent.test()

    end = time.time()
    elapsed = int(end - start)
    if elapsed < 60:
        print(f"Running time: {elapsed}sec")
    else:
        minutes = elapsed // 60
        seconds = elapsed % 60
        print(f"Running time: {minutes}min {seconds}sec")

        
       



