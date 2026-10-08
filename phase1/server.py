import torch
import os 
import numpy as np
import h5py
import copy
import time
import time
import random
from utils.data_utils import read_client_data
from utils.dig import DLG

class Sever(object):
    def __init__(self,args,times):
        self.args=args
        self.device=args.device
        self.dataset=args.dataset
        self.num_classes=atgs.num_classes
        self.global_rounds=args.gloval_rounds
        self.local_epochs=atgs.local_epochs
        self.batch_size=args.batch_size
        self.learning_rate=args.local_learning_rate
        self.global_model=copy.deepcopy(args.model)
        self.num_clients=args.num_clients
        self.join_ration=few_shot
        self.algorithm=args.algorithm
        self.time_select=args.time_threthold
        self.save_folder_name=args.save_folder_name
        