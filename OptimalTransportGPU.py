import numpy as np
import torch
import time

from scipy.spatial.distance import cdist
from transport import transport_torch
from matching import matching_torch_v1
from ImageUtility import images_To_Ndarrays


#destination set DA or demand set A : ndarray
#source set SB or supply set B : ndarray
#cost_matrix between elements of A to each element in B
#delta or the value epsilon in paper or accuracy**
#returns a cost tensor and the time in seconds the calculation took
def find_Cost(a, b, DA, SB, delta, is_Transport):
    with torch.no_grad():
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #for small data it seems cpu is a better device to use
        device = torch.device('cpu')
        cost_matrix = cdist(b, a, 'sqeuclidean')
        # print(cost_matrix, "\n")
        cost_tensor = torch.tensor(cost_matrix, device=device, requires_grad=False)
        DA_tensor = torch.tensor(DA, device=device, requires_grad=False)
        SB_tensor = torch.tensor(SB, device=device, requires_grad=False)
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        C_tensor = torch.tensor([cost_matrix.max()], device=device, requires_grad=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if is_Transport:
            F, yA, yB, total_cost, iteration = transport_torch(DA_tensor, SB_tensor, cost_tensor, delta_tensor, device=device)
        else:
            F, yA, yB, total_cost, iteration = matching_torch_v1(cost_tensor, C_tensor, delta_tensor, device=device)
        end = time.perf_counter()
        return F, yA, yB, total_cost, iteration, (end - start)
    
    # Finds the cost between two images and returns all params
def find_Cost_Between_Images(imageOne, imageTwo, is_Transport=True, delta=0.1, Threshold=100):
    a, b, DA, SB = images_To_Ndarrays(imageOne, imageTwo, Threshold)
    # print('Elements of A', a)
    # print('Elements of B', b)
    # print ('Demand A', DA)
    # print('Supply B', SB)
    F, yA, yB, total_cost, iteration, time = find_Cost(a, b, DA, SB, delta, is_Transport)
    return (a, b, F, yA, yB, total_cost, iteration, time, DA, SB)