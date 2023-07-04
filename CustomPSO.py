import numpy as np
import pyswarms.backend as P
from pyswarms.backend.topology import Star, Ring, VonNeumann, Random, Pyramid
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend.generators import generate_swarm, generate_velocity
from ImageUtility import lb, ub

def custom_pso(func, lb, ub, args=(), swarmsize=100, 
                w=0.5, c1=0.5, c2=0.5, maxiter=100, 
                minstep=1e-8, minfunc=1e-8, debug=False):
    """
    Perform a particle swarm optimization (PSO)
    Stylistically similar to pyswarm.pso, but with a few key differences:
        - The swarm is initialized around the center of the bounds
        - The velocity handler is a linear decrease as the search progresses
        - The boundary handler will shrink velocity when a particle hits a boundary

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    
    Optional
    ========
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    w : inertia weight
        Particle velocity scaling factor (Default: 0.5)
    c1 : cognitive parameter
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    c2 : social parameter
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        
    Returns
    =======
    swarm.best_pos : array
        The best position found during the search
    swarm.best_cost : scalar
        The objective value at ``swarm.best_pos``
    """
    
    # Check for bound shapes
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
    
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    # The velocity clamp is the max and min velocities allowed
    velocity_Clamp = (vlow, vhigh)
    
    # Initialize swarm
    # get dimensions
    dimensions = len(ub)
    bounds = (lb, ub)
    # set values for options
    options = {'c1': c1, 'c2': c2, 'w': w}
    # create the swarm
    swarm = P.create_swarm(n_particles=swarmsize, dimensions=dimensions, 
                           options=options, bounds=bounds,clamp=velocity_Clamp)
    swarm.pbest_pos = np.zeros((swarmsize, dimensions))
    swarm.pbest_cost = np.ones(swarmsize)*np.inf
    # define the topology
    topology = Star()
    
    # Shrink boundary handler will shrink the particle's velocity if it goes out of bounds
    bh = BoundaryHandler("shrink")
    # Velocity will have a linear decrease in inertia
    vh = VelocityHandler("linear")
    
    # Initialize objective function
    obj = lambda x: func(x, *args)
    
    # Iterate until termination criterion met
    for i in range(maxiter):
        
        # update current cost of each particle by objective function
        swarm.current_cost = P.compute_objective_function(swarm, obj)
        print(swarm.current_cost)
        
        # update particle best position and cost for each particle
        swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)
        
        # store the best cost and position of the swarm before updating
        best_cost = swarm.best_cost
        best_position = swarm.best_pos
        # update swarm's best position and cost as dimension vector, cost
        swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)
        
        # Calculate the stepsize of swarm's best position
        stepsize = np.sqrt(np.sum((best_position - swarm.best_pos)**2))
        
        # If the change in the swarm's best position is too small then stop
        if best_cost != [] and np.abs(best_cost - swarm.best_cost) < minfunc:
            print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
            return swarm.best_pos, swarm.best_cost
        # If the stepsize of swarm's best position is too small then stop
        elif stepsize < minstep:
            print('Stopping search: Swarm best position change less than {:}'.format(minstep))
            return swarm.best_pos, swarm.best_cost
            
        # update swarm velocity and position based on topology
        swarm.velocity = topology.compute_velocity(swarm, velocity_Clamp, vh, bounds)
        swarm.position = topology.compute_position(swarm, bounds, bh)
        
        if debug:
            # best after iteration print iteration number, cost, and position
            # Example: "Best after iteration 1: 0.3812 [-0.0012  0.0003]"
            print('Best after iteration {:}: {:} {:}'.format(i+1, swarm.best_cost, swarm.best_pos))
            
    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    return swarm.best_pos, swarm.best_cost