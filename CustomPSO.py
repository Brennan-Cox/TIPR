import numpy as np
from pyswarms.backend import operators
from pyswarms.backend.topology import Star
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from ImageUtility import lb, ub

def custom_pso(func, lb, ub, args=(), swarmsize=100, 
                w=0.5, c1=0.5, c2=0.5, maxiter=100, 
                minstep=1e-8, minfunc=1e-8, debug=False):
    """
    Perform a particle swarm optimization (PSO)
    

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
    """
    
    # Check for bound shapes
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
    
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Initialize swarm
    # get dimensions
    dimensions = len(ub)
    # set values for options
    options = {'c1': c1, 'c2': c2, 'w': w}
    # initialize the swarm positions
    initial_positions = np.random.uniform(lb, ub, (swarmsize, dimensions))
    # initialize the swarm velocities
    initial_velocities = np.random.uniform(vlow, vhigh, (swarmsize, dimensions))
    # create the swarm
    swarm = Swarm(initial_positions, initial_velocities, 
                  swarmsize, dimensions, options)
    
    # Initialize the topology and the handlers
    topology = Star()
    bounds = (lb, ub)
    bh = BoundaryHandler(strategy="shrink")
    # Velocity will have a linear decrease in inertia
    vh = VelocityHandler(strategy="linear")
    
    obj = lambda x: func(x, *args)
    
    # Iterate until termination criterion met
    for i in range(maxiter):
        
        # update swarm velocity
        swarm.velocity = topology.compute_velocity(swarm=swarm, clamp=None, vh=vh, bounds=bounds)
        swarm.position = topology.compute_position(swarm=swarm, bounds=bounds, bh=bh)
        
        # update swarm current cost matrix
        swarm.current_cost = operators.compute_objective_function(swarm, obj, args)
        
        swarm.pbest_pos, swarm.pbest_cost = operators.compute_pbest(swarm)
        swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)
        
        if debug:
            print('Iteration %i: Best Cost = %.3f' % (i+1, swarm.best_cost))
            
        if swarm.best_cost < minfunc:
            break