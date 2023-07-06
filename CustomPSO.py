import numpy as np
import pyswarms.backend as P
from pyswarms.backend.topology import Star, Ring, VonNeumann, Random, Pyramid
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend.generators import generate_swarm, generate_velocity
from ImageUtility import lb, ub

def custom_pso(func, lb, ub, args=(), swarmsize=100, 
                w=0.5, c1=0.5, c2=0.5, maxiter=100, 
                minstep=1e-8, minfunc=1e-8, debug=False, inertia_decay=1):
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
        (Default: False)
    intertia_decay : scalar
        The rate at which the inertia weight decreases by proportion scalar*100%
        (Default: 1)
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

    # Initialize objective function
    obj = lambda x: func(x, *args)
    
    # get dimensions
    dimensions = len(ub)
    
    vhigh = np.abs(ub - lb) / 5
    vlow = -vhigh
    # The velocity clamp is the max and min velocities allowed
    velocity_Clamp = (vlow, vhigh)
    
    # Initialize swarm
    bounds = (lb, ub)
    # set values for options
    options = {'c1': c1, 'c2': c2, 'w': w}

    # define the topology
    topology = Star()

    # create the swarm and initialize it
    swarm = P.create_swarm(n_particles=swarmsize, dimensions=dimensions, 
                           options=options, bounds=bounds,clamp=velocity_Clamp)
    # Initialize the swarm's positions and costs
    swarm.current_cost = P.compute_objective_function(swarm, obj)
    swarm.pbest_cost = np.inf * np.ones(swarmsize)
    swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)
    swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)

    # Shrink boundary handler will shrink the particle's velocity if it goes out of bounds
    bh = BoundaryHandler(strategy="random")
    # Velocity handler will decrease the particle's velocity as the search progresses
    vh = VelocityHandler(strategy="unmodified")
    
    # Iterate until termination criterion met
    for i in range(maxiter):

        # inertia weight decreases compoundingly
        swarm.options.update({'w': w * inertia_decay**i})

        # Update the velocity and position of the swarm
        swarm.velocity = topology.compute_velocity(swarm, velocity_Clamp, vh, bounds)
        swarm.position = topology.compute_position(swarm, bounds, bh)

        # Warning that the particles are not within the bounds and need to be clipped
        if not np.all(swarm.position <= ub): Warning('A particle moved out of bounds (upper): ' + str(swarm.position))
        if not np.all(swarm.position >= lb): Warning('A particle moved out of bounds (lower): ' + str(swarm.position))

        # Warning that the velocity of the particles are not within the bounds and need to be clipped
        if not np.all(swarm.velocity <= vhigh): Warning('A particle velocity was not within bounds (upper): ' + str(swarm.velocity))
        if not np.all(swarm.velocity >= vlow): Warning('A particle velocity was not within bounds (lower): ' + str(swarm.velocity))

        # update current cost of each particle by objective function
        swarm.current_cost = P.compute_objective_function(swarm, obj)
        
        # update particle best position and cost for each particle
        swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)
        
        # store the best cost and position of the swarm before updating
        best_cost = swarm.best_cost
        best_position = swarm.best_pos
        # update swarm's best position and cost as dimension vector, cost
        swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)

        # Calculate the step_size of swarm's best position if iteration > 0
        step_size = np.sqrt(np.sum((best_position - swarm.best_pos)**2))
        
        if debug:
            print('******************************')
            # best after iteration print iteration number, cost, and position
            # Example: "Best after iteration 1: 0.3812 [-0.0012  0.0003]"
            print('Best after iteration {:}: {:} {:} {:}'.format(i+1, swarm.best_cost, swarm.best_pos, swarm.options))
            # Print the pbest position and pbest cost for each particle without scientific notation
            print('pbest_pos: ' + str(np.array2string(swarm.pbest_pos, formatter={'float_kind':lambda x: "%.4f" % x})))
            print('pbest_cost: ' + str(np.array2string(swarm.pbest_cost, formatter={'float_kind':lambda x: "%.4f" % x})))
            # Print the velocity matrix for each particle without scientific notation
            print('velocity: ' + str(np.array2string(swarm.velocity, formatter={'float_kind':lambda x: "%.4f" % x})))
            print('******************************')
            
        # if swarm's best position is better than the best position of the swarm before updating
        if swarm.best_cost < best_cost:
            # print itteration number
            if np.abs(best_cost - swarm.best_cost) < minfunc:
                print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                return swarm.best_pos, swarm.best_cost
        # If the stepsize of swarm's best position is too small then stop
        if step_size < minstep and step_size != 0:
            print('Stopping search: Swarm best position change less than {:}'.format(minstep))
            return swarm.best_pos, swarm.best_cost
            
    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    return swarm.best_pos, swarm.best_cost