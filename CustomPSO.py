import numpy as np
import pyswarms.backend as P
from pyswarms.backend.topology import Star, Ring, VonNeumann, Random, Pyramid
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from ImageUtility import lb, ub
from matplotlib import pyplot as plt

def custom_pso(func, lb, ub, args=(), swarmsize=100, 
                w=0.5, c1=0.5, c2=0.5, maxiter=100, 
                minstep=1e-8, minfunc=1e-8, debug=False, inertia_decay=1, inital_position=None):
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
        Scaling factor to search towrds the particle's best known position
        (Default: 0.5)
    c2 : social parameter
        Scaling factor to search towards the swarm's best known position
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
    # take input positions
    if inital_position is not None:
        swarm.position = inital_position
    # Initialize the swarm's positions and costs
    swarm.current_cost = P.compute_objective_function(swarm, obj)
    swarm.pbest_cost = np.inf * np.ones(swarmsize)
    swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)
    swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)

    # assert that swarm is fully initialized with correct dimensions
    assert np.array(swarm.position).shape == (swarmsize, dimensions), 'Swarm position must be size (n_particles, dimensions)'
    assert np.array(swarm.velocity).shape == (swarmsize, dimensions), 'Swarm velocity must be size (n_particles, dimensions)'
    assert swarm.n_particles > 0 and swarm.n_particles == swarmsize, 'Swarm size must be a positive integer'
    assert swarm.dimensions == dimensions, 'swarm dimensions and instance dimenstions must be equal'
    assert np.array(swarm.current_cost).shape == (swarmsize, ), 'Swarm current_cost must be size (n_particles,)'
    assert np.array(swarm.pbest_pos).shape == (swarmsize, dimensions), 'Swarm pbest_pos must be size (n_particles, dimensions)'
    assert np.array(swarm.pbest_cost).shape == (swarmsize,), 'Swarm pbest_cost must be size (n_particles,)'
    assert np.array(swarm.best_pos).shape == (dimensions,), 'Swarm best_pos must be size (dimensions,)'
    assert np.isscalar(swarm.best_cost), 'Swarm best_cost must be a scalar'

    # Shrink boundary handler will shrink the particle's velocity if it goes out of bounds
    bh = BoundaryHandler(strategy="random")
    # Velocity handler will decrease the particle's velocity as the search progresses
    vh = VelocityHandler(strategy="unmodified")
    
    # Iterate until termination criterion met
    collisions = 0
    for i in range(maxiter):

        # inertia weight decreases compoundingly
        swarm.options.update({'w': w * inertia_decay**i})

        # Update the velocity and position of the swarm
        swarm.velocity = topology.compute_velocity(swarm, velocity_Clamp, vh, bounds)
        # Add some ratio matrix multiplied by velocity matrix
        swarm.position = topology.compute_position(swarm, bounds, bh)
        # if particles are too close to each other, move them apart
        for k in range(swarmsize):
            for j in range(swarmsize):
                if k == j: continue
                # take the distance between two particles
                # if that distance is less than the minimum step, move the particle
                # to a random position within the bounds
                if np.sqrt(np.sum((swarm.position[k] - swarm.position[j])**2)) < minstep:
                    collisions += 1
                    # move the particle just outside the bounds
                    # swarm.position[k] = swarm.position[k] + np.random.uniform(-1, 1, dimensions) * minstep
                    swarm.position[k] = np.random.uniform(lb, ub, dimensions)
                    swarm.velocity[k] = np.random.uniform(vlow, vhigh, dimensions)

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
                print('Stopping search: Swarm best objective change less than {:} iterations {:} collisions {:}'.format(minfunc, i + 1, collisions))
                return swarm.best_pos, swarm.best_cost, collisions, i
        # If the stepsize of swarm's best position is too small then stop
        if step_size < minstep and step_size != 0:
            print('Stopping search: Swarm best position change less than {:} iterations {:} collisions {:}'.format(minstep, i + 1, collisions))
            return swarm.best_pos, swarm.best_cost, collisions, i
            
    print('Stopping search: maximum iterations reached --> {:} collisions {:}'.format(maxiter, collisions))
    return swarm.best_pos, swarm.best_cost, collisions, i