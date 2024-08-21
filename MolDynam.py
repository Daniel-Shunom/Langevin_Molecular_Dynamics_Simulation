import numpy as np;
import matplotlib.pylab as plt;

Avogadro = 6.0221407e23
Boltzmann = 1.380649e-23

#enforces boundary conditions by reversing the direction of the particle if it is coming from the left or right
def wallHitCheck(pos, vels, box):
    ndims = len(box)

    for i in range(ndims):
        vels[(pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])] *= -1


def integrate(pos, vels, forces, mass, dt):
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T


#computes dissipated and random force and returns them as an array
def computeForce (mass, vels, temp, relax, dt):
    
    natoms, ndims = vels.shape

    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T

    force = - (vels * mass[np.newaxis].T)/ relax + noise

    return force


def run(**args):
    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps = args['mass'], args['relax'], args['steps'] 

    
    ndims = len(box)
    pos = np.random.rand(natoms, ndims)

    for i in range (ndims):
        pos[:, i] = box[i][0] + (box[i][1] - box[i][0]) * pos[:,i]

    vels = np.random.rand(natoms, ndims)
    mass = np.ones(natoms) * mass / Avogadro
    step = 0
    
    output = []
    
    #loops throug the number of steps
    while step <= nsteps:
        step += 1

        forces = computeForce(mass, vels, temp, relax, dt)

        integrate(pos, vels, forces, mass, dt)

        #Enforces the box boundary constraints
        wallHitCheck(pos, vels, box)

        #calculates the instantaenous temperature of the system
        insta_temp  = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2)) / (Boltzmann * ndims * natoms)
        output.append([dt * step, insta_temp])

    return np.array(output)

if __name__ == "__main__" :
    params = {
        'natoms': 1000,
        'radius': 120e-12,
        'mass': 1e-3,
        'dt': 1e-15,
        'relax': 1e-13,
        'temp': 300,
        'steps': 10000,
        'freq': 100,
        'box': ((0,1e-8), (0,1e-8), (0,1e-8)),
        'ofname': 'traj-hydrogen. dump'
    }

    output = run(**params)

    plt.plot(output[:,0] * 1e12, output[:,1])