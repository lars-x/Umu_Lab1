# --------------
# User Instructions
# 
# Finish the PID in the run function 
#
#
# --------------


 
from math import *
import random
import matplotlib.pyplot as plt

# ------------------------------------------------
# 
# this is the robot class
#

class robot:

    # --------
    # init: 
    #    creates robot and initializes location/orientation to 0, 0, 0
    #

    def __init__(self, length = 20.0):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.length = length
        self.steering_noise = 0.0
        self.distance_noise = 0.0
        self.steering_drift = 0.0

    # --------
    # set: 
    #	sets a robot coordinate
    #

    def set(self, new_x, new_y, new_orientation):

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation) % (2.0 * pi)


    # --------
    # set_noise: 
    #	sets the noise parameters
    #

    def set_noise(self, new_s_noise, new_d_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)

    # --------
    # set_steering_drift: 
    #	sets the systematical steering drift parameter
    #

    def set_steering_drift(self, drift):
        self.steering_drift = drift
        
    # --------
    # move: 
    #    steering = front wheel steering angle, limited by max_steering_angle
    #    distance = total distance driven, most be non-negative

    def move(self, steering, distance, 
             tolerance = 0.001, max_steering_angle = pi / 4.0):

        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0


        # make a new copy
        res = robot()
        res.length         = self.length
        res.steering_noise = self.steering_noise
        res.distance_noise = self.distance_noise
        res.steering_drift = self.steering_drift

        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # apply steering drift
        steering2 += self.steering_drift

        # Execute motion
        turn = tan(steering2) * distance2 / res.length

        if abs(turn) < tolerance:

            # approximate by straight line motion

            res.x = self.x + (distance2 * cos(self.orientation))
            res.y = self.y + (distance2 * sin(self.orientation))
            res.orientation = (self.orientation + turn) % (2.0 * pi)

        else:

            # approximate bicycle model for motion

            radius = distance2 / turn
            cx = self.x - (sin(self.orientation) * radius)
            cy = self.y + (cos(self.orientation) * radius)
            res.orientation = (self.orientation + turn) % (2.0 * pi)
            res.x = cx + (sin(res.orientation) * radius)
            res.y = cy - (cos(res.orientation) * radius)

        return res

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f]'  % (self.x, self.y, self.orientation)
   
    def cte(self, radius):           
        center1 = [radius, radius] #y,x
        center2 = [radius, radius]
        if self.x < radius:
            cte = ((self.x - center1[1])**2 + (self.y - center1[0])**2)**0.5 - radius
        else:
            cte = ((self.x - center2[1])**2 + (self.y - center2[0])**2)**0.5 - radius                
       
        return cte
# ------------------------------------------------------------------------
#
# run - does a single control run.


def run(params, radius):
    dt = params[3]
    myrobot = robot()
    myrobot.set(0.0, radius, pi / 2.0)
    speed = 10.0
    err = 0.0
    int_crosstrack_error = 0.0
    N = 1000
    crosstrack_error = myrobot.cte(radius)
    x_trajectory = []
    y_trajectory = []
    for i in range(N*2):
        diff_crosstrack_error = - crosstrack_error
        crosstrack_error = myrobot.cte(radius)
        diff_crosstrack_error += crosstrack_error
        diff_crosstrack_error = diff_crosstrack_error/dt
        int_crosstrack_error += crosstrack_error*dt       
        steer = -params[0] * crosstrack_error - params[1] * diff_crosstrack_error - params[2] * int_crosstrack_error
        myrobot = myrobot.move(steer, speed*dt)

        if i >= N:
            err += crosstrack_error ** 2
        
        x_trajectory.append(myrobot.x)
        y_trajectory.append(myrobot.y)
    
    return err / float(N), x_trajectory, y_trajectory

def limit_p_i(i, pi, printflag = False):
    p_min = 0.000001
    if pi < p_min:
        if printflag:
            print('limit_p_i :', i, pi)
        if i == 3:
            pi = p_min
    return pi

def twiddle(radius, params, twiddle_dt, printflag = False, tol=0.2):
    # P D I dt
    p = params
    n_params = len(p)
    if not twiddle_dt: n_params = n_params - 1
    dp = [1.0 for k in range(n_params)]
    
    best_err, _, _ = run(p, radius)
    it = 0
    while sum(dp) > tol:
        if printflag:
            print('********************** Iteration:', it)
            print('Parameters : ', p)
            print('dp         : ', dp)
            print('Error      : ', best_err)

        for i in range(n_params):
            p_i = p[i]
            p[i] = p_i + dp[i]            
            err, _, _ = run(p, radius)
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] = p_i - dp[i]
                p[i] = limit_p_i(i, p[i])
                err, _, _ = run(p, radius)
                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] = p_i
                    dp[i] *= 0.9
        
        it += 1

    return p, best_err

radius = 50.0
# P D I dt
# params_guess = [0.9, 0.0, 0.0, 1.0] # => Final parameters:  [0.814724445699, 1.0, 0.0, 1.0] Error:  43.96804735503897
# params_guess = [0.9, 0.0, 0.0, 0.8] # => Final parameters:  [1.8576035379000002, 2.547086236136181, 0.05814973700304011, 0.8] Error:  10.670094569381561
# params_guess = [0.9, 0.0, 0.0, 0.6] # => Final parameters:  [0.06429138897320935, 0.7963749277719623, 0.002541865828328993, 0.6] Error:  0.14811718599201817
# params_guess = [0.9, 0.0, 0.0, 0.5] # => Final parameters:  [0.08590672746124747, 0.9497655169465743, 0.0036055070229635486, 0.5] Error:  0.07110604484476168
params_guess = [0.9, 0.0, 0.0, 0.4] # => Final parameters:  [0.06569235094717615, 0.3476807028344309, 0.03814204568201396, 0.4] Error:  5.654554968624213e-29
# params_guess = [0.9, 0.0, 0.0, 0.1] # => Final parameters:  [1.084717830688594, 1.3120467467824615, 0.37284202899999963, 0.1] Error:  4.700348817668877e-29
params, err = twiddle(radius, params_guess, twiddle_dt=False, printflag=True)

# params_guess = [0.7, 0.19, 0.7, 0.17]
# params_guess = [0.7, 0.191, 0.7, 0.17]
# params_guess = [0.07, 0.191, 0.7, 0.17]
# params, err = twiddle(radius, params_guess, twiddle_dt=True, printflag=True)

err, x_trajectory, y_trajectory = run(params, radius)

print('**********************************************************************************')
print('Final parameters: ', params)
print('Error: ', err)

n = len(x_trajectory)
start = n // 2
end = n
# start = 0
# end = 20
x = x_trajectory[start:end]
y = y_trajectory[start:end]

plt.plot(x, y, color="r", linestyle="--", marker="*", linewidth=1.0)
title = '2-2 PID N=1000 dt=0.4'
plt.title(title)
plt.grid(True)
plt.axis('square')
# plt.savefig(title + '.png')
plt.show()
