import numpy as np
from matplotlib import pyplot as plt
from numpy.core.function_base import linspace


def NewTrajectory():
    x = np.linspace(-3,3,1000)
    y = np.linspace(-10,10,1000)
    #fx = np.exp(   x*np.exp(( y*1j )   ))
    z = x + (y * 1j)
    fx = np.exp(z)
    plt.plot(np.real(fx),np.imag(fx))
    plt.show()


def VelocityProfile(crusing_speed, time_for_crusing_speed):
    alpha = np.log(199)/time_for_crusing_speed #alpha is the accelerating_factor
    x = np.linspace(0,10,100)
    y = (1 / (1 + np.exp(-alpha*x))*2)-1
    omega = y*crusing_speed
    plt.plot(x,omega)
    plt.show()
    #plt.show()


# def Joaquins( crusing_speed, time_for_crusing_speed, time_in_crusing_speed, time_to_stop):
#     total_time = time_for_crusing_speed + time_in_crusing_speed + time_to_stop + time_to_stop*3
#     x = np.linspace(0,total_time,100)
#     y = np.zeros(x.shape)

#     alpha_acc = (-1/time_for_crusing_speed) * np.log(0.01) #alpha is the accelerating_factor
#     indexes_acceleration =
#     y[indexes_acceleration] = 1 - np.exp(-alpha_acc*x[indexes_acceleration])

#     alpha_dec = (-1/time_to_stop) * np.log(0.01)


#     omega = y*crusing_speed
#     plt.plot(x,omega)
#     plt.show()


def GettingPlot():
    time_for_crusing_speed = 3
    time_in_crusing_speed = 3
    time_to_stop = 1
    total_time = time_for_crusing_speed + time_in_crusing_speed + time_to_stop + time_to_stop*3
    x = np.linspace(0,total_time,1000)
    y = np.zeros(x.shape)
    # print(x)
    # print(x[:time_for_crusing_speed*10])
    # print(x[time_for_crusing_speed*10:(time_for_crusing_speed+time_in_crusing_speed)*10:])
    # print(x[(time_for_crusing_speed+time_in_crusing_speed)*10:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*10])
    # print(x[(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*10:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop+time_to_stop*3)*10])

    alpha_acc = (-1/time_for_crusing_speed) * np.log(0.01) #alpha is the accelerating_factor
    y[:time_for_crusing_speed*100] = 1 - np.exp(-alpha_acc*x[:time_for_crusing_speed*100])
    plt.plot(x[:time_for_crusing_speed*100], y[:time_for_crusing_speed*100], 'r', LineWidth = 2)
    y[time_for_crusing_speed*100:(time_for_crusing_speed+time_in_crusing_speed)*100:] = 1 - np.exp(-alpha_acc*x[time_for_crusing_speed*100:(time_for_crusing_speed+time_in_crusing_speed)*100:])
    plt.plot(x[time_for_crusing_speed*100:(time_for_crusing_speed+time_in_crusing_speed)*100:], y[time_for_crusing_speed*100:(time_for_crusing_speed+time_in_crusing_speed)*100:],'b', LineWidth = 2)
    alpha_dec = (-1/time_to_stop) * np.log(1 - y[(time_for_crusing_speed+time_in_crusing_speed)*100 - 1])
    y[(time_for_crusing_speed+time_in_crusing_speed)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100] = 1 - np.exp(-alpha_dec*((time_for_crusing_speed + time_in_crusing_speed + time_to_stop) - x[(time_for_crusing_speed+time_in_crusing_speed)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100]))
    plt.plot(x[(time_for_crusing_speed+time_in_crusing_speed)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100], y[(time_for_crusing_speed+time_in_crusing_speed)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100], 'g', LineWidth = 2)
    y[(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop+time_to_stop*3)*100] = 0
    plt.plot(x[(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop+time_to_stop*3)*100], y[(time_for_crusing_speed+time_in_crusing_speed+time_to_stop)*100:(time_for_crusing_speed+time_in_crusing_speed+time_to_stop+time_to_stop*3)*100], 'k', LineWidth = 2)
    plt.title(r'Parameters', fontsize=15, fontweight='bold')
    plt.ylabel(r'$\omega / \omega_{max}$', fontsize=15, fontweight='bold')
    plt.xlabel(r'$t$', fontsize=15, fontweight='bold')
    #plt.plot(x,y)
    plt.show()


class RotationCalculator(object):

    def __init__(self, crusing_speed, time_for_crusing_speed, time_in_crusing_speed, time_to_stop, delta_t):
        self.crusing_speed = crusing_speed
        self.time_for_crusing_speed = time_for_crusing_speed
        self.time_in_crusing_speed = time_in_crusing_speed
        self.time_to_stop = time_to_stop
        self.delta_t = delta_t
        self.old_theta = 0
        self.accelerating_parameter = (-1/self.time_for_crusing_speed) * np.log(0.01)
        self.decelerating_parameter = None

    def compute_rotation_angle(self, t):

        if t <= (self.time_for_crusing_speed + self.time_in_crusing_speed):
            omega = self.accelarating_function(t)


        elif t > (self.time_for_crusing_speed + self.time_in_crusing_speed) and t <= (self.time_for_crusing_speed + self.time_in_crusing_speed + self.time_to_stop):
            if self.decelerating_parameter is None:
                #setting up decelerating parameter
                self.decelerating_parameter = self.calculate_decelerating_parameter(t)
            omega = self.decelerating_function(t)

        elif t > (self.time_for_crusing_speed + self.time_in_crusing_speed + self.time_to_stop):
            omega = 0

        theta = self.old_theta +  omega * self.delta_t  * self.crusing_speed
        self.old_theta = theta

        #return theta
        return omega

    def calculate_accelerating_parameter(self):
        return (-1/self.time_for_crusing_speed) * np.log(0.01)

    def calculate_decelerating_parameter(self, t):
        return (-1/self.time_to_stop) * np.log(1 - self.accelarating_function(t - self.delta_t))

    def accelarating_function(self, t):
        return 1 - np.exp(- self.accelerating_parameter *t)

    def decelerating_function(self, t):
        return 1 - np.exp(-self.decelerating_parameter*((self.time_for_crusing_speed + self.time_in_crusing_speed + self.time_to_stop) - t))

if __name__=='__main__':
    #NewTrajectory() #this works!
    crusing_speed = 1
    time_for_crusing_speed = 1
    time_in_crusing_speed = 1
    time_to_stop = 1

    # # VelocityProfile(crusing_speed,time_for_crusing_speed)
    # Joaquins(crusing_speed,time_for_crusing_speed)  #This is easier XD XD XD
    #GettingPlot()
    time = np.linspace(0,15,1000)
    delta_t = time[1]
    CalculatorObject = RotationCalculator( crusing_speed, time_for_crusing_speed, time_in_crusing_speed, time_to_stop, delta_t )
    omega = np.zeros(len(time))
    for i in range(len(time)):
        #print('time:', time[i] , ' velocity: ', CalculatorObject.compute_rotation_angle(time[i]), '\n')
        omega[i] = CalculatorObject.compute_rotation_angle(time[i])
    plt.plot(omega)
    plt.show()
