# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:30:17 2018

@author: Stan

This module defines and models the network. It contains a list of nodes and can
graphically represent those.

Requirements: 
    - Energy consumption: star network??
    - Cluster head communicates over LoRa to base
    - Show differences in energy consumption.
    - Transmit power of 0 dBm (1 mW)
    - Payload 10 Bytes for sensors and 60 for Cluster Heads
    - Frequencies: 868 MHz for LoRa and 2.4GHz for BLE

"""


# Python imports
from functools import partial
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os.path

# Project imports
from node import (SensorNode, loss_CH, loss_ble_simple, 
                  f_lora, BLE, LORA, receiver_sens)

#=============================================
#
#   SET THE MODE
#
#=============================================


#=============================================
#   GLOBAL PARAMETERS
#=============================================

width = 1000
height = 1000
cell_width = 100
cell_height = 100

number_of_sensors = 2000
area = width*height
area_per_sensor = area/number_of_sensors
separation_distance = sqrt(area_per_sensor)
sim_params = {
        'urban': True,
        'smart': False,
        'dt': 1,    #s
        'cluster_send_interval': 3*60,
        'packet_rate': 0.01
}


end = 3600

def init():    
    global loss_lora, sensors
    loss_lora = partial(loss_CH, C=3) if sim_params['urban'] else loss_CH
    SensorNode.cluster_send_interval = sim_params['cluster_send_interval']
    sensors = []
    
# Choose the proper loss_models

if sim_params['urban']:
    loss_lora = partial(loss_CH, C=3)
else:
    loss_lora = loss_CH

loss_ble = loss_ble_simple


def calc_parameters():
    ble_range = calc_range(loss_ble, -receiver_sens['BLE'])
    lora7_range = calc_range(loss_lora, -receiver_sens['lora7'])
    
    return ble_range, lora7_range


#=============================================
#   MISC PARAMETERS
#=============================================


plot_opts = {'circles': False,
             'heads': True,
             'nodes': True,
             'plot_map': True,
             'plot_losses': False,
             'show_cons': True,
             'sim': True
             }


#=============================================
#   NETWORK GENERATION FUNCTIONS
#=============================================

sensors = []

def create_star():
    '''
    In the star network topology, we will determine whether the node can use
    BLE or else what kind of LoRa it will use. The LoRa is either Lora7 or 8.
    '''
    ble_range, lora7_range = calc_parameters()
    sens = []
    x_locs = np.linspace(-width/2,width/2, 45)
    y_locs = np.linspace(-height/2,height/2, 44)
    
    for x in x_locs:
        for y in y_locs:
            dist = np.sqrt(x**2+y**2)
            if dist > ble_range:
                mode = LORA
                if dist > lora7_range:
                    sf = 8
                else:
                    sf = 7
            else:
                mode = BLE
            sens.append(SensorNode(x,y, transmission_mode=mode,SF=sf, packet_rate=sim_params['packet_rate']))
    
    
    
    return sens

def create_smart():
    '''
    For the smart network, only the cluster heads need to transmit their data
    over LoRa. Since there will be (in our case) 64 nodes, we can use the bands
    with a 1% duty cycle, since we have less than 100 nodes, so the cluster 
    heads can send up to 36 seconds per hour which is far more than they need.
    
    '''
    ble_range, lora7_range = calc_parameters()
    
    sens = []
    
    # Creating cluster heads
    x0 = 1/2*np.sqrt(2)* ble_range
    y0 = x0
    dist0 = 2 * x0
    v_heads = int(np.ceil((height - dist0)/dist0))
    h_heads = int(np.ceil((width - dist0)/dist0))
    heads_x = np.linspace(-width/2+x0,width/2-x0, h_heads+2)
    heads_y = np.linspace(-height/2+y0,height/2-y0, v_heads+2)
    for x in heads_x:
        for y in heads_y:
            dist = np.sqrt(x**2+y**2)
            if dist < lora7_range:
                SF = 7
            else: 
                SF = 8
            sens.append(SensorNode(x,y,is_head=True, SF=SF))
           
    
    amount_of_heads = len(sens)
    vert_sens = int(np.ceil(np.sqrt(number_of_sensors-amount_of_heads)))
    hor_sens = int(np.floor(np.sqrt(number_of_sensors-amount_of_heads)))
    
    while vert_sens*hor_sens + amount_of_heads > number_of_sensors:
        if vert_sens < hor_sens: vert_sens -= 1
        else: hor_sens -= 1
    
    x_locs = np.linspace(-width/2,width/2, hor_sens+1)[:-1]
    x_locs += (x_locs[1] - x_locs[0])/2
    y_locs = np.linspace(-height/2,height/2, vert_sens+1)[:-1]
    y_locs += (y_locs[1] - y_locs[0])/2
    

    for x in x_locs:
        for y in y_locs:
            sens.append(SensorNode(x,y, transmission_mode=BLE, packet_rate=sim_params['packet_rate']))

    return sens


#=============================================
#   ANALYSIS FUNCTIONS
#=============================================


def visualize():
    # Plot Axis
    ble_range, lora7_range = calc_parameters()
    
    fig, ax = plt.subplots()
    
    
    plt.axis('equal')
   
    # Plot sensors
    xs = np.array([sensor.x for sensor in sensors])
    ys = np.array([sensor.y for sensor in sensors])
    heads = np.array([True if sensor.is_head else False for sensor in sensors])
    bles = np.array([1 if sensor.transmission_mode==BLE else 0 for sensor in sensors])
    print(f'There are {sum(bles)} nodes with Bluetooth enabled.')
    loras = np.array([1 if sensor.transmission_mode==LORA else 0 for sensor in sensors])
    sf7s = np.array([sensor.SF == 7 for sensor in sensors])
    if plot_opts['nodes']:
        plt.plot(xs[(heads==False) & (bles==True)],
                ys[(heads==False) & (bles==True)], 'b.')
      
        plt.plot(xs[(heads==False) & (loras==True) & (sf7s==True)],
                ys[(heads==False) & (loras==True) & (sf7s==True)], 'g.')
        
        plt.plot(xs[(heads==False) & (loras==True) & (sf7s==False)],
                ys[(heads==False) & (loras==True) & (sf7s==False)], 'yx')
    if plot_opts['heads']:
        plt.plot(xs[heads==True], ys[heads==True], 'r*', markersize=10)
    
    if plot_opts['circles']:
        for sens in sensors:
            if sens.is_head:
                c = plt.Circle((sens.x,sens.y), ble_range, color='blue', fill=False)
                ax.add_artist(c)
            
    plt.plot([-width/2,width/2], [0,0], 'y')
    plt.plot([0,0], [-height/2, height/2], 'y')
    # Plot border
    plt.plot([-width/2, width/2, width/2,-width/2,-width/2],
             [-height/2,-height/2,height/2,height/2,-height/2], linewidth=3)    
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Network topology for a {"smart" if sim_params["smart"] else "star"}-network in a ' 
                                       +f'{"urban" if sim_params["urban"] else "rural"} environment')
    
    plt.legend(('BLE-enabled sensors', 'LoRa7 sensors', 'LoRa 8 sensors', 'Cluster-heads'))
    plt.show()
    
    
def calc_range(func, threshold):
    '''
    Calculates the highest distance in meters for which lora is below the 
    threshold.
    '''    
    d = 500
    # Establish upper bound
    while func(d) < threshold:
        d *= 2
    max_d = d
    min_d = 0
    d = (min_d+max_d)/2
    while max_d-min_d > 1:
        if func(d) > threshold:
            max_d = d
            
        else:
            min_d = d
        d = (min_d+max_d)/2
    return np.floor(min_d)
        


def plot_losses():
    ble_range, lora7_range = calc_parameters()
    
    d = np.linspace(1,3000,100)
    plt.subplot(2,1,1)
    L_lora = np.array([loss_lora(d_) for d_ in d])
    plt.plot(d, L_lora)
    plt.plot([d[0], d[-1]], [-receiver_sens['lora7'],-receiver_sens['lora7']], 'r--')
    plt.plot([lora7_range, lora7_range], [0,loss_lora(lora7_range)], 'g--')
    
    plt.title('LoRa')
    plt.xlabel('d (m)')
    plt.ylabel('loss (dB)')
    
    plt.subplot(2,1,2)
    L_ble = np.array([loss_ble(d_) for d_ in d])
    plt.plot(d, L_ble)
    plt.plot([d[0], d[-1]], [-receiver_sens['BLE'],-receiver_sens['BLE']], 'r--')
    plt.plot([ble_range, ble_range], [0,loss_ble(ble_range)], 'g--')
    
    
    plt.title('BLE')
    plt.xlabel('d (m)')
    plt.ylabel('loss (dB)')
    
    plt.tight_layout()
    plt.show()

def plot_set(setname):
    prs = np.arange(0.01,0.21,0.01)
    
    data_smart_urban = extract_data(setname,True,True)
    data_smart_rural = extract_data(setname,True,False)
    data_star_urban = extract_data(setname,False,True)
    data_star_rural = extract_data(setname,False,False)
    
    #Markers: s(quare) for urban ^ for rural greem for smart, red for star
    
    for i in range(len(data_smart_urban)):
        plt.plot(prs, data_star_urban[i], 'rs')
        plt.plot(prs, data_star_rural[i], '^', color='orange')
        plt.plot(prs, data_smart_urban[i], 'ys')
        plt.plot(prs, data_smart_rural[i], 'g^')
        plt.legend(('Star Urban', 'Star Rural', 'Smart Urban', 'Smart Rural'))
        
    plt.title(f"Energy consumptions for various scenarios over 1 hour")
    plt.xlabel('Packet-rate')
    plt.ylabel('Energy (J)')

def extract_data(setname,smart,urban):
    prefix = setname + '/sim_'
    if smart:
        prefix += 'smart_'
    else:
        prefix += 'star_'
    if urban:
        prefix += 'urban_'
    else:
        prefix += 'rural_'
    total_energy = np.zeros(21)
    
    for i in range(1,21):
        arc = np.load(prefix+str(i)+'.npz')
        cons,bles,sf7s,heads = arc.f.result, arc.f.bles, arc.f.sf7s, arc.f.heads
        total_energy[i] = np.sum(cons)
        max_cons_lora = np.max(cons[bles==False])
        print(f"({i}) Duty cycle constraint: {'Fulfilled' if max_cons_lora < 0.036 else 'Failed'}")
        
    return total_energy[1:],
    
def plot_energy(setname,smart,urban):
    
    prs = np.arange(0.01,0.21,0.01)
    
    total_energy, = extract_data(setname,smart,urban)
        
    plt.plot(prs, total_energy, 'r.')
    plt.title(f"Energy consumptions for a {'smart' if smart else 'star'} network in a "
            + f"{'urban' if urban else 'rural'} environment")
    plt.xlabel('Packet-rate')
    plt.ylabel('Energy (J)')
    plt.legend('Total Smart')


def simulate(network_creation, urban=False):
    '''
    Simulates one run of the program and returns arrays of data
    '''
    global sensors
    global URBAN
    init()
    sensors = network_creation()
    URBAN = urban
    dt = sim_params['dt']
    now = 0
    while now < end:
        for sensor in sensors:
            sensor.tick(dt)
        now += dt
    #head_cons = np.array([sensor.energy_cons for sensor in sensors if sensor.is_head])
    #node_cons = np.array([sensor.energy_cons for sensor in sensors if not sensor.is_head])
    
    return np.array([sensor.energy_cons for sensor in sensors])
    
    
def show_energy_consumption(cons):
    
    
    bles = np.array([1 if sensor.transmission_mode==BLE else 0 for sensor in sensors])
    sf7s = np.array([sensor.SF == 7 for sensor in sensors])
    heads = np.array([sensor.is_head for sensor in sensors])
    
    plt.plot(np.arange(len(cons))[heads==1], cons[heads==1] , 'ro', fillstyle='none') 
    plt.plot(np.arange(len(cons))[(bles==0) & (sf7s == 1)],cons[(bles==0) & (sf7s == 1)] , 'g.')  
    plt.plot(np.arange(len(cons))[(bles==0) & (sf7s == 0)],cons[(bles==0) & (sf7s == 0)] , 'y.')  
    plt.plot(np.arange(len(cons))[bles==1],cons[bles==1] , 'b.')    
    
    plt.title('Consumed energy per node')
    plt.xlabel('Node ID')
    plt.ylabel('Total consumed energy in J')
    plt.legend(('Cluster Heads', 'LoRa7 nodes', 'LoRa8 nodes', 'BLE nodes'))
    
    head_avg = np.mean(cons[heads])
    node_avg = np.mean(cons[heads==False])
    print(f'Average energy consumption of cluster heads: {head_avg} J')
    print(f'Average energy consumption of nodes: {node_avg} J')   
    print(f'Total energy consumption: {np.sum(cons)} J')

def show_energy_consumption_from_file(fname):
    arc = np.load(fname)
    cons,bles,sf7s,heads = arc.f.result, arc.f.bles, arc.f.sf7s, arc.f.heads
    plt.plot(np.arange(len(cons))[heads==1], cons[heads==1] , 'ro', fillstyle='none') 
    plt.plot(np.arange(len(cons))[(bles==0) & (sf7s == 1)],cons[(bles==0) & (sf7s == 1)] , 'g.')  
    plt.plot(np.arange(len(cons))[(bles==0) & (sf7s == 0)],cons[(bles==0) & (sf7s == 0)] , 'y.')  
    plt.plot(np.arange(len(cons))[bles==1],cons[bles==1] , 'b.')   
    
    plt.title('Consumed energy per node')
    plt.xlabel('Node ID')
    plt.ylabel('Total consumed energy in J')
    plt.legend(('Cluster Heads', 'LoRa7 nodes', 'LoRa8 nodes', 'BLE nodes'))
    
    head_avg = np.mean(cons[heads])
    node_avg = np.mean(cons[heads==False])
    print(f'Average energy consumption of cluster heads: {head_avg} J')
    print(f'Average energy consumption of nodes: {node_avg} J')   
    print(f'Total energy consumption: {np.sum(cons)} J')
    
def save_consumption(filename):
    head_cons = np.array([sensor.energy_cons for sensor in sensors if sensor.is_head])
    node_cons = np.array([sensor.energy_cons for sensor in sensors if not sensor.is_head])
    np.savez(filename, head_cons=head_cons, node_cons=node_cons)

if __name__ == '__main__':
    
    if plot_opts['plot_losses']:
        plot_losses()
    
    if plot_opts['sim']:
        for i in range(1,21):
            sim_params['packet_rate'] = i*0.01
            name = 'sim_'
            if sim_params['smart']:
                result = simulate(create_smart)
                name += 'smart_'
            else:
                result = simulate(create_star)
                name += 'star_'
            if sim_params['urban']:
                name += 'urban_'
            else: 
                name += 'rural_'
            name += str(i)
            
            bles = np.array([1 if sensor.transmission_mode==BLE else 0 for sensor in sensors])
            sf7s = np.array([sensor.SF == 7 for sensor in sensors])
            heads = np.array([sensor.is_head for sensor in sensors])
            
            np.savez(name, result=result, bles=bles, sf7s=sf7s, heads=heads)
            
            
    if plot_opts['plot_map']:
        plt.figure()
        visualize()
    if plot_opts['show_cons']:
        plt.figure()
        show_energy_consumption(result)
    

    
    # result,bles,sf7s,heads = arc.f.result, arc.f.bles, arc.f.sf7s, arc.f.heads
    
    ble_range, lora7_range = calc_parameters()
    print(f'BLE reach: {ble_range}m')
    print(f'Lora7 reach: {lora7_range}m')
    print(f'Maximum distance in this scenario: {np.sqrt((width/2)**2 + (height/2)**2):.1f}m')
    
    