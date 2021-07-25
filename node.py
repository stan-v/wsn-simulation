# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:26:35 2018

@author: Stan

This module defined the SensorNode class which represents one sensor within a 
wireless Sensor Network

SF is also bits per chirp/symbol.

Coding rate will be chosen 4/5 since it is a good balance
This means that for every 4 information symbols it adds 1 error correction 
symbol. 

The corresponding data rates that will be used are DataRate No. 4 and 5
|   DR  |   SF  |
|    4  |    8  |
|    5  |    7  |

For the subband we can use either G1 or G4 which have a 1% duty cycle 
limitation. This means that we are allowed to transmit for 1% of an hour, per 
hour. 3600/100 = 36 seconds. 

"""

# Importing our main log function: the 10 base log.
from math import log10 as log
import numpy as np


#=============================================
#   GLOBAL CONSTANTS
#=============================================

BLE = 1
LORA = 2

BW = 125000         # Bandwidth for LoRa is 125kHz

#=============================================
#   GLOBAL PARAMETERS
#=============================================

# Transmission power for BLE/LoRa in dB
trans_power = 0 

# Sensor and cluster head payload in Bytes
sensor_payload = 10 
head_payload = 60

# Frequency of operation in MHz
f_lora = 868
f_ble = 2400

# Receiver sensitivity of LoRa# and BLE
# Range for BLE is about 112 m in optimal conditions.
# Range for loRa8 is about 643 m with hb=10 hr=1 and C=0
# LoRa11 can go up to about a kilometer with this.
receiver_sens = {'BLE': -116, 'lora7': -123, 'lora8': -126, 'lora9': -129,
                 'lora10': -132, 'lora11': -133, 'lora12': -136}

'''
The Spread Factor indicates the spread of frequencies used. 
Higher SF number means:
    - Longer transmission times (doubles for every +1)
    - More power consumption
    - Better signal
    - Reaches farther
'''


# Loss models
# Cost Hata 231 for LoRa
def loss_CH(d, hb=30, hr=1, f=f_lora, C=0): 
    '''
    f   -- career frequency in MHz
    hb  -- height of the base station in meter (default 20m)
    hr  -- height of the receiver in meter (default 1m)
    d   -- distance in meter
    C   -- a constant, 0 in suburbs 3 in metropoliton area
    '''
    #Convert the d to the d in km for the calculations
    d = d/1000
    a = (1.1*log(f)-0.7)*hr - (1.56*log(f) - 0.8)
    return (46.3+ 33.9*log(f) - 13.82*log(hb) - a
            + (44.9 - 6.55*log(hb))*log(d) + C)

# Simple model for BLE
def loss_ble_simple(d):
    '''
    d   -- distance in meter
    '''
    return 40.04 + 10*3.71*log(d)


#=============================================
#   COMMUNICATION CALCULATED PARAMETERS
#=============================================

# The BLE range does not depend on anything (only sensitivity and loss_model)
#ble_range = 111.5

# We can use LoRa SE9 or higher for the whole domain. 

def calc_toa_lora(SF, PL=60):
    '''Calculates the time on air for a LoRa transmission with spread factor 
    SF, which ranges from 7 to 12. All other parameters are filled in.
    
        PL: Payload of actual information you want to send (60 Bytes) 
    '''
    ts = 2**SF/BW
    npream = 8
    CR=1    # adds CR bits for every 4 bits.
    CRC=1   # Cyclic Redundancy Check (on -> 1)
    IH=1    # Implicit header mode (not used -> 1)
    DE=0    # Datarate optimization (not on -> 0)
    T = ((npream +12.25 )
            + max(
                    np.ceil(
                            (8*(13+PL) - 4*SF + 28 + 16*CRC - 20*IH)/
                          (4*(SF-2*DE))
                          )*CR+4, 0
                )
        )*ts
    return T
    

def calc_toa_ble(PL=10, SF=0):
    '''
    BLE has a protocol which sends 14 bytes of overhead along with the actual
    payload. The information is modulated on a 1 MBit/s speed, which gives a 
    Time on air of 1 microsecond per bit or 8 microsecond per byte/octet. 
    ''' 
    amount_of_bytes = 14 + PL
    T = amount_of_bytes*8/10**6
    return T

class SensorNode():
    '''
    The SensorNode class represents a sensor or cluster-head.
    It can transmit data or act as a cluster-head depending on the keyword
    argument is_head. It also tracks its energyconsumption to be summed and 
    compared at the end of a simulation.
    '''
    
    cluster_send_interval = 0
    
    def __init__(self, x, y, is_head=False, transmission_mode=0, SF=0,
                 packet_rate=0.01):
        self.x = x
        self.y = y
        self.is_head = is_head
        self.SF = SF
        if transmission_mode != 0:
            self.transmission_mode = transmission_mode
        elif is_head:
            self.transmission_mode = LORA
        else:
            self.transmission_mode = transmission_mode
                
        if self.transmission_mode == LORA:
            self.toa = calc_toa_lora
        else:
            self.toa = calc_toa_ble
        
        if self.is_head: # Clusterhead specifics
            self.clock = np.random.randint(0,round(SensorNode.cluster_send_interval))
            self.PL = 60
        else:       # Node specific
            self.PL = 10
            
        # Define the list of messages that are received or need to be transmitted
        self.energy_cons = 0
        # self.target = None
        self.packet_rate = packet_rate
       
        
#    def set_target_node(self,target_node):
#        ''' Sets the target node to the cluster head or 
#        the central base receiving node.
#        '''
#        self.target = target_node
        
    def tick(self, dt):
        if not self.is_head:
            # for mesh nodes: create packets randomly and transmit them when created.
            if np.random.random() < self.packet_rate*dt :
                self.transmit()
        else:
            # for cluster-heads, transmit once in a blue moon. 
            self.clock -= dt        # reduce the clock
            if self.clock <= 0:     # if the clock drops below zero
                self.transmit()     # transmit and reset clock. 
                self.clock = SensorNode.cluster_send_interval
            
    def transmit(self):
        '''Transmit a message to all nearby nodes.'''
        assert self.transmission_mode != 0, 'Transmission mode not set.'
        
        self.energy_cons += 0.001*self.toa(SF=self.SF, PL=self.PL)
        
    
if __name__=='__main__':
    sfs = np.arange(7,12)
    toas = [calc_toa_lora(sf) for sf in sfs]
    import matplotlib.pyplot as plt
    plt.plot(sfs, toas)
    plt.show()
    