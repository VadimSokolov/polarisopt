import os
import sys
import PolarisOpt
import numpy as np
import SALib.sample.morris as morris_s
import SALib.analyze.morris as morris_a
import SALib.plotting.morris as morris_p
import matplotlib.pyplot as plt

def cov_plot(ax2,var_SI,colors,fn=[]):
    n=len(var_SI)
    for item,color in zip(var_SI,colors):
        ax2.scatter(item[2], item[3], c = color[None,:], label = item[0])
    ax2.set_ylabel(r'$\sigma$')
    ax2.set_xlabel(r'$\mu^\star$')
    ax2.set_xlim(0,)
    ax2.set_ylim(0,)
    ax2.legend(loc=2, prop={'size': 6})
    x_axis_bounds = np.array(ax2.get_xlim())
    ax2.plot(x_axis_bounds, x_axis_bounds, 'k-')
    ax2.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'k--')
    ax2.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'k-.')
    if fn==[]:
        plt.show()
    else:
        plt.savefig(fn)

def mu_mustar(ax1, var_SI, colors, fn=[]):
    for item,color in zip(var_SI,colors):
        ax1.scatter(item[2], item[1], c = color[None,:], label= item[0]) 
    ax1.set_xlim(0,)
    ax1.set_ylabel(r'$\mu$')
    ax1.legend(loc=2, prop={'size': 6})
    ax1.set_xlabel(r'$\mu^\star$ ')
    if fn==[]:
        plt.show()
    else:
        plt.savefig(fn)

def graph_morris(var_SI):
    colors=plt.cm.Set1(np.linspace(0, 1, len(var_SI)))           
    #plot
    fig, (ax1, ax2) = plt.subplots(1,2)
    for item,color in zip(var_SI,colors):
        ax1.scatter(item[2], item[1], c = color[None,:], label= item[0])
        ax2.scatter(item[2], item[3], c = color[None,:], label = item[0])
    ax2.set_ylabel(r'$\sigma$')
    ax2.set_xlabel(r'$\mu^\star$')
    ax2.set_xlim(0,)
    ax2.set_ylim(0,)
    ax2.legend(loc=2, prop={'size': 6})
    x_axis_bounds = np.array(ax2.get_xlim())
    ax2.plot(x_axis_bounds, x_axis_bounds, 'k-')
    ax2.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'k--')
    ax2.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'k-.')
    ax1.set_xlim(0,)
    ax1.set_ylabel(r'$\mu$')
    ax1.legend(loc=2, prop={'size': 6})
    ax1.set_xlabel(r'$\mu^\star$ ')
    plt.show()


