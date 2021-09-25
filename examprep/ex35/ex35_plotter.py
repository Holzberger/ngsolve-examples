from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt



def get_last_period(time,val):
    lift_diff = val[1:]-val[:-1]
    lift_diffs = lift_diff[1:]*lift_diff[:-1]
    id_extr = np.where(lift_diffs<0)[0]+1
    if val[id_extr[1]]<val[id_extr[2]]:
        id_min = id_extr[1::2]
        id_max = id_extr[::2]
    else:
        id_min = id_extr[::2]
        id_max = id_extr[1::2]
    id_last_per = np.arange(id_min[-2], id_min[-1]+1)
    return id_min, id_max, id_last_per  

def get_period_props(time_per,val_per):
    val_max  = np.max(val_per)
    val_min  = np.min(val_per)
    val_mean = np.mean(val_per)
    per     = time_per[-1]-time_per[0]
    return val_max, val_min, val_mean, per

def plot_sols(sols):
    props_per_sol = []
    for sol in sols:
        time = sol[0][0,:]
        drag = sol[0][1,:]
        lift = sol[0][2,:]
        p1 = sol[0][3,:]
        p2 = sol[0][4,:]
        id_min, id_max, id_last_per = get_last_period(time, lift)
        time_per = time[id_last_per]
        drag_per = drag[id_last_per]
        lift_per = lift[id_last_per]
        p1p2_per = p1[id_last_per]-p2[id_last_per]
        lift_max, lift_min, lift_mean, per = get_period_props(time_per, lift_per)
        drag_max, drag_min, drag_mean, per = get_period_props(time_per, drag_per)
        props_per_sol.append([time_per, drag_per, lift_per, p1p2_per,
                              lift_max, lift_min, lift_mean,
                               drag_max, drag_min, drag_mean, per])
    fig0, ax0 = plt.subplots(figsize=(12,6))
    fig1, ax1 = plt.subplots(figsize=(12,6))
    fig2, ax2 = plt.subplots(figsize=(12,6))
    for n, props_sol in enumerate(props_per_sol):
        lab = make_name(sols[n][1])
        ax0.plot(props_sol[0]-props_sol[0][0], props_sol[2],label=lab)
        ax1.plot(props_sol[0]-props_sol[0][0], props_sol[1],label=lab)
        ax2.plot(props_sol[0]-props_sol[0][0], props_sol[3],label=lab)
    ax0.set_xlabel('time'); ax0.set_ylabel(r'C_L'); ax0.set_title('lift'); ax0.grid(True)
    ax0.legend(loc="best")
    ax1.set_xlabel('time'); ax1.set_ylabel('C_D'); ax1.set_title('drag'); ax1.grid(True)
    ax1.legend(loc="best")
    ax2.set_xlabel('time'); ax2.set_ylabel('p1-p2'); ax2.set_title('p1-p2'); ax2.grid(True)
    ax2.legend(loc="best")
    fig0.savefig("lift.pdf")
    fig1.savefig("drag.pdf")
    fig2.savefig("p1p2.pdf")
    return props_per_sol
    
       
def plot_full_sol(sols,sol_ids):
    fig0, ax0 = plt.subplots(figsize=(12,6))
    fig1, ax1 = plt.subplots(figsize=(12,6))
    fig2, ax2 = plt.subplots(figsize=(12,6))
    for n in sol_ids:
        lab = make_name(sols[n][1])
        current_sol = sols[n][0]
        ax0.plot(current_sol[0,:], current_sol[2,:],label=lab)
        ax1.plot(current_sol[0,:], current_sol[1,:],label=lab)
        ax2.plot(current_sol[0,:], current_sol[3,:]-current_sol[4,:],label=lab)
    ax0.set_xlabel('time'); ax0.set_ylabel('lift'); ax0.set_title('lift'); ax0.grid(True)
    ax0.legend(loc="best")
    ax1.set_xlabel('time'); ax1.set_ylabel('drag'); ax1.set_title('drag'); ax1.grid(True)  
    ax1.legend(loc="best")
    ax2.set_xlabel('time'); ax2.set_ylabel('p1-p2'); ax2.set_title('p1-p2'); ax2.grid(True)  
    ax2.legend(loc="best")
    fig0.savefig("lift.pdf")
    fig1.savefig("drag.pdf")
    fig2.savefig("p1p2.pdf")


def read_references(links, forces):
    datas = []
    for link in links:
        datas.append([np.concatenate((np.genfromtxt(link[0])[:,[1,3,4]].T,
                      np.genfromtxt(link[1])[:,[6,11]].T)), link[0].split("/")[-1]])
    return datas

def make_name(filename):
    props = filename.split("_")
    if len(props)==4:
        name = r"ref o={} {} {}".format(props[1][1], props[3], props[2])
    else:
        name =r"o={}  dt={}, th={}".format(props[1], props[2], props[4])
    return name

def plot_soltimes(sols):
    sol_times =[]
    labels = []
    fig, ax = plt.subplots(figsize=(12,6))
    for sol in sols:
        sol_times.append(float(sol[1].split("_")[-2]))
        labels.append(sol[1].split("_")[-3])
    id_sorted= np.argsort(np.array(labels,dtype="float"))
    ax.barh(np.array(labels)[id_sorted],np.array(sol_times)[id_sorted])
    ax.set_xlabel('solve time in s'); ax.set_ylabel('theta'); ax.set_title('solve times'); ax.grid(True)  
    ax.legend(loc="best")
    fig.savefig("solve_times_theta.pdf")


path = "./ex35data/"
my_files = listdir(path)

sols = []

for file in my_files:
    sols.append([np.load(path+file), file])


#plot_soltimes(sols)




link_ref_fold    = "reference_data/Q21/"
filenames        = [("bdforces_q2_lv6_dt3","pointvalues_q2_lv6_dt3")]
#
links = [[link_ref_fold+"drag_lift/"+name[0],link_ref_fold+"press/"+name[1]] for name in filenames]
datas = read_references(links,forces=True)

sols = sols+datas

#plot_full_sol(sols,[0,1,2])

#
props_per_sol = plot_sols(sols)

