
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as plt
import matplotlib.gridspec as gridspec

#Graphical options
import itertools
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText



def plot_2DHist(H, axis_edges, x_label, y_label, title, min=None, norm=None):

    fig = plt.figure(figsize=(10,10), dpi = 80)
    ax = plt.subplot(111)
    
    colormap= plt.cm.viridis #"jet" #
    
    #Main plot
    xedges = axis_edges[0]
    yedges = axis_edges[1]
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    Xaxis, Yaxis = np.meshgrid(yedges,xedges)
    i = ax.pcolor(Yaxis, Xaxis, H, cmap=colormap
            ,norm=colors.Normalize(vmin=min, vmax=norm))
    
    ax.set_xlabel(x_label, fontsize=36)
    ax.set_ylabel(y_label, fontsize=36)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    divider = make_axes_locatable(ax)
    
    #Legend
    Afont = {'family': 'DejaVu Sans',
             #'backgroundcolor': 'white',
             'color':  'white',
             'weight': 'bold',
             'size': 22,
            }
    anchored_text = AnchoredText(title, loc=2, frameon=False, prop=Afont)
    ax.add_artist(anchored_text)
    
    #Color bar
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    cbar = fig.colorbar(i, cax=cax)
    cbar.ax.set_title('Density', fontsize=24, loc='left')
    #cbar.ax.set_label("Density", rotation=270, fontsize=24)
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=22)
    #cbar.ax.ticklabel_format(style="sci",scilimits=(0, 0))
    plt.show()
    
    
def plot_projections(H, axis_edges, x_label, y_label, title,zlabel='Density', norm=None, min=None, zlogscale=False, savedir=None):

    # Define size of figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(10, 12)
    
    # Define the positions of the subplots.
    ax0 = plt.subplot(gs[6:10, 5:9])
    axx = plt.subplot(gs[5:6, 5:9])
    axy = plt.subplot(gs[6:10, 9:10])

    colormap = plt.cm.viridis #"jet" #
    
    #Main plot
    xedges = axis_edges[0]
    yedges = axis_edges[1]
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    Xaxis, Yaxis = np.meshgrid(yedges,xedges)
    if (zlogscale==False):
        i = ax0.pcolor(Yaxis, Xaxis, H, cmap=colormap
                    ,norm=colors.Normalize(vmin=min, vmax=norm))
    else:
        i = ax0.pcolor(Yaxis, Xaxis, H, cmap=colormap
            ,norm=colors.LogNorm(vmin=min, vmax=norm))                
    #Legend
    Afont = {'family': 'DejaVu Sans',
             #'backgroundcolor': 'white',
             'color':  'white',
             'weight': 'bold',
             'size': 22,
            }
    anchored_text = AnchoredText(title, loc=2, frameon=False, prop=Afont)
    ax0.add_artist(anchored_text)

    #Projected histograms inx and y
    hx, hy = H.T.sum(axis=0), H.T.sum(axis=1)
    #Top projection
    axx_lim = ax0.get_xlim()
    axx_mid = (xedges[1:] + xedges[:-1])/2.
    axx_width = (xedges[1:] - xedges[:-1])
    axx.bar(axx_mid, hx, align='center', width=axx_width, color='#404387',edgecolor='white',linewidth=0.5)
    axx.set_xlim(axx_lim)
    axx.grid(color='silver', linestyle='-', linewidth=0.2)
    #Right projection
    axy_lim = ax0.get_ylim()
    axy_mid = (yedges[1:] + yedges[:-1])/2.
    axy_width = (yedges[1:] - yedges[:-1])
    axy.barh(axy_mid, hy, height=axy_width, align='center', color='#404387',edgecolor='white',linewidth=0.5)
    axy.set_ylim(axy_lim)
    axy.grid(color='silver', linestyle='-', linewidth=0.2)
    
    #Axis labels
    ax0.set_xlabel(x_label, fontsize=22)
    ax0.set_ylabel(y_label, fontsize=22)
    ax0.tick_params(axis='both', which='minor', labelsize=22)
    # ax0.tick_params(labelrotation=45)
    axy.tick_params(axis='x',rotation=90)
    axx.tick_params(axis='both', which='minor', labelsize=22)
    axy.tick_params(axis='both', which='minor', labelsize=22)

    if(zlogscale):
        axx.semilogy()
        axy.semilogx()

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    
    cbar = fig.colorbar(i, cax=cax)
    cbar.ax.set_title(zlabel, fontsize=14, loc='left')
    #cax.set_ylabel('Density')
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.show()
    if savedir!=None:
        fig.savefig(savedir)
    return ax0, axx, axy

def Plot2D_sidebyside(H1, H2, Edges, title1="this", title2="that"):
    colormap= plt.cm.viridis #"jet" #

    E, psi = np.meshgrid(Edges[1], Edges[0])


    # Plot scramble and MC side by side per each PID bin
    fig, axs = plt.subplots(len(H1), 2)

    pdf = [H1, H2]

    plt.rcParams['figure.figsize'] = [20, 30]

    for i in range(len(H1)):
        for j in range(2):
            im=axs[i, j].pcolor(psi, E, pdf[j][i], cmap=colormap)
                    # , norm=colors.Normalize(vmin=0.0000, vmax=None))
            divider = make_axes_locatable(axs[i,j])

            if j==0:
                title = f"{title1} - PID {str(i+1)}"
            else:
                title = f"{title2} - PID {str(i+1)}"
        
            #Legend
            Afont = {'family': 'DejaVu Sans',
                    #'backgroundcolor': 'white',
                    'color':  'white',
                    'weight': 'bold',
                    'size': 10,
                    }
            anchored_text = AnchoredText(title, loc=2, frameon=False, prop=Afont)
            axs[i, j].add_artist(anchored_text)

            #Color bar
            cax = divider.append_axes("right", size="3.5%", pad=0.05)
            fig.colorbar(im, cax=cax)


def PlotProjection_comparison(H1, H2, edges, title1="this", title2="that", binE=None, binPsi=None):

    fig, axs = plt.subplots(3, 2)

    xedges = edges[0]
    yedges = edges[1]

    for i in range(len(H1)):
        # hx1, hy1 = H1[i].T.sum(axis=0), H1[i].T.sum(axis=1)
        # hx2, hy2 = H2[i].T.sum(axis=0), H2[i].T.sum(axis=1)
        if (binE is not None):
            hx1 = [H1[i][j][binE] for j in range(len(xedges)-1)]
            hx2 = [H2[i][j][binE] for j in range(len(xedges)-1)]
        else:
            hx1 = H1[i].T.sum(axis=0)
            hx2 = H2[i].T.sum(axis=0)

        if binPsi is not None:
            hy1 = [H1[i][binPsi][j] for j in range(len(yedges)-1)]
            hy2 = [H2[i][binPsi][j] for j in range(len(yedges)-1)]
        else:
            hy1 = H1[i].T.sum(axis=1)
            hy2 = H2[i].T.sum(axis=1)   

            


        x, y = hist_line(hx1, xedges)
        x2, y2 = hist_line(hx2, xedges)
        axs[i,0].plot(x, y, label=title1)
        axs[i,0].plot(x2, y2, label=title2)
        axs[i,0].legend()

        x, y = hist_line(hy1, yedges)
        x2, y2 = hist_line(hy2, yedges)
        axs[i,1].plot(x, y, label=title1)
        axs[i,1].plot(x2, y2, label=title2)
        axs[i,1].legend()

def hist_line(val, edges):
    left, right = edges[:-1], edges[1:]
    x = np.array([left, right]).T.flatten()
    y = np.array([val, val]).T.flatten()
    return x,y


def Plot2D_sidebyside_diff(H1, H2, Edges, title1="this", title2="that"):
    colormap= plt.cm.viridis #"jet" #

    E, psi = np.meshgrid(Edges[1], Edges[0])


    fig, axs = plt.subplots(len(H1), 3)

    diff = H1-H2
    pdf = [H1, H2, diff]

    plt.rcParams['figure.figsize'] = [40, 30]

    for i in range(len(H1)):
        for j in range(3):
            im=axs[i, j].pcolor(psi, E, pdf[j][i], cmap=colormap)
                    # , norm=colors.Normalize(vmin=0.0000, vmax=None))
            divider = make_axes_locatable(axs[i,j])

            if j==0:
                title = f"{title1} - PID {str(i+1)}"
            elif j==1:
                title = f"{title2} - PID {str(i+1)}"
            elif j==2:
                title = f"Difference - PID {str(i+1)}"
            #Legend
            Afont = {'family': 'DejaVu Sans',
                    #'backgroundcolor': 'white',
                    'color':  'white',
                    'weight': 'bold',
                    'size': 14,
                    }
            anchored_text = AnchoredText(title, loc=2, frameon=False, prop=Afont)
            axs[i, j].add_artist(anchored_text)

            #Color bar
            cax = divider.append_axes("right", size="3.5%", pad=0.05)
            fig.colorbar(im, cax=cax)

def Plot2D_sidebyside_noPID(H1, H2, Edges, title1="this", title2="that"):
    colormap= plt.cm.viridis #"jet" #

    E, psi = np.meshgrid(Edges[1], Edges[0])


    # Plot scramble and MC side by side per each PID bin
    fig, axs = plt.subplots(1, 2)

    pdf = [H1, H2]

    plt.rcParams['figure.figsize'] = [20, 8]

    for j in range(2):
        im=axs[j].pcolor(psi, E, pdf[j], cmap=colormap)
                # , norm=colors.Normalize(vmin=0.0000, vmax=None))
        divider = make_axes_locatable(axs[j])

        if j==0:
            title = f"{title1}"
        else:
            title = f"{title2}"
    
        #Legend
        Afont = {'family': 'DejaVu Sans',
                #'backgroundcolor': 'white',
                'color':  'white',
                'weight': 'bold',
                'size': 10,
                }
        anchored_text = AnchoredText(title, loc=2, frameon=False, prop=Afont)
        axs[j].add_artist(anchored_text)

        #Color bar
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        fig.colorbar(im, cax=cax)


def plot_Resolution(H, pointx, pointy, axis_edges, x_label, y_label, title, norm=None, min=None):

    # Define size of figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(10, 12)
    
    # Define the positions of the subplots.
    ax0 = plt.subplot(gs[6:10, 5:9])
    axx = plt.subplot(gs[5:6, 5:9])
    axy = plt.subplot(gs[6:10, 9:10])

    colormap = plt.cm.viridis #"jet" #
    
    #Main plot
    xedges = axis_edges[0]
    yedges = axis_edges[1]
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    Xaxis, Yaxis = np.meshgrid(yedges,xedges)
    i = ax0.pcolor(Yaxis, Xaxis, H, cmap=colormap
                   ,norm=colors.Normalize(vmin=min, vmax=norm))
    ax0.plot(pointx, pointy, marker='+', color='red')               
    #Legend
    Afont = {'family': 'DejaVu Sans',
             #'backgroundcolor': 'white',
             'color':  'white',
             'weight': 'bold',
             'size': 22,
            }
    anchored_text = AnchoredText(title, loc=2, frameon=False, prop=Afont)
    ax0.add_artist(anchored_text)

    #Projected histograms inx and y
    hx, hy = H.T.sum(axis=0), H.T.sum(axis=1)
    #Top projection
    axx_lim = ax0.get_xlim()
    axx_mid = (xedges[1:] + xedges[:-1])/2.
    axx_width = (xedges[1:] - xedges[:-1])
    axx.bar(axx_mid, hx, align='center', width=axx_width, color='#404387',edgecolor='white',linewidth=0.5)
    axx.set_xlim(axx_lim)
    axx.grid(color='silver', linestyle='-', linewidth=0.2)
    #Right projection
    axy_lim = ax0.get_ylim()
    axy_mid = (yedges[1:] + yedges[:-1])/2.
    axy_width = (yedges[1:] - yedges[:-1])
    axy.barh(axy_mid, hy, height=axy_width, align='center', color='#404387',edgecolor='white',linewidth=0.5)
    axy.set_ylim(axy_lim)
    axy.grid(color='silver', linestyle='-', linewidth=0.2)
    
    #Axis labels
    ax0.set_xlabel(x_label, fontsize=22)
    ax0.set_ylabel(y_label, fontsize=22)
    ax0.tick_params(axis='both', which='minor', labelsize=22)
    # ax0.tick_params(labelrotation=45)
    axy.tick_params(axis='x',rotation=90)
    axx.tick_params(axis='both', which='minor', labelsize=22)
    axy.tick_params(axis='both', which='minor', labelsize=22)


    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    
    cbar = fig.colorbar(i, cax=cax)
    cbar.ax.set_title('Density', fontsize=14, loc='left')
    #cax.set_ylabel('Density')
    fig.tight_layout()
    plt.show()
