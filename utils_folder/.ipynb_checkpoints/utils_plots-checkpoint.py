import matplotlib.pyplot as plt

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
def make_color_dict(color):
    
    dict_ = {'patch_artist': True,
             'boxprops': dict(color=color, facecolor='w'),
             'capprops': dict(color=color),
             'flierprops': dict(color=color, markeredgecolor=color),
             'medianprops': dict(color='k'),
             'whiskerprops': dict(color=color)}
    
    return dict_