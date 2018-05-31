# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:46:06 2018

@author: yongw
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144

from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure, show, output_file

def isNaN(num):
    return num != num

def isNotNaN(num):
    if num != num:
        return(False)
    else:
        return(True)

def NameToIndex(df_name, name):
#    index = Index(df_name).get_loc(name)
    if isNaN(name):
        return(-100)
    else:
        INDEX = df_name[df_name['Player'] == name].index[0]
        return(INDEX)

def SortData(df_name,df_stats):
    
    
    
    
    DATA = np.zeros((len(df_name),10)) #Enter year/PER_3/PTS_3/AST_3/TRB_3/ORB_3/DRB_3/BLK_3/TOV_3/per all time
    SUM3 = np.zeros((len(df_name),9)) #1 total game 2 PER*G 3 PTS*G 4.AST*G 5.TRB*G 6.ORB*G 7.DRB*G 8.BLK*G 9.TOV*G within 3 years
    SUM_alltime = np.zeros((len(df_name),3)) # 1.total games 2.PER*G #total years in NBA
    for i in range(len(df_stats)): #start to import data
        if NameToIndex(df_name,df_stats['Player'][i]) != -100:
            INDEX = NameToIndex(df_name,df_stats['Player'][i]) # indentify the player
            if DATA[INDEX][0] == 0:
                DATA[INDEX][0] = df_stats['Year'][i]   # The year enter NBA
            if df_stats['Year'][i] - DATA[INDEX][0] <3:  # Within first 3 years (player INDEX)
                if isNotNaN(df_stats['G'][i]):
                    SUM3[INDEX][0] += df_stats['G'][i]  # first three years total games
                    if isNotNaN(df_stats['PER'][i]):
                        SUM3[INDEX][1] += df_stats['PER'][i] * df_stats['G'][i] # first three years PER*G
                    if isNotNaN(df_stats['PTS'][i]):
                        SUM3[INDEX][2] += df_stats['PTS'][i]  #first three years total points
                    if isNotNaN(df_stats['AST'][i]):
                        SUM3[INDEX][3] += df_stats['AST'][i]  #first three years total ast
                    if isNotNaN(df_stats['TRB'][i]):
                        SUM3[INDEX][4] += df_stats['TRB'][i]  #first three years total rebounds
                    if isNotNaN(df_stats['ORB'][i]):
                        SUM3[INDEX][5] += df_stats['ORB'][i]  #first three years total offense rebounds
                    if isNotNaN(df_stats['DRB'][i]):
                        SUM3[INDEX][6] += df_stats['DRB'][i]  #first three years total defense rebounds
                    if isNotNaN(df_stats['BLK'][i]):
                        SUM3[INDEX][7] += df_stats['BLK'][i]  #first three years total blocks
                    if isNotNaN(df_stats['TOV'][i]):
                        SUM3[INDEX][3] += df_stats['TOV'][i]  #first three years total turnovers
            SUM_alltime[INDEX][0] += df_stats['G'][i]  # all time total games
            SUM_alltime[INDEX][1] += df_stats['G'][i]*df_stats['PER'][i] # all time total game*PER
            SUM_alltime[INDEX][2] +=1 #total years in NBA
    print(SUM3)
    for i in range(len(df_name)):  #sum3 and sum_alltime tranfer to DATA  i is the index of player
        if SUM3[i][0] == 0 or SUM_alltime[i][2] <2 or SUM_alltime[i][0] < 82 or SUM3[i][1] < 0: # play in NBA less than 3years or less than 82 games, does not count
            DATA[i] = 0
        else:
            DATA[i][1] = float(SUM3[i][1])/float(SUM3[i][0]) #PER first 3years
            DATA[i][2] = float(SUM3[i][2])/float(SUM3[i][0]) #pts per game first 3years
            DATA[i][3] = float(SUM3[i][3])/float(SUM3[i][0]) #AST per game first 3years
            DATA[i][4] = float(SUM3[i][4])/float(SUM3[i][0]) #Rebounds per game first 3years
            DATA[i][5] = float(SUM3[i][5])/float(SUM3[i][0]) #ORB per game first 3years
            DATA[i][6] = float(SUM3[i][6])/float(SUM3[i][0]) #DRB per game first 3years
            DATA[i][7] = float(SUM3[i][7])/float(SUM3[i][0]) #BLK per game first 3years
            DATA[i][8] = float(SUM3[i][8])/float(SUM3[i][0]) #TOV per game first 3years
        if SUM_alltime[i][0] == 0 or SUM_alltime[i][2] <2 or SUM_alltime[i][0] < 82 or SUM_alltime[i][1] <0: # play in NBA less than 3years or less than 82 games, does not count or all time PER<0
            DATA[i][9] = 0
        else:
            DATA[i][9] = float(SUM_alltime[i][1])/float(SUM_alltime[i][0]) # all carrer average PER
    i = 0
#    while i< len(df_name):
#        j = 0
#        while j< 10:
#            if isNaN(DATA[i][j]):
#                DATA[i][j] = 0
#            j+=1
#        i+=1
    DATA[DATA == 0] = np.nan  #dont show any zero data. transfer zero to NAN
#    print(DATA)
    np.save(path + "Matix_DATA1.npy", DATA)
    return(DATA)

def Plot_PER3(DATA):
    x = DATA[:,1] #first 3 years PER
    y = DATA[:,9]
    fig = plt.figure()
    fig.suptitle('3 years PER vs. career PER')
    plt.xlabel('first 3 years PER')
    plt.ylabel('career PER')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y,marker='o', s = 3 , c = 'blue')
    plt.savefig(path + "figure_PER3vsPER.png", dpi = 300,  transparent = False, aspect = 2 , axis = True)
    plt.show()
    
def Plot_PTS3(DATA):
    x = DATA[:,2] # first 3 years Points
    y = DATA[:,9]
    fig = plt.figure()
    fig.suptitle('3 years PTS per game vs. career PER')
    plt.xlabel('first 3 years PTS per game')
    plt.ylabel('career PER')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y,marker='o', s = 3 , c = 'skyblue')
    plt.savefig(path + "figure_PTS3.png", dpi = 300,  transparent = False, aspect = 2 , axis = True)
    plt.show()

def Plot_TRB3(DATA):
    x = DATA[:,4]
    y = DATA[:,9]
    fig = plt.figure()
    fig.suptitle('3 years TRB per game vs. career PER')
    plt.xlabel('first 3 years TRB per game')
    plt.ylabel('career PER')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y,marker='o', s = 3 , c = 'salmon')
    plt.savefig(path + "figure_TRB3.png", dpi = 300,  transparent = False, aspect = 2 , axis = True)
    plt.show()


def Plot_AST3(DATA):
    x = DATA[:,3]
    y = DATA[:,9]
    fig = plt.figure()
    fig.suptitle('3 years AST per game vs. career PER')
    plt.xlabel('first 3 years AST per game')
    plt.ylabel('career PER')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y,marker='o', s = 3 , c = 'hotpink')
    plt.savefig(path + "figure_AST3.png", dpi = 300,  transparent = False, aspect = 2 , axis = True)
    plt.show()


def Plot_PTS3_ML(df):
    fig = sns.jointplot(x='PTS_3', y='per all time', data= df)
#    PTS_ML_fig = fig.get_figure()
#    PTS_ML_fig.savefig('ML.png')
    fig.savefig(path + 'ML_PTS3.png')
#label='Volume > Average', ms=10, mfc=sns.color_palette()[4]
    
    
    
def Plot_test(df):
#    fig = sns.lmplot(x='PTS_3', y='per all time', data= df, palette="Set2")
#    fig.savefig(path + 'ML_test.png')
#
#    fig2= sns.jointplot(x='PTS_3', y='per all time',data= df, kind="hex", color="#4CB391")
#    fig2.savefig(path + 'ML_test2.png')

    g = sns.PairGrid(df.sort_values("per all time", ascending=False),
                 x_vars=df['PTS_3'], y_vars=df["birth_state"],
                 size=10, aspect=.25)
    g.map(sns.stripplot, size=10, orient="h",
      palette="Reds_r", edgecolor="gray")
    g.savefig(path + 'ML_test3.png', dpi = 600)
    g.show()

def Bokeh_PTS3(df1):
    
    source = ColumnDataSource(df1)
    
   
    
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset".split(',')
    hover = HoverTool(tooltips=[
    ("Player Name", "@{Player}"),
    ("Enter Year","@{Enter year}"),
    ("Collage", "@{collage}"),
    ("Birth State", "@{birth_state}"),
    ("(x,y)", "($x, $y)"),
    ("Career Player Efficiency Rating", "@{per all time}"),  # use {} to cite the dict.key
    ])
    TOOLS.append(hover)
    plot = figure(width=800, height=700, tools = TOOLS)
    
    
    plot.circle('PTS_3', 'per all time', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = "First 3 years: Points per Game"
    plot.yaxis.axis_label = "Career Player Efficiency Rating"
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file("PT3_PER all time.html", title="test")
    show(plot)


def Bokeh_PTS3_AST3(df1):
    palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    PER = df1["per all time"]
    low = min(PER)
    high = max(PER)
    PER_inds = [int(10*(x-low)/(high-low)) for x in PER] #gives items in colors a value from 0-10
    df1.loc[:,'PER_colors'] = [palette[i] for i in PER_inds]
    
    
    source = ColumnDataSource(df1)
    
   
    
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset".split(',')
    hover = HoverTool(tooltips=[
    ("Player Name", "@{Player}"),
    ("Enter Year","@{Enter year}"),
    ("Collage", "@{collage}"),
    ("Birth State", "@{birth_state}"),
    ("(x,y)", "($x, $y)"),
    ("Career Player Efficiency Rating", "@{per all time}"),  # use {} to cite the dict.key
    ])
    TOOLS.append(hover)
    plot = figure(width=800, height=700, tools = TOOLS)
    
    
    plot.circle('PTS_3', 'AST_3',color='PER_colors', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = "First 3 years: Points per Game"
    plot.yaxis.axis_label = "First 3 years: Assistance per Game"
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file("PT3_AST3_PerAllTime.html", title="PT3_AST3_PerAllTime")
    show(plot)

def Bokeh_TRB3_PER3(df1):
    palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    PER = df1["per all time"]
    low = min(PER)
    high = max(PER)
    PER_inds = [int(10*(x-low)/(high-low)) for x in PER] #gives items in colors a value from 0-10
    df1.loc[:,'PER_colors'] = [palette[i] for i in PER_inds]
    
    
    source = ColumnDataSource(df1)
    
   
    
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset".split(',')
    hover = HoverTool(tooltips=[
    ("Player Name", "@{Player}"),
    ("Enter Year","@{Enter year}"),
    ("Collage", "@{collage}"),
    ("Birth State", "@{birth_state}"),
    ("(x,y)", "($x, $y)"),
    ("Career Player Efficiency Rating", "@{per all time}"),  # use {} to cite the dict.key
    ])
    TOOLS.append(hover)
    plot = figure(width=800, height=700, tools = TOOLS)
    
    
    plot.circle('TRB_3', 'PER_3',color='PER_colors', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = "First 3 years: Total rebounds per game"
    plot.yaxis.axis_label = "First 3 years: Player Efficiency Rating"
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file("TRB3_PER3_PerAllTime.html", title="TRB3_PER3_PerAllTime")
    show(plot)
    

def Bokeh_Year_PTS3(df1):
    palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    PER = df1["per all time"]
    low = min(PER)
    high = max(PER)
    PER_inds = [int(10*(x-low)/(high-low)) for x in PER] #gives items in colors a value from 0-10
    df1.loc[:,'PER_colors'] = [palette[i] for i in PER_inds]
    
    
    source = ColumnDataSource(df1)
    
   
    
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset".split(',')
    hover = HoverTool(tooltips=[
    ("Player Name", "@{Player}"),
    ("Enter Year","@{Enter year}"),
    ("Collage", "@{collage}"),
    ("Birth State", "@{birth_state}"),
    ("(x,y)", "($x, $y)"),
    ("Career Player Efficiency Rating", "@{per all time}"),  # use {} to cite the dict.key
    ])
    TOOLS.append(hover)
    plot = figure(width=800, height=700, tools = TOOLS)
    
    
    plot.circle('Enter year', 'PTS_3',color='PER_colors', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = "Enter year"
    plot.yaxis.axis_label = "First 3 years: Points per game"
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file("Year_PTS3_PerAllTime.html", title="Year_PTS3_PerAllTime")
    show(plot)


def Bokeh_BLK3_PTS3(df1):
    palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    PER = df1["per all time"]
    low = min(PER)
    high = max(PER)
    PER_inds = [int(10*(x-low)/(high-low)) for x in PER] #gives items in colors a value from 0-10
    df1.loc[:,'PER_colors'] = [palette[i] for i in PER_inds]
    
    
    source = ColumnDataSource(df1)
    
   
    
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset".split(',')
    hover = HoverTool(tooltips=[
    ("Player Name", "@{Player}"),
    ("Enter Year","@{Enter year}"),
    ("Collage", "@{collage}"),
    ("Birth State", "@{birth_state}"),
    ("(x,y)", "($x, $y)"),
    ("Career Player Efficiency Rating", "@{per all time}"),  # use {} to cite the dict.key
    ])
    TOOLS.append(hover)
    plot = figure(width=800, height=700, tools = TOOLS)
    
    
    plot.circle('BLK_3', 'PTS_3',color='PER_colors', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = "First 3 years: Blocks per game"
    plot.yaxis.axis_label = "First 3 years: Points per game"
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file("BLK3_PTS3_PerAllTime.html", title="BLK3_PTS3_PerAllTime")
    show(plot)
#from bkcharts import Bar, show, output_file
    
    
def Bokeh_plot(df1, x, y, description):
    palette = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    PER = df1["per all time"]
    low = min(PER)
    high = max(PER)
    PER_inds = [int(10*(x-low)/(high-low)) for x in PER] #gives items in colors a value from 0-10
    df1.loc[:,'PER_colors'] = [palette[i] for i in PER_inds]
    
    
    source = ColumnDataSource(df1)
    
   
    
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset".split(',')
    hover = HoverTool(tooltips=[
    ("Player Name", "@{Player}"),
    ("Enter Year","@{Enter year}"),
    ("Collage", "@{collage}"),
    ("Birth State", "@{birth_state}"),
    ("(x,y)", "($x, $y)"),
    ("Career Player Efficiency Rating", "@{per all time}"),  # use {} to cite the dict.key
    ])
    TOOLS.append(hover)
    plot = figure(width=800, height=700, tools = TOOLS)
    
    
    plot.circle(str(x), str(y),color='PER_colors', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = description[x]
    plot.yaxis.axis_label = description[y]
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file(x + '_'+ y + ".html", title=description[x] + '_' + description[y])
    show(plot)

if __name__ == '__main__':
    path = "C:/Users/yongw/Desktop/Python/data incubator/Q3/"
    filename = "Players.csv"
    filename2 = "Seasons_Stats.csv"
    df_name= pd.read_csv(path + filename, sep=',',  engine = "python",error_bad_lines=False, skip_blank_lines=True,  encoding="utf-8-sig")
    df_stats= pd.read_csv(path + filename2, sep=',',  engine = "python",error_bad_lines=False,skip_blank_lines=True, encoding="utf-8-sig")
#    DATA = np.zeros((len(df_name),10))
#    DATA = SortData(df_name,df_stats)
    DATA = np.load(path + "Matix_DATA_2.npy")
    df_DATA = pd.DataFrame(DATA, columns=['Enter year','PER_3','PTS_3','AST_3','TRB_3','ORB_3','DRB_3','BLK_3', 'TOV_3', 'per all time', 'PTS_all time','AST_all time','TRB_all time','BLK_all time'])
    
    df_all = pd.concat([df_DATA, df_name], axis=1)
    df1 = df_all[~pd.isnull(df_all["per all time"])]
    description = dict()
    description = {'Enter year': 'The year enter NBA',
                   'PER_3': 'First 3 years: Player Efficiency Rating',
                   'PTS_3': 'First 3 years: Points per game',
                   'AST_3': 'First 3 years: Assistances per game',
                   'TRB_3': 'First 3 years: Rebounds per game',
                   'ORB_3': 'First 3 years: Offensive rebounds per game',
                   'DRB_3': 'First 3 years: Defensive rebounds per game',
                   'TOV_3': 'First 3 years: Turnovers per game',
                   'BLK_3': 'First 3 years: Blocks per game',
                   'per all time': 'Career player efficiency rating',
                   'PTS_all time': 'Career points per game',
                   'AST_all time': 'Career assistances per game',
                   'TRB_all time': 'Career recounds per game',
                   'BLK_all time': 'Career blocks per game',
                   'Player': 'Player name',
                   'height': 'Player Height',
                   'weight': 'Player Weight',
                   'collage': 'Player collage',
                   'born': 'Player born year',
                   'born_city': 'Player born city',
                   'born_state': 'Player born state'}
########################keep these plots
#    Bokeh_PTS3_AST3(df1)
#    Bokeh_TRB3_PER3(df1)
#    Bokeh_Year_PTS3(df1)
#    Bokeh_BLK3_PTS3(df1)
    Bokeh_plot(df1, 'PTS_3','PTS_all time',description)
    Bokeh_plot(df1, 'AST_3','AST_all time',description)
    Bokeh_plot(df1, 'TRB_3','TRB_all time',description)
    Bokeh_plot(df1, 'BLK_3','BLK_all time',description)
    Bokeh_plot(df1, 'height','PER_3',description)
    Bokeh_plot(df1, 'Enter year','PTS_all time',description)