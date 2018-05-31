# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:34:34 2018

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

from sklearn import preprocessing
from sklearn import utils
from matplotlib.mlab import griddata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


def randomforest_learn(df, y): # df is the DataFrame to train. y is the feature you want to predict. df_pred is the new dataframe to predict.
    df.loc[:,'random'] = (np.random.uniform(0,1,len(df)) <= 0.75)
    train, test = df[df['random']==True], df[df['random']==False]
    features = df.columns[1:9]
    clf = RandomForestClassifier(n_estimators=100,n_jobs=2, random_state=0)

    lab_enc = preprocessing.LabelEncoder() # encode the train Y to make sure CLF can fit
    encoded_trainY = lab_enc.fit_transform(train[str(y)])
#    print('encode Y is', encoded_trainY, 'max Y is', max(encoded_trainY))

    clf.fit(train[features], encoded_trainY)
    
    
    ####################rookie after 2014
#    df_test = df[df[:,'Enter year'] >2000]
#    print(len(df_test))
#    PER_predict = clf.predict(df_test[features])
    ####################
    
    PER_predict = clf.predict(test[features])
    
    
    PER_predict_transform = lab_enc.inverse_transform(PER_predict)    
    df_temp = pd.DataFrame(PER_predict_transform, columns=['predicted ' + str(y)])
    test= test.reset_index(drop = True)

    df_result = pd.concat([test, df_temp], axis=1)
#    df_result = pd.concat([df_test, df_temp], axis=1)
    
    return(df_result)
    
    
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
    plot.background_fill_color = "#7D7C7C"
    
#    plot.xgrid.grid_line_color = None  #remove grid
    
    plot.xgrid.grid_line_alpha = 0.5
    plot.xgrid.grid_line_dash = [6, 4]
    
    plot.ygrid.grid_line_alpha = 0.5
    plot.ygrid.grid_line_dash = [6, 4]

    
    plot.circle(str(x), str(y),color='PER_colors', source = source, size=8)
#    plot.background_fill_color = "#dddddd"
    plot.xaxis.axis_label = description[x]
    plot.yaxis.axis_label = description[y]
    
#    plot.circle(x=df1['PTS_3'], y=df1['per all time'], size=8)
    output_file(x + '_'+ y + ".html", title=description[x] + '_' + description[y])
    return(plot)


def mod_data():
    path = "C:/Users/yongw/Desktop/Python/data incubator/Q3/"
    filename = "Players.csv"
    filename2 = "Seasons_Stats.csv"
    df_name= pd.read_csv(path + filename, sep=',',  engine = "python",error_bad_lines=False, skip_blank_lines=True,  encoding="utf-8-sig")
    df_stats= pd.read_csv(path + filename2, sep=',',  engine = "python",error_bad_lines=False,skip_blank_lines=True, encoding="utf-8-sig")
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
                   'born_state': 'Player born state',
                   'predicted per all time': 'Predicted PER',
                   'predicted PTS_all time': 'Predicted points per game'}
    
#    df_result = randomforest_learn(df1,"per all time")
#    df2 = df_result[~pd.isnull(df_result["per all time"])]
#    plot = Bokeh_plot(df2, 'predicted per all time', 'per all time', description)
#    return(plot)
    
    df_result4 = randomforest_learn(df1,"PTS_all time")
    df4 = df_result4[~pd.isnull(df_result4["PTS_all time"])]
    plot = Bokeh_plot(df4, 'predicted PTS_all time', 'PTS_all time', description)
    return(plot)
    
#    df_pred = df1[df1['Enter year'] > 2010]
#    df_result3 = randomforest_learn(df1, "per all time" , df_pred)
#    df3 = df_result3[~pd.isnull(df_result3["per all time"])]
#    Bokeh_plot(df3, 'predicted per all time', 'per all time', description)

from flask import Flask
app = Flask(__name__)

import pandas as pd
import numpy as np
#import bokeh.charts as bc
from bokeh.resources import CDN
from bokeh.embed import components











@app.route("/")
def visualisation():
     plot = mod_data()
     # Generate the script and HTML for the plot
     script, div = components(plot)
    
     # Return the webpage
     return """
    <!doctype html>
    <head>
     <title>NBA test page</title>
     {bokeh_css}
    </head>
    <body>
     <h1>How rookies will score in the future!
     {div}
    
     {bokeh_js}
     {script}
    </body>
     """.format(script=script, div=div, bokeh_css=CDN.render_css(),
     bokeh_js=CDN.render_js())

if __name__ == "__main__":
 #    app.run(host='0.0.0.0', port=80)
     app.run(port=8000)
