#!/usr/bin/env python3
import pandas as pd
import io
import retentioneering
import numpy as np
import datetime
from copy import deepcopy
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# matplotlib inline

retentioneering.config.update({
    'event_col':'mapped_event',
    'event_time_col':'timestamp',
    'user_col': 'client_id'
})

def create_funnel_df(df, steps, from_date=None, to_date=None, step_interval=0, closed=True):
    """
    Function used to create a dataframe that can be passed to functions for generating funnel plots.
    channel,client_id,event,platform,region,timestamp,user_type
    """
    # filter df for only events in the steps list
    df = df[['client_id', 'mapped_event', 'timestamp']]
    df = df[df['mapped_event'].isin(steps)]

    values = []
    # for the rest steps, create a df and filter only for that step
    for i, step in enumerate(steps):
        if i == 0:
            dfs = {}

            dfs[step] = df[df['mapped_event'] == step] \
                .sort_values(['client_id', 'timestamp'], ascending=True) \
                .drop_duplicates(subset=['client_id', 'mapped_event'], keep='first')

            # filter df of 1st step according to dates
            if from_date:
                dfs[step] = dfs[step][(dfs[step]['timestamp'] >= from_date)]

            if to_date:
                dfs[step] = dfs[step][(dfs[step]['timestamp'] <= to_date)]

        else:
            dfs[step] = df[df['mapped_event'] == step]
            if not closed:
                dfs[step].drop_duplicates(subset=['client_id'], inplace=True)
            else:
                # outer join with previous step
                merged = pd.merge(dfs[steps[i - 1]], dfs[step], on='client_id', how='outer')

                # keep only rows for which the distinct_ids appear in the previous step
                valid_ids = dfs[steps[i - 1]]['client_id'].unique()
                merged = merged[merged['client_id'].isin(valid_ids)]

                # keep only events that happened after previous step and sort by time
                merged = merged[merged['timestamp_y'] >=
                                (merged['timestamp_x'] + pd.Timedelta(step_interval))].sort_values('timestamp_y', ascending=True)

                # take the minimum time of the valid ones for each user
                merged = merged.drop_duplicates(subset=['client_id', 'mapped_event_x', 'mapped_event_y'], keep='first')

                # keep only the necessary columns and rename them to match the original structure
                merged = merged[['client_id', 'mapped_event_y', 'timestamp_y']].rename({'mapped_event_y': 'mapped_event',
                                                                             'timestamp_y': 'timestamp'}, axis=1)

                # include the df in the df dictionary so that it can be joined to the next step's df
                dfs[step] = merged

        # append number of users to the "values" list
        values.append(len(dfs[step]))

    # create dataframe
    funnel_df = pd.DataFrame({'step': steps, 'val': values})
    # calculate percentage conversion for each step
    funnel_df['pct'] = (100 - 100 * abs(funnel_df['val'].pct_change()).fillna(0)).astype(int)
    # shift val by one to plot faded bars of previous step in background
    funnel_df['val-1'] = funnel_df['val'].shift(1)
    # calculate percentage conversion between each step and the first step in the funnel
    funnel_df['pct_from_first'] = (funnel_df['val'] / funnel_df['val'].loc[0] * 100).fillna(0).astype(int)

    return funnel_df


def group_funnel_dfs(events, steps, col, closed=True):
    """
    Function used to create a dict of funnel dataframes used to generate a stacked funnel plot.
    """
    dict_ = {}
    
    # get the distinct_ids for each property that we are grouping by
    ids = dict(events.groupby([col])['client_id'].apply(set))

    for entry in events[col].dropna().unique():
        ids_list = ids[entry]
        df = events[events['client_id'].isin(ids_list)].copy()
        if len(df[df['mapped_event'] == steps[0]]) > 0:
            dict_[entry] = create_funnel_df(df, steps, closed=closed)

    return dict_


def plot_stacked_funnel(events, steps, col=None, from_date=None, to_date=None, step_interval=0, closed=True):
    """
    Function used for producing a (stacked) funnel plot.
    """
    # create list to append each trace to
    # this will be passed to "go.Figure" at the end
    data = []

    # if col is provided, create a funnel_df for each entry in the "col"
    if col:
        # generate dict of funnel dataframes
        dict_ = group_funnel_dfs(events, steps, col, closed=closed)
        title = 'Funnel plot per {}'.format(col)
    else:
        funnel_df = create_funnel_df(events, steps, from_date=from_date, to_date=to_date, step_interval=step_interval, closed=closed)
        dict_ = {'Total': funnel_df}
        title = 'Funnel plot'

    for t in dict_.keys():
        trace = go.Funnel(
            name=t,
            y=dict_[t].step.values,
            x=dict_[t].val.values,
            textinfo="value+percent previous"
        )
        data.append(trace)

    layout = go.Layout(margin={"l": 180, "r": 0, "t": 30, "b": 0, "pad": 0},
                       funnelmode="stack",
                       showlegend=True,
                       hovermode='closest',
                       title='Funnel plot per {}'.format(col),
                       legend=dict(orientation="v",
                                   bgcolor='#E2E2E2',
                                   xanchor='left',
                                   font=dict(
                                       size=12)
                                   )
                       )

    return go.Figure(data, layout)

def map_event(event):
    if event in ('catalog/motobikes/CVO™ Limited® (FLHTKSE), 2020 Harley-Davidson', 'catalog/motobikes/Harley-Davidson, CVO™ Limited® (Flhtkse), 2020', 'catalog/motobikes/Road Glide Limited 114, Harley-Davidson (2020)'):
        return 'catalog/motobikes/choose'
    elif event in ('catalog/tools/49320-09 НАБОР ДУГ ЗАЩИТНЫХ ДВИГАТЕЛЯ, BLACK Harley Davidson', 'catalog/tools/55800646 РУЛЬ ДЛЯ МОТОЦИКЛА СОСТАВНОЙ', 'catalog/tools/67700455 KIT ФАРЫ МОТОЦИКЛА Harley Davidson'):
        return 'catalog/tools/choose'
    else:
        return event
    # elif 'catalog/tools' not in event:
    #     return event
    # else:
    #     return None


data = pd.read_csv('mobile-app-data.csv')

# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data = data.sort_values('timestamp')
# print(data)
# data.head()
# desc_table = data.rete.step_matrix(max_steps=20)
data['mapped_event'] = list(map(map_event, data['event']))
data.dropna(subset=['mapped_event'], inplace=True)
pd.set_option('display.max_rows', 1000)
print((data['event_name'].drop_duplicates())['event_name'])
# data = data.sort_values('mapped_event')
# print(data['mapped_event'].value_counts())
# custom_order = ['main',
#                 'catalog/motobikes',
#                 'catalog/motobikes/choose',
#                 'catalog/tools',
#                 'catalog/tools/choose',
#                 'request_call',
#                 'call_made',
#                 'ENDED']

# desc_table = data.rete.step_matrix(max_steps=20, targets=['call_made'], sorting=custom_order)

# stages = ['main', 
#           'catalog/motobikes',
#           'catalog/motobikes/choose', 
#           'request_call', 
#           'call_made']

# plot_stacked_funnel(data, stages, col='platform').show()
# plot_stacked_funnel(data, stages, col='channel').show()
# plot_stacked_funnel(data, stages, col='region').show()
# plot_stacked_funnel(data, stages, col='platform', closed=False).show()
# plot_stacked_funnel(data, stages, col='channel', closed=False).show()
# plot_stacked_funnel(data, stages, col='region', closed=False).show()




