import numpy as np
import pandas as pd

weightReturn = 0.8
weightRisk = 0.2


def fitness_function(ret_data, arr_weights, extermums):
    """ recieve return data as a Dataframe and array of weights"""

    arr_weights = np.array(arr_weights)
    # calculate mean return of portfolio
    mean_ret_arr = np.empty(len(arr_weights))
    for i in range(len(mean_ret_arr)):
        mean_ret_arr[i] = ret_data[ret_data.columns[i]].mean()
    mean_ret_portfo = np.dot(arr_weights.T, mean_ret_arr)

    # calculate standard deviation of portfolio
    cov_matrix_ret = np.array(ret_data.cov())
    std_portfo = np.sqrt(np.dot(arr_weights.T, np.dot(cov_matrix_ret, arr_weights)))

    ret_scale = (mean_ret_portfo-extermums[1])/(extermums[0]-extermums[1])
    if ret_scale > 1:
        ret_scale = 1
    elif ret_scale < 0:
        ret_scale = 0
    risk_scale = (std_portfo-extermums[3])/(extermums[2]-extermums[3])
    if risk_scale > 1:
        risk_scale = 1
    elif risk_scale < 0:
        risk_scale = 0
    fitness = (weightReturn*ret_scale)-(weightRisk*risk_scale)

    # # calculate risk adj return(if you want annual riskAdjret multiply it by 252)
    # if std_portfo == 0:
    #     risk_adj_ret = 0
    # else:
    #     risk_adj_ret = mean_ret_portfo / std_portfo
    
    return fitness

def calc_riskAdjRet(ret_data, arr_weights):
    """ recieve return data as a Dataframe and array of weights"""

    arr_weights = np.array(arr_weights)
    # calculate mean return of portfolio
    mean_ret_arr = np.empty(len(arr_weights))
    for i in range(len(mean_ret_arr)):
        mean_ret_arr[i] = ret_data[ret_data.columns[i]].mean()
    mean_ret_portfo = np.dot(arr_weights.T, mean_ret_arr)

    # calculate standard deviation of portfolio
    cov_matrix_ret = np.array(ret_data.cov())
    std_portfo = np.sqrt(np.dot(arr_weights.T, np.dot(cov_matrix_ret, arr_weights)))

    # calculate risk adj return(if you want annual riskAdjret multiply it by 252)
    if std_portfo == 0:
        risk_adj_ret = 0
    else:
        risk_adj_ret = mean_ret_portfo / std_portfo
    
    return risk_adj_ret

def calc_returnOfPoints(ret_data, arr_weights):
    """ recieve return data as a Dataframe and array of weights"""

    returns = []
    arr_weights = np.array(arr_weights)
    for i in range(len(ret_data)):
        returnOfPoint = np.dot(arr_weights.T, ret_data.iloc[i])
        returns.append(returnOfPoint)

    return np.array(returns)

def findMinMax(data, weights):

    maxRet = -999
    minRet = 999
    maxRisk = -999
    minRisk = 999
    for w in weights:
        arr_weights = np.array(w)
        # calculate mean return of portfolio
        mean_ret_arr = np.empty(len(arr_weights))
        for i in range(len(mean_ret_arr)):
            mean_ret_arr[i] = data[data.columns[i]].mean()
        mean_ret_portfo = np.dot(arr_weights.T, mean_ret_arr)

        if mean_ret_portfo > maxRet:
            maxRet = mean_ret_portfo
        if mean_ret_portfo < minRet:
            minRet = mean_ret_portfo

        # calculate standard deviation of portfolio
        cov_matrix_ret = np.array(data.cov())
        std_portfo = np.sqrt(np.dot(arr_weights.T, np.dot(cov_matrix_ret, arr_weights)))

        if std_portfo > maxRisk:
            maxRisk = std_portfo
        if std_portfo < minRisk:
            minRisk = std_portfo

    return [maxRet, minRet, maxRisk, minRisk]
    
