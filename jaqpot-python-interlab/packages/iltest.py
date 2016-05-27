#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
from random import randrange
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import cStringIO
from numpy import random
import scipy
from scipy.stats import chisquare
from copy import deepcopy
import operator 
import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter
app = Flask(__name__, static_url_path = "")

"""
    JSON Parser for interlabtest
"""
def getJsonContents (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]

        dataEntry = dataset.get("dataEntry", None)

        variables = dataEntry[0]["values"].keys() 

        owners_labs = []
        target_variable_values = []
        uncertainty_values = []
        for i in range(len(dataEntry)):
            owners_labs.append(dataEntry[i]["compound"].get("ownerUUID"))
            for j in variables:		
                temp = dataEntry[i]["values"].get(j)
                if isinstance (temp, list):
                    for k in range (len(temp)):
                        temp[k] = float(temp[k])
                    temp = numpy.average(temp)
                    temp = round(temp, 2)
                else:
                    try:
                        if isinstance (float(temp), float):
                            temp = float(temp)
                            temp = round(temp, 2)
                    except:
                        pass

                if j == predictionFeature:
                    target_variable_values.append(temp)
                else:
                    uncertainty_values.append(temp)
        """
        data_list = [[],[],[]]
        data_list[0] = owners_labs
        data_list[1] = target_variable_values
        data_list[2] = uncertainty_values
        """
        data_list = []
        data_list.append(owners_labs)
        data_list.append(target_variable_values)
        data_list.append(uncertainty_values)


    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"

    #data_list = sorted(data_list, key=itemgetter(1)) 
    data_list_transposed = map(list, zip(*data_list)) 
    return data_list_transposed, data_list # report in reverse


"""
    Get data. Returns list AND list_transposed
"""
def get_data ():
    path = "C:/Users/Georgios Drakakis/Desktop/Table8Unsorted.csv"
    #path = "C:/Users/Georgios Drakakis/Desktop/Table2.txt"
    
    # 1 =file has column names, 0 = no, just data
    has_header = 1

    # "," || "\t"
    delimitter = "," 
    #delimitter = "\t" 

    dataObj = open(path)
    my_list = []

    if has_header == 1:
        header = dataObj.readline()
        header = header.strip()
        header = header.split(delimitter)    

    while 1:
        line = dataObj.readline()
        if not line:
            break
        else:
            line = line.strip()
            line = line.replace("'", "") 
            temp = line.split(delimitter)

        for i in range (1,len(temp)):
            temp[i] = eval(temp[i])
            if isinstance (temp[i], list):
                for j in range (len(temp[i])):
                    temp[i][j] = float(temp[i][j])
                temp[i] = numpy.average(temp[i])
            else:
                try:
                    if isinstance (float(temp[i]), float):
                        temp[i] = float(temp[i])
                except:
                    pass
        #print temp
        my_list.append(temp)
    #print my_list
    # my_list = [ [Lab, Value, Uncertainty], [Lab, Value, Uncertainty], ...]
    # my_list_transposed = [ [Labs], [Values], [Uncertainties] ], sorted by Values
    my_list = sorted(my_list, key=itemgetter(1)) 
    my_list_transposed = map(list, zip(*my_list)) 

    return header, my_list, my_list_transposed 

"""
    initialise robust average and standard deviation
    *before* Algorithm A
"""
def init_robust_avg_and_std (sorted_data_list):
    x_star = numpy.median(sorted_data_list)
    median_list = abs(sorted_data_list - x_star)
    s_star =  1.483*numpy.median(median_list)
    x_star = round (x_star, 2)
    s_star = round (s_star, 2)
    return x_star, s_star

"""
    Algorithm A
"""
def algorithm_a(sorted_data_list, x_star, s_star):
    delta = 1.5*s_star
    new_sorted_data_list = []
    for i in range (len(sorted_data_list)):
        if sorted_data_list[i] < x_star - delta:
            new_sorted_data_list.append(x_star - delta)
        elif (sorted_data_list[i] > x_star + delta):
            new_sorted_data_list.append(x_star + delta)
        else: 
            new_sorted_data_list.append(sorted_data_list[i])
    new_sorted_data_list = numpy.array(new_sorted_data_list)
    x_star_new = sum(new_sorted_data_list)/len(new_sorted_data_list)
    s_star_new = 1.134*numpy.sqrt( (sum(numpy.power (new_sorted_data_list - x_star_new,2 ))) / (len(new_sorted_data_list) - 1) )
    x_star_new = round(x_star_new,2) 
    s_star_new = round(s_star_new,2)
    return x_star_new, s_star_new, new_sorted_data_list

"""
    Loop Algorithm A until values converge
"""
def loop_algorithm_a (sorted_data_list):
    x,s = init_robust_avg_and_std(sorted_data_list)
    tempx, temps = x,s
    temp_list = deepcopy(sorted_data_list)
    while 1:
        new_tempx, new_temps, new_temp_list = algorithm_a (temp_list, tempx, temps)
        if new_tempx == tempx and new_temps == temps: 
            break
        else:
            tempx = new_tempx
            temps = new_temps
            temp_list = deepcopy(new_temp_list)
    return new_temp_list, tempx, temps # rename????

"""
    Get standard uncertainty of assigned value from expert labs
"""
def get_uncertainty_for_assigned_value(s, p, uncertainties = []):
    #print "\n\n\n", s,p,uncertainties, "\n\n\n"
    if uncertainties != []:
        U_X_assigned = (1.25 / p) * numpy.sqrt(sum(numpy.power(uncertainties,2))) 
    else:
        U_X_assigned = (1.25 * s) / numpy.sqrt(p)

    U_X_assigned = round (U_X_assigned,2)
    return U_X_assigned

"""
    Removes data likely to make plots unreadable
"""
def kill_outliers(dataTransposed, robust_avg_x, robust_std_s):
    temp = [[],[],[]]
    for i in range (len(dataTransposed[1])):
        if dataTransposed[1][i] < robust_avg_x + 3.5* robust_std_s and dataTransposed[1][i] > robust_avg_x - 3.5* robust_std_s:
            temp[0].append(dataTransposed[0][i])
            temp[1].append(dataTransposed[1][i])
            temp[2].append(dataTransposed[2][i])
    dataTransposed = deepcopy(temp)
    return dataTransposed


"""
     Create histograms for report
"""
def hist_plots(num_bins, my_list_transposed, header):

    bins = numpy.linspace(min(my_list_transposed[1]), max(my_list_transposed[1]), num_bins+1)

    labels = []
    for i in range (num_bins):
        labels.append("")

    for i in range (len(my_list_transposed[1])):
        for j in range (num_bins):
            if my_list_transposed[1][i] >= bins[j] and my_list_transposed[1][i] <= bins[j+1]:
                labels[j] = labels[j] + " " + str(my_list_transposed[0][i])

    colours = []
    #print num_bins
    for i in range (num_bins):
        colours.append(random.rand(3,1)) # random colour scheme
                

    #myFIGA = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') ###
    myFIGA1 = plt.figure()

    # hist returns: num inst, bin bound, patches
    nn, bb, pp  = plt.hist(my_list_transposed[1], bins=num_bins, histtype='barstacked', color = random.rand(3,1), stacked=True) # , normed=True

    ## copy labels for 2nd histogram (first copy gets altered here)
    labels_copy = deepcopy(labels) 
    for i in range (len(pp)-1,0,-1):
        if labels[i] == "":
            labels.pop(i)
            pp.pop(i)
            colours.pop(i)
    for i in range (len(pp)):
        plt.setp(pp[i], color=colours[i])  

    ##################################################################################################
    #ax = plt.subplot(121) # <- with 2 we tell mpl to make room for an extra subplot
    #ax.plot([1,2,3], color='red', label='thin red line')
    #ax.plot([1.5,2.5,3.5], color='blue', label='thin blue line')
    #plt.legend(pp, labels, bbox_to_anchor=(1.05, 1), loc=2,  borderaxespad=0.)
    #plt.legend(pp, labels, bbox_to_anchor=(0, 1, 1, 1), loc=3, ncol =2, mode="expand", borderaxespad=0.,fontsize = 'x-small')
    #
    #works
    #johny = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2,  mode="expand", borderaxespad=0.,fontsize = 'x-small')
    #myFIGA.savefig('C:/Python27/interBLX.png', dpi=300, format='png', bbox_extra_artists=(johny,), bbox_inches='tight')
    #
    # local max window
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    ##################################################################################################
    
    myLegend = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2,  mode="expand", borderaxespad=0.,fontsize = 'small')
    sio = cStringIO.StringIO()
    myFIGA1.savefig(sio,format = 'PNG')
    saveas = pickle.dumps(myFIGA1)
    fig1_encoded = base64.b64encode(saveas)
    #plt.show()

    ##################################################################################################
    #plt.legend(pp, labels, bbox_to_anchor=(0, 1,1,2), loc=3,ncol =2, mode="expand", borderaxespad=0.)
    #plt.legend(pp, labels, bbox_to_anchor=(1, 0,0,1), loc=2, ncol=2, borderaxespad=0.)
    #
    #gg = plt.figure(figsize=(8, 6), dpi=80)#, facecolor='w', edgecolor='k')
    #gg.add_axes([0.1, 0.1, 0.6, 0.75])
    #plt.xlabel(header[1])
    #plt.ylabel(header[0])
    #plt.legend(pp, labels, bbox_to_anchor=(1, 0,0,1), loc=2, ncol=1, mode="expand", borderaxespad=0.)
    #plt.show() ######## HIDE show on development
    #
    #plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    #plt.legend(pp, labels, bbox_to_anchor=(0., 1.02, 1., .102))
    #plt.show()
    ##################################################################################################
	
    myFIGA2 = plt.figure()

    X = []
    Y = []
    for i in range (len(bb)-1):
        X.append( (bb[i] + bb[i+1]) / 2 )
        Y.append(0)
    plt.plot(Y,X, 'r--')
    for i in range (len(X)):
        plt.text(Y[i]+0.1, X[i], labels_copy[i])
    plt.axis([0, 1, min(bb), abs(min(bb))+max(bb)]) ##abs
    plt.xlabel(header[1])
    plt.ylabel(header[0])

    sio = cStringIO.StringIO()
    myFIGA2.savefig(sio,format = 'PNG')
    saveas = pickle.dumps(myFIGA2)
    fig2_encoded = base64.b64encode(saveas)
    #plt.show() ######## HIDE show on development
    return fig1_encoded, fig2_encoded

"""
    Start 
"""
def interlab_test (headers, myData, myDataTransposed ):

    # get data
    #headers, myData, myDataTransposed = get_data() # transposed == REVERSED INDICES
    #print "\n\n\n", myData, myDataTransposed , "\n\n\n"
    dataTransposed = deepcopy(myDataTransposed) ### data may be modified depending on extreme values - use for plots
    
    #print "\nHeaders: ", headers
    #print "\nSorted Data: ", myData
    #print "\nProcessed Data: ", myDataTransposed

    data_list, robust_avg_x, robust_std_s = loop_algorithm_a(myDataTransposed[1]) # 'values' index
    #print robust_avg_x, robust_std_s

    # if uncertainties have been reported:
    if len(myDataTransposed) == 3:
        tempU = get_uncertainty_for_assigned_value(robust_std_s, len(myDataTransposed[2]), uncertainties = myDataTransposed[2]) # 'uncertainty' index
    else:
        tempU = get_uncertainty_for_assigned_value(robust_std_s, len(myDataTransposed[2]), uncertainties = [])
    #print "\nUncertainty = ", tempU

    dataTransposed = kill_outliers(dataTransposed, robust_avg_x, robust_std_s)

    # Optimal No. bins accoring to Scott's rule
    test_bin = int ( (3.5*robust_std_s) / numpy.power(len(dataTransposed[1]), 0.3))
    if test_bin >20:
        test_bin = 20
    #test_bin = 18 # 'custom'
    #print "\nCalculated bins: ", test_bin
    fig1, fig2 = hist_plots(test_bin, dataTransposed, headers)

    """
        lab_bias = get_diff (myDataTransposed[0], tempX)
        print "\nLab Bias: ", myDataTransposed[0], lab_bias

        # lab_bias = x-X, tempS = rob std dev
        signals = check_bias(lab_bias, tempS) 
        # list of signals, lab names, lab bias
        print_labs_with_signals (signals, myDataTransposed[1], lab_bias) 

        lab_bias_percent = get_diff_percent (myDataTransposed[0], tempX)
        print "\nLab Bias %%: ", lab_bias_percent

        signals_percent = check_bias(lab_bias_percent, tempS) 
        print_labs_with_signals (signals, myDataTransposed[1], lab_bias_percent) 

        test_bin = 18 # normalise somehow
        #test_bin = int ( (3.5*tempS) / numpy.power(len(dataTransposed[1]), 0.3))

        # construct matrix with labels and values first
        percent_for_hist = []
        percent_for_hist.append(myDataTransposed[1]) 
        percent_for_hist.append(lab_bias_percent)
        hist_plots(test_bin, percent_for_hist, headers)

        # return ranks
        rank, rank_pc = get_ranks(myDataTransposed[1])
        print "\nRanks: ", rank
        print "\nRanks %%: ", rank_pc
    """
    return robust_avg_x, robust_std_s, fig1, fig2
"""
# only if assigned value exists (not calculated as robust avg)
comparison_of_assigned_value(ass_val, x_star, s_star, p, u)
"""
######################################
# RUN
#interlab_test()

@app.route('/pws/interlabtest', methods = ['POST'])
def create_task_interlabtest():

    if not request.json:
        abort(400)

    per_lab_data, per_attribute_data = getJsonContents(request.json)
    headers = "Labs", "Values", "Uncertainties"
    robAA, robSS, fig1, fig2 = interlab_test (headers, per_lab_data, per_attribute_data)
    task = {
        "singleCalculations": {"Robust Average": robAA, "Robust StDev": robSS},
        "arrayCalculations": {},
		"figures": {"Legend 1":fig1, "Hampos is awesome": fig2}
    }
    jsonOutput = jsonify( task )
    #xx = open("C:/Python27/ILTResponse", "w")
    #xx.writelines(str(fig1))
    #xx.writelines(str(task))
    #xx.close()
    return jsonOutput, 201 

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port = 5000, debug = True)	

# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ilt1.json http://localhost:5000/pws/interlabtest
#cd .. C:\Python27\Flask-0.10.1\python-api
#C:/Python27/python iltest.py
#header, my_list, my_list_transposed = get_data ()
#print my_list_transposed
#main_app()