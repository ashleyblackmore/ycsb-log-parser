#!/usr/bin/env python
import ycsb-io

import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as colmap
import os

from collections import OrderedDict
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

debug = False

def barplotc(ycsb, dototals):
    #print np.shape(ycsb)
    """
    Makes a bar plot of aggregate data
    Horrible rushed function made under serious time pressure.
    """
    updates = ycsb[0]
    reads = ycsb[1]
    checks = ycsb[2]

    utime, ulatency = splitbyrecordcount(updates)
    rtime, rlatency = splitbyrecordcount(reads)
    ctime, clatency = splitbyrecordcount(checks)

    uheights = []
    rheights = []
    cheights = []

    for i in range(1, len(utime)):
        uheights.append((integralof(ulatency[i], utime[i])/1000)/5)
    for i in range(1, len(rtime)):
        rheights.append((integralof(rlatency[i], rtime[i])/1000)/5)
    for i in range(1, len(ctime)):
        cheights.append((integralof(clatency[i], ctime[i])/1000)/5)

    heights = [ uheights, rheights, cheights ]
    return heights

def barplotb(heightsm, dototals, maxcutoff):
    l = np.shape(heightsm)[1]
    recs = []

    for m in maxcutoff:
        m = round(m, 0)
        recordcounts = np.arange(m/l, m + m/l, m/l)
        for i in range(0, len(recordcounts)):
            recordcounts[i] = int(round_sigfigs(recordcounts[i], 3))
            recs.append(recordcounts)

    btm = 0
    fig = plt.figure()
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 10}
    plt.rc('font', **font)
    fig.add_subplot(111)
    hatches = ['/', 'x', '.' ]

    legend = [ "Update", "Read", "Verification" ]

    for i in range(0, len(heightsm)):
        width = 0.5 # Bar width
        N = np.shape(heightsm)[1] # Number of bars
        ind = np.arange(N) # X-coords of left side of bars

        #c = colmap.gray((i+1)/3.,1)
        c = 'w'

        if i == 0:
            btm = [ 0 for x in range(0, len(heightsm[i])) ]
        elif i == 1:
            btm = heightsm[0]
        elif i == 2:
            btm = [ sum(a) for a in zip(heightsm[0], heightsm[1]) ]

        if debug:
            print "i = ", i
            print "heights:", heightsm[i]
            print "btm", btm

        plt.bar( ind, heightsm[i], width, color = c, hatch = hatches[i], bottom = btm )
        maxplotheight = max([ sum(a) for a in
                        zip(heightsm[0], heightsm[1], heightsm[2]) ])
        locs, labels = plt.yticks()
        plt.xticks(ind+width/2., recordcounts, rotation = 35)
        plt.yticks()

    plt.suptitle('Latency vs. Number of Records' + getTestInfo())
    plt.xlabel('Number of Records (Approx.)')

    if prunedtoken == 'P':
        plt.legend( legend, loc=2, bbox_to_anchor=(1.05, 1),
                borderaxespad=0. )

    fig.subplots_adjust( bottom=0.20, hspace=0.20, right=0.75 )

    if dototals:
        plt.ylabel('Total Latency (ms)')
        plt.savefig( getfilename("barplotTOTAL", i),
                     format='png', dpi=300, bbox_inches='tight',
                     transparent=True )
    else:
        plt.ylabel('Average Latency (ms)')
        plt.savefig( getfilename("barplotAVG", i),
                     format='png', dpi=300, bbox_inches='tight',
                     transparent=True )


def splitbyrecordcount(ycsblist):
    keys = map(lambda i:ycsblist[i],
        filter(lambda i:i%2 == 0, range(len(ycsblist))))
    vals = map(lambda i:ycsblist[i],
        filter(lambda i:i%2 == 1, range(len(ycsblist))))

    arrays_k = [ np.zeros(len(ycsblist)) for z in range(0, 10) ]
    arrays_v = [ np.zeros(len(ycsblist)) for z in range(0, 10) ]
    arr_it = 0

    for i in range(0, len(keys)):
        try:
            keys[i] = keys[i].replace(' ', '')
        except AttributeError:
            print "!!! Att error", keys[i], vals[i]
        arrayk = arrays_k[arr_it]
        arrayv = arrays_v[arr_it]
        if keys[i] == 'Operations':
            arr_it += 1
        elif (keys[i] == 'AverageLatency(us)' or
            keys[i] == 'MinLatency(us)' or
            keys[i] == 'MaxLatency(us)' or
            keys[i] == 'Return=0'):
            pass
        else:
            arrayk[i] = int(keys[i])
            arrayv[i] = float(vals[i])

    for i in range(0, len(arrays_k)):
        arrays_k[i] = np.trim_zeros(arrays_k[i], 'b')
        arrays_v[i] = np.trim_zeros(arrays_v[i], 'b')

    for i in range(0, len(arrays_k)):
        #trim array
        if len(arrays_k[i]) != len(arrays_v[i]):
            print len(arrays_k[i]), len(arrays_v[i])
            arrays_k[i] = arrays_v[i][0:len(arrays_k[i])]

    return arrays_k, arrays_v

def timedomain(ycsb, toplt):
    arrays_k, arrays_v = splitbyrecordcount(ycsb[toplt])
    arrays_ku, arrays_vu = splitbyrecordcount(ycsb[2])
    arrays_kr, arrays_vr = splitbyrecordcount(ycsb[1])
    arrays_kv, arrays_vv = splitbyrecordcount(ycsb[0])

    maxheightu = max([max(x) for x in arrays_vu[1:9]])
    maxheightr = max([max(x) for x in arrays_vr[1:9]])
    maxheightv = max([max(x) for x in arrays_vv[1:9]])
    maxheight = max(maxheightu, maxheightr, maxheightv)
    #print maxheight

    K = []
    K.extend(arrays_k)

    V = []
    V.extend(arrays_v)

    #K = [ K[1], K[11], K[21] ]
    #V = [ V[1], V[11], V[21] ]

    checktype = ( "Update", "Read", "Verification" )[toplt]

    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')

    it = 0
    for z in np.arange(1, 9):
        xs = K[z]
        ys = V[z]
        c = colmap.spectral(z/9.,1)
        ax.plot(xs, z * np.ones(xs.shape), zs=ys, zdir='z', color=c, zorder = -z)

    # Plot formatting
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    #plt.zlim(0, maxheight)

    #plt.legend(checktype, loc=2, bbox_to_anchor=(1.05, 1),
                #borderaxespad=0. )
    ax.set_zlim3d(0, maxheight)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Test Run')
    ax.set_zlabel('Runtime')
    ax.tick_params(axis='both', labelsize = 8)
    plt.savefig( getfilename("timeseries", checktype),
                 format='png', dpi=300, bbox_inches='tight',
                 transparent=True )

def integralof(latency, time):
    return integrate.trapz(latency, time)


def barplot(ycsb, maxcutoff, dototals):
    """
    Makes a bar plot of aggregate data
    Horrible rushed function made under serious time pressure.
    """
    updates = ycsb[0]
    reads = ycsb[1]
    checks = ycsb[2]

    hatches = ['/', 'x', '.' ]

    utime, ulatency = splitbyrecordcount(updates)
    rtime, rlatency = splitbyrecordcount(reads)
    ctime, clatency = splitbyrecordcount(checks)

    # Set up record counts for x-axis titles
    maxcutoff = round(maxcutoff, 0)
    recordcounts = np.arange(maxcutoff/9, maxcutoff + maxcutoff/9, maxcutoff/9)
    for i in range(0, len(recordcounts)):
        recordcounts[i] = int(round_sigfigs(recordcounts[i], 3))

    #for i in range(0, len(clatency)):
        #clatency[i] = clatency[i]/1000

    uheights = []
    rheights = []
    cheights = []

    if dototals:
        for i in range(1, len(utime)):
            uheights.append((integralof(ulatency[i], utime[i])/1000)/5)
        for i in range(1, len(rtime)):
            rheights.append((integralof(rlatency[i], rtime[i])/1000)/5)
        for i in range(1, len(ctime)):
            cheights.append((integralof(clatency[i], ctime[i])/1000)/5)
    else:
        for i in range(1, len(utime)):
            uheights.append(((sum(ulatency[i])/1000)/5)/len(ulatency[i]))
        for i in range(1, len(rtime)):
            rheights.append(((integralof(rlatency[i], rtime[i])/1000)/5)/len(rlatency[i]))
        for i in range(1, len(ctime)):
            cheights.append(((integralof(clatency[i], ctime[i])/1000)/5)/len(clatency[i]))

    if debug:
        print "U", len(ulatency[i]), len(utime[i])
        print "R", len(rlatency[i]), len(rtime[i])
        print "C", len(clatency[i]), len(ctime[i])

    btm = 0
    fig = plt.figure()
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 10}
    plt.rc('font', **font)
    fig.add_subplot(111)

    heights = [ uheights, rheights, cheights ]
    legend = [ "Update", "Read", "Verification" ]

    for i in range(0, len(heights)):
        width = 0.5 # Bar width
        N = 9 # Number of bars
        ind = np.arange(N) # X-coords of left side of bars

        #c = colmap.gray((i+1)/3.,1)
        c = 'w'

        if i == 0:
            btm = [ 0 for x in range(0, len(heights[i])) ]
        elif i == 1:
            btm = heights[0]
        elif i == 2:
            btm = [ sum(a) for a in zip(heights[0], heights[1]) ]

        if debug:
            print "i = ", i
            print "heights:", heights[i]
            print "btm", btm

        plt.bar( ind, heights[i], width, color = c, hatch = hatches[i], bottom = btm )
        maxplotheight = max([ sum(a) for a in
                        zip(heights[0], heights[1], heights[2]) ])
        locs, labels = plt.yticks()
        #print locs
        #labs = [ l/1000 for l in locs ]
        #print locs
        #plt.yticks(locs, labs)
        plt.yticks()

    plt.suptitle('Latency vs. Number of Records' + getTestInfo())
    plt.xlabel('Number of Records (Approx.)')
    plt.xticks(ind+width/2., recordcounts, rotation = 35)

    if prunedtoken == 'P':
        plt.legend( legend, loc=2, bbox_to_anchor=(1.05, 1),
                borderaxespad=0. )

    fig.subplots_adjust( bottom=0.20, hspace=0.20, right=0.75 )

    if dototals:
        plt.ylabel('Total Latency (ms)')
        plt.savefig( getfilename("barplot", i),
                     format='png', dpi=300, bbox_inches='tight',
                     transparent=True )
    else:
        plt.ylabel('Average Latency (ms)')
        plt.savefig( getfilename("barplotAVG", i),
                     format='png', dpi=300, bbox_inches='tight',
                     transparent=True )

def round_sigfigs(num, sig_figs):
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

def getTestInfo():
    numberpart = ''
    alphapart = ''

    for i in TEST:
        try:
            int(i)
            numberpart += i
        except ValueError:
            alphapart += i

    if alphapart == 'IBM':
        return prunetype + '\nIBM Academic Skills Cloud: Single Instance (' + numberpart + ' Operations)'
    if alphapart == 'AWS':
        return prunetype + '\nAmazon EC2: Four-Node Medium Instance (' + numberpart + ' Operations)'
    if alphapart == 'SIT':
        return prunetype + '\nSchool of IT: Single Instance (' + numberpart + ' Operations)'


def trim_zeros(filt, trim='fb'):
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filt:
            if i != 0.0: break
            else: first = first + 1
    last = len(filt)
    if 'B' in trim:
        for i in filt[::-1]:
            if i != 0.0: break
            else: last = last - 1
    return filt[first:last]

def keysep(remoutliers):
    """
    Determines the average algorithm runtime for a keyspace:
    Supports Record/Vertex/Edge Binning.
    Instructions: numberofbins determines width of bins.
    To choose recordspace, use 0
    To choose vertices, use 1
    To choose edges, use 2
    """
    arrays = [ np.zeros(numresults) for x in range (0,4) ]
    array_nkey = arrays[0]
    array_vset = arrays[1]
    array_eset = arrays[2]
    array_time = arrays[3]

    incr = 0

    # Fill the arrays with COLD HARD DATA >:I
    for k in funandprofit.keys():
        v = funandprofit.get(k)
        array_nkey[incr] = v[6]
        array_vset[incr] = v[7]
        array_eset[incr] = v[8]
        array_time[incr] = v[5]

        incr += 1

        if v[5] == 0.0:
            print "!!! Error: Algo runtime is 0"
        if v[6] == 0.0:
            print "!!! Error: Record space size is 0"

    if remoutliers:
        print "Time array length before outlier removal:", len(array_time)
        array_time = removeoutliers(array_time)
        print "Time array length after outlier removal:", len(array_time)

    # Convert from nanosecs to us
    for i in range(0, len(array_time)):
        array_time[i] = array_time[i]/1000

    arrays[0] = array_nkey
    arrays[1] = array_vset
    arrays[2] = array_eset
    arrays[3] = array_time
    return arrays

def getaverages(rangebins, barcutoffs):
    # Contains user choices for which array to output
    binview = [ len(b) for b in rangebins ]

    if debug and sum(binview) != numresults:
        print "! Not all results were binned."
        print "sum(binview) =", sum(binview), "numresults =", numresults
        print barcutoffs
        print "Verification times:", binview
        print "Bin sum is", sum(binview), "taken from", numresults, "results"

    # Return a dictionary of ranges vs. time averages.
    averages = OrderedDict()

    for i in range(1, len(rangebins) - 1):
        brange = int(barcutoffs[i]), int(barcutoffs[i+1])
        if len(rangebins[i]) == 0:
            runtimeavg = 0
        else:
            runtimeavg = sum(rangebins[i])/len(rangebins[i])
        averages[brange] = runtimeavg
    return averages

def histo(arrays):
    # Binning of values is determined by builtin histogram algorithm:
    # Change the shape here to alter range distribution
    # Plot histogram, if desired:
    array_time = arrays[3]
    choices = arrays[0:3]
    hatches = ['x', '.', '/' ]
    xtitles = [ 'Number of records in graph at time of operation',
                'Vertex space size at time of operation',
                'Edge space at time of operation' ]
    fig = plt.figure()
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)

    cn = 0
    ranges = []
    bars = []
    maxes = []
    for c in choices:
        sp = 311 + 1*(cn)
        ax = fig.add_subplot(sp)
        ax.set_yscale('log')
        histfreq, barcutoffs, _nix = ax.hist( c, bins=numbins,
                                              hatch=hatches[cn], color='w' )

        locs, labels = plt.xticks()
        newlocs = []
        for l in range(0, 2*len(locs)):
            #print l, max(locs), len(locs) - 1
            newlocs.append(l*max(locs)/(2*(len(locs) - 1)))
        plt.xticks( newlocs, rotation = 30 )
        maxes.append(max(barcutoffs))

        plt.xlim(0, max(c))

        rangebins = [ [] for x in range(0, len(barcutoffs)) ]

        pxlabel = xtitles[cn]
        plt.xlabel(pxlabel)
        plt.ylabel("Number of operations")
        #Separate into "size-of-recordspace buckets"
        for i in range(0, len(array_time) - 1):
            t = array_time[i]
            k = c[i]

            for j in range(0, len(barcutoffs)):
                xlim = barcutoffs[j]
                if k <= xlim:
                    rangebins[j].append(t)
                    break
            else:
                # N.B. The else belongs to the for loop, _not_ the if
                print "! A result was not binned:", k, t

        ranges.append(rangebins)
        bars.append(barcutoffs)
        cn += 1

    fig.subplots_adjust(hspace = 0.5)

    title = "Experimental Space Histogram "
    title += getTestInfo()

    plt.suptitle(title)
    plt.ylabel("Number of operations")
    fig.set_figheight(10)
    plt.savefig( getfilename("histogram", numresults),
                 format='png', dpi=120, bbox_inches='tight',
                 transparent=True )
    #plt.show()

    return ranges, bars, max(maxes)

def getfilename(graphtype, n):
    # Save the file
    filename = "../wab/" + str.lower(TESTNAME) + prunedtoken + str(mult) + "/"
    filename += graphtype
    filename += TEST
    filename += 'vs'
    filename += str(n)
    if remoutliers:
        filename += 'TRIMMED'
    filename += '.png'
    print filename
    return filename

def mad(x):
    """
    Calculates the Median Absolute Deviation (MAD)
    (the median of absolute deviations from the mean).
    """
    return np.median([abs(val-np.median(x)) for val in x])

def removeoutliers(num):
    """
    Remove outliers using the MAD algo. These are counted by the "discount"
    variable. By using j and not i to access the new array, all the zeros end up
    at the end of the array, making it simple to discount them later
    """
    it=0
    length = len(num)
    # Calculate the median and MAD, and calculate the
    # thresholds for classification as an outlier
    madd = mad(num)
    medd = np.median(num)
    lowerbound = 0#medd - 5 * madd
    upperbound = medd + 6 * madd

    # If a number is five times the Median Absolute Deviation away
    # from the Median, do not include it in the results
    print inout.getdatasetnames()[it], "MAD:", madd, "\tMedian:", medd,
    "\tBoundaries:", lowerbound, upperbound, "Shape:", np.shape(num)

    numbers_norm = np.zeros(length)
    discount, j = 0, 0

    for i in range(0, length):
        if (lowerbound <= num[i] <= upperbound):
            numbers_norm[j] = num[i]
            j += 1
        elif num[i] == 0:
            pass
        else:
            discount += 1

    # Discard only members of the array that are outliers.
    count_nonzeros = len(numbers_norm) - discount - 1
    if discount != 0:
        final_result = numbers_norm[0:count_nonzeros]
    else:
        final_result = numbers_norm

    print "Ignoring", discount, "outliers in", length, "result set."
    return final_result

def xy(ranges, bars):
    """ x vs. y plot of contents of input:
        a tuple key with a single float value"""
    for c in range(0, 3):

        averages = getaverages(ranges[c], bars[c])

        xtitles = [ 'Record', 'Vertex', 'Edge' ]
        fig=plt.figure()

        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 10}
        plt.rc('font', **font)

        x = np.arange(len(averages))
        y = averages.values()

        # Plot formatting
        ax = fig.add_subplot(111, clip_on = False)
        ptitle = "Runtime for "
        ptitle += xtitles[c]
        ptitle += "space: "
        ptitle += getTestInfo()
        plt.title(ptitle)
        pxlabel = "Total "
        pxlabel += xtitles[c]
        pxlabel += " Space Size"
        plt.xlabel(pxlabel)
        plt.ylabel( "Verification Runtime (us)" )

        binlabels = []
        for n, z in averages.keys():
            binlabels.append(n)
        binlabels = np.array(binlabels)

        plt.xticks( np.arange(len(averages)), binlabels[0:(len(binlabels)-1)],
                    rotation = 40 )

        ## Polyfitting
        fitdegree = 1
        z = np.polyfit(x, y, fitdegree)
        p = np.poly1d(z)
        xp = np.linspace(min(x), max(x), 100000)

        #ax.plot(xp, p(xp))
        ax.plot(x[1:len(x)-1], y[1:len(y)-1], 'o', xp, p(xp))
        #ax.plot(x, y, 'o')
        fig.subplots_adjust(bottom=0.20)

        # Save the file
        plt.savefig( getfilename(("xy" + xtitles[c]), len(funandprofit)),
                     format='png', bbox_inches='tight', dpi=300,
                     transparent=True, figsize=(8, 6) )
        #plt.show()

# I/O:
TEST='' # name of test AND folder
#TESTNAME = raw_input("A or I? ")
#TESTNAME = 'A'
#if TESTNAME.upper() == 'A':
    #TESTNAME = 'AWS'
#elif TESTNAME.upper() == 'I':
    #TESTNAME = 'IBM'
#else:
TESTNAME = 'AWS'

remoutliers = False

pardir = str.split(os.path.abspath(os.path.join(__file__, os.path.pardir)), '/')
prunetype = ''
prunedtoken = ''
print pardir[len(pardir) - 1]
if pardir[len(pardir) - 1] == 'PrunedSet':
    prunetype = ' (Pruning Enabled)'
    prunedtoken = 'P'
elif pardir[len(pardir) - 1] == 'NonPrunedSet':
    prunetype = ' (Pruning Disabled)'
    prunedtoken = 'NP'

mult = None
rangeset = []
barset = []
cutoffset = []
y = []
yav = []

for j in range(0, 2):
    if j == 0:
       mult = 1
    if j == 1:
       mult = 2
    if j == 2:
       mult = 5

    for i in range(0, 2): #4
        TEST = TESTNAME + str(10**3 * (int(mult)*10**i))
        print TEST
        inout.resetycsb()
        funandprofit, ycsb = inout.getdata(3, TEST)

        numresults =  len(funandprofit)
        numbins = 30

        # Categorise results
        arrays = keysep(remoutliers)
        ranges, bars, maxcutoff = histo(arrays)
        rangeset.append(ranges)
        barset.append(bars)
        cutoffset.append(maxcutoff)
        #xy(ranges, bars)

        # 0 is updates
        # 1 is reads
        # 2 is checks
        #timedomain(ycsb, 0)
        #timedomain(ycsb, 1)
        #timedomain(ycsb, 2)

        #barplot(ycsb, maxcutoff, True)
        #barplot(ycsb, maxcutoff, False)
        dt = True
        y.append(barplotc(ycsb, dt))
        yav.append(barplotc(ycsb, False))

ubanger = []
rbanger = []
vbanger = []

print np.shape(y)
for i in range(0, np.shape(y)[0]):
    ubanger.extend(y[i][0])
    rbanger.extend(y[i][1])
    vbanger.extend(y[i][2])
heightser = [ ubanger, rbanger, vbanger ]
barplotb(heightser, dt, cutoffset)

ubanger = []
rbanger = []
vbanger = []

for i in range(0, np.shape(y)[0]):
    ubanger.extend(yav[i][0])
    rbanger.extend(yav[i][1])
    vbanger.extend(yav[i][2])
heightser = [ ubanger, rbanger, vbanger ]
barplotb(heightser, dt, cutoffset)
