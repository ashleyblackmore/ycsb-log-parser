import glob
import tarfile
import os
import pickle
import csv
from collections import OrderedDict

DEBUG = True

class io:
    pass
    #def __init__(self):
    """
    i/o and parsing of gzipped ycsb log tarballs:
    decompression and de/serialisation.
    """
    def getdata(filetoanalyse, testname):
        """
        A function to pull data from YCSB log files.
        """
        setpath(testname)
        path = getpath()
        filetoanalyse = filetoanalyse - 1
        tgzfiles = False
        p = testname + '/' + '*.tar.gz'

        tgzfiles = sorted(glob.glob(p)) # Glob the tarballs in the dir

        if tgzfiles:
            n = len(tgzfiles)
            print "[info] number of data files is", n

            for i in range(0, n):
                print tgzfiles[i]
                parsedvals, filename = inflate(tgzfiles[i])
                if filename[0:7] == 'testout':
                    funandprofit, ycsb = lex(parsedvals)

                print "Processed", (i + 1), "/", len(tgzfiles),
                print type(funandprofit)

        return funandprofit, ycsb

#    def unpickle(filename):
        #"""
        #Deserialises data that has already been pulled from the tar files
        #Saves time. Currently defunct.
        #"""
        #try:
            #f = filename + path
            #picklefile = open(f, "r")
            #picklefile = pickle.load(picklefile)
            #print "[info] deserialised", f, ",", type(f)
            #return picklefile
        #except IOError:
#            print "[info] the pickle file does not exist: starting again."

    def stash(f):
        """
        Data serialisation.
        """
        fname = path
        fname += f + '.pickle'
        pickle.dump(f, open(fname, "w"))
        if os.path.isfile(fname):
            print "stashed", fname
            return True

    def inflate(tgz):
        """
        Open tarballs from glob
        """
        tar = tarfile.open(tgz, "r:gz")
        f = tar.extractfile(tar.getmembers()[0])
        parsedvals = csv.reader(f, delimiter=',')
        return parsedvals, tar.getnames()[0]

    def lex(parsedvals):
        """
        Takes the csv reader output and categorises by operation token
        """
        incr = 0

        for row in parsedvals:
            try:
                if row:
                    # Parse the tokens from the CSV:
                    if row[0] == '[SPECIAL]':
                        # tag, hashcode, walltime, threadid, iswrite,
                        # duration, keys, vset, edges
                        if len(row) != 9:
                            print row
                            print "ERROR: len(row) != 9"
                            continue
                        funandprofit[hash(str(row))] = row[0:9]
                    elif row[0] == '[UPDATE]':
                        ycsbu.append(row[1])
                        ycsbu.append(row[2])
                    elif row[0] == '[READ]':
                        ycsbr.append(row[1])
                        ycsbr.append(row[2])
                    elif row[0] == '[CHECK]':
                        ycsbc.append(row[1])
                        ycsbc.append(row[2])

                    incr += 1
            except ValueError:
                pass

        ycsb.append(ycsbu)
        ycsb.append(ycsbr)
        ycsb.append(ycsbc)

        return funandprofit, ycsb

    def getdatasetnames():
        """
        What it says on the tin.
        """
        return datasetnames

    def setpath(testname):
        """
        Set the path for data files
        """
        testname += '/'
        path = testname
        print "Path is", path

    def getpath():
        """
        Get the path for data files
        """
        return path

    def resetycsb():
        """
        Super-cheesy global reset. Pass the dunce's hat, thanks in advance.
        """
        global ycsb
        ycsb = []
        global ycsbu
        ycsbu = []
        global ycsbr
        ycsbr = []
        global ycsbc
        ycsbc = []

    path = ''

    tokens =       [ '[READ]', '[UPDATE]', '[Read operation]',
                    '[Update operation]', '[CHECK]', '[NUMKEYS]',
                    '[GRAPH_VSIZE]', '[GRAPH_ESIZE]', '[SPECIAL]' ]
    datasetnames = [ 'Reads', 'Updates', 'Reads aux',
                    'Updates aux', 'Checks', 'Keys',
                    'Vertices', 'Edges', 'ycsbmeasuretags' ]

    funandprofit = OrderedDict()
    ycsb = []
    ycsbu = []
    ycsbr = []
    ycsbc = []
