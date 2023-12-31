"""
Numpy Code implementation taken from https://github.com/brc7/PrivateRACE/

*****

Torch implemetation written based on CountSketch Class
Camel Casing for names

"""

import numpy as np
import copy
import torch

##==============================================================================
## NUMPY

from scipy.stats import norm # for P_L2
import math
from scipy.special import ndtr


class L2LSH_numpy():
    def __init__(self, N, d, r):
        # N = number of hashes
        # d = dimensionality
        # r = "bandwidth"
        self.N = N
        self.d = d
        self.r = r

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size = (N,d))
        self.b = np.random.uniform(low = 0,high = r,size = N)


    def hash(self,x):
        return np.floor( (np.squeeze(np.dot(self.W,x)) + self.b)/self.r )


def P_L2_numpy(c,w): ## %%%%  P-stable Eucledian LSH kernel
    try:
        p = 1 - 2*ndtr(-w/c) - 2.0/(np.sqrt(2*math.pi)*(w/c)) * (1 - np.exp(-0.5 * (w**2)/(c**2)))
        return p
    except:
        return 1
    # to fix nans, p[np.where(c == 0)] = 1


def P_SRP_numpy(x,y):
    x_u = x / np.linalg.norm(x)
    y_u = y / np.linalg.norm(y)
    angle = np.arccos(np.clip(np.dot(x_u, y_u), -1.0, 1.0))
    return 1.0 - angle / np.pi


class SRPMulti_numpy():
    # multiple SRP hashes combined into a set of N hash codes
    def __init__(self, reps, d, p):
        # reps = number of hashes (reps)
        # d = dimensionality
        # p = "bandwidth" = number of hashes (projections) per hash code
        self.N = reps*p # number of hash computations
        self.N_codes = reps # number of output codes
        self.d = d
        self.p = p

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size = (self.N,d))
        self.powersOfTwo = np.array([2**i for i in range(self.N)])

    def hash(self,x):
        # p is the number of concatenated hashes that go into each
        # of the final output hashes
        h = np.sign( np.dot(self.W,x) )
        h = np.clip( h, 0, 1)
        if self.p > 1:
            h = np.reshape(h,(-1,self.p))
            n_hashes = h.shape[0]
            powersOfTwo = np.array([2**i for i in range(self.p)])
            codes = np.zeros(n_hashes)
            for idx,hi in enumerate(h):
                codes[idx] = np.dot(hi,powersOfTwo)
            return codes
        else:
            return(h)


class FastSRPMulti_numpy():
    # multiple SRP hashes combined into a set of N hash codes
    def __init__(self, reps, d, p):
        # reps = number of hashes (reps)
        # d = dimensionality
        # p = "bandwidth" = number of hashes (projections) per hash code
        self.N = reps*p # number of hash computations
        self.N_codes = reps # number of output codes
        self.d = d
        self.p = p

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size = (self.N,d))
        self.powersOfTwo = np.array([2**i for i in range(self.p)])

    def hash(self,x):
        # p is the number of concatenated hashes that go into each
        # of the final output hashes
        h = np.sign( np.dot(self.W,x) )
        h = np.clip( h, 0, 1)
        h = np.reshape(h,(self.N_codes,self.p))
        return np.dot(h,self.powersOfTwo)



class SRP_numpy():
    def __init__(self, N, d):
        # N = number of hashes
        # d = dimensionality
        # r = "bandwidth"
        self.N = N
        self.d = d

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size = (N,d))
        self.powersOfTwo = np.array([2**i for i in range(self.N)])

    def hash(self,x):
        h = np.sign( np.dot(self.W,x) )
        h = np.clip( h, 0, 1)
        return np.dot( h, self.powersOfTwo)

    def hash_independent(self,x,p = 1):
        # p is the number of concatenated hashes that go into each
        # of the final output hashes
        h = np.sign( np.dot(self.W,x) )
        h = np.clip( h, 0, 1)
        if p > 1:
            h = np.reshape(h,(-1,p))
            n_hashes = h.shape[0]
            powersOfTwo = np.array([2**i for i in range(p)])
            codes = np.zeros(n_hashes)
            for idx,hi in enumerate(h):
                codes[idx] = np.dot(hi,powersOfTwo)
            return codes
        else:
            return(h)


class SRP_ALSH_numpy():
    def __init__(self, n, p, m, d):
        # n = number of hashes
        # p = power of hashes
        # m = dimensionality of transformation
        # d = dimensionality of input/query
        self.n = n
        self.p = p
        self.d = d
        self.m = m

        # set up the gaussian random projection vectors
        self.W = np.random.normal(size = (self.n*self.p,self.d + self.m))
        self.powersOfTwo = np.array([2**i for i in range(self.p)])

    def hash_query(self, q):
        q = np.pad(q,(0,self.m),'constant', constant_values = (0,0))
        return self._hash(q)

    def hash_input(self, x):
        norm_x = np.linalg.norm(x)
        x = np.pad(x,(0,self.m),'constant', constant_values = (0,0))
        for i in range(self.m):
            power = 2**(i+1)
            x[self.d + i] = 0.5 - norm_x**power
        return self._hash(x)

    def _hash(self, xt):
        # xt = transformed input / query x
        h = np.sign( np.dot(self.W,xt) )
        h = np.clip( h, 0, 1)
        if self.p > 1:
            h = np.reshape(h,(-1,self.p))
            n_hashes = h.shape[0]
            codes = np.zeros(n_hashes)
            for idx,hi in enumerate(h):
                codes[idx] = np.dot(hi,self.powersOfTwo)
            return codes
        else:
            return (h)

class HPH_numpy(): # hyperplanehash
    def __init__(self, n, d):
        # n = number of hashes
        # p = power of hashes
        # m = dimensionality of transformation
        # d = dimensionality of input/query
        self.n = n
        self.p = 2
        self.d = d
        # self.m = m

        # set up the gaussian random projection vectors
        self.W_data = np.random.normal(size = (self.n*2,self.d))
        self.W_query = self.W_data.copy()
        self.W_query[::2,:] = -1*self.W_query[::2,:]
        self.powersOfTwo = np.array([2**i for i in range(2)])

    def hash_query(self, q):
        h = np.sign( np.dot(self.W_query,q) )
        h = np.clip( h, 0, 1)
        h = np.reshape(h,(-1,2))
        n_hashes = h.shape[0]
        codes = np.zeros(n_hashes)
        for idx,hi in enumerate(h):
            codes[idx] = np.dot(hi,self.powersOfTwo)
        return codes

    def hash_input(self, x):
        h = np.sign( np.dot(self.W_data,x) )
        h = np.clip( h, 0, 1)
        h = np.reshape(h,(-1,2))
        n_hashes = h.shape[0]
        codes = np.zeros(n_hashes)
        for idx,hi in enumerate(h):
            codes[idx] = np.dot(hi,self.powersOfTwo)
        return codes


class RACE_numpy():
    def __init__(self, rows, cols, dtype = np.int32):
        self.dtype = dtype
        self.R = rows  # repetitions # number of ACEs (rows) in the array
        self.C = cols  # hash_range  # range of each ACE (width of each row)
        self.counts = np.zeros((self.R,self.C),dtype = self.dtype)
        self.real_counts = np.zeros((self.R,self.C),dtype = self.dtype)

    def add(self, hashvalues):
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.C
            self.real_counts[idx,rehash] += 1

    def remove(self, hashvalues):
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.C
            self.real_counts[idx,rehash] += -1

    def set_epsilon(self, epsilon):
        # make the whole RACE sketch epsilon-differentially private
        if epsilon is None:
            self.counts = self.real_counts.copy()
            return
        N = np.sum(self.real_counts[0,:])
        noise = np.random.laplace(scale = self.R / (N * epsilon), size=self.real_counts.shape)
        noise = np.floor(noise)
        self.counts = self.real_counts + np.array(noise,dtype = self.dtype)

    def clear(self):
        self.counts = np.zeros((self.R,self.C), dtype = self.dtype)

    def non_private_query(self, hashvalues):
        mean = 0
        N = np.sum(self.real_counts[0,:])
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.C
            mean = mean + self.real_counts[idx,rehash]
        return mean/(self.R * N)

    def query(self, hashvalues):
        mean = 0
        N = np.sum(self.counts) / self.R
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.C
            mean = mean + self.counts[idx,rehash]
        return mean/(self.R * N)

    def print(self):
        for i,row in enumerate(self.counts):
            print(i,'| \t',end = '')
            for thing in row:
                print(str(int(thing)).rjust(2),end = '|')
            print('\n',end = '')

    def counts(self):
        return self.counts


##==============================================================================
## TODO: Complete and Verify TORCH version of RACE


class L2LSH():
    def __init__(self, d, r, bw, device):
        # r = number of hashes
        # d = dimensionality
        # bw = "bandwidth"
        self.d = d
        self.r = r
        self.bw = bw
        self.device = device
        # set up the gaussian random projection vectors
        self.W = torch.randn((self.r, self.d)).to(self.device)
        self.B = (torch.rand(self.r) * self.bw).to(self.device)

    def hash(self,x):
        return torch.floor((torch.matmul(self.W, x) + self.B) / self.bw).to(torch.long)

    def to_(self, device):
        self.device = device
        self.B = self.B.to(device)
        self.W = self.W.to(device)


class RaceSketchVec(object):
    """ RaceSketch of a vector based on loc or value or each element

    NOTE:
    This class processes a vector stream treated as a sequence of tokens
    and counted,also extended for weights(values) addition instead of count .
    Functionality for value-based Locality-Sensitive Hashing (LSH)"
    NOT index based like CountSketch implementation.


    public methods: zero, unSketch, l2estimate, __add__, __iadd__
    """

    def __init__(self, d, r, c, bw,
                weighted_count = True,
                doInitialize=True,
                device=None,
                seed=42):
        """ Constructor for RSvec

        Args:
            d: the cardinality of the skteched vector
            c: the number of columns (buckets) in the sketch
            r: the number of rows in the sketch
            doInitialize: if False, you are responsible for setting
                self.table, self.signs, self.buckets, self.blockSigns,
                and self.blockOffsets
            device: which device to use (cuda or cpu). If None, chooses
                cuda if available, else cpu
        Note:

        """

        # save random quantities in a module-level variable so we can
        # reuse them if someone else makes a sketch with the same d, c, r
        global rs_cache

        self.r  = r      # rows - num of hashes used
        self.d  = int(d) # vector dimensionality                                 # need int() here b/c annoying np returning np.int64...
        self.c  = c      # hash_range number of Cols in each row
        self.bw = bw     # bandwidth - range of single bin

        self.weighted_count = weighted_count # bool to use count or values as weights for counting

        if self.weighted_count: self.dtype = torch.float32
        else : self.dtype = torch.int64


        # choose the device automatically if none was given
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if (not isinstance(device, torch.device) and
                    not ("cuda" in device or device == "cpu")):
                msg = "Expected a valid device, got {}"
                raise ValueError(msg.format(device))

        self.device = device

        # this flag indicates that the caller plans to set up
        # self.signs, self.buckets
        # itself (e.g. self.deepcopy does this)
        if not doInitialize: #NOTE: Not used now here in this class
            return


        # initialize the sketch to all zeros
        self.table = torch.zeros((r, c), dtype=self.dtype, device=self.device)

        # do all these computations on the CPU, since pytorch
        # is incapable of in-place mod, and without that, this
        # computation uses up too much GPU RAM
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)

        self.hashObj = L2LSH(self.d, self.r, self.bw, self.device)

        torch.random.set_rng_state(rand_state)


    def zero(self):
        """ Set all the entries of the sketch to zero """
        self.table.zero_()


    def to_(self, device): #for torch tensors
        self.device = device
        self.table = self.table.to(device)
        self.hashObj.to_(device)

    def cpu_(self): #for torch tensors
        self.to_("cpu")

    def cuda_(self, device="cuda"): #for torch tensors
        self.to_(device)


    def half_(self):
        self.table = self.table.half()

    def float_(self):
        self.table = self.table.float()

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newRSVec = RaceSketchVec(d=self.d, c=self.c, r=self.r, bw=self.bw,
                        weighted_count=self.weighted_count,
                        doInitialize=True, device=self.device,)
        newRSVec.table = copy.deepcopy(self.table)

        return newRSVec

    def __imul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.table = self.table.mul_(other)
        else:
            raise ValueError(f"Can't multiply a CSVec by {other}")
        return self


    def __mul__(self, other):
        returnCSVec = copy.deepcopy(self)
        returnCSVec *= other
        return returnCSVec


    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.table = self.table.div_(other)
        else:
            raise ValueError(f"Can't divide a CSVec by {other}")
        return self

    def __add__(self, other):
        """ Returns the sum of self with other

        Args:
            other: a CSVec with identical values of d, c, and r
        """
        # a bit roundabout in order to avoid initializing a new CSVec
        returnCSVec = copy.deepcopy(self)
        returnCSVec += other
        return returnCSVec

    def __iadd__(self, other):
        """ Accumulates another sketch

        Args:
            other: a RSVec with identical values of d, c, r, device, numBlocks
        """
        if isinstance(other, RaceSketchVec):
            # merges csh sketch into self
            assert(self.d == other.d)
            assert(self.c == other.c)
            assert(self.r == other.r)
            assert(self.bw == other.bw)
            assert(self.device == other.device)
            assert(type(self.hashObj) is type(other.hashObj))
            self.table += other.table
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    def accumulateTable(self, table):
        """ Adds a CSVec.table to self

        Args:
            table: the table to be added

        """
        if table.size() != self.table.size():
            msg = "Passed in table has size {}, expecting {}"
            raise ValueError(msg.format(table.size(), self.table.size()))

        self.table += table

    def accumulateVec(self, vec):
        """ Sketches a vector
        Args:
            vec: the vector to be sketched
        """

        assert(len(vec.size()) == 1 and vec.size()[0] == self.d)

        bins = self.hashObj.hash(vec)
        bins =  torch.remainder(bins.to(torch.long), 10)  ## torch.fmod gives signs
        bins = torch.stack([torch.tensor(range(0, self.r), dtype=torch.long, device=self.device),
                            bins], dim=1)

        if self.weighted_count:
            self.table[bins[:, 0], bins[:, 1]] += torch.mean(vec)
        else:
            self.table[bins[:, 0], bins[:, 1]] += 1

    def removeVec(self, vec):
        """ Removes a vector from sketch
        Args:
            vec: the vector to be removed
        """
        assert(len(vec.size()) == 1 and vec.size()[0] == self.d)

        bins = self.hashObj.hash(vec)
        bins =  torch.remainder(bins.to(torch.long), 10)  ## torch.fmod gives signs
        bins = torch.stack([torch.tensor(range(0, self.r), dtype=torch.long, device=self.device),
                            bins], dim=1)

        if self.weigted_count:
            self.table[bins[:, 0], bins[:, 1]] -= vec
        else:
            self.table[bins[:, 0], bins[:, 1]] -= 1


    def unSketch(self, k=None, epsilon=None, all=False):
        raise "NOT IMPLEMENTED"
        return

    def l2estimate(self):
        """ Return an estimate of the L2 norm of the sketch """
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())
