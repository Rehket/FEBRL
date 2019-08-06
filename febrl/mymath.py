# Licensed under GPLv3, please reference LICENSE for more details.
#
# Freely extensible biomedical record linkage (Febrl)
#
# See: http://datamining.anu.edu.au/linkage.html
#
# =============================================================================

"""Module mymath.py - Various mathematical routines.

   See doc strings of individual functions for detailed documentation.
"""

# =============================================================================
# Imports go here

import logging
import math
import random

from typing import List, Optional, Union

Num = Union[int, float]

# =============================================================================

# TODO: Switch Occurrences with math.log2()
# def log2(x):
#     """Compute binary logarithm (log2) for a floating-point number.
#
#     USAGE:
#       y = log2(x)
#
#     ARGUMENT:
#       x  An positive integer or floating-point number
#
#     DESCRIPTION:
#       This routine computes and returns the binary logarithm of a positive
#       number.
#     """
#
#     return math.log(x) / 0.69314718055994529  # = math.log(2.0)


# =============================================================================

# A function to create permutations of a list (from ASPN Python cookbook, see:
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66463)


def get_permutations(a):
    if len(a) == 1:
        yield a
    else:
        for i in range(len(a)):
            this = a[i]
            rest = a[:i] + a[i + 1 :]
            for p in get_permutations(rest):
                yield [this] + p


def permute(alist):
    reslist = []
    for l in get_permutations(alist):
        reslist.append(" ".join(l))

    return reslist


# =============================================================================


def perm_tag_sequence(in_tag_seq):

    """
    Create all permuations of a tag sequence.

    USAGE:
      seq_list = perm_tag_sequence(in_tag_seq)

    DESCRIPTION:
      This routine computes all permutations of the given input sequence. More
      than one permutation is created if at least one element in the input
      sequence contains more than one tag.

    :param in_tag_seq: Input sequence (list) with tags
    :return: A list containing tag sequences (lists).
    """

    if not isinstance(in_tag_seq, list):
        logging.exception("Input tag sequence is not a list: %s" % (str(in_tag_seq)))
        raise Exception

    out_tag_seq = [[]]  # List of output tag sequences, start with one empty list

    for elem in in_tag_seq:
        if "/" in elem:  # Element contains more than one tag, covert into a list
            elem = elem.split("/")

        tmp_tag_seq = []

        if isinstance(elem, str):  # Append a simple string
            for t in out_tag_seq:
                tmp_tag_seq.append(t + [elem])  # Append string to all tag sequences

        else:  # Process a list (that contains more than one tags)
            for tag in elem:  # Add each tag in the list to the temporary tag list
                for t in out_tag_seq:
                    tmp_tag_seq.append(t + [tag])  # Append string to all tag sequences

        out_tag_seq = tmp_tag_seq

    # A log message for high volume log output (level 3) - - - - - - - - - - - -
    #
    logging.debug("Input tag sequence: %s" % (str(in_tag_seq)))
    logging.debug("Output permutations:")
    for p in out_tag_seq:
        logging.debug("    %s" % (str(p)))

    return out_tag_seq


# =============================================================================


# TODO: Write a function to find a quantile.

def quantiles(in_data, quant_list):
    """
    Compute the quantiles for the given input data.

    USAGE:
      quant_val_list = quantiles(in_data, quant_list)

    DESCRIPTION:
      This routine computes and returns the values for the given quantiles and
      the give n data.

    :param in_data: A vector of numerical data, e.g. frequency counts
    :param quant_list: A list with quantile values, e.g. [0.5,0.25,0.50,0.75,0.95]
    :return:

    """

    len_in_data = len(in_data)
    in_data.sort()

    val_data = []

    for quant in quant_list:
        if (quant < 0.0) or (quant > 1.0):
            logging.exception(f"Quantile value not between 0 and 1: {quant}")
            raise Exception(f"Quantile value not between 0 and 1: {quant}")

        quant_ind = quant * (len_in_data - 1)  # Adjust for index start 0!

        quant_ind_floor = math.floor(quant_ind)
        quant_ind_int = int(quant_ind_floor)

        if quant_ind == quant_ind_floor:  # Check for fractionals
            val_data.append(in_data[quant_ind_int])
        else:
            quant_ind_frac = quant_ind - quant_ind_floor  # Fractional part

            tmp_val1 = in_data[quant_ind_int]
            tmp_val2 = in_data[quant_ind_int + 1]

            tmp_val = tmp_val1 + (tmp_val2 - tmp_val1) * quant_ind_frac

            val_data.append(tmp_val)

    return val_data


# =============================================================================
# Special random distributions


def random_linear(n):  # Return triangle distribution
    """Based on Paul Thomas' R code, 23 July 2007.

       Returns a random number 0 >= r < n, with a linear distribution, i.e.
       with p(n) < p(m) if n < m.
    """

    return math.sqrt(random.random() * n ** 2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def random_expo(n):  # Return expnential distribution
    """Returns a random number 0 >= r < n, with an exponential distribution.
    """

    r = n * random.expovariate(10.0)

    while r >= n:  # make sure r is not too large.. What is this doing?
        r = n * random.expovariate(10.0)

    return r


# =============================================================================
#
# Following code taken from Rational.py module
#
# changed: "import math as _math" as math already imported, then changed all
#          references of _math to math.

# Why not the built in one? -> was introduced in 3.5
# def _gcd(a, b):
#     if a > b:
#         b, a = a, b
#     if a == 0:
#         return b
#     while 1:
#         c = b % a
#         if c == 0:
#             return a
#         b = a
#         a = c


def _trim(n, d, max_d):
    if max_d == 1:
        return n / d, 1

    last_n, last_d = 0, 1
    current_n, current_d = 1, 0
    while 1:
        div, mod = divmod(n, d)
        n, d = d, mod
        before_last_n, before_last_d = last_n, last_d
        next_n = last_n + current_n * div
        next_d = last_d + current_d * div
        last_n, last_d = current_n, current_d
        current_n, current_d = next_n, next_d
        if mod == 0 or current_d >= max_d:
            break

    if current_d == max_d:
        return current_n, current_d
    i = (max_d - before_last_d) / last_d
    alternative_n = before_last_n + i * last_n
    alternative_d = before_last_d + i * last_d
    alternative = _Rational(alternative_n, alternative_d)
    last = _Rational(last_n, last_d)
    num = _Rational(n, d)
    if abs(alternative - num) < abs(last - num):
        return alternative_n, alternative_d
    else:
        return last_n, last_d


def _approximate(n, d, err):
    r = _Rational(n, d)
    last_n, last_d = 0, 1
    current_n, current_d = 1, 0
    while 1:
        div, mod = divmod(n, d)
        n, d = d, mod
        next_n = last_n + current_n * div
        next_d = last_d + current_d * div
        last_n, last_d = current_n, current_d
        current_n, current_d = next_n, next_d
        app = _Rational(current_n, current_d)
        if mod == 0 or abs(app - r) < err:
            break
    return app


# TODO: Why not use the builtin rational?
class _Rational:
    def __init__(self, n, d):
        if d == 0:
            return n / d
        n, d = list(map(int, (n, d)))
        if d < 0:
            n *= -1
            d *= -1
        f = math.gcd(abs(n), d)
        self.n = n / f
        self.d = d / f

    def __repr__(self):
        if self.d == 1:
            return "rational(%r)" % self.n
        return "rational(%(n)r, %(d)r)" % self.__dict__

    def __str__(self):
        if self.d == 1:
            return str(self.n)
        return "%(n)s/%(d)s" % self.__dict__

    # def __coerce__(self, other):
    #     for int_num in (type(1), type(1)):
    #         if isinstance(other, int):
    #             return self, rational(other)
    #     if type(other) == type(1.0):
    #         return float(self), other
    #     return NotImplemented

    # def __rcoerce__(self, other):
    #     return coerce(self, other)

    def __add__(self, other):
        return _Rational(self.n * other.d + other.n * self.d, self.d * other.d)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return _Rational(self.n * other.n, self.d * other.d)

    def __rmul__(self, other):
        return self * other

    def inv(self):
        return _Rational(self.d, self.n)

    def __div__(self, other):
        return self * other.inv()

    def __rdiv__(self, other):
        return self.inv() * other

    def __neg__(self):
        return _Rational(-self.n, self.d)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __long__(self):
        if self.d != 1:
            raise ValueError("cannot convert non-integer")
        return self.n

    def __int__(self):
        return int(int(self))

    def __float__(self):
        # Avoid NaNs like the plague
        if self.d > 1 << 1023:
            self = self.trim(1 << 1023)
        return float(self.n) / float(self.d)

    def __pow__(self, exp, z=None):
        if z is not None:
            raise TypeError("pow with 3 args unsupported")
        if isinstance(exp, _Rational):
            if exp.d == 1:
                exp = exp.n
        if isinstance(exp, type(1)) or isinstance(exp, type(1)):
            if exp < 0:
                return _Rational(self.d ** -exp, self.n ** -exp)
            return _Rational(self.n ** exp, self.d ** exp)
        return float(self) ** exp

    def __cmp__(self, other):
        return cmp(self.n * other.d, self.d * other.n)

    def __hash__(self):
        return hash(self.n) ^ hash(self.d)

    def __abs__(self):
        return _Rational(abs(self.n), self.d)

    def __complex__(self):
        return complex(float(self))

    def __bool__(self):
        return self.n != 0

    def __pos__(self):
        return self

    def __oct__(self):
        return "%s/%s" % (oct(self.n), oct(self.d))

    def __hex__(self):
        return "%s/%s" % (hex(self.n), hex(self.d))

    def __lshift__(self, other):
        if other.d != 1:
            raise TypeError("cannot shift by non-integer")
        return _Rational(self.n << other.n, self.d)

    def __rshift__(self, other):
        if other.d != 1:
            raise TypeError("cannot shift by non-integer")
        return _Rational(self.n, self.d << other.n)

    def trim(self, max_d):
        n, d = self.n, self.d
        if n < 0:
            n *= -1
        n, d = _trim(n, d, max_d)
        if self.n < 0:
            n *= -1
        r = _Rational(n, d)
        upwards = self < r
        if upwards:
            alternate_n = n - 1
        else:
            alternate_n = n + 1
        if self == _Rational(alternate_n + n, d * 2):
            new_n = min(alternate_n, n)
            return _Rational(new_n, d)
        return r

    def approximate(self, err):
        n, d = self.n, self.d
        if n < 0:
            n *= -1
        app = _approximate(n, d, err)
        if self.n < 0:
            app *= -1
        return app


def _parse_number(num):
    if "/" in num:
        n, d = num.split("/", 1)
        return _parse_number(n) / _parse_number(d)
    if "e" in num:
        mant, exp = num.split("e", 1)
        mant = _parse_number(mant)
        exp = int(exp)
        return mant * (rational(10) ** rational(exp))
    if "." in num:
        i, f = num.split(".", 1)
        i = int(i)
        f = rational(int(f), 10 ** len(f))
        return i + f
    return rational(int(num))


def rational(n, d=1):
    if type(n) in (type(""), type("")):
        n = _parse_number(n)
    if type(d) in (type(""), type("")):
        d = _parse_number(d)
    if isinstance(n, type(1.0)):
        n = _float_to_ratio(n)
    if isinstance(d, type(1.0)):
        d = _float_to_ratio(d)
    for arg in (n, d):
        if isinstance(arg, type(1j)):
            raise TypeError("cannot convert arguments")
    if isinstance(n, _Rational):
        return rational(n.n, n.d * d)
    if isinstance(d, _Rational):
        return rational(n * d.d, d.n)
    return _Rational(n, d)


import builtins

builtins.rational = rational

# =============================================================================
#
# Arthmetic coder, taken from:
#
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/306626
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  A very slow arithmetic coder for Python.
#
#  "Rationals explode quickly in term of space and ... time."
#              -- comment in Rational.py (probably Tim Peters)
#
# Really.  Don't use this for real work.  Read Mark Nelson's
# Dr. Dobb's article on the topic at
#    http://dogma.net/markn/articles/arith/part1.htm
# It's readable, informative and even includes clean sample code.
#
# Contributed to the public domain
# Andrew Dalke < dalke @ dalke scientific . com >


def arith_coder_train(text):
    """text -> 0-order probability statistics as a dictionary

    Text must not contain the NUL (0x00) character because that's
    used to indicate the end of data.
    """
    assert "\x00" not in text
    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    counts["\x00"] = 1
    tot_letters = sum(counts.values())

    tot = 0
    probs = {}
    prev = rational(0)
    for c, count in list(counts.items()):
        next = rational(tot + count, tot_letters)
        probs[c] = (prev, next)
        prev = next
        tot = tot + count
    assert tot == tot_letters

    return probs


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def arith_coder_encode(text, probs):
    """text and the 0-order probability statistics -> longval, nbits

    The encoded number is rational(longval, 2**nbits)
    """

    minval = rational(0)
    maxval = rational(1)
    for c in text + "\x00":
        prob_range = probs[c]
        delta = maxval - minval
        maxval = minval + prob_range[1] * delta
        minval = minval + prob_range[0] * delta

    # I tried without the /2 just to check.  Doesn't work.
    # Keep scaling up until the error range is >= 1.  That
    # gives me the minimum number of bits needed to resolve
    # down to the end-of-data character.
    delta = (maxval - minval) / 2
    nbits = 0
    while delta < 1:
        nbits = nbits + 1
        delta = delta << 1
    if nbits == 0:
        # return 0, 0
        return 0  # Only number of bits needed
    else:
        avg = (maxval + minval) << (nbits - 1)  # using -1 instead of /2
    # Could return a rational instead ...

    # return avg.n//avg.d, nbits  # the division truncation is deliberate
    return nbits  # Only number of bits needed


# =============================================================================
