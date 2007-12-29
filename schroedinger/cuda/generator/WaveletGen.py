#!/usr/bin/python
from Horizontal import *
from Vertical import *
from Preamble import *
from wl import *
import w5_3_s, w5_3_a, w9_3_s, w9_3_a, w13_5_a, w13_5_s, wfidelity_a, wfidelity_s, w9_7_a, w9_7_s

import sys
path = sys.argv[1]

def gen(path, wl):
    rv = open(path+"iiwt"+wl+".cu", "w")
    rv.write(preamble)
    rv.write(s_params[wl])
    rv.write(s_transform_h_begin_unroll)
    rv.write(s_transform_h[wl])
    rv.write(s_transform_h_end_unroll2)
    rv.write(param_v)
    rv.write(s_transform_v[wl])
    rv.write(transform_v_unroll % {"dir":"s"})
    rv.write(function_iiwt % wl)

    rv = open(path+"iwt"+wl+".cu", "w")
    rv.write(preamble)
    rv.write(a_params[wl])
    rv.write(a_transform_h_begin)
    rv.write(a_transform_h[wl])
    rv.write(a_transform_h_end)
    rv.write(param_v)
    rv.write(a_transform_v[wl])
    rv.write(transform_v % {"dir":"a"})
    rv.write(function_iwt % wl)

gen(path, "5_3")
gen(path, "9_3")
gen(path, "13_5")
gen(path, "fidelity")
gen(path, "9_7")
