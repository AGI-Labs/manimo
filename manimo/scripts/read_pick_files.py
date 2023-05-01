import pickle as p

fname = "/home/sudeep/Documents/manimo/manimo/scripts/demos/pick_nsh_220_demos.pkl"

with open(fname, "rb") as f:
    data = p.load(f)
    print(data)