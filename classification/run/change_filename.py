# read all teh file in a direcrtory an change anmes
import os
import sys
sys.path.append(os.getcwd())
from src.utils.formatting import hash_name
import shutil
path = sys.argv[1]
files = os.listdir(path)


# filter files with only pcsd and alpha values to get only names and processedd files
files = [f for f in files if '.pcsd' in f]
files = [f for f in files if 'alpha' in f]

# convert names to hash
# # COMMENTED for SAFE GUARD
# for f in files:
#     name = f[:-5]
#     hashname = hash_name(name)
#     hashname = hashname + '.pcsd'
#     # copy file content and re
#     shutil.move(path + f , path + hashname)
