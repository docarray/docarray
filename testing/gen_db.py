# standard imports
import argparse
import os
import sys

# install packages
import numpy



def parse_cmd_line():
    '''Parse command line arguments.'''

    parser = argparse.ArgumentParser(
                    prog = sys.argv[0],
                    description = 'A cmd line tool for vector search evaluation')
    parser.add_argument('-d', '--dimensions',   type=int,   required=True )
    parser.add_argument('-r', '--records',      type=int,   required=True )
    parser.add_argument('-f', '--file',                     required=True )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''Main entry point of this program.'''

    args = parse_cmd_line()

    filename = args.file if args.file.endswith(".npy") else args.file + ".npy"
    if os.path.exists(filename):
        raise Exception("This output file (%s) already exists" % filename)

    print("Create a floating point array of size (%d, %d)..." % (args.records, args.dimensions) )
    rand_array = numpy.random.rand( args.records, args.dimensions )

    print("Saving array to npy file...")
    numpy.save(filename, rand_array)
    print("Wrote %s" % filename)
 
    
