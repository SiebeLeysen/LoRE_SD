import numpy as np
import argparse

def read_array_from_file(file_path):
    return np.loadtxt(file_path)

def modify_first_column(array):
    array[0] *= -1
    # array[0], array[1] = array[1], array[0]
    return array

def write_array_to_file(array, file_path):
    np.savetxt(file_path, array, fmt='%f')

def main():
    parser = argparse.ArgumentParser(description='Process a 2D array from a file.')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')

    args = parser.parse_args()

    # Read the array from the input file
    array = read_array_from_file(args.input_file)

    # Modify the first column
    modified_array = modify_first_column(array)

    # Write the modified array to the output file
    write_array_to_file(modified_array, args.output_file)

    print("Modified array has been written to", args.output_file)

if __name__ == "__main__":
    main()
