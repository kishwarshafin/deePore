import pysam
import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

'''
    Assumes all bam files in specified nanopore_prefix directory
    are chromosomes in the NA12878 nanopore consortium.

    Returns full paths to all files in the directory that end in .bam
'''
def retrieve_nanopore_bams(nanopore_prefix):
    return [x for x in os.listdir(nanopore_prefix) if '.bam' in x[-4:]]

'''
    Simple conversion function to turn counts in a histogram of 
    counts into percentages
'''
def count_to_distribution(nparray):
    total_count = float(sum(nparray))
    return nparray / total_count

'''
    Gets the histogram from a list of coverages according to 
    a defined set of bins
'''
def coverage_to_histogram(coverage_list, bins):
    temp_hist = np.histogram(coverage_list, bins=bins)
    return np.array(temp_hist[0])

'''
    Find the first and last read from a bam file.

    Set the START index to the first read, check every 1000 positions.

    Find the END index by checking every millionth position, 
    and then iterating after the last nonzero read by 1000's
    to find the last read.
'''
def find_boundaries(data_prefix, filename, chr_label):
    current_position = 0
    coverages = []
    bamfile = pysam.AlignmentFile(data_prefix + filename, "rb")

    while 1:
        if bamfile.count(chr_label, current_position, current_position+1) > 0:
            coverages.append(current_position)
            break
        current_position += 1000
    START = coverages[0]

    # Add nonzero reads to coverages, break once we've found
    # 100 consecutive reads
    consecutive = 0
    while consecutive < 100:
        if bamfile.count(chr_label, current_position, current_position+1) > 0:
            coverages.append(current_position)
            consecutive = 0
        else:
            consecutive += 1
        current_position += 1000000

    end = coverages[-1:][0]
    post = end + 1000000
    for current_position in range(end, post, 1000):
        if bamfile.count(chr_label, current_position, current_position+1) > 0:
            coverages.append(current_position)

    return START, coverages[-1:][0]

'''
    Dump the zero_positions list to a file
'''
def output_zeroes(seq_type, chr_label, zero_positions):
    with open("{}.{}.zeros.txt".format(seq_type, chr_label), 'w') as zero_file:
        for entry in zero_positions:
            zero_file.write('{}\n'.format(entry))
        print("Dumping {}.{}.zeros.txt".format(seq_type, chr_label))
        zero_file.flush()

'''
    Process for reading coverages from each chromosome in 
    each bamfile, and producing the visualizations. 
'''
def bam_management(START, END, STEPS, data_prefix, bamfilename, chr_label, outfile):
    coverage_list  = []
    hist_list      = []
    zero_positions = []

    bins = range(0,105,5)
    running_total = np.zeros(len(bins)-1)

    # Get reads from START to END in BAMFILE
    bamfile = pysam.AlignmentFile(data_prefix + bamfilename, "rb")

    # Print coverage for every STEPS
    for i in range(START, END, STEPS):
        value = bamfile.count(chr_label, i, i+1)

        # Debugging zero length problems
        if value == 0:
            zero_positions.append(i)

        # Coverage Distribution
        coverage_list.append(value)
        if len(coverage_list) > 1000:
            running_total += coverage_to_histogram(coverage_list, bins)
            coverage_list = []
            outfile.write('{}\n'.format(i))
            outfile.flush()

    # Pick up any stragglers
    if len(coverage_list) > 0:
        running_total += coverage_to_histogram(coverage_list, bins)

    bamfile.close()

    # change running_total to distribution
    distribution = count_to_distribution(running_total)
    num_bins = int((END - START) / STEPS)
    seq_type = "Illumina" if chr_label not in bamfilename else "Nanopore"
    outfile.write("Finished reading {}, now writing distribution.\n".format(chr_label))
    outfile.flush()

    # Dump zero positions
    output_zeroes(seq_type, chr_label, zero_positions)
    plot_distribution(distribution, seq_type, chr_label)

def plot_distribution(distribution, seq_type, chr_label):
    y_pos = np.arange(20)*5
    plt.clf()
    plt.bar(y_pos, distribution, align='center', alpha=1.0, facecolor='black')
    plt.xticks(y_pos, y_pos)
    plt.ylabel('Frequency')
    plt.xlabel('Coverage')
    plt.title('{} {} Coverage Distribution'.format(seq_type, chr_label))
    print("Printing {}.{}.coverage_percentage_distribution.png".format(seq_type, chr_label))
    plt.savefig('{}.{}.coverage_percentage_distribution.png'.format(seq_type, chr_label))

if __name__=="__main__":
    # Illumina NA12878
    #test_prefix = "/data/users/ryan/"
    #test_file   = "chr3.bam"
    #test_label  = "chr3"

    # Testing 
    #START, END = find_boundaries(test_prefix, test_file, test_label)
    #bam_management(START, END, STEPS, test_prefix, test_file, test_label)

    # Nanopore chr3
    nanopore_prefix = '/data/users/common/nanopore/'
    illumina_prefix = '/data/users/ryan/'
    illumina_file   = 'NA12878_S1.bam'
    STEPS = 1000

    # Nanopore Processing
    for chr_file in retrieve_nanopore_bams(nanopore_prefix):
        chr_label = chr_file.split('.')[0]
        #if chr_label in ["chr1", "chr3", "chr6", "chr9", "chr11", "chr12", "chr13", "chr14", "chr16", "chr18", "chr19"]:
        #    continue
        outfile = open("output.nanopore.{}.txt".format(chr_label), 'w')
        print("Printing to output.nanopore.{}.txt".format(chr_label))
        sys.stdout.flush()
        outfile.write("Starting nanopore validation for {}\n".format(chr_label))
        START, END = find_boundaries(nanopore_prefix, chr_file, chr_label)
        outfile.write("Bounds: {} {}\n".format(START, END))
        outfile.flush()
        bam_management(START, END, STEPS, nanopore_prefix, chr_file, chr_label, outfile)
        outfile.close()

    # Illumina Processing
    for i in range(1, 20):
        chr_label = "chr" + str(i)
        outfile = open("output.illumina.{}.txt".format(chr_label), 'w')
        print("Printing to output.illumina.{}.txt".format(chr_label))
        sys.stdout.flush()
        outfile.write("Starting illumina validation for {}\n".format(chr_label))
        START, END = find_boundaries(illumina_prefix, illumina_file, chr_label)
        outfile.write("Bounds: {} {}\n".format(START, END))
        outfile.flush()
        bam_management(START, END, STEPS, illumina_prefix, illumina_file, chr_label, outfile)
        outfile.close()