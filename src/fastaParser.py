from pysam import VariantFile
import argparse

def generatePileupBasedOnVCF(fastaFile):
    fasta_file = open(fastaFile, "r")
    for line in fasta_file:
        line = line.rstrip()
        if not line:
            continue
        if line[0] == '>':
            lineList = line[1:].split(' ')
            lineList[0] = 'chr' + lineList[0]
            print('>'+' '.join(lineList))
        else:
            print(line)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--fasta",
        type=str,
        required = True,
        help="Fasta file."
    )
    FLAGS, unparsed = parser.parse_known_args()
    generatePileupBasedOnVCF(FLAGS.fasta)