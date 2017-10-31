import pysam
from pyfaidx import Fasta
from collections import defaultdict
from datetime import datetime
from matplotlib import cm
from matplotlib import image


class Pileup:
    '''
    A child of PileupGenerator which contains all the information and methods to produce a pileup at a given
    position, using the pysam and pyfaidx object provided by PileupGenerator
    '''

    def __init__(self,sam,fasta,chromosome,queryStart,regionLength):
        self.queryStart = queryStart
        self.queryEnd = queryStart + regionLength
        self.chromosome = chromosome
        self.length = regionLength

        # pysam uses 0-based coordinates
        self.localReads = sam.fetch("chr"+self.chromosome, start=self.queryStart, end=self.queryEnd)

        # pyfaidx uses 1-based coordinates
        self.coverage = sam.count("chr"+self.chromosome, start=self.queryStart, end=self.queryEnd)
        self.refSequence = fasta.get_seq(name="chr"+self.chromosome, start=self.queryStart+1, end=self.queryEnd+10)
        self.referenceBMP = list()

        # stored during cigar string parsing to save time
        self.inserts = defaultdict(list)

        # initialize array assuming length adjusted for insertion is < 2*reference length...
        self.pileupBMP = [[0 for i in range(self.length*3*2)] for j in range(self.coverage)]

        self.deltaRef = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # key for whether reference advances
        self.deltaRead = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # key for whether read sequence advances
        # ['M','I','D','N','S','H','P','=','X','B','NM']

        self.deltaRefRelative = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        self.noneChar = '_'     # character to use for empty positions in the pileup

        self.SNPtoBMP = {'M': [1, 1, 1],
                         'A': [0, 0, 1],
                         'C': [0, 1, 0],
                         'G': [0, 1, 1],
                         'T': [1, 0, 0],
                         'D': [1, 0, 1],
                         self.noneChar: [0, 0, 0], }

        self.BMPtoSNP = [[[self.noneChar, 'A'], ['C', 'G']], [['T', 'D'], [None, 'M']]]

        self.cigarLegend = ['M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X', 'B', '?']  # ?=NM

        self.cigarTuples = None
        self.refStart = None
        self.refEnd = None
        self.refPositions = None
        self.refPosition = None
        self.readSequence = None
        self.readPosition = None
        self.relativeIndex = None
        self.relativeIndexRef = None


    def generateBMPtoSNP(self):
        '''
        Generate inverse of SNPtoBMP automatically, from any SNPtoBMP dictionary
        :return:
        '''

        self.BMPtoSNP = [[[None for i in range(2)] for j in range(2)] for k in range(2)]

        for item in self.SNPtoBMP.items():
            i1,i2,i3 = item[1]
            self.BMPtoSNP[i1][i2][i3] = item[0]


    def iterateReads(self):
        '''
        For all the reads mapped to the region specified, collect their alignment data and build a draft pileup
        '''

        for r, read in enumerate(self.localReads):
            refPositions = read.get_reference_positions()
            self.refStart = refPositions[0]
            self.refEnd = refPositions[-1]
            self.cigarTuples = read.cigartuples

            refPositions = None  # dump

            self.readSequence = read.query_alignment_sequence

            self.refPosition = 0    #
            self.readPosition = -1  #
            self.relativeIndex = self.refStart-self.queryStart
            self.relativeIndexRef = 0

            for c, entry in enumerate(self.cigarTuples):
                snp = entry[0]
                n = entry[1]

                self.absolutePosition = self.refPosition+self.refStart

                if self.absolutePosition < self.queryStart-n:       # skip by blocks if not in query region yet
                    self.refPosition += self.deltaRef[snp]*n
                    self.readPosition += self.deltaRead[snp]*n
                    self.relativeIndex += self.deltaRef[snp]*n

                elif self.absolutePosition >= self.queryStart-n:    # read one position at a time for query region
                    for i in range(entry[1]):                       # this should be switched to slicing for speed
                        self.absolutePosition = self.refPosition+self.refStart
                        self.refPosition += self.deltaRef[snp]
                        self.readPosition += self.deltaRead[snp]

                        if self.absolutePosition >= self.queryStart and self.absolutePosition < self.queryEnd:

                            self.addSNPtoPileup(r,snp,n,i)
                            self.relativeIndexRef += self.deltaRefRelative[snp]

                            if snp == 4:
                                break

                        self.relativeIndex += self.deltaRef[snp]

                elif self.absolutePosition > self.queryEnd:  # stop iterating after query region
                    break

    def addSNPtoPileup(self,r,snp,n,i):
        '''
        For a given read and SNP, add the corresponding encoding to the pileup array
        :param r:
        :param snp:
        '''

        index = self.relativeIndex*3
        encoding = None

        if snp == 0:                                            # match
            nt = self.readSequence[self.readPosition]
            ntRef = self.refSequence[self.relativeIndexRef]

            if nt != ntRef:
                encoding = self.SNPtoBMP[nt]
            else:
                encoding = self.SNPtoBMP['M']

        elif snp == 1:                                          # insert
            nt = self.readSequence[self.readPosition]

            encoding = self.SNPtoBMP[nt]

            if i == 0:  # save the insert position and length upon encountering it
                position = self.absolutePosition-self.queryStart
                self.inserts[position].append([r, n])

            self.relativeIndex += 1

        elif snp == 2:                                          # delete
            encoding = self.SNPtoBMP['D']

        elif snp == 3:                                          # soft clip
            encoding = self.SNPtoBMP['']

        if snp < 4:
            self.pileupBMP[r][index] = encoding[0]
            self.pileupBMP[r][index+1] = encoding[1]
            self.pileupBMP[r][index+2] = encoding[2]
        else:                                                   # something else?
            print("WARNING: unencoded SNP: ", self.cigarLegend[snp]," at position ",self.relativeIndex)

    def reconcileInserts(self):
        '''
        For all recorded insertion indexes in the pileup, ensure reads across the column have uniform length,
         despite differences in insertion lengths among reads.
        '''

        offsets = [0 for n in range(self.coverage)]

        for p in sorted(self.inserts.keys()):
            i = 0

            # sort this position's inserts by length and select longest
            longestInsert = sorted(self.inserts[p], key=lambda x: x[1], reverse=True)[0]
            n = longestInsert[1]    #length of the insert

            for r in range(self.coverage):
                if r not in [insert[0] for insert in self.inserts[p]]:
                    # update the position based on previous inserts
                    pAdjusted = p+offsets[r]
                    pAdjusted *= 3

                    # add the insert to the pileup
                    self.pileupBMP[r] = self.pileupBMP[r][:pAdjusted]+self.SNPtoBMP[self.noneChar]*n+self.pileupBMP[r][pAdjusted:]

                    if i == 0:
                        # add the insert to the reference sequence
                        self.refSequence = self.refSequence[:int(pAdjusted/3)]+self.noneChar*n+self.refSequence[int(pAdjusted/3):]
                        self.length += n
                    i += 1

                else:
                    # find this read's insert length
                    insertLength = [insert[1] for insert in self.inserts[p] if insert[0] == r][0]

                    if insertLength < n:  # if this read has an insert, but of shorter length than the max
                        pAdjusted = p+offsets[r]+insertLength  # start adding insert chars after this insertion
                        nAdjusted = n-insertLength  # number of inserts to add

                        pAdjusted *= 3

                        self.pileupBMP[r] = self.pileupBMP[r][:pAdjusted]+self.SNPtoBMP[self.noneChar]*nAdjusted+self.pileupBMP[r][pAdjusted:]

                offsets[r] += n     # <-- the magic happens here

    def encodeReference(self):
        '''
        Encode the reference sequence into BMP triplets and add add it as the header line to the pileup BMP
        :return:
        '''
        for character in self.refSequence:
            self.referenceBMP += self.SNPtoBMP[character]

        self.pileupBMP = [self.referenceBMP] + self.pileupBMP

    def savePileupBMP(self,filename):
        '''
        Save the pileup binary array as a bitmap image using a gray color map
        :param filename:
        :return:
        '''

        if not filename.endswith(".bmp"):
            filename += ".bmp"

        self.pileupBMP = [row[:self.length*3] for row in self.pileupBMP]

        image.imsave(filename,self.pileupBMP,cmap=cm.gray)


    def decodeBMP(self,filename):
        '''
        Read a BMP and convert to a text alignment
        :param filename:
        :return:
        '''

        bmp = image.imread(filename)
        text = ''

        n = int(len(bmp[0])/3)  # length of the decoded sequence

        for row in bmp:
            for i in range(0,n):
                rgb1,rgb2,rgb3 = row[i*3:(i*3)+3]   # one triplet is one character in pileup sequence

                if rgb1[:3].all(0):
                    i1 = 1
                else:
                    i1 = 0

                if rgb2[:3].all(0):
                    i2 = 1
                else:
                    i2 = 0

                if rgb3[:3].all(0):
                    i3 = 1
                else:
                    i3 = 0

                text += self.BMPtoSNP[i1][i2][i3]
            text += '\n'

        return text



class PileUpGenerator:
    '''
    Creates pileups of aligned reads given a SAM/BAM alignment and FASTA reference
    '''

    def __init__(self,alignmentFile,referenceFile):
        self.sam = pysam.AlignmentFile(alignmentFile, "rb")
        self.fasta = Fasta(referenceFile, as_raw=True, sequence_always_upper=True)

    def generatePileup(self, chromosome, position, flankLength, outputFileName):
        '''
        Generate a pileup at a given position
        :param queryStart:
        :param regionLength:
        :return:
        '''

        outputFileName = outputFileName + ".bmp"
        queryStart = position-flankLength
        regionLength = flankLength*2+1

        chromosome = str(chromosome)

        pileup = Pileup(self.sam,self.fasta,chromosome,queryStart,regionLength)
        pileup.iterateReads()
        pileup.reconcileInserts()
        pileup.encodeReference()
        pileup.savePileupBMP(outputFileName)

        print(pileup.decodeBMP(outputFileName))



# bamFile = "NA12878.np.chr3.100kb.0.bam"
# fastaFile = "hg19.chr3.9mb.fa.txt"

# piler = PileUpGenerator(bamFile,fastaFile)
# piler.generatePileup(chromosome=3,position=79005,flankLength=25)
# piler.generatePileup(chromosome=3,position=99999+25,flankLength=25)
