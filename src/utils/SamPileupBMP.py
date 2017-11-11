import pysam
from pyfaidx import Fasta
from collections import defaultdict
from datetime import datetime
from PIL import Image
from math import floor
import sys


class Pileup:
    '''
    A child of PileupGenerator which contains all the information and methods to produce a pileup at a given
    position, using the pysam and pyfaidx object provided by PileupGenerator
    '''

    def __init__(self,sam,fasta,chromosome,queryStart,flankLength,outputFilename,label,variantLengths,coverageCutoff,mapQualityCutoff,subsampleRate=0,forceCoverage=False,arrayInitializationFactor=2):
        self.length = flankLength*2+1
        self.label = label
        self.variantLengths = variantLengths
        self.subsampleRate = subsampleRate
        self.forceCoverage = forceCoverage
        self.coverageCutoff = coverageCutoff
        self.outputFilename = outputFilename
        self.queryStart = queryStart
        self.queryEnd = queryStart + self.length
        self.chromosome = chromosome
        self.mapQualityCutoff = mapQualityCutoff

        # pysam uses 0-based coordinates
        self.localReads = sam.fetch("chr"+self.chromosome, start=self.queryStart, end=self.queryEnd)

        # pyfaidx uses 1-based coordinates
        self.coverage = sam.count("chr"+self.chromosome, start=self.queryStart, end=self.queryEnd)
        self.singleCoverage = sam.count("chr"+self.chromosome, start=self.queryStart+flankLength, end=self.queryStart+flankLength+1)
        self.refSequence = fasta.get_seq(name="chr"+self.chromosome, start=self.queryStart+1, end=self.queryEnd+1)
        self.referenceBMP = list()

        # stored during cigar string parsing to save time
        self.inserts = defaultdict(list)


        self.deltaRef  = [1,0,1,1,0,0,0,0,0,0,0]    # key for whether reference advances
        self.deltaRead = [1,1,0,0,0,0,0,0,0,0,0]    # key for whether read sequence advances
                                                    #  ['M','I','D','N','S','H','P','=','X','B','NM']

        self.deltaRefRelative  = [1,0,1,0,0,0,0,0,0,0,0]    # key for whether reference advances (for match comparison within query region)

        self.noneChar = '_'     # character to use for empty positions in the pileup
        self.noneLabel = '3'      # character to use for (non variant called) inserts in the label

        self.SNPtoBMP = {'M': (255,255,255),
                         'A': (255,0,  0),
                         'C': (255,255,0),
                         'G': (0,  255,0),
                         'T': (0,  0,  255),
                         'D': (0,  255,255),
                         'N': (0,  0,  0),  # redundant in case of read containing 'N'... should this be independent?
               self.noneChar: (0,  0,  0),}

        # initialize array assuming length adjusted for insertion is < 2*reference length...
        self.pileupBMP = [[self.SNPtoBMP[self.noneChar] for i in range(self.length*2)] for j in range(self.coverageCutoff)]


        self.BMPtoSNP = [[[' ', 'A'], ['C', 'G']], [['T', 'D'], [None, '.']]]

        self.cigarLegend = ['M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X', 'B', '?']  # ?=NM

        # self.three = range(3)   # used repeatedly, and in the inner loop

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
        For all the reads mapped to the region specified, (subsample and) collect their alignment data and build a draft
        pileup
        '''
        index_iterator = 0
        for r, read in enumerate(self.localReads):
            if r == self.coverageCutoff:
                break
            if read.mapping_quality < self.mapQualityCutoff:
                continue
            self.parseRead(index_iterator, read)
            index_iterator += 1


    def parseRead(self,r,read):
        '''
        Create a draft pileup representation of a read using cigar string and read data. This method iterates through
        a read and its cigar string from the start (so it needs to be optimized for long reads).
        :param r:
        :param read:
        :return:
        '''

        refPositions = read.get_reference_positions()

        if len(refPositions) == 0:
            sys.stderr.write("WARNING: read contains no reference alignments: %s\n" % read.query_name)
        else:
            self.refStart = refPositions[0]
            self.refEnd = refPositions[-1]
            self.cigarTuples = read.cigartuples

            refPositions = None  # dump

            self.readSequence = read.query_alignment_sequence

            self.refPosition = 0  #
            self.readPosition = -1  #
            self.relativeIndex = self.refStart-self.queryStart
            self.relativeIndexRef = 0

            if self.refStart > self.queryStart:                         # if the read starts during the query region
                self.relativeIndexRef += self.refStart-self.queryStart  # add the difference to ref index

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
                            self.addSNPtoPileup(r, snp, n, i)

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

        index = self.relativeIndex
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

            if i == 0:      # record the insert for later
                position = self.absolutePosition-self.queryStart
                self.inserts[position].append([r, n])

            self.relativeIndex += 1

        elif snp == 2:                                          # delete
            encoding = self.SNPtoBMP['D']

        elif snp == 3:                                          # refskip (?)
            encoding = self.SNPtoBMP['N']

        else:                                                   # anything else
            sys.stderr.write("WARNING: unencoded SNP: %s at position %d\n" % (self.cigarLegend[snp],self.relativeIndex))

        if snp < 4:
            self.pileupBMP[r][index] = encoding


    def reconcileInserts(self):
        '''
        For all recorded insertion indexes in the pileup, ensure reads across the column have uniform length,
         despite differences in insertion lengths among reads.
        '''

        offsets = [0 for n in range(self.coverageCutoff)]

        for p in sorted(self.inserts.keys()):
            i = 0

            # sort this position's inserts by length and select longest
            longestInsert = sorted(self.inserts[p], key=lambda x: x[1], reverse=True)[0]
            n = longestInsert[1]    #length of the insert

            for r in range(self.coverageCutoff):
                if r not in [insert[0] for insert in self.inserts[p]]:
                    # update the position based on previous inserts
                    pAdjusted = p+offsets[r]

                    print(r,len(self.pileupBMP))
                    print(pAdjusted,len(self.pileupBMP))

                    # add the insert to the pileup
                    # for ridx in self.three:
                    self.pileupBMP[r] = self.pileupBMP[r][:pAdjusted]+[self.SNPtoBMP[self.noneChar]]*n+self.pileupBMP[r][pAdjusted:]


                    if i == 0:
                        # add the insert to the reference sequence and the label string
                        self.refSequence = self.refSequence[:int(pAdjusted)]+self.noneChar*n+self.refSequence[int(pAdjusted):]

                        pVariant = p-1
                        if pVariant in self.variantLengths:                    # if the position is a variant site
                            l = self.variantLengths[pVariant] -1  # -1 for ref
                            if l >= n:                                  # correctly modify the label to fit the insert
                                labelInsert = self.label[pAdjusted-1]*n     # using the length of the called variant at pos.
                            else:
                                labelInsert = self.label[pAdjusted-1]*l + self.noneLabel*(n-l)

                        else:
                            labelInsert = self.noneLabel*n              # otherwise the insert is labeled with None code

                        self.label = self.label[:int(pAdjusted)]+labelInsert+self.label[int(pAdjusted):]
                        self.length += n
                    i += 1

                else:
                    # find this read's insert length
                    insertLength = [insert[1] for insert in self.inserts[p] if insert[0] == r][0]

                    if insertLength < n:    # if this read has an insert, but of shorter length than the max
                        pAdjusted = p+offsets[r]+insertLength  # start adding insert chars after this insertion
                        nAdjusted = n-insertLength             # number of inserts to add

                        self.pileupBMP[r] = self.pileupBMP[r][:pAdjusted]+[self.SNPtoBMP[self.noneChar]]*nAdjusted+self.pileupBMP[r][pAdjusted:]

                offsets[r] += n     # <-- the magic happens here


    def encodeReference(self):
        '''
        Encode the reference sequence into BMP triplets and add add it as the header line to the pileup BMP
        :return:
        '''

        for character in self.refSequence:
            triplet = self.SNPtoBMP[character]
            self.referenceBMP.append(triplet)

        self.pileupBMP = [self.referenceBMP] + self.pileupBMP


    def savePileupBMP(self,filename):
        '''
        Save the pileup binary array as a bitmap image using a gray color map
        :param filename:
        :return:
        '''


        # for row in self.pileupBMP:
        #     print(row)

        if not filename.endswith(".png"):
            filename += ".png"

        self.pileupBMP = self.pileupBMP[:self.coverageCutoff]
        self.pileupBMP = [row[:self.length] for row in self.pileupBMP]


        image = Image.new("RGB",(self.length,self.coverageCutoff))
        pixels = image.load()

        for i in range(image.size[0]):
            for j in range(image.size[1]):
                pixels[i, j] = self.pileupBMP[j][i]

        image.save(filename,"PNG")


    def getOutputLabel(self):
        return self.label


    def decodeBMP(self,filename):   # fix function for RGB !!
        '''
        Read a BMP and convert to a text alignment
        :param filename:
        :return:
        '''

        bmp = image.imread(filename)
        text = ''

        width = int(len(bmp[0]))  # length of the decoded sequence
        height = int(len(bmp)/3)

        for h in range(height):
            for w in range(width):
                rgb1,rgb2,rgb3 = [bmp[3*h+i][w] for i in range(0,3)]   # one vertical triplet is one character in pileup

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
        self.sam = pysam.AlignmentFile(alignmentFile,"rb")
        self.fasta = Fasta(referenceFile,as_raw=True,sequence_always_upper=True)


    def generatePileup(self,chromosome,position,flankLength,outputFilename,label,variantLengths,forceCoverage=False,coverageCutoff=100,mapQualityCutoff=30):
        '''
        Generate a pileup at a given position
        :param queryStart:
        :param regionLength:
        :return:
        '''

        queryStart = position-flankLength

        chromosome = str(chromosome)

        # startTime = datetime.now()
        pileup = Pileup(self.sam,self.fasta,chromosome,queryStart,flankLength,outputFilename,label,variantLengths,forceCoverage=forceCoverage,coverageCutoff=coverageCutoff,mapQualityCutoff=mapQualityCutoff)
        # print(datetime.now() - startTime, "initialized")
        pileup.iterateReads()
        # print(datetime.now() - startTime, "drafted")
        pileup.reconcileInserts()
        # print(datetime.now() - startTime, "finalized")
        pileup.encodeReference()
        pileup.savePileupBMP(outputFilename)
        # print(datetime.now() - startTime, "encoded and saved")

        # print(pileup.label)
        # print(pileup.decodeBMP(outputFilename + ".bmp"))

        return pileup.getOutputLabel()

#
# bamFile = "deePore/data/chr3_200k.bam"
# fastaFile = "deePore/data/chr3.fa"
#
# vcf_region = "3"
# window_size = 25
# filename = "deePore/data/training_chr3_350/"
# labelString = '0'*51
# variantLengths = dict()
# position = 111945
#
# piler = PileUpGenerator(bamFile,fastaFile)
# outputLabelString = piler.generatePileup(chromosome=vcf_region, position=position, flankLength=window_size,
#                                      outputFilename=filename, label=labelString, variantLengths=variantLengths,forceCoverage=True,coverageCutoff=50)

