import pysam
from pyfaidx import Fasta
from collections import defaultdict
from datetime import datetime
from PIL import Image
from math import floor
import sys
import numpy
import copy
"""
This  module takes an alignment file and produces a pileup across all alignments in a query region, and encodes the
pileup as an image where each x pixel corresponds to a position and y corresponds to coverage depth
"""

class Pileup:
    '''
    A child of PileupGenerator which contains all the information and methods to produce a pileup at a given
    position, using the pysam and pyfaidx object provided by PileupGenerator
    '''

    def __init__(self,sam,fasta,chromosome,queryStart,flankLength,outputFilename,label,variantLengths,coverageCutoff,mapQualityCutoff,windowCutoff,sortColumns,subsampleRate=0,forceCoverage=False,arrayInitializationFactor=2):
        self.length = flankLength*2+1
        self.label = label
        self.variantLengths = variantLengths
        self.subsampleRate = subsampleRate
        self.coverageCutoff = coverageCutoff
        self.outputFilename = outputFilename
        self.queryStart = queryStart
        self.queryEnd = queryStart + self.length
        self.chromosome = chromosome
        self.mapQualityCutoff = mapQualityCutoff
        self.windowCutoff = windowCutoff
        self.sortColumns = sortColumns

        # pysam uses 0-based coordinates
        self.localReads = sam.fetch("chr"+self.chromosome, start=self.queryStart, end=self.queryEnd)

        # pyfaidx uses 1-based coordinates
        self.coverage = sam.count("chr"+self.chromosome, start=self.queryStart, end=self.queryEnd)
        self.singleCoverage = sam.count("chr"+self.chromosome, start=self.queryStart+flankLength, end=self.queryStart+flankLength+1)
        self.refSequence = fasta.get_seq(name="chr"+self.chromosome, start=self.queryStart+1, end=self.queryEnd)
        self.referenceRGB = list()

        # stored during cigar string parsing to save time
        self.inserts = defaultdict(list)

        self.deltaRef  = [1,0,1,1,0,0,0,0,0,0,0]    # key for whether reference advances
        self.deltaRead = [1,1,0,0,0,0,0,0,0,0,0]    # key for whether read sequence advances
                                                    #  ['M','I','D','N','S','H','P','=','X','B','NM']

        self.deltaRefRelative  = [1,0,1,0,0,0,0,0,0,0,0]    # key for whether reference advances (for match comparison within query region)

        self.noneChar = '_'       # character to use for empty positions in the pileup
        self.noneLabel = '0'      # character to use for (non variant called) inserts in the label

        self.SNPtoRGB = {'M': (255,255,255),
                         'A': (255,0,  0),
                         'C': (255,255,0),
                         'G': (0,  255,0),
                         'T': (0,  0,  255),
                         'I': (255,0,  255),
                         'D': (0,  255,255),
                         'N': (0,  0,  0),  # redundant in case of read containing 'N'... should this be independent?
               self.noneChar: (0,  0,  0),}

        self.sortingKey = {'M':5,
                           'A':0,
                           'C':1,
                           'G':2,
                           'T':3,
                           'I':6,
                           'D':4,
                           'N':7}

        self.RGBtoSNP = [[['N', 'T'], ['G', 'D']], [['A', 'I'], ['C', 'M']]]

        self.RGBtoSNPtext = [[[' ', 'T'], ['G', 'D']], [['A', '_'], ['C', '.']]]

        # initialize array using windowCutoff as the initial width
        self.pileupRGB = [[self.SNPtoRGB[self.noneChar] for i in range(self.windowCutoff)] for j in range(self.coverage)]

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
        self.readPileupStarted = False
        self.insertOffsets = dict()
        self.readMap = dict()           # a mapping of the reads' vertical index after repacking
        self.readEnds = dict()          # track where reads end for packing purposes
        self.breakpoints = defaultdict(list)    # don't even ask


    def generateRGBtoSNP(self):
        '''
        Generate inverse of SNPtoRGB automatically, from any SNPtoRGB dictionary
        :return:
        '''

        self.RGBtoSNP = [[[None for i in range(2)] for j in range(2)] for k in range(2)]

        for item in self.SNPtoRGB.items():
            bits = [int(i/255) for i in item[1]]

            i1,i2,i3 = bits
            self.RGBtoSNP[i1][i2][i3] = item[0]


    def iterateReads(self):
        '''
        For all the reads mapped to the region specified, (subsample and) collect their alignment data and build a draft
        pileup
        '''

        pileupIteratorIndex = 0
        for r, read in enumerate(self.localReads):
            if read.mapping_quality < self.mapQualityCutoff:
                continue
            self.parseRead(pileupIteratorIndex, read)
            pileupIteratorIndex += 1


    def parseRead(self,r,read):
        '''
        Create a draft pileup representation of a read using cigar string and read data. This method iterates through
        a read and its cigar string from the start (so it needs to be optimized for long reads).
        :param r:
        :param read:
        :return:
        '''

        refPositions = read.get_reference_positions()
        readQualities = read.query_qualities
        mapQuality = read.mapping_quality
        firstPass = True

        if len(refPositions) == 0:
            sys.stderr.write("WARNING: read contains no reference alignments: %s\n" % read.query_name)
        else:
            self.refStart = refPositions[0]
            self.refEnd = refPositions[-1]
            self.cigarTuples = read.cigartuples

            refPositions = None  # dump

            self.readSequence = read.query_alignment_sequence

            self.refPosition = 0
            self.readPosition = -1
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
                            if firstPass and snp == 0:
                                self.generateReadMapping(r,self.relativeIndex)
                                firstPass = False

                            if not firstPass:
                                self.addSNPtoPileup(r, snp, n, i, readQualities[self.readPosition], mapQuality)

                                self.relativeIndexRef += self.deltaRefRelative[snp]

                            if snp == 4:
                                break

                        self.relativeIndex += self.deltaRef[snp]

                elif self.absolutePosition > self.queryEnd:  # stop iterating after query region
                    break


    def generateReadMapping(self,r,startIndex):
        '''
        Find the topmost available space to insert into the pileup
        :param r:
        :param startIndex:
        :return:
        '''

        i = 0
        unmapped = True
        for item in sorted(self.readEnds.items(), key=lambda x: x[1]):
            i += 1
            if item[1] < startIndex:
                self.readMap[r] = item[0]                               # store for use during insert homogenization
                self.breakpoints[self.readMap[r]].append(item[1])       # god this is so convoluted
                unmapped = False
                break

        if unmapped:
            self.readMap[r] = i


    def addSNPtoPileup(self,r,snp,n,i,readQuality,mapQuality):
        '''
        For a given read and SNP, add the corresponding encoding to the pileup array
        :param r:
        :param snp:
        '''

        index = self.relativeIndex
        encoding = None

        if snp == 0:                                                # match
            nt = self.readSequence[self.readPosition]
            ntRef = self.refSequence[self.relativeIndexRef]

            if nt != ntRef:
                encoding = self.SNPtoRGB[nt]    # Mismatch (specify the alt)
            else:
                encoding = self.SNPtoRGB['M']   # Match

            # update the end index array
            self.readEnds[self.readMap[r]] = index

        elif snp == 1:                                              # insert
            nt = self.readSequence[self.readPosition]

            encoding = self.SNPtoRGB[nt]

            if i == 0:      # record the insert for later
                position = self.absolutePosition-self.queryStart
                self.inserts[position].append([self.readMap[r], n])

            self.relativeIndex += 1

        elif snp == 2:                                              # delete
            encoding = self.SNPtoRGB['D']

        elif snp == 3:                                              # refskip (?)
            encoding = self.SNPtoRGB['N']

        # else:                                                       # anything else
            # sys.stderr.write("WARNING: unencoded SNP: %s at position %d\n" % (self.cigarLegend[snp],self.relativeIndex))

        if snp < 4:
            quality = (1-(10**((mapQuality)/-10)))*(1-(10**((readQuality)/-10)))*255   # calculate product of P_no_error

            # print(quality,mapQuality,readQuality)
            # if quality < 180:
            #     print("LOW QUALITY PIXEL FOUND: ", round(quality), " | mapQ: ", mapQuality, " | readQ: ", readQuality, " | column: ", index)

            encoding = list(copy.deepcopy(encoding)) + [int(round(quality))]    # append the quality Alpha value
            self.pileupRGB[self.readMap[r]][index] = tuple(encoding)       # Finally add the code to the pileup


    def reconcileInserts(self):
        '''
        For all recorded insertion indexes in the pileup, ensure reads across the column have uniform length,
         despite differences in insertion lengths among reads.
        '''

        offsets = [0 for n in range(len(self.pileupRGB))]
        breakpointOffsets = {r:0 for r in self.breakpoints.keys()}

        for p in sorted(self.inserts.keys()):
            i = 0

            # sort this position's inserts by length and select longest
            longestInsert = sorted(self.inserts[p], key=lambda x: x[1], reverse=True)[0]
            n = longestInsert[1]    #length of the insert

            for r in range(len(self.pileupRGB)):
                if r not in [insert[0] for insert in self.inserts[p]]:
                    # update the position based on previous inserts
                    pAdjusted = p+offsets[r]

                    # add the insert to the pileup
                    self.pileupRGB[r] = self.pileupRGB[r][:pAdjusted]+[self.SNPtoRGB['I']]*n+self.pileupRGB[r][pAdjusted:]

                    if i == 0:
                        # add the insert to the reference sequence and the label string
                        self.refSequence = self.refSequence[:int(pAdjusted)]+'I'*n+self.refSequence[int(pAdjusted):]

                        pVariant = p-1
                        if pVariant in self.variantLengths:             # if the position is a variant site
                            l = self.variantLengths[pVariant] -1  # -1 for ref
                            if l >= n:                                  # correctly modify the label to fit the insert
                                labelInsert = self.label[pAdjusted-1]*n # using the length of the called variant at pos.
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
                    pAdjusted = p+offsets[r]+insertLength  # start adding insert chars after this insertion

                    if insertLength < n:    # if this read has an insert, but of shorter length than the max
                        nAdjusted = n-insertLength             # number of inserts to add

                        self.pileupRGB[r] = self.pileupRGB[r][:pAdjusted]+[self.SNPtoRGB['I']]*nAdjusted+self.pileupRGB[r][pAdjusted:]

                    if r in breakpointOffsets:
                        breakpointOffsets[r] += insertLength
                        # find the nearest break point that comes after this insert and put some spaces in that _____
                        for breakpoint in self.breakpoints[r]:
                            if breakpoint > pAdjusted:
                                bAdjusted = breakpoint + breakpointOffsets[r]
                                self.pileupRGB[r] = self.pileupRGB[r][:bAdjusted]+[self.SNPtoRGB[self.noneChar]]*n+self.pileupRGB[r][bAdjusted:]

                offsets[r] += n     # <-- the magic happens here


    def encodeReference(self):
        '''
        Encode the reference sequence into RGB triplets and add add it as the header line to the pileup RGB
        :return:
        '''

        for character in self.refSequence:
            encoding = list(self.SNPtoRGB[character]) + [255]
            self.referenceRGB.append(tuple(encoding))


    def savePileupRGB(self,filename):
        '''
        Save the pileup binary array as a bitmap image using a gray color map
        :param filename:
        :return:
        '''

        if not filename.endswith(".png"):
            filename += ".png"

        self.pileupRGB = self.pileupRGB[:self.coverageCutoff]
        self.pileupRGB = [row[:self.windowCutoff] for row in self.pileupRGB]

        if self.sortColumns:
            self.pileupRGB = [list(entry) for entry in zip(*self.pileupRGB)] #transpose
            self.pileupRGB = [sorted(row, key=lambda x: self.RGBtoSortingKey(x[:3])) for row in self.pileupRGB] # sort
            self.pileupRGB = [list(entry) for entry in zip(*self.pileupRGB)] #transpose back

        self.pileupRGB = [self.referenceRGB] + self.pileupRGB

        image = Image.new("RGB",(self.windowCutoff,self.coverageCutoff))
        pixels = image.load()

        jlength = len(self.pileupRGB)

        for i in range(image.size[0]):
            for j in range(image.size[1]):

                if j < jlength:
                    pixels[i,j] = self.pileupRGB[j][i] if i < len(self.pileupRGB[j]) else self.SNPtoRGB[self.noneChar]
                else:
                    pixels[i,j] = self.SNPtoRGB[self.noneChar]
        image.save(filename,"PNG")


    def RGBtoBinary(self,rgb):
        return [int(value/255) for value in rgb]


    def RGBtoSortingKey(self,rgb):
        i1,i2,i3 = self.RGBtoBinary(rgb)
        code = self.RGBtoSNP[i1][i2][i3]
        return self.sortingKey[code]


    def getOutputLabel(self):
        blankLength = self.windowCutoff - len(self.label)

        self.label += self.noneLabel*blankLength
        return self.label


    def decodeRGB(self,filename):
        '''
        Read a RGB and convert to a text alignment
        :param filename:
        :return:
        '''

        img = Image.open(filename)          # <-- switch this to RGBA
        pixels = numpy.array(img.getdata())
        text = ''

        width,height = img.size
        depth = 3

        pixels = numpy.reshape(pixels,(height,width,depth))

        for h in range(height):
            for w in range(width):
                r,g,b = self.RGBtoBinary(pixels[h][w])

                text += self.RGBtoSNPtext[r][g][b]
            text += '\n'

        return text


class PileUpGenerator:
    '''
    Creates pileups of aligned reads given a SAM/BAM alignment and FASTA reference
    '''

    def __init__(self,alignmentFile,referenceFile):
        self.sam = pysam.AlignmentFile(alignmentFile,"rb")
        self.fasta = Fasta(referenceFile,as_raw=True,sequence_always_upper=True)


    def generatePileup(self,chromosome,position,flankLength,outputFilename,label,variantLengths,forceCoverage=False,coverageCutoff=150,mapQualityCutoff=0,windowCutoff=300,sortColumns=True):
        '''
        Generate a pileup at a given position
        :param queryStart:
        :param regionLength:
        :return:
        '''

        queryStart = position-flankLength

        chromosome = str(chromosome)

        # print(outputFilename)
        # startTime = datetime.now()
        pileup = Pileup(self.sam,self.fasta,chromosome,queryStart,flankLength,outputFilename,label,variantLengths,windowCutoff=windowCutoff,forceCoverage=forceCoverage,coverageCutoff=coverageCutoff,mapQualityCutoff=mapQualityCutoff,sortColumns=sortColumns)
        # print(datetime.now() - startTime, "initialized")
        pileup.iterateReads()
        # print(datetime.now() - startTime, "drafted")
        pileup.reconcileInserts()
        # print(datetime.now() - startTime, "finalized")
        pileup.encodeReference()
        pileup.savePileupRGB(outputFilename)
        # print(datetime.now() - startTime, "encoded and saved")

        # print(pileup.getOutputLabel())
        # print(pileup.decodeRGB(outputFilename + ".png"))

        return pileup.getOutputLabel()


# bamFile = "deePore/data/chr3_200k.bam"
# fastaFile = "deePore/data/chr3.fa"
#
# vcf_region = "3"
# window_size = 100
# filename = "deePore/data/test_packed/"
# labelString = '0'*(window_size+1)
# variantLengths = dict()
# position = 66164
#
# piler = PileUpGenerator(bamFile,fastaFile)
# outputLabelString = piler.generatePileup(chromosome=vcf_region, position=position, flankLength=window_size,
#                                      outputFilename=filename, label=labelString, variantLengths=variantLengths,forceCoverage=True,coverageCutoff=100)
