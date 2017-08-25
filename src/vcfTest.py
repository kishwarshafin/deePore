from pysam import VariantFile

def generatePileupBasedOnVCF(vcfFile):
    vcf_in = VariantFile(vcfFile)
    recordList = []
    cnt = 0
    for rec in vcf_in.fetch():
        print(rec)
        cnt+=1
        recordList.append(rec)
    return recordList


generatePileupBasedOnVCF("/hive/users/kishwar/GRCh37.vcf")