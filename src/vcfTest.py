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


generatePileupBasedOnVCF("../../../../Downloads/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.vcf")