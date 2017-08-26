from pysam import VariantFile
import argparse

def generatePileupBasedOnVCF(vcfFile, region):
    vcf_in = VariantFile(vcfFile)
    recordList = []
    cnt = 0
    for rec in vcf_in.fetch(region):
        #print(rec)
        cnt+=1
        recordList.append(rec)
    print("Total records found: "+str(cnt)+" Region: "+region)
    return recordList


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--vcf",
        type=str,
        required = True,
        help="VCF file with records."
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region of VCF file to parse."
    )
    FLAGS, unparsed = parser.parse_known_args()
    generatePileupBasedOnVCF(FLAGS.vcf, FLAGS.region)