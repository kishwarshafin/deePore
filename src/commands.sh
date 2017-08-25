##VCF FILE
python3 parser-withVCF.py --bam ../data/NA12878.np.chr3.100kb.0.bam --ref ../data/hg19.chr3.9mb.fa --vcf ../data/NA12878.hg19.PG.chr3.full.vcf --region chr3 --site_start 100000 --site_end 110000 --coverage 34 --window_size 50 --vcf_region_only 1 --matrix_out 1

##JUST PILEUP
python3 parser-withVCF.py --bam ../data/NA12878.np.chr3.100kb.0.bam --ref ../data/hg19.chr3.9mb.fa --region chr3 --site_start 100000 --site_end 110000 --coverage 34 --matrix_out 1

python parser-withVCF.py --bam /hive/users/kishwar/sorted_final_merged.bam --ref /hive/users/kishwar/human_g1k_v37.fasta --region chr3 --site_start 100000 --site_end 110000 --coverage 34 --matrix_out 1
