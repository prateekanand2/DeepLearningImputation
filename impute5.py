# %%
import numpy as np
import pandas as pd
import subprocess
import os
import math
import sys
from sklearn.metrics import r2_score
import shutil
import tempfile

k = int(sys.argv[1]) - 1
threads = int(sys.argv[2])
train_prefix = str(sys.argv[3])
test_prefix = str(sys.argv[4])
chrnumber = int(sys.argv[5])
method_full = str(sys.argv[6])
snp_index_file = str(sys.argv[7])
# %%
def process_plink_data(folder, plink_file_prefix):
    vcf_file = f"{folder}/{plink_file_prefix}.vcf"

    # print("Exporting PLINK data to VCF...")
    export_command = [
        "../plink2", "--bfile", f"{folder}/{plink_file_prefix}",
        "--recode", "vcf", "--out", f"{folder}/{plink_file_prefix}",
        "--silent"
    ]
    
    if not os.path.exists(vcf_file):
        subprocess.run(export_command, check=True)

    bcftools_command = [
        "/scratch2/prateek/bcftools/bcftools", "view", "-Ou", "-o", f"{folder}/{plink_file_prefix}.bcf", vcf_file
    ]

    if not os.path.exists(f"{folder}/{plink_file_prefix}.bcf"):
        subprocess.run(bcftools_command, check=True)

        # print("BCF conversion complete.")

        # print("Adding AC field..")
        subprocess.run([
        '/scratch2/prateek/bcftools/bcftools', '+fill-tags', f'{folder}/{plink_file_prefix}.bcf', '-Ou', '-o', f'{folder}/{plink_file_prefix}_AC.bcf', '--', '-t', 'AN,AC'
        ])
        # print("Adding index file..")
        subprocess.run(['/scratch2/prateek/bcftools/bcftools', 'index', f'{folder}/{plink_file_prefix}_AC.bcf'])

def process_plink_data_with_drop(folder, plink_file_prefix, rs_ids, temp_dir):
    vcf_file = f"{folder}/{plink_file_prefix}.vcf"

    modified_vcf_file = f"{temp_dir}/modified_test.vcf"
    with open(vcf_file, 'r') as f:
        lines = f.readlines()

    with open(modified_vcf_file, 'w') as out_file:
        for line in lines:
            if line.startswith('#'):
                out_file.write(line)
                continue
            
            fields = line.strip().split('\t')
            snp_id = fields[2]

            if snp_id in rs_ids:
                # fields[9:] = ['./.' for _ in fields[9:]]
                continue
                
            out_file.write('\t'.join(fields) + '\n')

    # print("VCF modification complete.")

    bcftools_command = [
        "/scratch2/prateek/bcftools/bcftools", "view", "-Ou", '--threads', f'{threads}', "-o", f"{temp_dir}/modified_test.bcf", modified_vcf_file
    ]
    subprocess.run(bcftools_command)

    # print("BCF conversion complete.")

    # print("Adding AC field..")
    subprocess.run([
     '/scratch2/prateek/bcftools/bcftools', '+fill-tags', f'{temp_dir}/modified_test.bcf', '-Ou', '-o', f'{temp_dir}/modified_test_AC.bcf', '--', '-t', 'AN,AC'
    ], stdout=subprocess.DEVNULL)
    # print("Adding index file..")
    subprocess.run(['/scratch2/prateek/bcftools/bcftools', 'index', f'{temp_dir}/modified_test_AC.bcf'])

def extract_imputed_genotype_array(vcf_file, snp_set, correct_genotype_array, num_samples):
    results = {}
    
    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id in snp_set:
                c = correct_genotype_array[snp_id]

                genotype_data = fields[9:]
                genotype_array = []
                prob1 = []
                expected_counts = []

                for i, genotype in enumerate(genotype_data):
                    gt_info, _, gp_info = genotype.split(':')
                    
                    # Convert genotype format to 0, 1, or 2
                    if gt_info == "0":
                        genotype_array.append(0)
                    # elif gt_info == "0|1" or gt_info == "1|0":
                    #     genotype_array.append(1)
                    elif gt_info == "1":
                        genotype_array.append(1)
                    else:
                        genotype_array.append(-1)

                    prob_values = list(map(float, gp_info.split(',')))
                    prob_0 = prob_values[0]
                    prob_1 = prob_values[1]
                    # prob_2 = prob_values[2]
                    expected_count = prob_0 * 0 + prob_1 * 1 # + prob_2 * 2
                    expected_counts.append(expected_count)

                    prob1.append(prob_1)

                results[snp_id] = (genotype_array, prob1, expected_counts)

    return results

def extract_test_genotype_array(vcf_file, snp_set):
    results = {}

    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id in snp_set:
                genotype_data = fields[9:]
                genotype_array = []

                for genotype in genotype_data:
                    if genotype == "0":
                        genotype_array.append(0)
                    # elif genotype == "0/1" or genotype == "1/0":
                    #     genotype_array.append(1)
                    elif genotype == "1":
                        genotype_array.append(1)
                    else:
                        genotype_array.append(-1)

                results[snp_id] = genotype_array

    return results 

def compute_r_squared_old(array1, array2):
    # Ensure both arrays are NumPy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Calculate the correlation coefficient
    correlation_matrix = np.corrcoef(array1, array2)
    correlation_xy = correlation_matrix[0, 1]
    
    # Calculate R^2
    r_squared = correlation_xy ** 2
    
    return r_squared

def analyze_vcf(vcf_file):
    min_pos = float('inf')
    max_pos = float('-inf')
    num_snps = 0
    num_samples = 0

    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith("##"):
                continue
            
            if line.startswith("#CHROM"):
                columns = line.strip().split("\t")
                num_samples = len(columns) - 9
                continue
            
            columns = line.strip().split("\t")
            pos = int(columns[1])
            
            min_pos = min(min_pos, pos)
            max_pos = max(max_pos, pos)
            
            num_snps += 1

    return min_pos, max_pos, num_snps, num_samples

def extract_rs_ids_from_vcf(vcf_file):
        """
        Extract rsIDs from a VCF file.

        Parameters:
        - vcf_file (str): Path to the VCF file.

        Returns:
        - list: List of rsIDs extracted from the VCF file.
        """
        rs_ids = []

        with open(vcf_file, 'r') as f:
            for line in f:
                # Skip header lines starting with '##'
                if line.startswith('##'):
                    continue

                # Skip the column header line
                if line.startswith('#CHROM'):
                    continue

                # Split the VCF line into fields
                fields = line.split()

                # Extract rsID from the third column (ID column)
                rs_id = fields[2]  # This is where rsID is typically found in a VCF file
                rs_ids.append(rs_id)
        
        return rs_ids
########################################################################################################################
#  %%
def main():
    temp_dir = tempfile.mkdtemp(prefix=f"temp{k}")

    folder = "/scratch2/prateek/diffusion/data/UKBB"
    plink_file_train_prefix = train_prefix
    plink_file_test_prefix = test_prefix
    chrnum = chrnumber

    vcf_file = f"{folder}/{plink_file_test_prefix}.vcf"
    idx1, idx2, num_snps, num_samples = analyze_vcf(vcf_file)

    print(f"Min position: {idx1}")
    print(f"Max position: {idx2}")
    print(f"Num SNPs: {num_snps}")
    print(f"Num samples: {num_samples}")

    buffer_region = f"{chrnum}:{idx1}-{idx2}"

    rs_ids = extract_rs_ids_from_vcf(vcf_file)

    os.environ['BCFTOOLS_PLUGINS'] = '/scratch2/prateek/bcftools/plugins'

    # Load SNP indices to drop from file
    # snp_index_file = "snp_index_file.txt"  # path to your file
    with open(snp_index_file, "r") as f:
        snp_indices_to_drop = [int(line.strip()) for line in f if line.strip()]

    print(f"Dropping {len(snp_indices_to_drop)} SNPs from {snp_index_file}")
    
    snp_set = [rs_ids[idx] for idx in snp_indices_to_drop]
    # print(f"Dropping SNP set: {snp_set}")

    # modify plink data dropping multiple SNPs
    process_plink_data(folder, plink_file_train_prefix)
    process_plink_data_with_drop(folder, plink_file_test_prefix, snp_set, temp_dir)

    # run imputation
    subprocess.run([
        '/scratch2/prateek/impute5_v1.2.0/impute5_v1.2.0_static',
        '--h', f'{folder}/{plink_file_train_prefix}_AC.bcf',
        '--m', f'/scratch2/prateek/b37_recombination_maps/chr{chrnum}.b37.gmap.gz',
        '--g', f'{temp_dir}/modified_test_AC.bcf',
        '--r', buffer_region,
        '--buffer-region', buffer_region,
        '--o', f"{temp_dir}/imputed_custom.vcf",
        '--l', f"{temp_dir}/imputed_custom.log",
        '--haploid',
        '--threads', f'{threads}'
    ])

    # extract results for the whole set
    test_arrays = extract_test_genotype_array(vcf_file, snp_set)
    results = extract_imputed_genotype_array(f'{temp_dir}/imputed_custom.vcf', snp_set, test_arrays, num_samples)

    r2_list = []
    r2_geno_list = []

    outdir = f"results/bootstrap/{method_full}_dosages/"
    # Remove existing directory and its contents if it exists
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
        
    for snp in snp_set:
        test_genotype_array = test_arrays[snp]
        imputed_genotype_array = results[snp][0]
        prob1 = results[snp][1]

        r2 = compute_r_squared_old(test_genotype_array, prob1)
        r2_geno = compute_r_squared_old(test_genotype_array, imputed_genotype_array)

        r2_list.append(r2)
        r2_geno_list.append(r2_geno)

        # print(f"{snp}\tR2={r2:.4f}, R2_geno={r2_geno:.4f}")

        np.savetxt(f"{outdir}{snp}.txt", prob1, fmt="%.5f")

    # === print averages at the end ===
    # avg_r2 = np.mean(r2_list)
    # avg_r2_geno = np.mean(r2_geno_list)
    avg_r2 = np.mean(np.nan_to_num(r2_list, nan=0.0))
    avg_r2_geno = np.mean(np.nan_to_num(r2_geno_list, nan=0.0))
    print(f"\nAverage R2={avg_r2:.4f}, Average R2_geno={avg_r2_geno:.4f}")

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
# %%
