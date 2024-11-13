import subprocess
import os
import argparse
import tqdm

def generate_full_brain_tractogram(odfs, tissue_segmentation, gmwmi, save_dir):
    for odf in odfs:
        odf_name = odf.split('/')[-1].split('.')[0]
        if not os.path.exists(f'{save_dir}/{odf_name}.tck'):
            tckgen_cmd = f'tckgen {odf} {save_dir}/{odf_name}.tck ' + \
                    f'-select 10M -nthreads 100 -force -seed_gmwmi {gmwmi} -act {tissue_segmentation} -backtrack'
            
            print(f'Generating full-brain tractography for {odf}...')
            
            subprocess.run(tckgen_cmd, shell=True)

def filter_tracts(tracts, voi, custom_voi, save_dir):
    include_dirs = [f'{voi}/{d}' for d in os.listdir(voi) if 'incs' in d]
    exclude_dirs = [f'{voi}/{d}' for d in os.listdir(voi) if 'excs' in d]
    for tract in tracts:
        tract_name = tract.split('/')[-1].split('.')[0]
        tckedit_cmd = f'tckedit {tract} {save_dir}/{tract_name}.tck -number 20k -force -quiet -exclude {custom_voi}/cerebellum_Bil_X.nii.gz -nthreads 100 '
        if 'LT' in voi:
            tckedit_cmd += f'-exclude {custom_voi}/cerebrum_hemi_RT_X_nv.nii.gz '
        elif 'RT' in voi:
            tckedit_cmd += f'-exclude {custom_voi}/cerebrum_hemi_LT_X_nv.nii.gz '
        for exc in exclude_dirs:
            f = os.listdir(exc)[0]
            tckedit_cmd += f'-exclude {exc}/{f} '
        for inc in include_dirs:
            f = os.listdir(inc)[0]
            tckedit_cmd += f'-include {inc}/{f} '
    
        subprocess.run(tckedit_cmd, shell=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment tracts')
    parser.add_argument('dwi', type=str, help='Path to the preprocessed DWI data')
    parser.add_argument('odfs', type=str, help='List of ODFs separated by commas')
    parser.add_argument('voi_dir', type=str, help='Path to the directory containing the VOIs')
    parser.add_argument('tissue_segmentation', type=str, help='Path to the tissue segmentation file')
    parser.add_argument('gmwmi', type=str, help='Path to the GM-WM interface file')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')

    args = parser.parse_args()

    voi_dir = args.voi_dir
    # Find all subdirs that to no have MNI in their name
    voi_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'MNI' not in d and 'done' not in d and 'custom' not in d]
    
    odfs = args.odfs.split(',')
    odfs = [os.path.join(os.getcwd(), o) for o in odfs]

    os.makedirs(args.output_dir, exist_ok=True)
    generate_full_brain_tractogram(odfs, args.tissue_segmentation, args.gmwmi, args.output_dir)

    tracts = [f'{args.output_dir}/{f}' for f in os.listdir(args.output_dir) if f.endswith('.tck')]

    # Copy all the VOIs to the output directory
    for voi in voi_dirs:
        for subdir in list(filter(lambda f: 'custom_' not in f and 'MNI' not in f, os.listdir(voi))):
            for f in list(filter(lambda f: '_bin.nii.gz' in f, os.listdir(f'{voi}/{subdir}'))):
                os.makedirs(f'{args.output_dir}/{voi.split("/")[-1]}/{subdir}', exist_ok=True)
                cmd = f'cp {voi}/{subdir}/{f} {args.output_dir}/{voi.split("/")[-1]}/{subdir}/{f}'
                subprocess.run(cmd, shell=True)

        # Filter the tracts
        save_dir = f'{args.output_dir}/{voi.split("/")[-1]}'
        os.makedirs(save_dir, exist_ok=True)
        filter_tracts(tracts, save_dir, f'{os.path.join(os.getcwd(), voi_dir)}/custom_VOIs', save_dir)



    # Now copy the custom_VOIs as well
    copy_cmd = f'cp -r {voi_dir}/custom_VOIs {args.output_dir}/'
    subprocess.run(copy_cmd, shell=True)

