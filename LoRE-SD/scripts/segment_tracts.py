import subprocess
import os
import argparse
import tqdm

# def high_res_2_lore(high_res_fd, b0, save_dir):
#     if os.path.exists(f'{save_dir}/high_res_2_lore_0GenericAffine.mat'):
#         print('Transformation matrix already calculated')
#         return f'{save_dir}/high_res_2_lore_0GenericAffine.mat'
#     print(f'Calculating transformation from high res to LoRE space...')
#     cmd = f'antsRegistrationSyN.sh -d 3 -f {b0} -m {high_res_fd} -n 100 -o {save_dir}/high_res_2_lore_ -t r'
#     subprocess.run(cmd, shell=True)

#     tf_matrix = f'{save_dir}/high_res_2_lore_0GenericAffine.mat'
#     return tf_matrix

# def dwi_2_b0(dwi, save_dir):
# 
#     if os.path.exists(f'{save_dir}/b0.nii.gz'):
#         print('b0 already extracted')
#         return f'{save_dir}/b0.nii.gz'

#     print(f'Extracting b0 from dwi...')
#     cmd = f'dwiextract {dwi} -bzero - | mrmath - mean -axis 3 {save_dir}/b0.nii.gz'
#     subprocess.run(cmd, shell=True)

#     b0 = f'{save_dir}/b0.nii.gz'
#     return b0
# 
# def transform_VOIs(voi_dir, transform, b0, save_dir):
#     # For every directory in the VOI directory

#     for voi in tqdm.tqdm(list(filter(lambda x: 'MNI' not in x and 'custom_' not in x and os.path.isdir(os.path.join(voi_dir, x)), os.listdir(voi_dir)))):
#         for subdir in os.listdir(f'{voi_dir}/{voi}'):
#             for f in os.listdir(f'{voi_dir}/{voi}/{subdir}'):
#                 # If the file is a nii.gz file
#                 if 'bin.nii.gz' in f:
#                     # Apply the transformation to the file
#                     transform_VOI(f'{voi_dir}/{voi}/{subdir}/{f}', transform, b0, os.path.join(save_dir, voi, subdir))

#     # For the custom VOIs, we only consider a few
#     to_transform_in_custom = ['cerebellum_Bil_X.nii.gz', 'cerebrum_hemi_LT_X_nv.nii.gz', 'cerebrum_hemi_RT_X_nv.nii.gz']
#     for custom_voi in to_transform_in_custom:
#         transform_VOI(f'{voi_dir}/custom_VOIs/{custom_voi}', transform, b0, os.path.join(save_dir, 'custom_VOIs'))

# def transform_VOI(voi, transform, b0, save_dir):
#     if os.path.exists(f'{save_dir}/{voi.split("/")[-1]}'):
#         print(f'{voi} already transformed')
#         return
    
#     else:
#         os.makedirs(save_dir, exist_ok=True)

#     cmd = f'antsApplyTransforms -d 3 -i {voi} -o {save_dir}/{voi.split("/")[-1]} ' + \
#         f'-t {transform} -r {b0} -n GenericLabel'
    
#     subprocess.run(cmd, shell=True)

def regrid_VOI(voi, template, save_dir):
    if os.path.exists(f'{save_dir}/{voi.split("/")[-1]}'):
        # print(f'{voi} already regridded')
        return
    
    else:
        os.makedirs(save_dir, exist_ok=True)

    cmd = f'mrgrid {voi} regrid -template {template} {save_dir}/{voi.split("/")[-1]} -interp nearest -force -quiet'
    
    subprocess.run(cmd, shell=True)

def regrid_VOIs(voi_dir, template, save_dir):
    # For every directory in the VOI directory
    for voi in tqdm.tqdm(list(filter(lambda x: 'MNI' not in x and 'custom_' not in x and os.path.isdir(os.path.join(voi_dir, x)), os.listdir(voi_dir)))):
        for subdir in os.listdir(f'{voi_dir}/{voi}'):
            for f in os.listdir(f'{voi_dir}/{voi}/{subdir}'):
                # If the file is a nii.gz file
                if 'bin.nii.gz' in f:
                    # Apply the transformation to the file
                    regrid_VOI(f'{voi_dir}/{voi}/{subdir}/{f}', template, os.path.join(save_dir, voi, subdir))

    # For the custom VOIs, we only consider a few
    to_regrid_in_custom = ['cerebellum_Bil_X.nii.gz', 'cerebrum_hemi_LT_X_nv.nii.gz', 'cerebrum_hemi_RT_X_nv.nii.gz']
    for custom_voi in to_regrid_in_custom:
        regrid_VOI(f'{voi_dir}/custom_VOIs/{custom_voi}', template, os.path.join(save_dir, 'custom_VOIs'))

def generate_tract(voi_dir, custom_voi_dir, odfs, save_dir):
    include_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'incs' in d]
    exclude_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'excs' in d]

    cerebellum_bundles = ['ICP', 'MCP', 'DRT']

    for odf in odfs:
        odf_name = odf.split('/')[-1].split('.')[0]
        
        tckgen_cmd = f'tckgen {odf} {save_dir}/{odf_name}.tck ' + \
                f'-select 20k -nthreads 150 -force -seeds 100M '
        # If none of the 'cerebellum_bundles' is in the name of the VOI directory, exclude the cerebellum
        if not any([cb in voi_dir for cb in cerebellum_bundles]):
            tckgen_cmd += f'-exclude {custom_voi_dir}/cerebellum_Bil_X.nii.gz '
        if 'LT' in voi_dir:
            tckgen_cmd += f'-exclude {custom_voi_dir}/cerebrum_hemi_RT_X_nv.nii.gz '
        elif 'RT' in voi_dir:
            tckgen_cmd += f'-exclude {custom_voi_dir}/cerebrum_hemi_LT_X_nv.nii.gz '

        for exc in exclude_dirs:
            f = os.listdir(exc)[0]
            tckgen_cmd += f'-exclude {exc}/{f} '
        
        for inc in include_dirs:
            f = os.listdir(inc)[0]
            tckgen_cmd += f'-include {inc}/{f} '
            tckgen_cmd += f'-seed_image {inc}/{f} '
            
        subprocess.run(tckgen_cmd, shell=True)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment tracts')
    parser.add_argument('voi_dir', type=str, help='Path to the directory containing the VOIs')
    parser.add_argument('odfs', type=str, help='List of ODFs separated by commas')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')

    args = parser.parse_args()

    voi_dir = args.voi_dir
    # Find all subdirs that to no have MNI in their name
    voi_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'MNI' not in d and 'done' not in d and 'custom' not in d]
    
    odfs = args.odfs.split(',')
    odfs = [os.path.join(os.getcwd(), o) for o in odfs]

    os.makedirs(args.output_dir, exist_ok=True)

    # b0 = dwi_2_b0(args.dwi, args.output_dir)
    subprocess.run(f'mrconvert {odfs[0]} {args.output_dir}/template.mif -coord 3 0 -force -quiet', shell=True)
    regrid_VOIs(voi_dir, f'{args.output_dir}/template.mif', args.output_dir)

    # # Copy all the VOIs to the output directory
    # for voi in voi_dirs:
    #     for subdir in list(filter(lambda f: 'custom_' not in f and 'MNI' not in f, os.listdir(voi))):
    #         for f in list(filter(lambda f: '_bin.nii.gz' in f, os.listdir(f'{voi}/{subdir}'))):
    #             os.makedirs(f'{args.output_dir}/{voi.split("/")[-1]}/{subdir}', exist_ok=True)
    #             cmd = f'cp {voi}/{subdir}/{f} {args.output_dir}/{voi.split("/")[-1]}/{subdir}/{f}'
    #             subprocess.run(cmd, shell=True)
    # Now copy the custom_VOIs as well
    # copy_cmd = f'cp -r {voi_dir}/custom_VOIs {args.output_dir}/'
    # subprocess.run(copy_cmd, shell=True)

    tracts = list(filter(lambda x: os.path.isdir(os.path.join(args.output_dir, x)) and 'SLF_III_RT' in x, os.listdir(args.output_dir)))
    for tract in tqdm.tqdm(tracts):
        generate_tract(f'{args.output_dir}/{tract}', f'{args.output_dir}/custom_VOIs', 
                       odfs, os.path.join(args.output_dir, tract))
