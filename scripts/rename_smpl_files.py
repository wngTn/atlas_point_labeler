import os
import glob
import shutil

PATH_TO_DATA = 'data/trials'
PATH_TO_SMPL = 'data/atlas_3d_keypoints'

def get_trial_frames(trial, phase):
    all_images = glob.glob(os.path.join(PATH_TO_DATA, trial, phase) + '/**/*.jpg', recursive=True)
    base_names = list(map(os.path.basename, all_images))

    unique_base_names = set(base_names)

    frame_ids = []
    for name in unique_base_names:
        name = name[:-10]
        frame_ids.append(int(name))
    
    frame_ids = list(sorted(frame_ids))

    return frame_ids


def rename_smpl_fs():
    for trial in os.listdir(PATH_TO_SMPL):
        for phase in os.listdir(os.path.join(PATH_TO_SMPL, trial)):

            original_frame_ids = get_trial_frames(trial, phase)
            smpl_files = list(sorted(glob.glob(os.path.join(PATH_TO_SMPL, trial, phase, "smpl_files") + '/*.json')))

            if len(original_frame_ids) != len(smpl_files):
                print(f"Skipping {trial}, {phase}")
                continue

            for smpl_file, frame_id in zip(smpl_files, original_frame_ids):
                new_file_name = os.path.join(PATH_TO_SMPL, trial, phase, "smpl_files", f"{str(frame_id).zfill(4)}_smpl.json")
                shutil.move(smpl_file, new_file_name)
                print(f"Renamed {smpl_file} to {new_file_name}")

if __name__=='__main__':
    rename_smpl_fs()



