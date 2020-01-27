import os



for seed in seeds:
    out_dir = os.path.join(
        '/path/to/save/to',
            'blr_face_seed_{}'.format(seed))
    # Save output and parameters to text file in the localhost node,
    # which is where the computation is performed.
    command = [
        "python", "run_models.py",
        "--type", "simple",
        "--outcome-name", 'pct_turnout_ge2017'
        "--out-dir", 'experiments/simple-rbf'
        "--split-seed", str(seed),
        "--out-dir ", out_dir
            ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)
