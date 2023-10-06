==========================================
srun -p gpu -A legacy-projects --gpus-per-node v100:1 --pty bash
srun -p gpu -A r00043 --gpus-per-node v100:1 --time=3:00:00 --pty bash
module load deeplearning
pip install [package]

or for the gpu node
module load anaconda
module load deeplearning
conda activate tc 

For the debug node: 
module load anaconda
conda
conda activate py3.9


==========================================
#create a new git
git init
git add *
git commit -m "Start new repository"
git branch -M main
git remote add origin https://github.com/kieucq/da_3dvar_L40.git
git push -u origin main

# adding files/folders
git add letkf/*.ctl
git commit -m "Adding Grads ctl files of letkf"
git push -u origin main

# update all changes
git status -uno # opt -uno will not list all untracked files
git add -u
git commit -m "Adding all modified files only"
git push -u origin main

# remove files/folder
git rm letkf/*.dat
git commit -m "Remove all dat files"
git push -u origin main

#checkout a single file
git fetch
git checkout -m <yourfilepath>
git add <yourfilepath>
git commit

#Personal access token
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
