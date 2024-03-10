Pre-requisites for running on raspberry-pi-4 and macOS 13.2.1:
- python 3.7.3 (restriction on pi-4)
- downgrade numpy<1.19.0 (eg. 1.18.5) to avoid build braking according to #4 referring to code change in #5
- building tensorflow-2.2.0 for macOS:
-- bazel-2.0.0 and GCC-7.3.1 (see #1)
-- compiler bugfix mentioned in #2 and described in #3
-- compiler bugfix mentioned in #6 but at different code lines
- build wheel:
-- install gsed and replace call sed by gsed: https://github.com/tensorflow/tensorflow/issues/45434
-


How to install

GIT REMOTE handling: https://stackoverflow.com/questions/11935633/git-diff-between-a-remote-and-local-repository

To compare a local working directory against a remote branch, for example origin/master:

git fetch origin master
This tells git to fetch the branch named 'master' from the remote named 'origin'.  git fetch will not affect the files in your working directory; it does not try to merge changes like git pull does.
git diff --summary FETCH_HEAD
When the remote branch is fetched, it can be referenced locally via FETCH_HEAD. The command above tells git to diff the working directory files against FETCHed branch's HEAD and report the results in summary format. Summary format gives an overview of the changes, usually a good way to start. If you want a bit more info, use --stat instead of --summary.
git diff FETCH_HEAD -- mydir/myfile.js
If you want to see changes to a specific file, for example myfile.js, skip the --summary option and reference the file you want (or tree).
As noted, origin references the remote repository and master references the branch within that repo. By default, git uses the name origin for a remote, so if you do git clone <url> it will by default call that remote origin. Use git remote -v to see what origin points to.

You may have more than one remote. For example, if you "fork" a project on GitHub, you typically need a remote referencing the original project as well as your own fork. Say you create https://github.com/yourusername/someproject as a fork of https://github.com/theoriginal/someproject. By convention, you would name the remote to the original repo upstream, while your own fork would be origin. If you make changes to your fork on GitHub and want to fetch those changes locally, you would use git fetch origin master. If the upstream has made changes that you need to sync locally before making more changes, you would use git fetch upstream master.

References:
#1 https://www.tensorflow.org/install/source#macos
#2 issue_60191_patch.txt
#3 https://github.com/tensorflow/tensorflow/issues/60191
#4 https://github.com/tensorflow/tensorflow/issues/40688
#5 https://github.com/numpy/numpy/pull/15355
#6 https://github.com/tensorflow/tensorflow/commit/75ea0b31477d6ba9e990e296bbbd8ca4e7eebadf
