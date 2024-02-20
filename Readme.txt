Pre-requisites for running on raspberry-pi-4 and macOS 13.2.1:
- python 3.7.3 (restriction on pi-4)
- downgrade numpy<1.19.0 (eg. 1.18.5) to avoid build braking according to #4 referring to code change in #5
- building tensorflow-2.2.0 for macOS:
-- bazel-2.0.0 and GCC-7.3.1 (see #1)
-- compiler bugfix mentioned in #2 and described in #3
-- compiler bugfix mentioned in #6 but at different code lines


-


How to install



References:
#1 https://www.tensorflow.org/install/source#macos
#2 issue_60191_patch.txt
#3 https://github.com/tensorflow/tensorflow/issues/60191
#4 https://github.com/tensorflow/tensorflow/issues/40688
#5 https://github.com/numpy/numpy/pull/15355
#6 https://github.com/tensorflow/tensorflow/commit/75ea0b31477d6ba9e990e296bbbd8ca4e7eebadf
