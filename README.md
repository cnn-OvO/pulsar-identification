# Pulsar Identification Program

The pulsar identifiacation program is a tool for scoring pulsar candidates from radio astronomy surveys. It is designed to be user-friendly and customizable. The program handles input data in __.pfd__ file format, which is commonly used in radio astronomy surveys.

## CONTENT
- [Install](#index1)
- [Usage](#index2)

## <span id="index1">Install</span>

0. The program may need the following environments to ensure that it runs smoothly: 

   - python = 3.8
   - torch = 1.10.0
   - torchvision = 0.11.1
   - cv2 = 3.4.10

1.  Clone the repository onto your local machine.
   
2.  Navigate to the root directory of the repository.
   
3.  Ensure that all `.py` file is executable by running `chmod +x xxx.py` in your terminal.
   
4. You can add the program to your bash environment:
	export PYTHONPATH=/path/:$PYTHONPATH
	alias pred='python3 /path/pred.py'


## <span id="index2">Usage</span>

- __Prediction__

The program score the pfd file in two ways:
1. Navigate to a directory containing .pfd files, run `pred` if you add the program to your bash environment or run `python3 /path/pred.py`.
2. Prepare a txt file list the names of .pfd file which you want to identify. Then run `pred -f xxx.txt` or run `python3 /path/pred.py xxx.txt`.

- __Train__

update later...
