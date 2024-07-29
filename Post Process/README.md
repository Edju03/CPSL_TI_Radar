# Post Processing code

## Code Descriptions


Useful Codes:

MMWaveDevice: Used to set up the profile and configuration 

NoVisualization: Estimator class (Two profile) 

estimator: estimator class but reads cfg file instead of json file in NoVisualiation (Two profile)

MultipleProfileSeperatePeriod: Plots adc data, range doppler plot, and other useful plots for post analysis

GraphPrediction: GraphGenration + Prediction (Two profile)

----------


To run the code you need to have a json file and bin file (or cfg and bin file for NoVisualization)



You can generate the json and bin file using mmWaveStudio software.

First follow the steps in https://github.com/davidmhunt/mmWaveStudioProcessing/blob/main/Guides/Data%20Capture%20Quick%20Guide.md

After running you can go to your post process file to download the bin file. You can also press the export json file button in your left to download the json file.

------------

In the code
Identify the file path and modify the file path in the code to be able to run.
