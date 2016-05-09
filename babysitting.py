import sys

def progress_bar(percent, barLen, measure1, measure2, measure3 = 0, trainmode = 0):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
            
    if (trainmode):      
        sys.stdout.write("[ %s ] %.2f%% - Training loss: %0.4f - L2 loss: %0.4f - Accuracy: %0.4f" % (progress, percent * 100, measure1, measure2, measure3))
    else:
        sys.stdout.write("[ %s ] %.2f%% - Validation loss: %0.4f - Accuracy: %0.4f" % (progress, percent * 100, measure1, measure2))   
    sys.stdout.flush()