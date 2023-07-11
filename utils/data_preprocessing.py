from fdsobj import FdsObj
import numpy as np
import pandas as pd
import os



def reload_data(fname, numShots):
    fds = FdsObj(fname)
    data = np.asarray([shot for _,shot in zip(range(numShots), fds)])
    data = np.transpose(data)
    return fds, data #data shape: (bin, shot)

def get_start_end_shot(car_label, file_index, direction = None):
    """
    Our aim is to draw two linear regressions by collecting multiple pair of (shot_start[i], shot_end[i], bin[i]). 
    Since a human labeller can't label start_shot and end_shot for all the possible bin in a 2D-DAS map, we draw the two lines to find a DAS signal in between the lines.   
    """
    bins_clean = [] # A placeholder that stores bin containing "signals we can define when they start and end", i.e, clean signal. Some cars may not need this list because they have clean signals in a list pre-defined bins. 
#     shot_start = [] # A placeholder stores start shot of a signal.
    if file_index == "023":
        if car_label == "1":
            print('overlap with other car signal, not consider using it')
        elif car_label == "2":
            shot_start = np.array([100000,96000,92500,88000,84000,80000,75500,72000,67000,64000,59000])
            shot_end = np.array([107000,102000,97000,92500,89000,84000,79000,76000,72000,68000,64000])
        elif car_label == "3":
            shot_end = np.array([209000,203000,198000,192500,189000,184000,179000,175000,171000,167000,161000])
            shot_start = np.array([202000,197000,193000,188500,184000,180000,175000,171000,165000,161500,157500])
        elif car_label == "4":
            shot_start = np.array([270000,265500,262000,256500,252000,248000,242500,238500,234000,228500,224500])
            shot_end = np.array([277500,271500,266500,262000,257000,252000,246500,242500,237500,233500,229500])
        else:
            shot_start = np.array([332500,330600,324000,319000,315000,312000,306000,302000,297000,293500,289500])
            shot_end = np.array([338500,331000,328000,323000,319000,315000,310500,306500,303000,298500,293500])
    elif file_index == "024":
        if car_label == "1":
            shot_start = np.array([20000,22500,24500,29000,32000,36500,38500,42500,46500,49000])
            shot_end = np.array([25000,26000,30000,33000,36000,40000,43000,46500,49500,52500])
        elif car_label == "2":
            bins_clean = np.array([300, 350, 400,450, 500,550, 600, 700, 750])
            shot_start = np.array([64000,68000,71500,74000,77500,81500,83500,91000,94000])
            shot_end = np.array([71000,71500,75000,79000,81500,84500,87500,94000,96500])
        elif car_label == "3":
            shot_start = np.array([109000,113000,116000,119500,123000,127500,130000,134000,138500,140000])
            shot_end = np.array([114000,116000,120500,123000,127500,131000,133500,137500,140500,144000])
        elif car_label == "4":
            bins_clean = np.array([300, 350, 400,450, 500, 600, 650, 700, 750])
            shot_start = np.array([158000,162000,165500,169000,172500,180000,184000,188500,192000])
            shot_end = np.array([164000,166500,170000,173000,177500,184000,188000,192000,196000])
        else:
            bins_clean = np.array([300, 350, 400, 500, 550,600, 650, 700, 750])
            shot_start = np.array([210500,214000,217000,223000,227500,230500,233500,237000,240000])
            shot_end = np.array([216000,218000,221500,227500,231000,233000,237000,240000,242500])
    elif file_index == "025":
        if car_label == "1":
            shot_start = np.array([43000,42000,38500,36000,33500,30500,28500,26000,23500,19500])
            shot_end = np.array([48000,44500,42000,40500,37500,35000,32500,29500,27500,24500])
        elif car_label == "2":
            shot_start = np.array([82000,80000,76500,74500,72000,68500,67000,63500,60000,58500])
            shot_end = np.array([86500,83500,81000,79000,76000,73500,71500,67000,65500,63000])
        elif car_label == "3":
            shot_start = np.array([126500,125500,122000,119500,117000,112500,111000,108000,105000,102500])
            shot_end = np.array([132500,129000,126000,123500,121500,119000,116000,113000,112000,107500])
        elif car_label == "4":
            shot_start = np.array([167000,165500,162000,160000,157000,154500,152000,149000,146000,143000])
            shot_end = np.array([172000,168500,166000,163500,162000,157500,156500,153000,151000,148000])
        else:
            shot_start = np.array([230000,228000,225000,222500,220000,216500,215000,211000,210000,207000])
            shot_end = np.array([235000,231500,228500,226500,224000,222000,219500,216000,214500,211000])
    elif file_index == "026":
        if car_label == "1":
            shot_start = np.array([0,2500,4500,6500,8500,10500,12500,15500,17000,19000])
            shot_end = np.array([3500,4500,7500,10000,11500,14000,16500,18000,20000,22500])
        elif car_label == "2":
            shot_start = np.array([53500,55500,57500,58500,61000,63500,65000,68000,70000,72500])
            shot_end = np.array([56000,57500,60000,62000,64000,67000,69000,71000,73000,76000])
        elif car_label == "3":
            shot_start = np.array([86500,89000,91500,93000,95500,98000,100500,103000,105500,107500])
            shot_end = np.array([90500,91500,94000,96000,98500,102000,103500,105500,108000,110000])
        elif car_label == "4":
            shot_start = np.array([122500,125500,128500,130000,132500,134000,136500,139500,141000,143500])
            shot_end = np.array([127500,128500,131000,133500,135000,138500,140000,142000,145500,147500])
        else:
            shot_start = np.array([168500,172000,174500,176000,178500,181000,182500,184500,187500,189500])
            shot_end = np.array([173500,174500,177000,179000,181000,183500,185000,188000,190000,193500])
    elif file_index == "027":
        if car_label == "1":
            shot_start = np.array([31000,29500,27500,26000,24000,22500,20500,18500,17000,15000,13500,11500])
            shot_end = np.array([36000,33500,31500,29000,27000,25500,24000,22500,20500,18500,16500,14000])
        elif car_label == "2":
            shot_start = np.array([61000,59000,57000,55500,53000,51500,50000,48000,47000,44000,43000,40000])
            shot_end = np.array([66000,62500,60500,58000,56500,54500,53000,51500,49500,48500,46000,44000])
        elif car_label == "3":
            shot_start = np.array([91000,89500,88000,86500,83500,82500,80500,78000,76500,74000,73000,70500])
            shot_end = np.array([96000,94000,91000,88550,87000,85000,83500,81500,79500,77500,75500,74000])
        elif car_label == "4":
            shot_start = np.array([123000,121000,119000,117500,115000,113500,111500,109000,108000,105000,103500,101500])
            shot_end = np.array([127500,125000,123000,120500,118500,116500,114500,113000,111000,109500,107000,105000])
        else:
            shot_start = np.array([343000,340000,338000,336500,334500,332500,331000,329000,327500,325000,323000,320000])
            shot_end = np.array([348000,344000,341500,339000,337000,335500,334000,332500,330500,328000,326500,324000])
    elif file_index == "058.5":
        if car_label == "1":
            if direction == "west":
                shot_start = np.array([8000,10000,13000,15000,17000,19500,20500,23000,25000,27500])
                shot_end = np.array([12000,13000,15000,18500,20000,22000,23500,26500,28500,30000])
            else:
                shot_start = np.array([76000,74450,70000,68500,67500,65000,63500,61000,59500,57500])
                shot_end = np.array([79550,77000,74450,72500,71000,68500,67000,64500,62500,59500])
                
        elif car_label == "2":
            if direction == "west":
                bins_clean = np.array([300,350,410,460,650,703,750])
                shot_start = np.array([103500,106500,109000,111000,119000,121000,123500])
                shot_end = np.array([107500,109000,112000,114500,122500,124500,127000])
            else:
                shot_start = np.array([169000,167500,164500,162550,161000,158500,157000,154500,152500,150500])
                shot_end = np.array([173500,170000,167500,165500,164000,162000,160000,157500,156000,153500])
        
        elif car_label == "3":
            if direction == "west":
                shot_start = np.array([208000,211000,213500,215500,216500,219500,221000,223500,225500,227500])
                shot_end = np.array([212000,213500,216000,218500,220000,222500,224000,226500,228500,230500])
            else:
                shot_start = np.array([269000,268000,265000,263000,261000,259000,257000,254500,253000,250500])
                shot_end = np.array([274000,270500,268000,266000,264500,262500,260500,258000,256000,254000])
        elif car_label == "4":
            if direction == "west":
                shot_start = np.array([301000,303500,305500,307500,309000,311500,313500,316000,318500,320500])
                shot_end = np.array([305000,306000,309000,311500,313000,315000,317000,319500,322000,324000])
            else:
                shot_start = np.array([361000,359500,357000,355000,353500,351000,349500,347000,345000,343500])
                shot_end = np.array([366000,363000,360000,358500,357000,355000,352500,350000,348500,346000])
        else: #car_label == "5"
            if direction == "west":
                shot_start = np.array([408000,410550,413550,415550,417000,419000,420500,423500,425500,427500])
                shot_end = np.array([412000,413500,415500,419000,420000,422000,423500,426500,428500,430500])
            else:
                bins_clean = np.array([460,600,650,703,750])
                shot_start = np.array([492500,487000,485000,482500,480500])
                shot_end = np.array([495500,490500,488500,486500,484000])
    elif file_index == "058":
        if car_label == "1":
            if direction == "west":
                    shot_start = np.array([17500,20000,21000,23500,25000,26000,29000,31000])
                    shot_end = np.array([21000,23000,25000,26000,29000,31000,32500,34500])
            else:
                print("not support 058_5p east")
        else:
            print("no other car label other than 1: 5p")
    else:
        print('not support other dataset')
    
    if shot_start != []:
        if bins_clean !=[]:
            return bins_clean, shot_start, shot_end
        else:
            return shot_start, shot_end
    else:
        print('nothing to return')
        
        
        
def fit_linear(b, shot):
    order = 1
    w = np.polyfit(b, shot, order)
    p = np.poly1d(w)
    return p.c[0], p.c[1]

def get_a1b1_a2b2(b, shot_start, shot_end): #if noisy, bin will be fed bins_clean, ignore noisy window's (bin,  start_shot and end_shot)
    a1,b1 = fit_linear(b, shot_start)
    a2,b2 = fit_linear(b, shot_end)
    return a1, b1, a2, b2 #note this func doesn't delete any noisy signal but only produces linear coefficients for clean signals 

def get_shot_window(i, a1, b1, a2, b2):
    LR_shot_start = int(a1*i+b1)
    LR_shot_end=int(a2*i+b2)
    return LR_shot_start, LR_shot_end # LR prediction of shot start and shot end.



def labeling_shot_given_bin(car_label, b, a1, b1, a2, b2, output_labels):
    """
    ex, in a given bin,
        shot     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10...N (1 shot = 1/1000 sec)
        label    1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5....1..(we have 5 cars) 
    
    """
    
    LR_shot_start, LR_shot_end = get_shot_window(b, a1, b1, a2, b2)
    output_labels[b][LR_shot_start:LR_shot_end] = car_label
    return LR_shot_start, LR_shot_end, output_labels

def get_labels(output_labels, start_bin, end_bin, a1, b1, a2, b2, car_label, noisy_bin_start=-1, noisy_bin_end=-1):
    window_width_list = [] #collect all possible window width
    for b in range(start_bin, end_bin+1):
        if noisy_bin_start >=0 and noisy_bin_end>= 0:
            if b not in range(noisy_bin_start, noisy_bin_end):
                LR_shot_start, LR_shot_end, output_labels = labeling_shot_given_bin(car_label, b, a1, b1, a2, b2, output_labels)
                window_width_list.append(LR_shot_end-LR_shot_start+1)
            else:
                continue
        else:
            LR_shot_start, LR_shot_end, output_labels = labeling_shot_given_bin(car_label, b, a1, b1, a2, b2, output_labels)
            window_width_list.append(LR_shot_end-LR_shot_start+1)
    return output_labels, window_width_list


def check_slope(a1, a2, car_label, direction=None):
    try:
        if abs(a1-a2)>6.06:
            print('Window slopes are too different: ',  abs(a1-a2))
            print("car {0}".format(car_label))
            if direction != None:
                print("Direction {0}".format(direction))
            
    except:
        raise SystemExit("two LR slopes are too diverse!")

        



def padding_symmetrically(i, data, start_shot, end_shot, biggest):
    window_width = end_shot-start_shot+1
    window = data[i][start_shot:end_shot+1]
    output = ""
    for s in window[:-1]:
        output += str(s) + ' '
    output += str(window[-1])

    if window_width<biggest: #do padding if window width< biggest  
        diff = biggest - window_width
        if diff%2 == 0:
            padding_1 = int(diff/2)*(str(0)+ ' ')
            padding_2 =  (int(diff/2)-1)*(str(0)+ ' ') + str(0)
            output = padding_1 + output + ' ' + padding_2
        else:
            diff = diff-1
            padding = int(diff/2)*(str(0)+ ' ')
            output = padding +output + ' ' + padding +str(0)
    
    return output


        
def export_data_v1(data, output_labels, start_bin, end_bin, biggest_window_len, output_fname_x, output_fname_y):
    car_unique_labels = np.unique(output_labels)
    car_unique_labels = np.delete(car_unique_labels, 0) # get rid of background noise label 
    with open (output_fname_x, "w") as f_x, open (output_fname_y, "w") as f_y:
        for i in range(start_bin, end_bin+1):
            for car_label in car_unique_labels:
                car_signal = [idx for idx, shot_label in enumerate(output_labels[i]) if shot_label==car_label]
                if car_signal != []:
                    start_shot, end_shot = min(car_signal), max(car_signal)
                    output= padding_symmetrically(i, data, start_shot, end_shot, biggest_window_len)
                    f_x.write(output+"\n")
                    f_y.write(str(car_label)+"\n")
                else:
                    continue

                    
def export_data_v2(data, output_labels, start_bin, end_bin, biggest_window_len, output_fname_x, output_fname_y): #only support two directions data 058
    unique_labels = np.unique(output_labels) # include noise which has label 0 
    num_shots = len(output_labels[start_bin]) #total shots
    unique_labels = np.delete(unique_labels, 0) # get rid of background noise label 
    with open (output_fname_y, "w") as f_y, open (output_fname_x, "w") as f_x:
            for i in range(start_bin, end_bin+1):
#                 for car_label in range(1,(len(unique_labels)-1)+1): # -1 to get rid of noise label and +1 because range(a, b) only iterate to b-1 
                for car_label in unique_labels:
                    car_label = int(car_label)
                    is_window_1 = False
                    is_window_2 = False
                    
                    for s in range(num_shots):
                        if output_labels[i][s] == car_label:
                            start_shot_1 = s
                            is_window_1 = True
                            break # get the first shot and out from this loop
                    for s in range(start_shot_1 + 1,num_shots): # iterate from the shot after start shot 
                        if output_labels[i][s] != car_label: #if the new shot has differnt label from current one
                            end_shot_1 = s-1 #get end shot of current label
                            break #out from this loop
                    for s in range(end_shot_1+1,num_shots):
                        if output_labels[i][s] == car_label:
                            start_shot_2 = s
                            is_window_2 = True
                            break # get the first shot of the second window for this car and out from this loop
                    for s in range(start_shot_2 + 1,num_shots): # iterate from the shot after start shot 
                        if output_labels[i][s] != car_label: #if the new shot has differnt label from current one
                            end_shot_2 = s-1 #get end shot of current label
                            break #out from this loop
                    
                    #if both windows dpn't exist, quite the program
                    if is_window_1 == False and is_window_2 == False:
                        print("can't find a window with car {0} in bin {1} and shot {2}".format(car_label, i, s))
                        exit()
                    
                    #write to  x and y
                    if is_window_1:
                        f_y.write(str(car_label)+"\n")
                        output_1 = padding_symmetrically(i, data, start_shot_1, end_shot_1, biggest_window_len)
                        f_x.write(output_1+"\n")
                    
                    if is_window_2:
                        f_y.write(str(car_label)+"\n")
                        output_2 = padding_symmetrically(i, data, start_shot_2, end_shot_2, biggest_window_len)
                        f_x.write(output_2+"\n")                    






