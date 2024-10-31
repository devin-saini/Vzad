import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import nlopt
import glob2

yes = ["y", "yes", "Y", "YES", "Yes"]
no = ["n", "no", "N", "NO", "No"]
class SimResults:
    def __init__(self, Sim, SStatesFinal):
        self.Sim = Sim
        self.SStatesFinal = SStatesFinal

while True:
    print("Input folder name, without extension")
    analysis_name = input()

    print("Enter 1 for FRR and 2 for O2 data")
    analysisType = input()

    print("Enter 1 for COBYLA, 2 for BOBYQA, 3 for Sbplx")
    algorithm = input()

    if analysisType == "1":
        filename = "FRRdata.txt"
        os.chdir(analysis_name)
        O2ExpFile = open(filename)
        C = O2ExpFile.read().split("\n")
        if "\t" in C[0]:
            O2ExpOriginal = [float(C[i].split("\t")[1]) for i in range(len(C))]
        else:
            O2ExpOriginal = [float(C[i]) for i in range(len(C))]
        O2ExpFile.close()
        # normalize Experimental Data
        minVal = min([O2ExpOriginal[2], O2ExpOriginal[3]])
        O2ExpReduced = [O2ExpOriginal[i] - minVal for i in range(len(O2ExpOriginal))]
        avg = sum(O2ExpReduced) / len(O2ExpReduced)
        O2ExpNormalized = [O2ExpReduced[i] / avg for i in range(len(O2ExpReduced))]
        filename_norm = "FRRdata_normalized" + algorithm + ".txt"
        O2ExpNormFile = open(filename_norm, "w")
        for i in range(len(O2ExpNormalized)):
            O2ExpNormFile.write(str(i + 1) + "\t" + str(O2ExpNormalized[i]) + "\n")
        O2ExpNormFile.close()
        os.chdir("..")
    elif analysisType == "2":
        filename = "O2data.txt"
        os.chdir(analysis_name) #change directory to analysis_name
        O2ExpFile = open(filename) #make variable equal open file
        C = O2ExpFile.read().split("\n") #C = read file and turn into list,
        if "\t" in C[0]: #if first value has "\t", checks to see how flash # is seperated by flash time
            O2ExpOriginal = [float(C[i].split("\t")[1]) for i in range(len(C))] #make list of time of each flash as float
        elif " " in C[0]: #if first value has " " (space)
            O2ExpOriginal = [float(C[i].split(" ")[1]) for i in range(len(C))] #make list of time of each flash as float
        else: #if anything else
            O2ExpOriginal = [float(C[i]) for i in range(len(C))] #make each time of flash float value
        O2ExpFile.close() #close the file

        # normalize Experimental Data
        minVal = min([O2ExpOriginal[0], O2ExpOriginal[1]]) #find lowest value in O2ExpOriginal list for each column
        O2ExpReduced = [O2ExpOriginal[i] - minVal for i in range(len(O2ExpOriginal))] #make list of reduced values
        avg = sum(O2ExpReduced) / len(O2ExpReduced) #find average reduced value
        O2ExpNormalized = [O2ExpReduced[i] / avg for i in range(len(O2ExpReduced))] #make list of normalized values
        filename_norm = "O2data_normalized.txt" #assign variable to normalized file
        O2ExpNormFile = open(filename_norm, "w") #assign variable to open file and write in it
        for i in range(len(O2ExpNormalized)): #make text file of normalized values
            O2ExpNormFile.write(str(i + 1) + "\t" + str(O2ExpNormalized[i]) + "\n")
        O2ExpNormFile.close() #close file
        os.chdir("..") #change directory to parent directory

    print("Total number of flashes:" + str(len(O2ExpNormalized)) + ". Enter Number Of Flashes")
    nFlashesTot = int(input())

    print("Do you want to split the data set in two? This option will calculate two set of inefficiency parameters, and one initial S-state distribution")
    splitDataStr = input()

    print ("Do you want to put comments in the parameter file?")
    comments_param = input()

    splitData = -1
    while splitData == -1: #while split data is not answered
        if splitDataStr in yes: #if split data
            splitData = 1
            print("Please enter the number of pulses in the first part")
            nFlashes1 = int(input())
            nflashes2 = nFlashesTot - nFlashes1
        elif splitDataStr in no: #if don't split data
            splitData = 0
            nFlashes = nFlashesTot
        else:
            splitDataStr = input()

    def En(x, grad, Exp, analysisType): #x=file values, at line 244
        A = np.zeros((5, 5)) #make 5 by 5 array (table) with 0's

        # change values of 1st row
        A[0, 0] = x[0]
        A[0, 1] = 0
        A[0, 2] = x[1] - x[1] * x[3]
        A[0, 3] = 1 - x[0] - x[1] - x[2] - x[3]
        A[0, 4] = 0

        #change values of 2nd row
        A[1, 0] = 1 - x[0] - x[1]
        A[1, 1] = x[0]
        A[1, 2] = x[2]
        A[1, 3] = x[1] - x[1] * x[3]
        A[1, 4] = 0

        #change values of 3rd row
        A[2, 0] = x[1]
        A[2, 1] = 1 - x[0] - x[1]
        A[2, 2] = x[0]
        A[2, 3] = x[3]
        A[2, 4] = 0

        #change values of 4th row
        A[3, 0] = 0
        A[3, 1] = x[1]
        A[3, 2] = 1 - x[0] - x[1] - x[2]
        A[3, 3] = x[0]
        A[3, 4] = 0

        #change values of 5th row
        A[4, 0] = 0
        A[4, 1] = 0
        A[4, 2] = x[1] * x[3]
        A[4, 3] = x[3] + x[1] * x[3]
        A[4, 4] = 1

        S = x[4:] #assign s to be array of all values after 4th index
        En = 0
        for i in range(len(Exp)):
            if analysisType == "2": #if O2 data
                En += ((1 - x[0] - x[2] - x[3]) * S[3] + x[1] * S[2] - Exp[i]) ** 2 #calculate energy value
                # print("Energy: ", str(En))
            elif analysisType == "1": #if FRR data
                En += (x[0] * S[1] + (1 - x[0] - x[1]) * S[0] + x[1] * (1 - x[3]) * S[3] + x[3] * S[2] - Exp[i]) ** 2 #calculate enrgy value
                #print("Energy: ", str(En))
            S = np.dot(A, S) #dot product of array A * S

        # print("Energy: ", str(En))
        return En.astype(float)


    def EnSplit(x, grad, Exp1, Exp2, analysisType): #ran if choose to split data
        x1 = [x[i] for i in range(9)]
        En1 = En(x1, [], Exp1, analysisType)
        Sim1 = forwardSimulation(x1, Exp1, analysisType)
        S02 = Sim1.SStatesFinal
        x2 = [x[i] for i in range(9, len(x))]
        for i in range(len(S02)):
            x2.append(S02[i])
        En2 = En(x2, [], Exp2, analysisType)
        return En1 + En2


    def forwardSimulation(x, Exp, analysisType):
        A = np.zeros((5, 5)) #make 5 by 5 array of 0s

        #Change values of 1st row
        A[0, 0] = x[0]
        A[0, 1] = 0
        A[0, 2] = x[1] - x[1] * x[3]
        A[0, 3] = 1 - x[0] - x[1] - x[2] - x[3]
        A[0, 4] = 0

        #change values of 2nd row
        A[1, 0] = 1 - x[0] - x[1]
        A[1, 1] = x[0]
        A[1, 2] = x[2]
        A[1, 3] = x[1] - x[1] * x[3]
        A[1, 4] = 0

        #change values of 3rd row
        A[2, 0] = x[1]
        A[2, 1] = 1 - x[0] - x[1]
        A[2, 2] = x[0]
        A[2, 3] = x[3]
        A[2, 4] = 0

        #change values of 4th row
        A[3, 0] = 0
        A[3, 1] = x[1]
        A[3, 2] = 1 - x[0] - x[1] - x[2]
        A[3, 3] = x[0]
        A[3, 4] = 0

        #change values of 5th row
        A[4, 0] = 0
        A[4, 1] = 0
        A[4, 2] = x[1] * x[3]
        A[4, 3] = x[3] + x[1] * x[3]
        A[4, 4] = 1

        S = x[4:] #assign s to be array of all values after 4th index of x
        Sim = [] #make empty list
        for i in range(len(Exp)):
            #make simulated data based on analysis type
            if analysisType == "2": #O2 data
                Sim.append((1 - x[0] - x[2] - x[3]) * S[3] + x[1] * S[2])
                # print("Energy: ", str(En))
            elif analysisType == "1":
                Sim.append(x[0] * S[1] + (1 - x[0] - x[1]) * S[0] + x[1] * (1 - x[3]) * S[3] + x[3] * S[2])
                # print("Energy: ", str(En))
            S = np.dot(A, S) #S = dot product of array A * S
        results = SimResults(Sim, S)
        return results

    def constr(x, grad):
        return x[0] + x[1] + x[2] + x[3] - 1

    def constr2(x, grad):
        return x[9] + x[10] + x[11] + x[12] - 1

    # if analysisType == '2':
    #    O2Exp = [O2ExpNormalized[i] for i in range(nFlashesTot)]
    # elif analysisType == '1':
    #    O2Exp = [O2ExpOriginal[i] for i in range(nFlashesTot)]

    O2Exp = [O2ExpNormalized[i] for i in range(nFlashesTot)]
    if splitData == 0:
        if algorithm == "1":
            opt = nlopt.opt(nlopt.LN_COBYLA, 9)
            # opt.add_inequality_constraint(constr, 1e-8)
            alg = "COBYLA"
        elif algorithm == "2":
            opt = nlopt.opt(nlopt.LN_BOBYQA, 9)
            alg = "BOBYQA"
        elif algorithm == "3":
            opt = nlopt.opt(nlopt.LN_SBPLX, 9)
            alg = "SBPLX"

        params0 = np.array([0, 0.1, 0.1, 0.1])
        S0 = np.array([1, 1, 1, 1, 1])
        x0 = np.concatenate((params0, S0))
        lb = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        ub = np.array([1, 1, 1, 1, 1000, 1000, 1000, 1000, 0])
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_ftol_rel(1e-10)
        opt.set_xtol_rel(1e-10)
        opt.set_initial_step(0.01)
        opt.set_min_objective(lambda x, grad: En(x, grad, O2Exp, analysisType))
        # opt.set_min_objective(testFunc)
        # x0 = [0.1,0.1,0.1,0.1,1,1,1,1,1]
        opt.maxeval = 10000
        x = opt.optimize(x0) #this is x

        paramsOpt = x[0:4]
        Sopt = x[4:]
        SoptNorm = Sopt / (Sopt.sum())

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        SimOpt = forwardSimulation(x, O2Exp, analysisType)
        residual = [SimOpt.Sim[i] - O2Exp[i] for i in range(len(SimOpt.Sim))]

        ax1.plot(range(len(SimOpt.Sim)), SimOpt.Sim, label="Simulated")
        # ax1.plot(range(len(simRes.O2)),simRes.O2)
        ax1.plot(range(len(O2Exp)), O2Exp, label="Experimental")
        ax1.scatter(range(len(SimOpt.Sim)), SimOpt.Sim, s=6)
        ax1.scatter(range(len(O2Exp)), O2Exp, s=6)
        ax1.plot(range(len(residual)), residual, label="Simulated - Experimental")
        ax1.scatter(range(len(residual)), residual, s=6)
        ax1.legend()
        plt.title("VZAD python")

        os.chdir(analysis_name)

        residual_file = open ("residual" + alg + str(nFlashesTot) + ".txt", "w")
        for n in range (len(residual)):
            residual_file.write(str(n+1) + "\t" + str(residual[n]) + "\n")
        residual_file.close()

        fft_points = len(O2ExpNormalized)
        x = np.linspace(0, fft_points, endpoint=False)  #x axis indicators
        x_fft = fftfreq(fft_points)[:fft_points // 2]  #make x axis to be half of number of values
        y_fft = fft(O2ExpNormalized)  # fft on data
        with open("FFT" + str(nFlashesTot) + ".txt", "w") as f:
            for n in range(1,len(x_fft)):
                f.write(f"{x_fft[n]} \t {y_fft[n]}\n")

        fig1.savefig("figVZADPython" + alg + str(nFlashesTot) + ".png", dpi=300)

        fileSim = open("O2simulated" + alg + str(nFlashesTot) + ".txt", "w")
        for i in range(len(SimOpt.Sim)):
            fileSim.write(str(i + 1) + "\t" + str(SimOpt.Sim[i]) + "\n")
        fileSim.close()

        fileParams = open("params" + alg + str(nFlashesTot) + ".txt", "w")
        fileParams.write("alpha\t" + str(paramsOpt[0]) + "\n")
        fileParams.write("beta\t" + str(paramsOpt[1]) + "\n")
        fileParams.write("delta\t" + str(paramsOpt[2]) + "\n")
        fileParams.write("epsilon\t" + str(paramsOpt[3]) + "\n")

        fileParams.write("S0norm\t" + str(SoptNorm[0]) + "\n")
        fileParams.write("S1norm\t" + str(SoptNorm[1]) + "\n")
        fileParams.write("S2norm\t" + str(SoptNorm[2]) + "\n")
        fileParams.write("S3norm\t" + str(SoptNorm[3]) + "\n")
        fileParams.write("Senorm\t" + str(SoptNorm[4]) + "\n")

        fileParams.write("S0\t" + str(Sopt[0]) + "\n")
        fileParams.write("S1\t" + str(Sopt[1]) + "\n")
        fileParams.write("S2\t" + str(Sopt[2]) + "\n")
        fileParams.write("S3\t" + str(Sopt[3]) + "\n")
        fileParams.write("Senorm\t" + str(SoptNorm[4]) + "\n")
        fileParams.write ("Q Value with epsilon\t" + str (1 / (sum(paramsOpt))) + "\n")
        fileParams.write("Q Value without epsilon\t" + str (1 / (sum(paramsOpt[:3]))) + "\n")

        fileParams.close()
    elif splitData == 1:
        O2Exp1 = [O2Exp[i] for i in range(nFlashes1)]
        O2Exp2 = [O2Exp[i] for i in range(nFlashes1, nFlashesTot)]
        if algorithm == "1":
            opt = nlopt.opt(nlopt.LN_COBYLA, 13)
            # opt.add_inequality_constraint(constr, 1e-8)
            # opt.add_inequality_constraint(constr2, 1e-8)
            alg = "COBYLA"
        elif algorithm == "2":
            opt = nlopt.opt(nlopt.LN_BOBYQA, 13)
            alg = "BOBYQA"
        elif algorithm == "3":
            opt = nlopt.opt(nlopt.LN_SBPLX, 13)
            alg = "SBPLX"

        params01 = np.array([0.1, 0.1, 0.1, 0.1])
        params02 = np.array([0.1, 0.1, 0.1, 0.1])
        S0 = np.array([1, 1, 1, 1, 1])
        x0 = np.concatenate((params01, S0, params02))
        lb = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ub = np.array([1, 1, 1, 1, 1000, 1000, 1000, 1000, 0, 1, 1, 1, 1])
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_ftol_rel(1e-10)
        opt.set_xtol_rel(1e-10)
        opt.set_initial_step(0.01)
        opt.set_min_objective(lambda x, grad: EnSplit(x, grad, O2Exp1, O2Exp2, analysisType))
        # opt.set_min_objective(testFunc)
        # x0 = [0.1,0.1,0.1,0.1,1,1,1,1,1]
        opt.maxeval = 10000
        x = opt.optimize(x0)

        paramsOpt1 = x[0:4]
        Sopt = x[4:9]
        paramsOpt2 = x[9:]
        SoptNorm = Sopt / (Sopt.sum())

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        x1opt = x[0:9]
        Sim1 = forwardSimulation(x1opt, O2Exp1, analysisType)
        x2opt = np.concatenate((paramsOpt2, Sim1.SStatesFinal))
        Sim2 = forwardSimulation(x2opt, O2Exp2, analysisType)

        SimTot = Sim1.Sim + Sim2.Sim

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        residual1 = [Sim1.Sim[i] - O2Exp1[i] for i in range(len(Sim1.Sim))]
        residual2 = [Sim2.Sim[i] - O2Exp2[i] for i in range(len(Sim2.Sim))]

        ax1.plot(range(len(Sim1.Sim)), Sim1.Sim, label="Simulated 1")
        # ax1.plot(range(len(simRes.O2)),simRes.O2)
        ax1.plot(range(len(O2Exp1)), O2Exp1, label="Experimental 1")
        ax1.scatter(range(len(Sim1.Sim)), Sim1.Sim, s=6)
        ax1.scatter(range(len(O2Exp1)), O2Exp1, s=6)
        ax1.plot(range(len(residual1)), residual1, label="Simulated 1 - Experimental 1")
        ax1.scatter(range(len(residual1)), residual1, s=6)

        ax1.plot(range(nFlashes1, nFlashesTot), Sim2.Sim, label="Simulated 2")
        # ax1.plot(range(len(simRes.O2)),simRes.O2)
        ax1.plot(range(nFlashes1, nFlashesTot), O2Exp2, label="Experimental 2")
        ax1.scatter(range(nFlashes1, nFlashesTot), Sim2.Sim, s=6)
        ax1.scatter(range(nFlashes1, nFlashesTot), O2Exp2, s=6)
        ax1.plot(range(nFlashes1, nFlashesTot), residual2, label="Simulated 2 - Experimental 2")
        ax1.scatter(range(nFlashes1, nFlashesTot), residual2, s=6)
        ax1.legend()
        plt.title("VZAD python")

        os.chdir(analysis_name)
        fig1.savefig("figVZADPython" + alg + "Split" + str(nFlashes1) + "_" + str(nFlashesTot - nFlashes1) + ".png",
                     dpi=300)

        fileSim = open("O2simulated" + alg + ".txt", "w")
        for i in range(len(SimTot)):
            fileSim.write(str(i + 1) + "\t" + str(SimTot[i]) + "\n")
        fileSim.close()

        fileParams = open("params" + alg + ".txt", "w")
        fileParams.write("alpha1\t" + str(paramsOpt1[0]) + "\n")
        fileParams.write("beta1\t" + str(paramsOpt1[1]) + "\n")
        fileParams.write("delta1\t" + str(paramsOpt1[2]) + "\n")
        fileParams.write("epsilon1\t" + str(paramsOpt1[3]) + "\n")
        fileParams.write("S0norm\t" + str(SoptNorm[0]) + "\n")
        fileParams.write("S1norm\t" + str(SoptNorm[1]) + "\n")
        fileParams.write("S2norm\t" + str(SoptNorm[2]) + "\n")
        fileParams.write("S3norm\t" + str(SoptNorm[3]) + "\n")
        fileParams.write("Senorm\t" + str(SoptNorm[4]) + "\n")
        fileParams.write("alpha2\t" + str(paramsOpt2[0]) + "\n")
        fileParams.write("beta2\t" + str(paramsOpt2[1]) + "\n")
        fileParams.write("delta2\t" + str(paramsOpt2[2]) + "\n")
        fileParams.write("epsilon2\t" + str(paramsOpt2[3]) + "\n")
        fileParams.close()

    if comments_param in yes:
        params = glob2.glob('params' + alg + str(nFlashesTot) + '*')
        with open(params[0], 'a') as file:
            file.write("Comment:" + input('Enter comments: '))

    again = str(input('Do you want to run another file?\n'))
    os.chdir('..')
    if again in no:
        break
