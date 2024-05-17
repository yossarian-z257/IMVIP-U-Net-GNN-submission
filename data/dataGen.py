################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
import utils

samples           = 100           # no. of datasets to produce
freestream_angle  = math.pi / 8.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
freestream_length_factor = 10.    # length factor

airfoil_database  = "./airfoil_database_test/"
output_dir        = "./test/"

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

def genMesh(airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)

    if os.system("gmsh airfoil.geo -3 -format msh2 -o airfoil.msh > /dev/null") != 0:
        print("error during mesh creation!")
        return(-1)

    if os.system("gmshToFoam airfoil.msh > /dev/null") != 0:
        print("error during conversion to OpenFoam mesh!")
        return(-1)

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

    return(0)

def runSim(freestreamX, freestreamY):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    os.system("./Allclean && simpleFoam > foam.log")

# def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, pfile='OpenFOAM/postProcessing/internalCloud/500/cloud_p.xy', ufile='OpenFOAM/postProcessing/internalCloud/500/cloud_U.xy', res=128, imageIndex=0):
#     # output layout channels:
#     # [0] freestream field X + boundary
#     # [1] freestream field Y + boundary
#     # [2] binary mask for boundary
#     # [3] pressure output
#     # [4] velocity X output
#     # [5] velocity Y output
#     npOutput = np.zeros((6, res, res))


#     pfile = os.path.join(os.getcwd(), pfile)  # Construct absolute path
#     ufile = os.path.join(os.getcwd(), ufile)  # Construct absolute path


#     ar = np.loadtxt(pfile)
#     curIndex = 0

#     for y in range(res):
#         for x in range(res):
#             xf = (x / res - 0.5) * 2 + 0.5
#             yf = (y / res - 0.5) * 2
#             if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
#                 npOutput[3][x][y] = ar[curIndex][3]
#                 curIndex += 1
#                 # fill input as well
#                 npOutput[0][x][y] = freestreamX
#                 npOutput[1][x][y] = freestreamY
#             else:
#                 npOutput[3][x][y] = 0
#                 # fill mask
#                 npOutput[2][x][y] = 1.0

#     ar = np.loadtxt(ufile)
#     curIndex = 0

#     for y in range(res):
#         for x in range(res):
#             xf = (x / res - 0.5) * 2 + 0.5
#             yf = (y / res - 0.5) * 2
#             if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
#                 npOutput[4][x][y] = ar[curIndex][3]
#                 npOutput[5][x][y] = ar[curIndex][4]
#                 curIndex += 1
#             else:
#                 npOutput[4][x][y] = 0
#                 npOutput[5][x][y] = 0

#     utils.saveAsImage('data_pictures/pressure_%04d.png'%(imageIndex), npOutput[3])
#     utils.saveAsImage('data_pictures/velX_%04d.png'  %(imageIndex), npOutput[4])
#     utils.saveAsImage('data_pictures/velY_%04d.png'  %(imageIndex), npOutput[5])
#     utils.saveAsImage('data_pictures/inputX_%04d.png'%(imageIndex), npOutput[0])
#     utils.saveAsImage('data_pictures/inputY_%04d.png'%(imageIndex), npOutput[1])

#     #fileName = dataDir + str(uuid.uuid4()) # randomized name
#     fileName = dataDir + "%s_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100) )
#     print("\tsaving in " + fileName + ".npz")
#     np.savez_compressed(fileName, a=npOutput)


# def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, file='OpenFOAM/postProcessing/internalCloud/500/cloud.xy', res=128, imageIndex=0):
#     # Ensure the output directory exists
#     # makeDirs([dataDir])

#     npOutput = np.zeros((6, res, res))
#     filepath = os.path.join(os.getcwd(), file)
#     data = np.loadtxt(filepath)
#     curIndex = 0
    
#     for y in range(res):
#         for x in range(res):
#             xf = (x / res - 0.5) * 2 + 0.5
#             yf = (y / res - 0.5) * 2
#             if abs(data[curIndex][1] - xf) < 1e-4 and abs(data[curIndex][2] - yf) < 1e-4:
#                 npOutput[3][x][y] = data[curIndex][4]  # Pressure
#                 npOutput[4][x][y] = data[curIndex][5]  # Velocity X
#                 npOutput[5][x][y] = data[curIndex][6]  # Velocity Y
#                 curIndex += 1
#             else:
#                 npOutput[0][x][y] = freestreamX
#                 npOutput[1][x][y] = freestreamY
#                 npOutput[2][x][y] = 1.0

#     # Save outputs using updated file paths
#     utils.saveAsImage(f'{dataDir}/pressure_{imageIndex:04d}.png', npOutput[3])
#     utils.saveAsImage(f'{dataDir}/velX_{imageIndex:04d}.png', npOutput[4])
#     utils.saveAsImage(f'{dataDir}/velY_{imageIndex:04d}.png', npOutput[5])

#     fileName = os.path.join(dataDir, f"{basename}_{int(freestreamX*100)}_{int(freestreamY*100)}")
#     print("\tsaving in " + fileName + ".npz")
#     np.savez_compressed(fileName, a=npOutput)
# def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, file='OpenFOAM/postProcessing/internalCloud/500/cloud.xy', res=128, imageIndex=0):
#     # makeDirs([dataDir])
#     print(output_dir)
#     npOutput = np.zeros((6, res, res))
#     filepath = os.path.join(os.getcwd(), file)
#     data = np.loadtxt(filepath)
#     curIndex = 0
#     match_count = 0

#     for y in range(res):
#         for x in range(res):
#             xf = (x / res - 0.5) * 2 + 0.5
#             yf = (y / res - 0.5) * 2
#             if abs(data[curIndex][1] - xf) < 1e-4 and abs(data[curIndex][2] - yf) < 1e-4:
#                 npOutput[3][x][y] = data[curIndex][4]
#                 npOutput[4][x][y] = data[curIndex][5]
#                 npOutput[5][x][y] = data[curIndex][6]
#                 curIndex += 1
#                 match_count += 1
#             else:
#                 npOutput[0][x][y] = freestreamX
#                 npOutput[1][x][y] = freestreamY
#                 npOutput[2][x][y] = 1.0

#     print(f"Total matches found: {match_count} out of {res*res}")

#     utils.saveAsImage(f'{dataDir}pressure_{imageIndex:04d}.png', npOutput[3])
#     utils.saveAsImage(f'{dataDir}velX_{imageIndex:04d}.png', npOutput[4])
#     utils.saveAsImage(f'{dataDir}velY_{imageIndex:04d}.png', npOutput[5])

#     fileName = os.path.join(dataDir, f"{basename}_{int(freestreamX*100)}_{int(freestreamY*100)}")
#     print("\tsaving in " + fileName + ".npz")
#     np.savez_compressed(fileName, a=npOutput)
def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, res=128, imageIndex=0): 
    # Define file paths for the pressure and velocity data at different timesteps
    initialFile = '/home/melkor/projects/fluid_stuff/Deep-Flow-Prediction/data/OpenFOAM/postProcessing/internalCloud/0/cloud.xy'#os.path.join(dataDir, f"internalCloud/0/cloud.xy")
    finalFile = '/home/melkor/projects/fluid_stuff/Deep-Flow-Prediction/data/OpenFOAM/postProcessing/internalCloud/500/cloud.xy'
    
    # Ensure the directory for saving images exists
    os.makedirs('data_pictures', exist_ok=True)
    
    # Initialize the output data array
    npOutput = np.zeros((6, res, res))

    # Load data from the initial and final files
    data_initial = np.loadtxt(initialFile)
    data_final = np.loadtxt(finalFile)
    
    # Process the initial and final data separately if needed
    # Example processing for final data
    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            
            # Find the closest data point in the loaded data
            distances = np.sqrt((data_final[:, 0] - xf)**2 + (data_final[:, 1] - yf)**2)
            nearest_index = np.argmin(distances)
            
            if distances[nearest_index] < 0.01:  # Check if the nearest data point is close enough to consider
                npOutput[3][y][x] = data_final[nearest_index, 3]  # Pressure
                npOutput[4][y][x] = data_final[nearest_index, 4]  # U_x
                npOutput[5][y][x] = data_final[nearest_index, 5]  # U_y
            else:
                npOutput[0][y][x] = freestreamX
                npOutput[1][y][x] = freestreamY
                npOutput[2][y][x] = 1.0  # Mask for boundary or no data

    # Saving images
    utils.saveAsImage(f'data_pictures/pressure_{imageIndex:04d}.png', npOutput[3])
    utils.saveAsImage(f'data_pictures/velX_{imageIndex:04d}.png', npOutput[4])
    utils.saveAsImage(f'data_pictures/velY_{imageIndex:04d}.png', npOutput[5])
    utils.saveAsImage(f'data_pictures/inputX_{imageIndex:04d}.png', npOutput[0])
    utils.saveAsImage(f'data_pictures/inputY_{imageIndex:04d}.png', npOutput[1])

    # Save numpy array
    fileName = os.path.join(dataDir, f"{basename}_{int(freestreamX*100)}_{int(freestreamY*100)}")
    print(f"\tsaving in {fileName}.npz")
    np.savez_compressed(fileName, a=npOutput)





files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

utils.makeDirs( ["./data_pictures", "./train", "./test", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"] )


# main
for n in range(samples):
    print("Run {}:".format(n))

    fileNumber = np.random.randint(0, len(files))
    basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]
    print("\tusing {}".format(files[fileNumber]))

    length = freestream_length * np.random.uniform(1.,freestream_length_factor)
    angle  = np.random.uniform(-freestream_angle, freestream_angle)
    fsX =  math.cos(angle) * length
    fsY = -math.sin(angle) * length

    print("\tUsing len %5.3f angle %+5.3f " %( length,angle )  )
    print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))

    os.chdir("./OpenFOAM/")
    if genMesh("../" + airfoil_database + files[fileNumber]) != 0:
        print("\tmesh generation failed, aborting");
        os.chdir("..")
        continue

    runSim(fsX, fsY)
    os.chdir("..")

    outputProcessing(basename, fsX, fsY, imageIndex=n)
    print("\tdone")
