from pickle import FALSE
import time
import os
import random
import math
from tokenize import Double
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
board = [[[[" "," "," "],[" "," "," "],[" "," "," "]] for i in range(3)] for j in range(3)]
currentBoard = [0,0]
bigBoard = [[" "," "," "],[" "," "," "],[" "," "," "]]
gen = 70
acted = False
playergo = ["",""]
population_size = 40
def prepicked(playergo):
    if(board[currentBoard[0]][currentBoard[1]][playergo[0]-1][playergo[1]-1] == " "):
        return True
    else:
        return False
def prepickedai(ai):
    if(board[currentBoard[0]][currentBoard[1]][ai[0]][ai[1]] == " "):
        return True
    else:
        return False
def prepickedGlobal(ai):
    if(bigBoard[ai[0]][ai[1]] == " "):
        return True
    else:
        return False
def highest(nextMove):
    high = float(-1)
    winner = 0
    for x in range(9):
        for i in range(9):
            if(nextMove[0][i]>=high):
                high = nextMove[0][i]
                winner = i
        if(prepickedai(ToBoard(winner))==False):
            nextMove[0][winner]=-1
            high = float(-1)
            winner = 0
        else:
            return ToBoard(winner)

def ToBoard(i):
    if(i==0):
        return [0,0]
    if(i==1):
        return [0,1]
    if(i==2):
        return [0,2]
    if(i==3):
        return [1,0]
    if(i==4):
        return [1,1]
    if(i==5):
        return [1,2]
    if(i==6):
        return [2,0]
    if(i==7):
        return [2,1]
    if(i==8):
        return [2,2]
def flattenBoarda(myx, myy):#turns the current Board into input for a NN
    newBoard = [float(0) for x in range(18)]
    for i in range(3):
        for j in range(3):
            if(board[myx][myy][i][j] == "X"):
                newBoard[i*3+j] = 1
            elif(board[myx][myy][i][j] == "O"):
                newBoard[i*3+j+9] = 1
    return np.array([newBoard]).astype('float32')
def flattenBoardb(myx, myy):#turns the current Board into input for a NN
    newBoard = [float(0) for x in range(18)]
    for i in range(3):
        for j in range(3):
            if(board[myx][myy][i][j] == "X"):
                newBoard[i*3+j+9] = 1
            elif(board[myx][myy][i][j] == "O"):
                newBoard[i*3+j] = 1
    return np.array([newBoard]).astype('float32')

def AIMove():
    return False
def updateBigBoard():
    for i in range(3):
        for j in range(3):
            empty = False
            for k in range(3):
                for l in range(3):
                    if(board[i][j][k][l] == " "):
                        empty = True
            if(empty == False):
                bigBoard[i][j] = "T"
            if board[i][j][0][0] == "X" and board[i][j][0][1] == "X" and board[i][j][0][2] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][1][0] == "X" and board[i][j][1][1] == "X" and board[i][j][1][2] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][2][0] == "X" and board[i][j][2][1] == "X" and board[i][j][2][2] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][0][0] == "X" and board[i][j][1][0] == "X" and board[i][j][2][0] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][0][1] == "X" and board[i][j][1][1] == "X" and board[i][j][2][1] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][0][2] == "X" and board[i][j][1][2] == "X" and board[i][j][2][2] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][0][0] == "X" and board[i][j][1][1] == "X" and board[i][j][2][2] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][0][2] == "X" and board[i][j][1][1] == "X" and board[i][j][2][0] == "X":
                bigBoard[i][j] = "X"
            elif board[i][j][0][0] == "O" and board[i][j][0][1] == "O" and board[i][j][0][2] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][1][0] == "O" and board[i][j][1][1] == "O" and board[i][j][1][2] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][2][0] == "O" and board[i][j][2][1] == "O" and board[i][j][2][2] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][0][0] == "O" and board[i][j][1][0] == "O" and board[i][j][2][0] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][0][1] == "O" and board[i][j][1][1] == "O" and board[i][j][2][1] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][0][2] == "O" and board[i][j][1][2] == "O" and board[i][j][2][2] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][0][0] == "O" and board[i][j][1][1] == "O" and board[i][j][2][2] == "O":
                bigBoard[i][j] = "O"
            elif board[i][j][0][2] == "O" and board[i][j][1][1] == "O" and board[i][j][2][0] == "O":
                bigBoard[i][j] = "O"
def printBigBoard():
    if(bigBoard[currentBoard[0]][currentBoard[1]] == " "):
        bigBoard[currentBoard[0]][currentBoard[1]] = "#"
    print("")
    print("overall, the board looks like (# is the CURRENT AREA)")
    print("")
    print(" "+bigBoard[0][2]+" | "+bigBoard[1][2]+" | "+bigBoard[2][2])
    print(" ---------")
    print(" "+bigBoard[0][1]+" | "+bigBoard[1][1]+" | "+bigBoard[2][1])
    print(" ---------")
    print(" "+bigBoard[0][0]+" | "+bigBoard[1][0]+" | "+bigBoard[2][0])
    print("")
    if(bigBoard[currentBoard[0]][currentBoard[1]] == "#"):
        bigBoard[currentBoard[0]][currentBoard[1]] = " "
def printBoard():
    os.system("cls")
    print(board[0][2][0][2]+" "+board[0][2][1][2]+" "+board[0][2][2][2]+"|"+board[1][2][0][2]+" "+board[1][2][1][2]+" "+board[1][2][2][2]+"|"+board[2][2][0][2]+" "+board[2][2][1][2]+" "+board[2][2][2][2])
    print("")
    print(board[0][2][0][1]+" "+board[0][2][1][1]+" "+board[0][2][2][1]+"|"+board[1][2][0][1]+" "+board[1][2][1][1]+" "+board[1][2][2][1]+"|"+board[2][2][0][1]+" "+board[2][2][1][1]+" "+board[2][2][2][1])
    print("")
    print(board[0][2][0][0]+" "+board[0][2][1][0]+" "+board[0][2][2][0]+"|"+board[1][2][0][0]+" "+board[1][2][1][0]+" "+board[1][2][2][0]+"|"+board[2][2][0][0]+" "+board[2][2][1][0]+" "+board[2][2][2][0])
    print("------------------")
    print(board[0][1][0][2]+" "+board[0][1][1][2]+" "+board[0][1][2][2]+"|"+board[1][1][0][2]+" "+board[1][1][1][2]+" "+board[1][1][2][2]+"|"+board[2][1][0][2]+" "+board[2][1][1][2]+" "+board[2][1][2][2])
    print("")
    print(board[0][1][0][1]+" "+board[0][1][1][1]+" "+board[0][1][2][1]+"|"+board[1][1][0][1]+" "+board[1][1][1][1]+" "+board[1][1][2][1]+"|"+board[2][1][0][1]+" "+board[2][1][1][1]+" "+board[2][1][2][1])
    print("")
    print(board[0][1][0][0]+" "+board[0][1][1][0]+" "+board[0][1][2][0]+"|"+board[1][1][0][0]+" "+board[1][1][1][0]+" "+board[1][1][2][0]+"|"+board[2][1][0][0]+" "+board[2][1][1][0]+" "+board[2][1][2][0])
    print("------------------")
    print(board[0][0][0][2]+" "+board[0][0][1][2]+" "+board[0][0][2][2]+"|"+board[1][0][0][2]+" "+board[1][0][1][2]+" "+board[1][0][2][2]+"|"+board[2][0][0][2]+" "+board[2][0][1][2]+" "+board[2][0][2][2])
    print("")
    print(board[0][0][0][1]+" "+board[0][0][1][1]+" "+board[0][0][2][1]+"|"+board[1][0][0][1]+" "+board[1][0][1][1]+" "+board[1][0][2][1]+"|"+board[2][0][0][1]+" "+board[2][0][1][1]+" "+board[2][0][2][1])
    print("")
    print(board[0][0][0][0]+" "+board[0][0][1][0]+" "+board[0][0][2][0]+"|"+board[1][0][0][0]+" "+board[1][0][1][0]+" "+board[1][0][2][0]+"|"+board[2][0][0][0]+" "+board[2][0][1][0]+" "+board[2][0][2][0])
def wincomingLocal():
	wins = [0,0,0,0,0,0,0,0]
	if board[currentBoard[0]][currentBoard[1]][0][0] == "X":
		wins[0] = wins[0]+1
		wins[3] = wins[3]+1
		wins[6] = wins[6]+1
	if board[currentBoard[0]][currentBoard[1]][0][1] == "X":
		wins[0] = wins[0]+1
		wins[4] = wins[4]+1
	if board[currentBoard[0]][currentBoard[1]][0][2] == "X":
		wins[0] = wins[0]+1
		wins[5] = wins[5]+1
		wins[7] = wins[7]+1
	if board[currentBoard[0]][currentBoard[1]][1][0] == "X":
		wins[1] = wins[1]+1
		wins[3] = wins[3]+1
	if board[currentBoard[0]][currentBoard[1]][1][1] == "X":
		wins[1] = wins[1]+1
		wins[4] = wins[4]+1
		wins[6] = wins[6]+1
		wins[7] = wins[7]+1
	if board[currentBoard[0]][currentBoard[1]][1][2] == "X":
		wins[1] = wins[1]+1
		wins[5] = wins[5]+1
	if board[currentBoard[0]][currentBoard[1]][2][0] == "X":
		wins[2] = wins[2]+1
		wins[3] = wins[3]+1
		wins[7] = wins[7]+1
	if board[currentBoard[0]][currentBoard[1]][2][1] == "X":
		wins[2] = wins[2]+1
		wins[4] = wins[4]+1
	if board[currentBoard[0]][currentBoard[1]][2][2] == "X":
		wins[2] = wins[2]+1
		wins[5] = wins[5]+1
		wins[6] = wins[6]+1
	return wins
def wincomingGlobal():
	wins = [0,0,0,0,0,0,0,0]
	if bigBoard[0][0] == "X":
		wins[0] = wins[0]+1
		wins[3] = wins[3]+1
		wins[6] = wins[6]+1
	if bigBoard[0][1] == "X":
		wins[0] = wins[0]+1
		wins[4] = wins[4]+1
	if bigBoard[0][2] == "X":
		wins[0] = wins[0]+1
		wins[5] = wins[5]+1
		wins[7] = wins[7]+1
	if bigBoard[1][0] == "X":
		wins[1] = wins[1]+1
		wins[3] = wins[3]+1
	if bigBoard[1][1] == "X":
		wins[1] = wins[1]+1
		wins[4] = wins[4]+1
		wins[6] = wins[6]+1
		wins[7] = wins[7]+1
	if bigBoard[1][2] == "X":
		wins[1] = wins[1]+1
		wins[5] = wins[5]+1
	if bigBoard[2][0] == "X":
		wins[2] = wins[2]+1
		wins[3] = wins[3]+1
		wins[7] = wins[7]+1
	if bigBoard[2][1] == "X":
		wins[2] = wins[2]+1
		wins[4] = wins[4]+1
	if bigBoard[2][2] == "X":
		wins[2] = wins[2]+1
		wins[5] = wins[5]+1
		wins[6] = wins[6]+1
	return wins
def wincon():
    updateBigBoard()
    if bigBoard[0][0] == "X" and bigBoard[0][1] == "X" and bigBoard[0][2] == "X":
        return 1
    elif bigBoard[1][0] == "X" and bigBoard[1][1] == "X" and bigBoard[1][2] == "X":
        return 1
    elif bigBoard[2][0] == "X" and bigBoard[2][1] == "X" and bigBoard[2][2] == "X":
        return 1
    elif bigBoard[0][0] == "X" and bigBoard[1][0] == "X" and bigBoard[2][0] == "X":
        return 1
    elif bigBoard[0][1] == "X" and bigBoard[1][1] == "X" and bigBoard[2][1] == "X":
        return 1
    elif bigBoard[0][2] == "X" and bigBoard[1][2] == "X" and bigBoard[2][2] == "X":
        return 1
    elif bigBoard[0][0] == "X" and bigBoard[1][1] == "X" and bigBoard[2][2] == "X":
        return 1
    elif bigBoard[0][2] == "X" and bigBoard[1][1] == "X" and bigBoard[2][0] == "X":
        return 1
    elif bigBoard[0][0] == "O" and bigBoard[0][1] == "O" and bigBoard[0][2] == "O":
        return 2
    elif bigBoard[1][0] == "O" and bigBoard[1][1] == "O" and bigBoard[1][2] == "O":
        return 2
    elif bigBoard[2][0] == "O" and bigBoard[2][1] == "O" and bigBoard[2][2] == "O":
        return 2
    elif bigBoard[0][0] == "O" and bigBoard[1][0] == "O" and bigBoard[2][0] == "O":
        return 2
    elif bigBoard[0][1] == "O" and bigBoard[1][1] == "O" and bigBoard[2][1] == "O":
        return 2
    elif bigBoard[0][2] == "O" and bigBoard[1][2] == "O" and bigBoard[2][2] == "O":
        return 2
    elif bigBoard[0][0] == "O" and bigBoard[1][1] == "O" and bigBoard[2][2] == "O":
        return 2
    elif bigBoard[0][2] == "O" and bigBoard[1][1] == "O" and bigBoard[2][0] == "O":
        return 2
    elif(bigBoard[0][0] != " " and bigBoard[0][1] != " " and bigBoard[0][2] != " " and bigBoard[1][0] != " " and bigBoard[1][1] != " " and bigBoard[1][2] != " " and bigBoard[2][0] != " " and bigBoard[2][1] != " " and bigBoard[2][2] != " "):
        return 3
    else:
        return 0

def exampleTrainGenetic(layer1w, layer1b, layer2w, layer2b): #an example of how to train a genetic algorithm
        # Define Sequential model
        #print(x)
        #print(y)
        #input("hmmm?")
        model = tf.keras.Sequential([
            layers.Dense(2, activation='relu', name="layer1", input_shape=(2,)),
            layers.Dense(2 ,activation='sigmoid', name="layer2")
            #layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                loss='categorical_crossentropy',
                #loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        data = np.array([[1,1],[1,0],[0,1],[0,0]])
        labels = np.array([[1,0],[1,0],[0,1],[0,1]])
        evaluations = np.array([float(0) for i in range(population_size)])
        for i in range(population_size):
            model.layers[0].set_weights([np.array(layer1w[i]),np.array(layer1b[i])])
            model.layers[1].set_weights([np.array(layer2w[i]),np.array(layer2b[i])])

            #model.layers[0].set_weights([np.array([[-2.475, 0.197],[.138, -0.117]]),np.array([2.328,-.37])])
            #model.layers[1].set_weights([np.array([[-2.993, 4.36],[-.639,.588]]),np.array([3.65,-3.65])])
            
            #model.fit(data, labels, epochs=200, batch_size=30)
            #model.save('my_model.h5')
            print(model.predict(data))
            #print("###########")
            #print(model.predict(np.array([data[0]])))
            #print(model.get_weights())
            #for layer in model.layers:
            #    print(layer.get_weights())
            print(model.evaluate(data, labels)[0])
            print("")
            evaluations[i] = model.evaluate(data,labels)[0]
        print(tf.constant(evaluations))
        return tf.constant(evaluations)
    

def trainGenetic(fungusLayer1w, fungusLayer1b, fungusLayer2w, fungusLayer2b, chungusLayer1w, chungusLayer1b, amongusLayer1w, amongusLayer1b, amongusLayer2w, amongusLayer2b): #an example of how to train a genetic algorithm
    global currentBoard
    global board
    global bigBoard
    global gen
    slow = False
    #Define Sequential model
    #print(x)
    #print(y)
    #input("hmmm?")

    #first need to create two networks
    fungus = tf.keras.Sequential([#fungus does whole board sentiment
        layers.Dense(8, activation='relu', name="layer1", input_shape=(18,), dtype='float32'),
        layers.Dense(2, activation='sigmoid', name="layer2")
        #layers.Dense(2, activation='softmax')
    ])
    fungus.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='categorical_crossentropy',
            #loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    chungus = tf.keras.Sequential([#chungus does local sentiment
        layers.Dense(12, activation='sigmoid', name="layer1", input_shape=(18,), dtype='float32'),
        #layers.Dense(2, activation='sigmoid', name="layer2")
        #layers.Dense(2, activation='softmax')
    ])
    chungus.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='categorical_crossentropy',
            #loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    amongus = tf.keras.Sequential([#chungus does local sentiment
        layers.Dense(10, activation='relu', name="layer1", input_shape=(36,), dtype='float32'),
        layers.Dense(9, activation='sigmoid', name="layer2")
        #layers.Dense(2, activation='softmax')
    ])
    amongus.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='categorical_crossentropy',
            #loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


    kitten = tf.keras.Sequential([#fungus does whole board sentiment
        layers.Dense(8, activation='relu', name="layer1", input_shape=(18,), dtype='float32'),
        layers.Dense(2, activation='sigmoid', name="layer2")
        #layers.Dense(2, activation='softmax')
    ])
    kitten.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='categorical_crossentropy',
            #loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    ballista = tf.keras.Sequential([#chungus does local sentiment
        layers.Dense(12, activation='sigmoid', name="layer1", input_shape=(18,), dtype='float32'),
        #layers.Dense(2, activation='sigmoid', name="layer2")
        #layers.Dense(2, activation='softmax')
    ])
    ballista.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='categorical_crossentropy',
            #loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    fiftyfour = tf.keras.Sequential([#chungus does local sentiment
        layers.Dense(10, activation='relu', name="layer1", input_shape=(36,), dtype='float32'),
        layers.Dense(9, activation='sigmoid', name="layer2")
        #layers.Dense(2, activation='softmax')
    ])
    fiftyfour.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='categorical_crossentropy',
            #loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    gen += 1
    victories = np.array([float(0) for i in range(population_size)])
    for i in range(int(population_size/2)):
        for w in range(2):
            if(w==1 and i>=10):    
                fungus.layers[0].set_weights([np.array(fungusLayer1w[i+int(population_size/4)]),np.array(fungusLayer1b[i+int(population_size/4)])])
                fungus.layers[1].set_weights([np.array(fungusLayer2w[i+int(population_size/4)]),np.array(fungusLayer2b[i+int(population_size/4)])])
                chungus.layers[0].set_weights([np.array(chungusLayer1w[i+int(population_size/4)]),np.array(chungusLayer1b[i+int(population_size/4)])])
                amongus.layers[0].set_weights([np.array(amongusLayer1w[i+int(population_size/4)]),np.array(amongusLayer1b[i+int(population_size/4)])])
                amongus.layers[1].set_weights([np.array(amongusLayer2w[i+int(population_size/4)]),np.array(amongusLayer2b[i+int(population_size/4)])])
                
                kitten.layers[0].set_weights([np.array(fungusLayer1w[i+int(population_size/2)]),np.array(fungusLayer1b[i+int(population_size/2)])])
                kitten.layers[1].set_weights([np.array(fungusLayer2w[i+int(population_size/2)]),np.array(fungusLayer2b[i+int(population_size/2)])])
                ballista.layers[0].set_weights([np.array(chungusLayer1w[i+int(population_size/2)]),np.array(chungusLayer1b[i+int(population_size/2)])])
                fiftyfour.layers[0].set_weights([np.array(amongusLayer1w[i+int(population_size/2)]),np.array(amongusLayer1b[i+int(population_size/2)])])
                fiftyfour.layers[1].set_weights([np.array(amongusLayer2w[i+int(population_size/2)]),np.array(amongusLayer2b[i+int(population_size/2)])])
            else:
                fungus.layers[0].set_weights([np.array(fungusLayer1w[i]),np.array(fungusLayer1b[i])])
                fungus.layers[1].set_weights([np.array(fungusLayer2w[i]),np.array(fungusLayer2b[i])])
                chungus.layers[0].set_weights([np.array(chungusLayer1w[i]),np.array(chungusLayer1b[i])])
                amongus.layers[0].set_weights([np.array(amongusLayer1w[i]),np.array(amongusLayer1b[i])])
                amongus.layers[1].set_weights([np.array(amongusLayer2w[i]),np.array(amongusLayer2b[i])])
                
                kitten.layers[0].set_weights([np.array(fungusLayer1w[i+int(population_size/2)-w*int(population_size/4)]),np.array(fungusLayer1b[i+int(population_size/2)-w*int(population_size/4)])])
                kitten.layers[1].set_weights([np.array(fungusLayer2w[i+int(population_size/2)-w*int(population_size/4)]),np.array(fungusLayer2b[i+int(population_size/2)-w*int(population_size/4)])])
                ballista.layers[0].set_weights([np.array(chungusLayer1w[i+int(population_size/2)-w*int(population_size/4)]),np.array(chungusLayer1b[i+int(population_size/2)-w*int(population_size/4)])])
                fiftyfour.layers[0].set_weights([np.array(amongusLayer1w[i+int(population_size/2)-w*int(population_size/4)]),np.array(amongusLayer1b[i+int(population_size/2)-w*int(population_size/4)])])
                fiftyfour.layers[1].set_weights([np.array(amongusLayer2w[i+int(population_size/2)-w*int(population_size/4)]),np.array(amongusLayer2b[i+int(population_size/2)-w*int(population_size/4)])])
                
            #needs to scan over all boards for sentiment
            printBoard()
            printBigBoard()
            print("Choose which area of the board to play in")
            high = float(-1)
            winner = [0,0]
            inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
            for j in range(3):
                for k in range(3):
                    inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoarda(j,k))[0]
            for x in range(9):
                inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                myBoard = np.array([float(0) for z in range(6)])
                inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                #run a deep network over 18 inputs(x's and o's)
                inputs1 = chungus.predict(flattenBoarda(ToBoard(x)[0],ToBoard(x)[1]))[0]
                #merge sentiment and deep outputs
                myBoard[ToBoard(x)[0]] = 1
                myBoard[3+ToBoard(x)[1]] = 1
                inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                #run merged input_data through a deep net to find next move
                nextMove = amongus.predict(inputs3.astype('float32'))
                for j in range(9):
                    if(nextMove[0][j]>=high and board[ToBoard(x)[0]][ToBoard(x)[1]][ToBoard(j)[0]][ToBoard(j)[1]] == " " and bigBoard[ToBoard(x)[0]][ToBoard(x)[1]] == " "):
                        high = nextMove[0][j]
                        winner = ToBoard(x)
            currentBoard[0] = winner[0]
            currentBoard[1] = winner[1]
            while True:
                printBoard()
                printBigBoard()
                print("Generation "+str(gen))
                if(w==1 and i>=10):
                    print("Individuals X="+str(i+int(population_size/4))+', O='+str(i+int(population_size/2)))
                else:
                    print("Individuals X="+str(i)+', O='+str(i+int(population_size/2)-w*int(population_size/4)))
                print("Where does X play?")
                if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                    print("The chosen area was already claimed!!!!")
                    high = float(-1)
                    winner = [0,0]
                    inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                    for j in range(3):
                        for k in range(3):
                            inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoarda(j,k))[0]
                    for x in range(9):
                        if(bigBoard[ToBoard(x)[0]][ToBoard(x)[1]] == " "):
                            inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                            myBoard = np.array([float(0) for z in range(6)])
                            inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                            #run a deep network over 18 inputs(x's and o's)
                            inputs1 = chungus.predict(flattenBoarda(ToBoard(x)[0],ToBoard(x)[1]))[0]
                            #merge sentiment and deep outputs
                            myBoard[ToBoard(x)[0]] = 1
                            myBoard[3+ToBoard(x)[1]] = 1
                            inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                            #run merged input_data through a deep net to find next move
                            nextMove = amongus.predict(inputs3.astype('float32'))
                            for j in range(9):
                                if(nextMove[0][j]>=high and board[ToBoard(x)[0]][ToBoard(x)[1]][ToBoard(j)[0]][ToBoard(j)[1]] == " "):
                                    high = nextMove[0][j]
                                    winner = ToBoard(x)
                    currentBoard[0] = winner[0]
                    currentBoard[1] = winner[1]

                inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                myBoard = np.array([float(0) for z in range(6)])
                inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                for j in range(3):
                    for k in range(3):
                        inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoarda(j,k))[0]
                #run a deep network over 18 inputs(x's and o's)
                inputs1 = chungus.predict(flattenBoarda(currentBoard[0],currentBoard[1]))[0]
                #merge sentiment and deep outputs
                myBoard[currentBoard[0]] = 1
                myBoard[3+currentBoard[1]] = 1
                inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                #run merged input_data through a deep net to find next move
                nextMove = amongus.predict(inputs3)
                playergo = highest(nextMove)
                print(nextMove)
                print(playergo)
                board[currentBoard[0]][currentBoard[1]][playergo[0]][playergo[1]] = "X"
                currentBoard[0] = playergo[0]
                currentBoard[1] = playergo[1]
                #make next move
                #model.save('my_model.h5')
                #pick open square with highest output
                if(slow):
                    time.sleep(1)

                if(wincon() != 0):
                    break
                printBoard()
                printBigBoard()
                print("Generation "+str(gen))
                if(w==1 and i>=10):
                    print("Individuals X="+str(i+int(population_size/4))+', O='+str(i+int(population_size/2)))
                else:
                    print("Individuals X="+str(i)+', O='+str(i+int(population_size/2)-w*int(population_size/4)))
                print("Where does O go?")
                if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                    print("The chosen area was already claimed!!!!")
                    high = float(-1)
                    winner = [0,0]
                    inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                    for j in range(3):
                        for k in range(3):
                            inputs2[3*j+k],inputs2[3*j+k+9] = kitten.predict(flattenBoardb(j,k))[0]
                    for x in range(9):
                        if(bigBoard[ToBoard(x)[0]][ToBoard(x)[1]] == " "):
                            inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                            myBoard = np.array([float(0) for z in range(6)])
                            inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                            #run a deep network over 18 inputs(x's and o's)
                            inputs1 = ballista.predict(flattenBoardb(ToBoard(x)[0],ToBoard(x)[1]))[0]
                            #merge sentiment and deep outputs
                            myBoard[ToBoard(x)[0]] = 1
                            myBoard[3+ToBoard(x)[1]] = 1
                            inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                            #run merged input_data through a deep net to find next move
                            nextMove = fiftyfour.predict(inputs3.astype('float32'))
                            for j in range(9):
                                if(nextMove[0][j]>=high and board[ToBoard(x)[0]][ToBoard(x)[1]][ToBoard(j)[0]][ToBoard(j)[1]] == " "):
                                    high = nextMove[0][j]
                                    winner = ToBoard(x)
                    currentBoard[0] = winner[0]
                    currentBoard[1] = winner[1]

                inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                myBoard = np.array([float(0) for z in range(6)])
                inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                for j in range(3):
                    for k in range(3):
                        inputs2[3*j+k],inputs2[3*j+k+9] = kitten.predict(flattenBoardb(j,k))[0]
                #run a deep network over 18 inputs
                inputs1 = ballista.predict(flattenBoardb(currentBoard[0],currentBoard[1]))[0]
                #merge sentiment and deep outputs
                myBoard[currentBoard[0]] = 1
                myBoard[3+currentBoard[1]] = 1
                inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                #run merged input_data through a deep net to find next move
                nextMove = fiftyfour.predict(inputs3.astype('float32'))
                playergo = highest(nextMove)
                print(nextMove)
                print(playergo)
                board[currentBoard[0]][currentBoard[1]][playergo[0]][playergo[1]] = "O"
                currentBoard[0] = playergo[0]
                currentBoard[1] = playergo[1]
                #make next move
                #model.save('my_model.h5')
                #pick open square with highest output
                if(slow):
                    time.sleep(1)
                if(wincon() != 0):
                    break
            if(wincon() == 1):
                if(w==1 and i>=10):    
                    victories[i+int(population_size/4)] += .025
                    victories[i+int(population_size/2)] += .4
                else:
                    victories[i] += .025
                    victories[i+int(population_size/2)-w*int(population_size/4)] += .4
                print("X WINNNNS")
                if(slow):
                    time.sleep(1)
            elif(wincon() == 2):
                if(w==1 and i>=10):    
                    victories[i+int(population_size/4)] += .4
                    victories[i+int(population_size/2)] += .025
                else:
                    victories[i] += .4
                    victories[i+int(population_size/2)-w*int(population_size/4)] += .025
                print("O WINNNNS")
                if(slow):
                    time.sleep(1)
            elif(wincon() == 3):
                if(w==1 and i>=10):    
                    victories[i+int(population_size/4)] += .15
                    victories[i+int(population_size/2)] += .15
                else:
                    victories[i] += .15
                    victories[i+int(population_size/2)-w*int(population_size/4)] += .15
                print("TIEEEEEE")
                if(slow):
                    time.sleep(1)
            board = [[[[" "," "," "],[" "," "," "],[" "," "," "]] for i in range(3)] for j in range(3)]
            currentBoard = [0,0]
            bigBoard = [[" "," "," "],[" "," "," "],[" "," "," "]]
        
            if(w==0):
                fungus.save('./AI/fungus'+str(i)+'.h5')
                chungus.save('./AI/chungus'+str(i)+'.h5')
                amongus.save('/AI/amongus'+str(i)+'.h5')

                kitten.save('./AI/kitten'+str(i)+'.h5')
                ballista.save('./AI/ballista'+str(i)+'.h5')
                fiftyfour.save('./AI/fiftyfour'+str(i)+'.h5')

    print("loss")
    print(victories)
    time.sleep(2)
    return tf.constant(victories.astype('float32'))
playType = input("friend or AI\n")
if playType == "friend":
    printBoard()
    printBigBoard()
    print("Choose which area of the board to play in")
    playergo = input("which AREA does X play in? (input as 'x,y')\n")
    playergo = eval("["+playergo+"]")
    currentBoard[0] = playergo[0]-1
    currentBoard[1] = playergo[1]-1
    while True:
        while True:
            printBoard()
            printBigBoard()
            if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                print("The chosen area was already claimed!!!!")
                try:
                    playergo = input("which AREA does X play in? (input as 'x,y')\n")
                    playergo = eval("["+playergo+"]")
                    currentBoard[0] = playergo[0]-1
                    currentBoard[1] = playergo[1]-1
                except:
                    os.system("cls")
                    print("Invalid Input")
                    time.sleep(2)
            else:
                try:
                    playergo = input("where does X move? (input as 'x,y')\n")
                    playergo = eval("["+playergo+"]")
                    if prepicked(playergo):
                        board[currentBoard[0]][currentBoard[1]][playergo[0]-1][playergo[1]-1] = "X"
                        break
                    else:
                        print("you can't do that")
                except:
                    os.system("cls")
                    print("Invalid Input")
                    time.sleep(2)
            time.sleep(1)
        if wincon():
            break
        currentBoard[0] = playergo[0]-1
        currentBoard[1] = playergo[1]-1
        while True:
            os.system("cls")
            printBoard()
            printBigBoard()
            if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                print("The chosen area was already claimed!!!!")
                try:
                    playergo = input("which AREA does O play in? (input as 'x,y')\n")
                    playergo = eval("["+playergo+"]")
                    currentBoard[0] = playergo[0]-1
                    currentBoard[1] = playergo[1]-1
                except:
                    os.system("cls")
                    print("Invalid Input")
                    time.sleep(2)
            else:
                try:
                    playergo = input("where does O move? (input as 'x,y')\n")
                    playergo = eval("["+playergo+"]")
                    if prepicked(playergo):
                        board[currentBoard[0]][currentBoard[1]][playergo[0]-1][playergo[1]-1] = "O"
                        break
                    else:
                        print("you can't do that")
                except:
                    os.system("cls")
                    print("Invalid Input")
                    time.sleep(2)
            time.sleep(1)
        if wincon() != 0:
            break
        currentBoard[0] = playergo[0]-1
        currentBoard[1] = playergo[1]-1
elif playType == "AI" or playType == "ai" or playType == "Ai":
    playType = input("Easy or Hard\n")
    if(playType == "easy" or playType == "Easy"):
        printBoard()
        printBigBoard()
        print("Choose which area of the board to play in")
        playergo = input("which AREA does X play in? (input as 'x,y')\n")
        playergo = eval("["+playergo+"]")
        currentBoard[0] = playergo[0]-1
        currentBoard[1] = playergo[1]-1
        while True:
            while True:
                printBoard()
                printBigBoard()
                if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                    print("The chosen area was already claimed!!!!")
                    try:
                        playergo = input("which AREA does X play in? (input as 'x,y')\n")
                        playergo = eval("["+playergo+"]")
                        currentBoard[0] = playergo[0]-1
                        currentBoard[1] = playergo[1]-1
                    except:
                        os.system("cls")
                        print("Invalid Input")
                        time.sleep(2)
                else:
                    try:
                        playergo = input("where does X move? (input as 'x,y')\n")
                        playergo = eval("["+playergo+"]")
                        if prepicked(playergo):
                            board[currentBoard[0]][currentBoard[1]][playergo[0]-1][playergo[1]-1] = "X"
                            currentBoard[0] = playergo[0]-1
                            currentBoard[1] = playergo[1]-1
                            break
                        else:
                            os.system("cls")
                            print("Already Taken")
                            time.sleep(2)
                    except:
                        os.system("cls")
                        print("Invalid Input")
                        time.sleep(2)
                time.sleep(1)
            if wincon():
                break
            printBoard()
            printBigBoard()
            wins = wincomingGlobal()
            acted = False
            if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                if wins[0] >= 2:
                    if prepickedGlobal([0,0]):
                        acted = True
                        currentBoard = [0,0]
                    elif prepickedGlobal([0,1]):
                        acted = True
                        currentBoard = [0,1]
                    elif prepickedGlobal([0,2]):
                        acted = True
                        currentBoard = [0,2]
                if wins[1] >= 2 and acted == False:
                    if prepickedGlobal([1,0]):
                        acted = True
                        currentBoard = [1,0]
                    elif prepickedGlobal([1,1]):
                        acted = True
                        currentBoard = [1,1]
                    elif prepickedGlobal([1,2]):
                        acted = True
                        currentBoard = [1,2]
                if wins[2] >= 2 and acted == False:
                    if prepickedGlobal([2,0]):
                        acted = True
                        currentBoard = [2,0]
                    elif prepickedGlobal([2,1]):
                        acted = True
                        currentBoard = [2,1]
                    elif prepickedGlobal([2,2]):
                        acted = True
                        currentBoard = [2,2]
                if wins[3] >= 2 and acted == False:
                    if prepickedGlobal([0,0]):
                        acted = True
                        currentBoard = [0,0]
                    elif prepickedGlobal([1,0]):
                        acted = True
                        currentBoard = [1,0]
                    elif prepickedGlobal([2,0]):
                        acted = True
                        currentBoard = [2,0]
                if wins[4] >= 2 and acted == False:
                    if prepickedGlobal([0,1]):
                        acted = True
                        currentBoard = [0,1]
                    elif prepickedGlobal([1,1]):
                        acted = True
                        currentBoard = [1,1]
                    elif prepickedGlobal([2,1]):
                        acted = True
                        currentBoard = [2,1]
                if wins[5] >= 2 and acted == False:
                    if prepickedGlobal([0,2]):
                        acted = True
                        currentBoard = [0,2]
                    elif prepickedGlobal([1,2]):
                        acted = True
                        currentBoard = [1,2]
                    elif prepickedGlobal([2,2]):
                        acted = True
                        currentBoard = [2,2]
                if wins[6] >= 2 and acted == False:
                    if prepickedGlobal([0,0]):
                        acted = True
                        currentBoard = [0,0]
                    elif prepickedGlobal([1,1]):
                        acted = True
                        currentBoard = [1,1]
                    elif prepickedGlobal([2,2]):
                        acted = True
                        currentBoard = [2,2]
                if wins[7] >= 2 and acted == False:
                    if prepickedGlobal([0,2]):
                        acted = True
                        currentBoard = [0,2]
                    elif prepickedGlobal([1,1]):
                        acted = True
                        currentBoard = [1,1]
                    elif prepickedGlobal([2,0]):
                        acted = True
                        currentBoard = [2,0]
                if acted == False:
                    while True:
                        guess = [math.floor(3*random.random()),math.floor(3*random.random())]
                        if prepickedGlobal([guess[0],guess[1]]):
                            currentBoard = [guess[0],guess[1]]
                            break
            acted = False
            wins = wincomingLocal()
            if wins[0] >= 2:
                if prepickedai([0,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][0] = "O"
                    currentBoard = [0,0]
                elif prepickedai([0,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][1] = "O"
                    currentBoard = [0,1]
                elif prepickedai([0,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][2] = "O"  
                    currentBoard = [0,2]
            if wins[1] >= 2 and acted == False:
                if prepickedai([1,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][0] = "O"
                    currentBoard = [1,0]
                elif prepickedai([1,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][1] = "O"
                    currentBoard = [1,1]
                elif prepickedai([1,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][2] = "O"
                    currentBoard = [1,2]
            if wins[2] >= 2 and acted == False:
                if prepickedai([2,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][0] = "O"
                    currentBoard = [2,0]
                elif prepickedai([2,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][1] = "O"
                    currentBoard = [2,1]
                elif prepickedai([2,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][2] = "O"
                    currentBoard = [2,2]
            if wins[3] >= 2 and acted == False:
                if prepickedai([0,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][0] = "O"
                    currentBoard = [0,0]
                elif prepickedai([1,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][0] = "O"
                    currentBoard = [1,0]
                elif prepickedai([2,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][0] = "O"
                    currentBoard = [2,0]
            if wins[4] >= 2 and acted == False:
                if prepickedai([0,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][1] = "O"
                    currentBoard = [0,1]
                elif prepickedai([1,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][1] = "O"
                    currentBoard = [1,1]
                elif prepickedai([2,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][1] = "O"
                    currentBoard = [2,1]
            if wins[5] >= 2 and acted == False:
                if prepickedai([0,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][2] = "O"
                    currentBoard = [0,2]
                elif prepickedai([1,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][2] = "O"
                    currentBoard = [1,2]
                elif prepickedai([2,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][2] = "O"
                    currentBoard = [2,2]
            if wins[6] >= 2 and acted == False:
                if prepickedai([0,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][0] = "O"
                    currentBoard = [0,0]
                elif prepickedai([1,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][1] = "O"
                    currentBoard = [1,1]
                elif prepickedai([2,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][2] = "O"
                    currentBoard = [2,2]
            if wins[7] >= 2 and acted == False:
                if prepickedai([0,2]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][0][2] = "O"
                    currentBoard = [0,2]
                elif prepickedai([1,1]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][1][1] = "O"
                    currentBoard = [1,1]
                elif prepickedai([2,0]):
                    acted = True
                    board[currentBoard[0]][currentBoard[1]][2][0] = "O"
                    currentBoard = [2,0]
            if acted == False:
                while True:
                    guess = [math.floor(3*random.random()),math.floor(3*random.random())]
                    if prepickedai([guess[0],guess[1]]):
                        board[currentBoard[0]][currentBoard[1]][guess[0]][guess[1]] = "O"
                        currentBoard = [guess[0],guess[1]]
                        break
            time.sleep(2)
            if wincon() != 0:
                break
    else:
        if(math.floor(2*random.random()) == 1):
            print("You are X")
            time.sleep(3)
            fungus = tf.keras.models.load_model('./AI/fungus1.h5')
            chungus = tf.keras.models.load_model('./AI/chungus1.h5')
            amongus = tf.keras.models.load_model('./AI/amongus1.h5')
            printBoard()
            printBigBoard()
            print("Choose which area of the board to play in")
            playergo = input("which AREA does X play in? (input as 'x,y')\n")
            playergo = eval("["+playergo+"]")
            currentBoard[0] = playergo[0]-1
            currentBoard[1] = playergo[1]-1
            while True:
                while True:
                    printBoard()
                    printBigBoard()
                    if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                        print("The chosen area was already claimed!!!!")
                        try:
                            playergo = input("which AREA does X play in? (input as 'x,y')\n")
                            playergo = eval("["+playergo+"]")
                            currentBoard[0] = playergo[0]-1
                            currentBoard[1] = playergo[1]-1
                        except:
                            os.system("cls")
                            print("Invalid Input")
                            time.sleep(2)
                    else:
                        try:
                            playergo = input("where does X move? (input as 'x,y')\n")
                            playergo = eval("["+playergo+"]")
                            if prepicked(playergo):
                                board[currentBoard[0]][currentBoard[1]][playergo[0]-1][playergo[1]-1] = "X"
                                currentBoard[0] = playergo[0]-1
                                currentBoard[1] = playergo[1]-1
                                break
                            else:
                                os.system("cls")
                                print("Already Taken")
                                time.sleep(2)
                        except:
                            os.system("cls")
                            print("Invalid Input")
                            time.sleep(2)
                    time.sleep(1)
                if wincon():
                    break
                printBoard()
                printBigBoard()
                print("Where does O play?(AI)")
                if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                    print("The chosen area was already claimed!!!!")
                    high = float(-1)
                    winner = [0,0]
                    inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                    for j in range(3):
                        for k in range(3):
                            inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoardb(j,k))[0]
                    for x in range(9):
                        if(bigBoard[ToBoard(x)[0]][ToBoard(x)[1]] == " "):
                            inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                            myBoard = np.array([float(0) for z in range(6)])
                            inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                            #run a deep network over 18 inputs(x's and o's)
                            inputs1 = chungus.predict(flattenBoardb(ToBoard(x)[0],ToBoard(x)[1]))[0]
                            #merge sentiment and deep outputs
                            myBoard[ToBoard(x)[0]] = 1
                            myBoard[3+ToBoard(x)[1]] = 1
                            inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                            #run merged input_data through a deep net to find next move
                            nextMove = amongus.predict(inputs3.astype('float32'))
                            for j in range(9):
                                if(nextMove[0][j]>=high and board[ToBoard(x)[0]][ToBoard(x)[1]][ToBoard(j)[0]][ToBoard(j)[1]] == " "):
                                    high = nextMove[0][j]
                                    winner = ToBoard(x)
                    currentBoard[0] = winner[0]
                    currentBoard[1] = winner[1]

                inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                myBoard = np.array([float(0) for z in range(6)])
                inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                for j in range(3):
                    for k in range(3):
                        inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoardb(j,k))[0]
                #run a deep network over 18 inputs(x's and o's)
                inputs1 = chungus.predict(flattenBoardb(currentBoard[0],currentBoard[1]))[0]
                #merge sentiment and deep outputs
                myBoard[currentBoard[0]] = 1
                myBoard[3+currentBoard[1]] = 1
                inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                #run merged input_data through a deep net to find next move
                nextMove = amongus.predict(inputs3)
                playergo = highest(nextMove)
                board[currentBoard[0]][currentBoard[1]][playergo[0]][playergo[1]] = "O"
                currentBoard[0] = playergo[0]
                currentBoard[1] = playergo[1]
                #make next move
                #model.save('my_model.h5')
                #pick open square with highest output
                time.sleep(3)

                if(wincon() != 0):
                    break
        else:
            print("You are O")
            time.sleep(3)
            fungus = tf.keras.models.load_model('./AI/fungus1.h5')
            chungus = tf.keras.models.load_model('./AI/chungus1.h5')
            amongus = tf.keras.models.load_model('./AI/amongus1.h5')
            printBoard()
            printBigBoard()
            print("Where does X go?(AI)")
            high = float(-1)
            winner = [0,0]
            inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
            for j in range(3):
                for k in range(3):
                    inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoarda(j,k))[0]
            for x in range(9):
                if(bigBoard[ToBoard(x)[0]][ToBoard(x)[1]] == " "):
                    inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                    myBoard = np.array([float(0) for z in range(6)])
                    inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                    #run a deep network over 18 inputs(x's and o's)
                    inputs1 = chungus.predict(flattenBoarda(ToBoard(x)[0],ToBoard(x)[1]))[0]
                    #merge sentiment and deep outputs
                    myBoard[ToBoard(x)[0]] = 1
                    myBoard[3+ToBoard(x)[1]] = 1
                    inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                    #run merged input_data through a deep net to find next move
                    nextMove = amongus.predict(inputs3.astype('float32'))
                    for j in range(9):
                        if(nextMove[0][j]>=high and board[ToBoard(x)[0]][ToBoard(x)[1]][ToBoard(j)[0]][ToBoard(j)[1]] == " "):
                            high = nextMove[0][j]
                            winner = ToBoard(x)
            currentBoard[0] = winner[0]
            currentBoard[1] = winner[1]

            while True:
                printBoard()
                printBigBoard()
                print("Where does X play?(AI)")
                if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                    print("The chosen area was already claimed!!!!")
                    high = float(-1)
                    winner = [0,0]
                    inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                    for j in range(3):
                        for k in range(3):
                            inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoarda(j,k))[0]
                    for x in range(9):
                        if(bigBoard[ToBoard(x)[0]][ToBoard(x)[1]] == " "):
                            inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                            myBoard = np.array([float(0) for z in range(6)])
                            inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                            #run a deep network over 18 inputs(x's and o's)
                            inputs1 = chungus.predict(flattenBoarda(ToBoard(x)[0],ToBoard(x)[1]))[0]
                            #merge sentiment and deep outputs
                            myBoard[ToBoard(x)[0]] = 1
                            myBoard[3+ToBoard(x)[1]] = 1
                            inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                            #run merged input_data through a deep net to find next move
                            nextMove = amongus.predict(inputs3.astype('float32'))
                            for j in range(9):
                                if(nextMove[0][j]>=high and board[ToBoard(x)[0]][ToBoard(x)[1]][ToBoard(j)[0]][ToBoard(j)[1]] == " "):
                                    high = nextMove[0][j]
                                    winner = ToBoard(x)
                    currentBoard[0] = winner[0]
                    currentBoard[1] = winner[1]

                inputs1 = np.array([float(0) for z in range(12)])#output from local analysis
                inputs2 = np.array([float(0) for z in range(18)])#outputs from whole board analysis
                myBoard = np.array([float(0) for z in range(6)])
                inputs3 = np.array([float(0) for z in range(12+18+6)])#merge into inputs for the final network
                for j in range(3):
                    for k in range(3):
                        inputs2[3*j+k],inputs2[3*j+k+9] = fungus.predict(flattenBoarda(j,k))[0]
                #run a deep network over 18 inputs(x's and o's)
                inputs1 = chungus.predict(flattenBoarda(currentBoard[0],currentBoard[1]))[0]
                #merge sentiment and deep outputs
                myBoard[currentBoard[0]] = 1
                myBoard[3+currentBoard[1]] = 1
                inputs3 = np.array([np.concatenate([inputs1,inputs2,myBoard])]) #effectively
                #run merged input_data through a deep net to find next move
                nextMove = amongus.predict(inputs3)
                playergo = highest(nextMove)
                board[currentBoard[0]][currentBoard[1]][playergo[0]][playergo[1]] = "X"
                currentBoard[0] = playergo[0]
                currentBoard[1] = playergo[1]
                #make next move
                #model.save('my_model.h5')
                #pick open square with highest output
                time.sleep(3)

                if(wincon() != 0):
                    break
        
                while True:
                    printBoard()
                    printBigBoard()
                    if(bigBoard[currentBoard[0]][currentBoard[1]] != " "):
                        print("The chosen area was already claimed!!!!")
                        try:
                            playergo = input("which AREA does O play in? (input as 'x,y')\n")
                            playergo = eval("["+playergo+"]")
                            currentBoard[0] = playergo[0]-1
                            currentBoard[1] = playergo[1]-1
                        except:
                            os.system("cls")
                            print("Invalid Input")
                            time.sleep(2)
                    else:
                        try:
                            playergo = input("where does O move? (input as 'x,y')\n")
                            playergo = eval("["+playergo+"]")
                            if prepicked(playergo):
                                board[currentBoard[0]][currentBoard[1]][playergo[0]-1][playergo[1]-1] = "O"
                                currentBoard[0] = playergo[0]-1
                                currentBoard[1] = playergo[1]-1
                                break
                            else:
                                os.system("cls")
                                print("Already Taken")
                                time.sleep(2)
                        except:
                            os.system("cls")
                            print("Invalid Input")
                            time.sleep(2)
                    time.sleep(1)
                if wincon():
                    break
                
elif playType == "train":
    # With an initial population and a multi-part state.
    
    funguses = [np.zeros((population_size,18,8)), np.zeros((population_size,8)), np.zeros((population_size,8,2)), np.zeros((population_size,2))]  #same as above, network is overall, 2 input, 2 hidden, 2 output
    chunguses = [np.zeros((population_size,18,12)), np.zeros((population_size,12))] #2x2 weights for 2 input to 2 hidden nodes, then 2 bias weights
    amonguses = [np.zeros((population_size,36,10)), np.zeros((population_size,10)),np.zeros((population_size,10,9)), np.zeros((population_size,9))]  #same as above, network is overall, 2 input, 2 hidden, 2 output

    
    for j in range(10):#runs epochs
        try:#reloads old weights
            for i in range(int(population_size/2)):
                fungus = tf.keras.models.load_model('./AI/fungus'+str(i)+'.h5')
                chungus = tf.keras.models.load_model('./AI/chungus'+str(i)+'.h5')
                amongus = tf.keras.models.load_model('./AI/amongus'+str(i)+'.h5')
                
                kitten = tf.keras.models.load_model('./AI/kitten'+str(i)+'.h5')
                ballista = tf.keras.models.load_model('./AI/ballista'+str(i)+'.h5')
                fiftyfour = tf.keras.models.load_model('./AI/fiftyfour'+str(i)+'.h5')
                
                funguses[0][i] = fungus.layers[0].get_weights()[0]
                funguses[1][i] = fungus.layers[0].get_weights()[1]
                funguses[2][i] = fungus.layers[1].get_weights()[0]
                funguses[3][i] = fungus.layers[1].get_weights()[1]

                chunguses[0][i] = chungus.layers[0].get_weights()[0]
                chunguses[1][i] = chungus.layers[0].get_weights()[1]

                amonguses[0][i] = amongus.layers[0].get_weights()[0]
                amonguses[1][i] = amongus.layers[0].get_weights()[1]
                amonguses[2][i] = amongus.layers[1].get_weights()[0]
                amonguses[3][i] = amongus.layers[1].get_weights()[1]


                funguses[0][i+int(population_size/2)] = kitten.layers[0].get_weights()[0]
                funguses[1][i+int(population_size/2)] = kitten.layers[0].get_weights()[1]
                funguses[2][i+int(population_size/2)] = kitten.layers[1].get_weights()[0]
                funguses[3][i+int(population_size/2)] = kitten.layers[1].get_weights()[1]

                chunguses[0][i+int(population_size/2)] = ballista.layers[0].get_weights()[0]
                chunguses[1][i+int(population_size/2)] = ballista.layers[0].get_weights()[1]

                amonguses[0][i+int(population_size/2)] = fiftyfour.layers[0].get_weights()[0]
                amonguses[1][i+int(population_size/2)] = fiftyfour.layers[0].get_weights()[1]
                amonguses[2][i+int(population_size/2)] = fiftyfour.layers[1].get_weights()[0]
                amonguses[3][i+int(population_size/2)] = fiftyfour.layers[1].get_weights()[1]

            population = (
                tf.constant(funguses[0].astype('float32')), tf.constant(funguses[1].astype('float32')), tf.constant(funguses[2].astype('float32')), tf.constant(funguses[3].astype('float32')), 
                tf.constant(chunguses[0].astype('float32')), tf.constant(chunguses[1].astype('float32')),
                tf.constant(amonguses[0].astype('float32')), tf.constant(amonguses[1].astype('float32')), tf.constant(amonguses[2].astype('float32')), tf.constant(amonguses[3].astype('float32'))
            ) 
        except:
            population = (
                tf.random.normal([population_size,18,8]), tf.random.normal([population_size,8]), #2x2 weights for 2 input to 2 hidden nodes, then 2 bias weights
                tf.random.normal([population_size,8,2]), tf.random.normal([population_size,2]),  #same as above, network is overall, 2 input, 2 hidden, 2 output
                tf.random.normal([population_size,18,12]), tf.random.normal([population_size,12]), #2x2 weights for 2 input to 2 hidden nodes, then 2 bias weights
                tf.random.normal([population_size,36,10]), tf.random.normal([population_size,10]), #2x2 weights for 2 input to 2 hidden nodes, then 2 bias weights
                tf.random.normal([population_size,10,9]), tf.random.normal([population_size,9]),  #same as above, network is overall, 2 input, 2 hidden, 2 output
            )
            print(population)
            input("SAVES NOT FOUND, PRESS ENTER TO CREATE NEW SPECIES")
        optim_results = tfp.optimizer.differential_evolution_minimize(
            trainGenetic,
            initial_population=population,
            seed = int(time.time()*400),
        )
    """
    for i in range(50):
        optim_results = tfp.optimizer.differential_evolution_one_step(
            trainGenetic,
            population=population,
            seed = 43210,
            crossover_prob=tf.Variable(0.9)
        )
        print("bbababasbbdabsdbasdba")
        print(optim_results)
        print("booboo")
        print(optim_results[0])
        population = optim_results[0]
    #"""

    """
    #exampleTrainGenetic population
    population = (
        tf.random.normal([population_size,2,2]), tf.random.normal([population_size,2]), #2x2 weights for 2 input to 2 hidden nodes, then 2 bias weights
        tf.random.normal([population_size,2,2]), tf.random.normal([population_size,2])  #same as above, network is overall, 2 input, 2 hidden, 2 output
    )

    for i in range(50):
        optim_results = tfp.optimizer.differential_evolution_one_step(
            exampleTrainGenetic,
            population=population,
            seed = 43210,
            crossover_prob=tf.Variable(0.9)
        )
        print("bbababasbbdabsdbasdba")
        print(optim_results)
        print("booboo")
        print(optim_results[0])
        population = optim_results[0]
    #"""
    """
    optim_results = tfp.optimizer.differential_evolution_minimize(
      exampleTrainGenetic,
      initial_population=initial_population,
      seed=43210
    )
    

    print(optim_results.converged)
    print(optim_results.position)  # Should be (close to) [pi, pi].
    print(optim_results.objective_value)    # Should be -1.
    print(optim_results)
    #"""
    input("FINISHED")
    """
    def easom_fn(x, y):
        print(x)
        print(y)
        print(-(tf.math.cos(x) * tf.math.cos(y) * tf.math.exp(-(x-np.pi)**2 - (y-np.pi)**2)))
        input("hmmm?")
        return -(tf.math.cos(x) * tf.math.cos(y) * tf.math.exp(-(x-np.pi)**2 - (y-np.pi)**2))
    
    # With a single starting point
    initial_position = (tf.constant(1.0), tf.constant(1.0))

    optim_results = tfp.optimizer.differential_evolution_minimize(
      easom_fn,
      initial_position=initial_position,
      population_size=40,
      population_stddev=2.0,
      seed=43210)
    #"""

if wincon() == 1:
    os.system("cls")
    printBoard()
    printBigBoard()
    time.sleep(2)
    os.system("cls")
    time.sleep(.5)
    print("X WINS!!!")
    print("x")
    time.sleep(1.5)
    os.system("cls")
    print("X WINS!!!")
    print("xx")
    time.sleep(1.5)
    os.system("cls")
    print("X WINS!!!")
    print("xxx")
    time.sleep(1.5)
    os.system("cls")
if wincon() == 2:
    os.system("cls")
    printBoard()
    printBigBoard()
    time.sleep(2)
    os.system("cls")
    time.sleep(.5)
    print("O WINS!!!")
    print("o")
    time.sleep(1.5)
    os.system("cls")
    print("O WINS!!!")
    print("oo")
    time.sleep(1.5)
    os.system("cls")
    print("O WINS!!!")
    print("ooo")
    time.sleep(1.5)
    os.system("cls")
if wincon() == 3:
    os.system("cls")
    printBoard()
    printBigBoard()
    time.sleep(2)
    os.system("cls")
    print("TIE")
    time.sleep(1)
    os.system("cls")
    print("TIE")
    print("WA")
    time.sleep(1)
    os.system("cls")
    print("TIE")
    print("WA - WA")
    time.sleep(1)
    os.system("cls")
    print("TIE")
    print("WA - WA - WAAAAAAAAA")
    time.sleep(2.5)

