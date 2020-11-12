import cv2
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import math

# Main Code
if __name__ == "__main__":


    numPlayers = int(input('Number of Players? '))
    playerScore = np.zeros(numPlayers)
    nRows = 4
    nCols = 5


    lineMatrix = np.zeros((nRows,nCols,2))
    scoreMatrix = np.zeros((nRows-1,nCols-1))

    playerTurn = 0
    scoredPreviousTurn = 0

    playerColour = ['k','r','g']
    
    for i in range(nRows):
        for j in range(nCols):
            plt.plot(j+1,i+1,'ob', linewidth = 10) 
            

    plt.show(block=False) 

    # Game running infinitely
    while True:

        # Current Player 
        if scoredPreviousTurn == 0:
            playerTurn = playerTurn + 1
        else :
            scoredPreviousTurn = 0
        
        if playerTurn > numPlayers :
            playerTurn = 1

        print("Player " + str(playerTurn) + " turn:")

        validInput = 0
            
        # Player input
        while validInput == 0:
            x1 = int(input('x1: '))
            y1 = int(input('y1: '))
            x2 = int(input('x2: '))
            y2 = int(input('y2: '))
            
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Check to see if valid line (e.g. coordinates are in grid and adjacent)
            if x1 <= nCols and x2 <= nCols and y1 <= nRows and y2 <= nRows and distance == 1:

                # Check to see if line is free or occupied by another player
                if x1 == x2 and lineMatrix[min(y1,y2)-1,x1-1,1] < 1 :
                    lineMatrix[min(y1,y2)-1,x1-1,1] = playerTurn
                    validInput = 1
                    plt.plot([x1, x2], [y1, y2], playerColour[playerTurn-1], linewidth = 10)
                elif y1 == y2 and lineMatrix[y1-1,min(x1,x2)-1,0] < 1 :
                    lineMatrix[y1-1,min(x1,x2)-1,0] = playerTurn
                    validInput = 1
                    plt.plot([x1, x2], [y1, y2], playerColour[playerTurn-1], linewidth = 10)
                else :
                    print('Line already occupied')

            else :
                print('Invalid input')


        plt.show(block=False) 

        # Check to see if complete square and increase score if it is
        for i in range(nRows-1) : 
            for j in range(nCols-1) :    
                if lineMatrix[i-1,j-1,0] > 0 and lineMatrix[i-1,j-1,1] > 0  and lineMatrix[i,j-1,0] > 0 and lineMatrix[i-1,j,1] > 0 and scoreMatrix[i-1,j-1] == 0 :
                   
                    scoreMatrix[i-1,j-1] = playerTurn
                    scoredPreviousTurn = 1 

                    plt.plot(j+0.5, i+0.5, 'x', linewidth = 10, color = playerColour[playerTurn-1])  

                    playerScore[playerTurn-1] += 1

                    
        plt.show(block=False) 



