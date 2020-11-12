clear;
clc;
close all;

numPlayers = input('Number of Players? ');

nRows = 4;
nCols = 5;

lineMatrix = zeros(nRows, nCols, 2);
scoreMatrix = zeros(nRows-1, nCols-1);

playerTurn = 0;
scoredPreviousTurn = 0;

figure();
grid on;
hold on;

playerColour = ['k','r','g'];

for i = 1:nRows
    for j = 1:nCols
        plot(j,i,'ob','LineWidth',10) 
    end
end

while(1)
    
    % Current Player
    if(scoredPreviousTurn == 0)
        playerTurn = playerTurn + 1;
    else
        scoredPreviousTurn = 0;
    end
    
    if(playerTurn > numPlayers)
        playerTurn = 1;
    end
    disp(["Player " + playerTurn + " turn:"])

    validInput = 0;
    while(validInput == 0)
        x1 = input('x1: ');
        y1 = input('y1: ');
        x2 = input('x2: ');
        y2 = input('y2: ');

        % Check to see if valid line
        if(x1 <= nCols && x2 <= nCols && y1 <= nRows && y2 <= nRows && ...
                sqrt((x2-x1)^2 + (y2-y1)^2) == 1)

            % Check to see if occupied line
            if(x1 == x2 && lineMatrix(min(y1,y2),x1,2) < 1 ) 
                lineMatrix(min(y1,y2),x1,2) = playerTurn;
                validInput = 1;
                plot([x1 x2], [y1 y2],playerColour(playerTurn), 'LineWidth',10)
            elseif(y1 == y2 && lineMatrix(y1,min(x1,x2),1) < 1 )
                lineMatrix(y1,min(x1,x2),1) = playerTurn;
                validInput = 1;
                plot([x1 x2], [y1 y2],playerColour(playerTurn), 'LineWidth',10)
            else
                disp('Line already occupied')
            end

        else
            disp('Invalid input')
        end

    end

    % Check to see if complete square and increase score
    for i = 1:nRows-1    
        for j = 1:nCols-1     
            if(lineMatrix(i,j,1) > 0 && lineMatrix(i,j,2) > 0 ...
                    && lineMatrix(i+1,j,1) > 0 && lineMatrix(i,j+1,2) > 0 && scoreMatrix(i,j) == 0)
                scoreMatrix(i,j) = playerTurn;
                scoredPreviousTurn = 1; 
                plot(j+0.5, i+0.5, 'x','LineWidth',10, 'Color' ,playerColour(playerTurn)) 
            end 
        end
    end  
end
