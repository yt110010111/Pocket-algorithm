import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from Perceptron import LinearPerceptron



def pocket(perceptron: LinearPerceptron) -> np.ndarray:
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  
    """

    st = time.time()
    weight_matrix = np.zeros(3)
    converged = False
    count = 0
    data = perceptron.data
    labels = perceptron.label
    best_weight = weight_matrix.copy()
    best_error_count = 2000
    max=0
    while  max<10000:
        
        max+=1
        error_count = 0  
        for i in range(len(data)):
            x = np.array(data[i])
            y = int(labels[i])
            count += 1
            if np.sign(np.dot(weight_matrix, x)) != np.sign(y):
                error_count += 1  
                weight_matrix += y * x
                
                  
        if error_count <= best_error_count:
            best_error_count = error_count
            best_weight = weight_matrix.copy()
    
        

    weight_matrix = best_weight
    et = time.time()
    execution_time = et - st
    
    
    lll=0
    for i in range(len(data)):
        z = np.array(data[i])
        hhh=int(labels[i])
        if np.sign(np.dot(weight_matrix, z)) != np.sign(hhh):

            lll+=1
            
    
    print("Best error count:",lll)
    print("Accuracy: ", 100*(1 - (lll / len(data))),"%")
    print("Execution time:", execution_time, "seconds")
    print("Count:", count)

    return weight_matrix

   
def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    perceptron = LinearPerceptron(args.path)
    updated_weight = pocket(perceptron=perceptron)

    #############################################
    x_values = np.linspace(min(perceptron.cor_x_pos + perceptron.cor_x_neg), 
                       max(perceptron.cor_x_pos + perceptron.cor_x_neg), 100)

    
    y_values = 3 * x_values + 5
    #weight line
    a=updated_weight[1]
    b=updated_weight[2]
    c=updated_weight[0]
    xl=np.linspace(-1000,1000,1000)
    yl = (-a / b) * xl - (c / b)
    plt.plot(xl, yl, label=f'updated_weight:  {a}x + {b}y + {c} = 0',color='black')
    
    # Plot the line
    plt.plot(x_values, y_values, label='baseline:  y = 3x + 5', color='green')
    plt.plot(perceptron.cor_x_pos, perceptron.cor_y_pos, 'bo', label='Positive')
    plt.plot(perceptron.cor_x_neg, perceptron.cor_y_neg, 'ro', label='Negative')
    
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Points')
    plt.grid(True)
    plt.legend() 
    plt.show()
    #                                           #
    #                                           #
    #############################################

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser(description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
