from pulp import *
from random import *
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

def test_read(csv_file_name):
    data_frame = pd.read_csv(csv_file_name)
    print(data_frame.head())
    # Y = data_frame["B"].iloc[0]
    # print(Y)
    # print(type(Y))
    # Y = np.asarray(eval(Y))
    # print(Y[0])

def generate_csv(file_name ,num_var, num_const, num_prob):

    dict_input = {"A": [], "B": [] , "C" : [], "Solution" : []}

    for k in tqdm(range(num_prob)):
        C = []
        for i in range(num_var):
            C.append(uniform( 0,1))


        A = []
        B = []
        for j in range(num_const):
            A.append([])
            for i in range(num_var):
                A[j].append(uniform(0,5))
            B.append(uniform(2,5))

        # print("A ",A)
        # print(len(A))
        # print("B ",B)
        # print(len(B))
        # print("C ",C)
        # print(len(C))
        dict_input["A"].append(A)
        dict_input["B"].append(B)
        dict_input["C"].append(C)

        
        prob = LpProblem("TheCEIProblem",LpMinimize)

        

        prob_var = [LpVariable("Var{}".format(i),0)for i in range(num_var)]
        # prob_var = LpVariable.dicts("Vars",list_var,0)

        prob+= lpSum([prob_var[i] * C[i] for i in range(num_var)]), "CostFunction"

        for j in range(num_const):
            prob += lpSum([prob_var[i] * A[j][i] for i in range(num_var)]) >= B[j], "ProblemLine{}".format(j)

        # The problem data is written to an .lp file
        prob.writeLP("test.lp")

        # The problem is solved using PuLP's choice of Solver
        prob.solve()

        # The status of the solution is printed to the screen
        # print("Status:", LpStatus[prob.status]) 

        # Each of the variables is printed with it's resolved optimum value
        Solution = []
        for v in prob.variables():
            # print(v.name, "=", v.varValue)
            Solution.append(v.varValue)
        # The optimised objective function value is printed to the screen    
        # print("Total Cost = ", value(prob.objective))
        dict_input["Solution"].append(Solution)

    df_input = pd.DataFrame(dict_input)
    # df_input = df_input.sample(frac=1).reset_index(drop=True)
    # print(df_input.head())
    df_input.to_csv(file_name, index=False)




    # prob.solve()
    
    # for v in prob.variables():
    #     Vars.append(v.varValue)






    test_read(file_name)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_var", type=int, default=5,
                        help="Number of variables (default: 5)")
    parser.add_argument("--num_const", type=int, default=4,
                        help="number of constrains (default: 4)")
    parser.add_argument("--num_prob", type=int, default=10,
                        help="number of problems to generate (default: 10)")


    args = parser.parse_args()

    generate_csv('./../DATA/input.csv', args.num_var, args.num_const, args.num_prob)

    



    


if __name__ == "__main__":
    main()