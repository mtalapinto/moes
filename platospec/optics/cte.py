
def recalc(par, mat, t):
    file_cte = open('optics/CTE.txt')
    tref = 25.
    for line in file_cte:
        lineaux = line.split()
        if str(lineaux[0]) == str(mat):
            if par != 0.:                
                delta_par = par*float(lineaux[1])*(tref - t)
                par = par + delta_par
            else:
                delta_par = float(lineaux[1])*(tref - t)
                par = par + delta_par

    return par


if __name__ == '__main__':

    d = recalc(0, 'zerodur', 12)
    print(d)



