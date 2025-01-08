import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import scipy
import math
import scipy.interpolate

def Plegendre (x,n):
    if n < 1: return 1
    elif n==1: return x
    else: return (((2*n-1)/n)*x*Plegendre(x,n-1)-((n-1)/n)* Plegendre(x,n-2))

##1.a. plot of the 35 legendre polonymial:
x = np.linspace(-1, 1, 1000)
p35_x= Plegendre (x,35)
f1 = plt.figure()
plt.plot (x, p35_x)
plt.title('Legendre Polymonomial n=35')
plt.show()

#1.b plot of the 35 legendre polonymial using build in functions:
f2 = plt.figure()
plt.plot (x, np.polyval(scipy.special.legendre(35), x))
plt.title('Legendre Polymonomial (with build-in function) n=35')
plt.show()

#The difference between the functions
f3 = plt.figure()
plt.plot (x, np.polyval(legendre(35), x)-p35_x)
plt.title('Difference between Legendre Polymonomial n=35')
plt.show()

##1.d
def Gaussian_quadrature_Jn (n):
    ##calculating gamaN:
    def gamaN (n):
        return (np.sqrt(n**2/(4*n**2-1)))
    ##creating a matrix (when betaN=0)
    matrix = np.zeros(shape=(n+1,n+1))
    for y in range(n):
        matrix[y,y+1] = gamaN(y+1)
        matrix[y+1,y] = gamaN(y+1)
    eigon_values,eigon_vectors= np.linalg.eig(matrix)
    ##gama0 is sqrt(2) add calaculation
    W_values =np.zeros(shape=(1,n+1))
    for y in range (n+1):
        W_values[0,y]=2*(eigon_vectors[0,y])**2    
    return ([eigon_values, W_values])
##check: the values of P35(eigon_values of Jn( x0,x1...,x34) should be zero, and theortical sum is 2.
#eigon_values, W_values=Gaussian_quadrature_Jn (34)
#print(np.sum(W_values))
#print (Plegendre (eigon_values,35))

##1.e function that calculates the integral using Gauss method for given n
def Integral_Approx_Gauss (n):
    eigon_values, W_values=Gaussian_quadrature_Jn(n-1)
    sum_integral=0
    for x in range(n):
        sum_integral= sum_integral+W_values[0,x]*np.cos(((eigon_values[x]+11)/10)**24)/10
    return (sum_integral)

##function that calculates the integral using Simpson method for given n
def Integral_Approx_Simpson (n):
    sum_integral=0
    for x in range(n):
        if x==0:
            sum_integral= sum_integral+np.cos((1)**24)
        elif x==n-1:
            sum_integral= sum_integral+np.cos((1.2)**24)
        elif x % 2 == 0:
            sum_integral= sum_integral+2*np.cos((1+0.2*x/(n-1))**24)
        else:
            sum_integral= sum_integral+4*np.cos((1+0.2*x/(n-1))**24)
    return (sum_integral*0.2/(3*(n-1)))

##creating the graphs of the error using two meathods.    
x = (np.linspace(10, 1000, 991, dtype=int))
I_theoretical=-0.01521513770698
#I_Jauss=np.zeros(len(x))
#N=0
#for y in range(10,1001):
#    I_Jauss[y-10]=Integral_Approx_Gauss (y)
#    if abs(I_Jauss[y-10]-I_theoretical)<(10**(-15)):
#        N=y
#        print(y)
#        break
    

#f4 = plt.figure()
#plt.plot (x[0:N-10], abs((I_Jauss-I_theoretical*np.ones(len(x)))[0:N-10]))
#plt.title('Error in Gauss Integration')
#plt.yscale('log')
#plt.show()
## N is the number of points that the error would be smaller than the machine error 
## I stopped at that point because it took a lot of time to make the graph with the entire 991 points.
I_Simpson=np.zeros(len(x))
N=495
for y in range(495):
    I_Simpson[y]=Integral_Approx_Simpson (2*y+11)
    if abs(I_Simpson[y]-I_theoretical)<(10**(-15)):
        N=y
        break
    
#print(N)
## N is the number of points that the error would be smaller than the machine error 
## if it's 495 than the error doesn't smaller than the machine error.
##creating graph of the theoretical error.
y=np.zeros(len(x))
for n in range(len(x)):
    y[n]=524/((9+n)**(4))
    

f5 = plt.figure()
plt.plot (x[1: 1001: 2], abs((I_Simpson-I_theoretical*np.ones(len(x)))[0:N]))
plt.plot(x ,y)
plt.title('Error in Simpson Integration')
plt.yscale('log')
plt.show()


    
##2.a
def f (n):
    return (n**n/scipy.special.factorial(n))

#x = np.linspace(1, 1001,1001)
#sum1= 0
#for y in x:
    #if math.isinf (f(y)):
        #print ("n**n is too large")
        #break
    #else:
        #sum1=sum1+1
#print(sum1)  

##2.b
def new_f (n):
    y=1
    for x in range(n-1):
        y=y*n/(x+1)
    return (y)

#for y in x:
    #if math.isinf (new_f(int(y))):
        #print ("Result is too large")
        #break
    #else:
        #print (y, new_f(int(y)))


##3.a
def Newtons_Divided_Difference (x,y):   
    ##function to calculate the f[xi,..xj]
    def F (n1,n2):
        if n2==0:
            return (y[0])
        if n2-n1==1:
            return ((y[n2]-y[n1])/(x[n2]-x[n1]))
        elif n2-n1>1:
            return ((F(n1+1,n2)-F(n1,n2-1))/(x[n2]-x[n1]))
    ##creating output list.
    list1=[] 
    for n in range(len(x)):
        list1.append(F(0,n))
    return(list1)

##3.b
def Newton_Interpolation (c,x,xnew):
    ##creating function for qi:
    def qi(n):
        qi_list=np.zeros(shape=(len(x)))
        qi_list[0]=1
        for z in range(len(x)-1):
            qi_list[z+1]=(xnew[n]-x[z])*qi_list[z]
        return(qi_list)
    ynew=[]
    for n in range(len(xnew)):
        ynew.append(np.sum((np.array(c)*(qi(n)).transpose())))
    return (ynew)
    
##3.c
xnew=np.random.uniform(-math.pi, math.pi, size=20)
def Interpolation_of_cosx(n,z,a):
    ##equally distance point
    xnew1=xnew
    if z==1:
        x= np.linspace(-math.pi, math.pi, n)
    else:
        xnew1=xnew/math.pi
        x=np.zeros(shape=(n))
        for j in range(n):
            x[j]=np.cos(math.pi*(2*j+1)/(2*n))
    if z==3:
        np.random.shuffle(x)
    y=np.cos(x)
    c=Newtons_Divided_Difference(x,y)
    if a=='Newton':
        return(Newton_Interpolation(c,x,xnew1))
    if a=='Lagrange':
        return(scipy.interpolate.barycentric_interpolate(x, y, xnew1))

numbers_N= np.array([10,20,30,40,50,60])

def y_max(z,a):
    y_max=np.zeros(len(numbers_N))
    for n in range(len(numbers_N)):
        y_max[n]=max(np.absolute(Interpolation_of_cosx(numbers_N[n],z,a)-np.cos(xnew)))
    return(y_max)



f5 = plt.figure()
plt.plot (numbers_N, y_max(1,'Newton'))
plt.yscale('log')
plt.title('Error in Newton Interpolation of cos(x) with equally distributed points')
plt.show()

f6 = plt.figure()
plt.plot (numbers_N, y_max(2,'Newton'))
plt.yscale('log')
plt.title('Error in Newton Interpolation of cos(x) with arranged Chebyshev zeros')
plt.show()

f7 = plt.figure()
plt.plot (numbers_N, y_max(1,'Lagrange'))
plt.yscale('log')
plt.title('Error in Lagrange Interpolation of cos(x) with equally distributed points')
plt.show()

f8 = plt.figure()
plt.plot (numbers_N, y_max(2,'Lagrange'))
plt.yscale('log')
plt.title('Error in Lagrange Interpolation of cos(x) with arranged Chebyshev zeros')
plt.show()

f9 = plt.figure()
plt.plot (numbers_N, y_max(3,'Newton'))
plt.yscale('log')
plt.title('Error in Newton Interpolation of cos(x) with unarranged Chebyshev zeros')
plt.show()

f10 = plt.figure()
plt.plot (numbers_N, y_max(3,'Lagrange'))
plt.yscale('log')
plt.title('Error in Lagrange Interpolation of cos(x) with unarranged Chebyshev zeros')
plt.show()

##3.d let's asuume the taylor series of cosx centered at zero.
def taylor_cosx(n,xnew):
    sum1=np.zeros(shape=(len(xnew)))
    for z in range(n+1):
        if z%2==0:
            sum1=sum1+((-1)**(z/2)/scipy.special.factorial(z))*xnew**z
    return(sum1)

##making the graph of taylor error)
y_max=np.zeros(len(numbers_N))
for n in range(len(numbers_N)):
    y_max[n]=max(np.absolute(taylor_cosx(n,xnew)-np.cos(xnew)))

f11 = plt.figure()
plt.plot (numbers_N, y_max)
plt.yscale('log')
plt.title('Error in Taylor Series of cos(x)')
plt.show()
