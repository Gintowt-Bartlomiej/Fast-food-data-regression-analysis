import csv
import  numpy as np
from numpy.random import *
from numpy import*
import math
import matplotlib.pyplot as plt
import scipy.stats
import statistics
import seaborn as sns
import csv
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis
from distfit import distfit
import statsmodels.tsa.api as smt

def dystrybuanta_emp(x):
    n = len(x)
    F = empty(n, dtype= float)
    t = np.linspace(min(x), max(x), n)
    for i in range(n):
        F[i] = sum(x <= t[i])
        F[i] = F[i]/n
    return F

def raport_1(plik):
    with open(plik, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    x = [float(i[3]) for i in data[1:]]
    y = [float(i[5]) for i in data[1:]]
    x2 = np.array([float(i[3]) for i in data[1:]])
    y2 = np.array([float(i[5]) for i in data[1:]])

    """

    ZMIENNA ZALEŻNA
    
    """
    """Tłuszcze"""
    dist = distfit()
    model = dist.fit_transform(y2)
    dist.plot()
    t = np.linspace(min(y), max(y), len(y))
    plt.hist(y, bins=15, alpha=0.7, edgecolor='k', density=True)
    plt.plot(t, gamma.pdf(t, a=2.36122, loc=-0.679726, scale=11.5491), label="Gamma(2.36, -0.678, 11.5)")
    plt.title("Histogram i gęstość empiryczna tłuszczu")
    plt.xlabel('Tłuszcz')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.show()


    t = np.linspace(min(y), max(y), len(y))
    plt.title("Dystrybuanta empiryczna tłuszczu")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.plot(t, dystrybuanta_emp(y))
    plt.show()
    
    dane = y
    kwartyle=np.quantile(dane,[0.25,0.5,0.75])
    print("TŁUSZCZE")
    print("X średnie",np.average(dane))
    print("Mediana",kwartyle[1])
    print("Kwartyle",kwartyle[0],kwartyle[2])
    print("Rozstęp z próby",max(dane)-min(dane))
    print("Rozstęp międzykwartylowy",kwartyle[2]-kwartyle[0])
    print("Wariancja z próby",np.var(dane))
    print("Odchylenie standardowe z próby",np.std(dane))
    print("Wspólczynnik zmienności",np.std(dane)/np.average(dane))
    print("Skośność",skew(dane))
    print("Kurtoza",kurtosis(dane))
    
    
    """
    
    ZMIENNA NIEZALEŻNA
    
    """
    """Kalorie"""
    # dist = distfit()
    # model = dist.fit_transform(x2)
    # dist.plot()
    t = np.linspace(min(x), max(x), len(x))
    plt.hist(x, bins=15, alpha=0.7, edgecolor='k', density=True)
    plt.plot(t, beta.pdf(t, a=3.66778, b=32.1245, loc=-21.8256, scale=5416.93), label="Beta(3.7, 32.12, -21.83, 5416.93)")
    plt.title("Histogram i gęstość empiryczna kalorii")
    plt.xlabel('Kalorie')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.show()
    
    t = np.linspace(min(x), max(x), len(x))
    plt.title("Dystrybuanta empiryczna kalorii")
    plt.plot(t, dystrybuanta_emp(x))
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.show()
    
    dane = x
    kwartyle=np.quantile(dane,[0.25,0.5,0.75])
    print(" ")
    print("Kalorie")
    print("X średnie",np.average(dane))
    print("Mediana",kwartyle[1])
    print("Kwartyle",kwartyle[0],kwartyle[2])
    print("Rozstęp z próby",max(dane)-min(dane))
    print("Rozstęp międzykwartylowy",kwartyle[2]-kwartyle[0])
    print("Wariancja z próby",np.var(dane))
    print("Odchylenie standardowe z próby",np.std(dane))
    print("Wspólczynnik zmienności",np.std(dane)/np.average(dane))
    print("Skośność",skew(dane))
    print("Kurtoza",kurtosis(dane))
    
    
    """"
    
    
    ANALIZA ZALEŻNOŚCI LINIOWEJ
    
    
    """
    plt.title("Wykres rozproszczenia tłuszczu od kalorii")
    plt.scatter(x,y)
    plt.xlabel('Kalorie')
    plt.ylabel('Tłuszcz')
    plt.show()
    
    """PROSTA REGRESJI"""
    # #Punktowa
    xs=np.mean(x)
    ys=np.mean(y)
    g = [ x[i]*(y[i]-ys) for i in range(len(x))]
    d = [ (x[i]-xs)**2 for i in range(len(x))]
    b1 = sum(g)/sum(d)
    b0 = np.mean(y) - b1*np.mean(x)
    yd = [b0+b1*i for i in x]
    

    print(b1)
    print(b0)

    plt.title("Dopasowana prosta regresji")
    plt.scatter(x, y)
    plt.plot(x, yd, label="Prosta regresji", color='red')
    plt.xlabel('Kalorie')
    plt.ylabel('Tłuszcz')
    plt.legend()
    plt.show()
    #Przedziałowa
    alfa=0.05
    n=len(x)
    q=scipy.stats.t.ppf(1 - alfa / 2, n - 2)
    pom = [(y[i] - b0 - b1 * x[i]) ** 2 for i in range(len(y))]
    S = (sum(pom) / (n - 2)) ** (1 / 2)
    ub0 = [b0 - q * S * (1 / n + xs ** 2 / sum(d)) ** (1 / 2),
           b0 + q * S * (1 / n + xs ** 2 / sum(d)) ** (1 / 2)]
    ub1 = [b1 - q * S / sum(d) ** (1 / 2), b1 + q * S / sum(d) ** (1 / 2)]
    print("Przedział ufności b0",ub0)
    print("Przedział ufności b1",ub1)
    
    """Ocena poziomu zależnowści"""
    SST=sum([(y[i]-ys)**2 for i in range(len(y))])
    SSE=sum([(y[i]-yd[i])**2 for i in range(len(y))])
    SSR=sum([(yd[i]-ys)**2 for i in range(len(y))])
    print("Korelacja Pearsona",scipy.stats.pearsonr(x,y)[0])
    print("SST",SST)
    print("SSE",SSE)
    print("SSR",SSR)
    
    """Predykcja"""
    nn=485
    alfa=0.05
    q2 = scipy.stats.t.ppf(1 - alfa / 2, nn - 2)
    xsort=[i for i in x]
    ysort = [i for i in y]
    x2 =[]
    y2=[]
    for i in range(len(x)):
        inde=xsort.index(min(xsort))
        x2.append(xsort[inde])
        y2.append(ysort[inde])
        xsort.pop(inde)
        ysort.pop(inde)
    ys2 = sum(y2[:nn]) / nn
    xs2 = sum(x2[:nn]) / nn
    g = [(x2[i] - xs2) * y2[i] for i in range(nn)]
    d = [(x2[i] - xs2) ** 2 for i in range(nn)]
    b1 = sum(g) / sum(d)
    b0 = ys2 - b1 * xs2
    yd2 = [b0 + b1 * i for i in x2[:nn]]
    pom = [(y2[i] - b0 - b1 * x2[i]) ** 2 for i in range(nn)]
    S = (sum(pom) / (nn - 2))
    nd = []
    ng = []
    for i in range(len(y2)-nn):
        n1 = b1 * x2[nn + i] + b0 - q2 * S ** (1 / 2) * (1 + 1 / nn + (x2[nn + i] - xs2) ** 2 / sum(d)) ** (
                    1 / 2)
        n2 = b1 * x2[nn + i] + b0 + q2 * S ** (1 / 2) * (1 + 1 / nn + (x2[nn + i] - xs2) ** 2 / sum(d)) ** (
                    1 / 2)
        nd.append(n1)
        ng.append(n2)
    plt.scatter(x2, y2, s=5)
    plt.plot(x2[nn:], ng, color="lime",label="Górny przedział ufności")
    plt.plot(x2[nn:], nd, color="lime",label="Dolny przedział ufności")
    plt.plot(x2[:nn], yd2, color="red",label="Prosta regresji")
    plt.title("Predykcja dla "+str(len(x)-nn)+" ostatnich danych testowych")
    plt.xlabel('Kalorie')
    plt.ylabel('Tłuszcz')
    plt.legend()
    plt.show()
    
    
    
    #wywołanie na nowo
    xs = np.mean(x)
    ys = np.mean(y)
    g = [x[i] * (y[i] - ys) for i in range(len(x))]
    d = [(x[i] - xs) ** 2 for i in range(len(x))]
    b1 = sum(g) / sum(d)
    b0 = np.mean(y) - b1 * np.mean(x)
    yd = [b0 + b1 * i for i in x]
    #
    """
    
    
    RESIDUA
    
    
    """
    e=[y[i]-yd[i] for i in range(len(y))]
    print(np.mean(e))
    print(np.var(e))
    sr_e = [np.mean(e) for i in range(len(y))]
    sr_teor = [0 for i in range(len(y))]
    ind = [i for i in range(len(y))]
    plt.title("Błędy")
    plt.xlabel('n')
    plt.ylabel('e_i')
    plt.plot(ind, sr_e, label="Średnia błędów", color='red')
    plt.plot(ind, sr_teor, label="Średnia = 0", color='green')
    plt.scatter(ind, e,  s=0.6)
    plt.show()
    
    e2 = np.array(e)
    dist = distfit()
    model = dist.fit_transform(e2)
    dist.plot()
    t = np.linspace(min(e), max(e), len(e))
    plt.title("Histogram błędów")
    plt.hist(e, bins=15, alpha=0.7, edgecolor='k', density=True)
    plt.plot(t, norm.pdf(t, loc=0, scale=np.sqrt(np.var(e))), label="Norm(0, 7.55)")
    #plt.plot(tx, scipy.stats.t.pdf(tx, loc=0.404, scale=5.196), label="t(0.404, 5.196)")
    plt.xlabel('Błędy')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.show()
    
    
    acf = smt.graphics.plot_acf(e, lags=40 , alpha=0.05, title="Empiryczna autokorelacja residuów")
    plt.xlabel('Lag')
    plt.ylabel('Gamma(h)')
    plt.show()
    
    print(" ")
    print("Test normalności Szapiro-Wilka")
    print(shapiro(e))
    
    
    
    """ Pozbycie sie wartosci odstających kwartylami """
    kwartyle = np.quantile(e, [0.25, 0.5, 0.75])
    IQR=kwartyle[2]-kwartyle[0]
    dolna=kwartyle[0]-IQR
    gorna=kwartyle[2]+IQR
    xn=[]
    yn=[]
    xu=[]
    yu=[]
    ino=[]
    iu=[]
    for i in range(len(y)):
        if (dolna<=e[i]<=gorna):
            ino.append(i)
            yn.append(y[i])
            xn.append(x[i])
        else:
            iu.append(i)
            yu.append(y[i])
            xu.append(x[i])
    ys2=np.mean(yn)
    xs2=np.mean(xn)
    g=[x[i]*(y[i]-ys2) for i in ino]
    d=[(x[i]-xs2)**2 for i in ino]
    b1=sum(g)/sum(d)
    b0=ys2-b1*xs2
    yd2 = [b0 + b1 * i for i in xn]
    e2=[y[i]-yd2[i] for i in range(len(yd2))]
    
    t = np.linspace(min(e2), max(e2), len(e2))
    plt.title("Histogram błędów")
    plt.hist(e2, bins=15, alpha=0.7, edgecolor='k', density=True)
    plt.plot(t, norm.pdf(t, loc= np.mean(e2), scale=sqrt(np.var(e2))), label="Norm(1.9, 21)")
    #plt.plot(tx, scipy.stats.t.pdf(tx, loc=0.404, scale=5.196), label="t(0.404, 5.196)")
    plt.xlabel('Błędy')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.show()
    print(np.mean(e2))
    print(np.var(e2))
    
    acf = smt.graphics.plot_acf(e2, lags=40 , alpha=0.05, title="Empiryczna autokorelacja residuów")
    plt.xlabel('Lag')
    plt.ylabel('Gamma(h)')
    plt.show()
    
    print(" ")
    print("Test normalności Szapiro-Wilka")
    print(shapiro(e2))
    
    
    """" MODULACJA DANYCH WEJŚCIOWYCH """
    #yn=[np.log(i) for i in y]
    #yn=[np.sqrt(i) for i in y]
    yn=[i**(1/3) for i in y]
    ys = np.mean(yn)
    xs = np.mean(x)
    g = [x[i] * (yn[i] - ys) for i in range(len(x))]
    d = [(x[i] - xs) ** 2 for i in range(len(x))]
    b1 = sum(g) / sum(d)
    b0 = ys - b1 * xs
    yd = [b0 + b1 * i for i in x]
    xe=[i for i in x]
    xe.sort()
    #ye=[np.exp(b0)*np.exp(b1*i) for i in xe]
    #ye=[b0**2*(b1*i)**2 for i in xe]
    ye=[b0**3*(b1*i)**3 for i in xe]
    plt.scatter(x, y, s=0.8)
    plt.plot(xe,ye,color="red")
    plt.show()
    plt.scatter(x, yn, s=0.8)
    plt.plot(x,yd,color="red")
    plt.show()

raport_1(r"fastfood.csv")