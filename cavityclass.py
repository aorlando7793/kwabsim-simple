import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

#======================================================
#Define preliminary functions that are not contained in the Cavity class.

def prop_q(q0,rtm):
    a,b,c,d =[rtm[0][0],rtm[0][1],rtm[1][0],rtm[1][1]]
    q = (q0*a + b)/(q0*c + d)
    return q

#Ray Trace Matricies
#================================================

#Free Space
def m_freeSpace(dist):
    #|1 d|
    #|0 1|
    return np.array([[1,dist],[0,1]])

#Lens
def m_lens(f):
    #|1    0|
    #|-1/f 1|
    if(f==0):
        return np.array([[1,0],[0,1]])
    return np.array([[1,0],[-1/f,1]])

#Mirror
def m_mirror(roc):
    #|1     0|
    #|-2/f  1|
    return m_lens(roc/2)

#Brewster Surface
def m_brewster(n):
    #|1    0|
    #|0    k|
    theta_I = np.arctan(1/n)
    theta_R = np.arcsin((1/n)*np.sin(theta_I))
    k = np.cos(theta_I)/np.cos(theta_R)
    return np.array([[1, 0],[0, k]])

def m_brewster_inv(n):
    theta_I = np.arctan(1/n)
    theta_R = np.arcsin((1/n)*np.sin(theta_I))
    k = np.cos(theta_I)/np.cos(theta_R)
    return np.array([[1, 0],[0, 1/k]])
#======================================================
#Dictionary of optics that corrrespond to their respective ray trace matrices

optics = {
    'D':m_freeSpace,
    'L':m_lens,
    'M':m_mirror,
    'Cx':m_lens,
    'Cy':m_lens,
    'B':m_brewster,
    'B_inv':m_brewster_inv
}

#======================================================
#Preliminary Functions End Here
#======================================================
#Cavity Class:

class Cavity(object):

    def __init__(self, cavity_input, lam):
        if isinstance(cavity_input, list):
            self.cavity = cavity_input
        else:
            self.cavity_file = open(cavity_input, 'r+')
            #self.scavity = self.get_string_cav()
            self.cavity = self.get_cavity()
            #print(self.cavity)
        self.lam = lam
        self.rtm = self.get_RTM()
        if self.check_stab():
            self.is_stable = True
            self.q0 = self.get_q0()
            self.L = self.L()
            #print('Cavity is Stable!!!!')
        else:
            self.is_stable = False
            #print('Cavity is Unstable :(')

    #=========================================
    #Interpretting cavity files:

    def get_cavity(self):
        cavity = []
        for line in self.cavity_file:
            part = []
            for s in line.split():
                try:
                    part.append(float(s))
                except ValueError:
                    part.append(s)
            cavity.append(part)
        return cavity

    #==================================================================
    # Class Methods:

    def unfold_Cav(self):
        return self.cavity[0:]+list(reversed(self.cavity))[1:-1]

    
    def get_RTM(self):
        cavity = self.unfold_Cav()
        rtm = np.array([[1,0],[0,1]])
        second_pass = False
        for optic in cavity:
            if optic[0] == 'B' and not second_pass:
                second_pass = True
                m = optics[optic[0]](optic[1])
            elif optic[0] == 'B' and second_pass:
                m = optics['B_inv'](optic[1])
            else:
                m = optics[optic[0]](optic[1])
            rtm = np.dot(m,rtm)
        return rtm


    def check_stab(self):
        x = (self.rtm[0][0]+self.rtm[1][1]+2)/4.0
        #print('x =',x )
        if((x>0) & (x<1)):
            return True
        return False

    def get_q0(self):
        rtm = self.rtm
        a,b,c,d =[rtm[0][0],rtm[0][1],rtm[1][0],rtm[1][1]]
        rover1 = (d-a)/(2*b)
        w2 = ((self.lam/math.pi)*abs(b)/(math.sqrt(1 - ((a+d)/2)**2)))
        #print(w2*math.pi/(1064*10**(-7)))
        q0 = 1/(rover1 - (self.lam/(math.pi*w2))*1j)
        q0 = q0.real + q0.imag*1j
        return q0

    def q(self, z): 
        q = self.q0
        d = 0 #running cumulative distance of free space elements
        for optic in self.cavity:
            if optic[0] == 'D' and ((z - d) > optic[1]):
                q = prop_q(q, optics[optic[0]](optic[1]))
                d += optic[1]
            elif optic[0] != 'D':
                q = prop_q(q, optics[optic[0]](optic[1]))
            else:
                break
        q += z - d
        return q

    def w(self, z):
        z0 = self.q(z).real
        zR = self.q(z).imag
        #Multiply by 10^3 to get answer in microns
        w = (10**4)*np.sqrt((4*self.lam/math.pi)*(zR + ((z0)**2)/zR))
        return w

    def get_Z_and_W(self,n_points):
        Z = np.linspace(0,self.L,num=n_points)
        W = [self.w(z) for z in Z]
        return Z,W
        

    def insert_optic(self, optic, par, pos):
        #Inserts ['optic',par] as the pos'th position in [cavity]
        cav1 = self.cavity[:pos-1]
        cav2 = self.cavity[pos-1:]
        new_cavity = cav1 + [[optic,par]]+ cav2        
        return Cavity(new_cavity,self.lam)

    def remove_optic(self, pos):
        #Removes pos'th optic
        new_cavity = self.cavity[:pos-1]+self.cavity[pos:]
        return Cavity(new_cavity, self.lam)

    def get_xcav(self):
        xcav_input = []
        for optic in self.cavity:
            if (optic[0] == 'M' and len(optic) == 3):
                #this will account for mirrors at a non-normal AOI
                #effective roc = roc*cos(AOI)
                new_optic = [optic[0], optic[1]*np.cos(optic[2]*math.pi/180)]
                xcav_input.append(new_optic)
            elif optic[0] != 'Cy':
                xcav_input.append(optic)
        return Cavity(xcav_input, self.lam)

    def get_ycav(self):
        ycav_input = []
        for optic in self.cavity:
            if (optic[0] == 'M' and len(optic) == 3):
                #this will account for mirrors at a non-normal AOI
                #effective roc = roc/cos(AOI)
                new_optic = [optic[0], optic[1]/np.cos(optic[2]*math.pi/180)]
                ycav_input.append(new_optic)
            elif (optic[0] != 'B' and optic[0] != 'Cx'):
                ycav_input.append(optic)
        return Cavity(ycav_input, self.lam)

    def L(self):
        space = [optic[1] for optic in self.cavity if optic[0] == 'D']
        return sum(space)

    def div(self, z):
        #Convert back to centimeters
        w = (10**-4)*self.w(z)
        #Calculate divergence and convert from radians to milliradians
        d = (10**3)*(self.q(z).real*4*self.lam/math.pi)/(self.q(z).imag*w)
        return d

    def astigmatism(self, z):
        xcav = self.get_xcav()
        ycav = self.get_ycav()
        q_x = xcav.q(z)
        q_y = ycav.q(z)
        diff_wl = abs(q_x.real - q_y.real)
        avg_Rayleigh = (q_x.imag + q_y.imag)/2
        return diff_wl/avg_Rayleigh

    def analysis(self):
        q = self.q0
        Q = []
        Z_R = []
        W = []
        D = []
        optic_abrev = []
        optic_param = []
        for optic in self.cavity:
            optic_abrev.append(optic[0])
            optic_param.append(optic[1])
            q = prop_q(q, optics[optic[0]](optic[1]))
            Q.append(q)
            #Rayleigh Range
            z_r = q.imag
            Z_R.append(z_r)
            #Spot Size
            w = (10**4)*np.sqrt((4*self.lam/math.pi)*(z_r + ((q.real)**2)/z_r))
            W.append(w)
            #Divergence
            d = (10**3)*(q.real*4*self.lam/math.pi)/(q.imag*w*(10**-4))
            D.append(d)
            
            
        #Rayleigh range at each optic
        #R = [q.imag for q in Q]
        #Spot-size at each optic
        #W = [np.sqrt((4*self.lam/math.pi)*(q.imag + ((q.real)**2)/q.imag)) for q in Q]
        #Divergence at each optic
        #D = [(q.real*4*self.lam/math.pi)/(q.imag*np.sqrt((4*self.lam/math.pi)*(((q.real)**2)/q.imag))) for q in Q]
        #store data in dataframe
        df = pd.DataFrame({'Parameter': optic_param, 'Rayleigh Range(cm)': Z_R, 'Spot Size(microns)': W, 'Divergence(mrads) ': D}, index=optic_abrev)
        return df

    def plot_w(self, n_points):
        M_2x = float(input('Enter M-Squared for x-axis:    '))
        M_2y = float(input('Enter M-Squared for y-axis:    '))
        LAM_x = self.lam * M_2x
        LAM_y = self.lam * M_2y

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Z(cm)')
        ax.set_ylabel('Spot Size(microns)')
        ax.set_ylim(0,3000)

        xcav = Cavity(self.get_xcav().cavity, LAM_x)
        ycav = Cavity(self.get_ycav().cavity, LAM_y)

        if xcav.cavity == ycav.cavity:
            Z,W = self.get_Z_and_W(n_points)
            self.Z, self.W = Z, W
            ax.plot(Z,W)
        else:
            Zx, Wx = xcav.get_Z_and_W(n_points)
            Zy, Wy = ycav.get_Z_and_W(n_points)
            ax.plot(Zx,Wx,label='xcav')
            ax.plot(Zy,Wy,label='ycav')
            ax.legend()
            #Create class attributes
            self.Zx, self.Wx = Zx, Wx
            self.Zy, self.Wy = Zy, Wy
        print('X-Axis')
        print(xcav.analysis())
        print('Y-Axis')
        print(ycav.analysis())

if __name__ == "__main__":
    M_2 = 1
    LAM = 1064*10**(-7) * M_2
    
    cav_parts = [
        ['M', 100, 0],
        ['D', 10],
        ['D', 10],
        ['M', -70, 25],
        ['D', 3.5],
        ['L', 19],
        ['D', 3.5],
        ['M', -100, 25],
        ['D', 19.5],
        ['B', 1.5],
        ['D', 5],
        ['M', 0, 0]
        ]
    
    laser = Cavity(cav_parts, LAM)
    
    #test spot size plot
    laser.plot_w(250)
    plt.show()


    #test divergence plot
    x_laser = laser.get_xcav()
    y_laser = laser.get_ycav()
    Dx = [x_laser.div(zx) for zx in laser.Zx]#running laser.plot creates class attribute laser.Zx, so this works
    Dy = [y_laser.div(zy) for zy in laser.Zy]#see above comment
    plt.plot(laser.Zx,Dx)
    plt.plot(laser.Zy,Dy)
    plt.show()

    #print astigmatism at THG
    print('Astigmatism at THG', laser.astigmatism(laser.L - 5.1))
    Z = np.linspace(0,laser.L,num=200)
    A = [laser.astigmatism(z) for z in Z]
    plt.plot(Z,A)
    plt.show()

    #print dataframe of cavity analysis
    #print('X-Axis')
    #print(x_laser.analysis())
    #print('Y-Axis')
    #print(y_laser.analysis())