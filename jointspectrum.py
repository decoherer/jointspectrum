
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,pi,log,sin,cos,inf,sqrt
from wavedata import Wave2D,Wave,list2str,track,gaussianproduct,sinc,wrange
import datetime
from joblib import Memory
memory = Memory('j:/backup', verbose=0) # use as @memory.cache

class JointSpectrum(Wave2D):
    def ghz2invnm(self,λx,λy):
        return JointSpectrum(self,xs=self.xs/299792458+1/λx,ys=self.ys/299792458+1/λy)
    def invnm2ghz(self,x0=None,y0=None):
        x0 = x0 if x0 is not None else self.xs[len(self.xs)//2]
        y0 = y0 if y0 is not None else self.ys[len(self.ys)//2]
        return JointSpectrum(self,xs=299792458*(self.xs-x0),ys=299792458*(self.ys-y0))
    def fplot(self,**plotargs):
        self.invnm2ghz().plot(x='Δf (GHz)',y='Δf (GHz)',legendtext=f"P = {1/schmidtK(self):g}",**plotargs)
        return self
def invnm2ghz(w):
    if 1==len(w.shape):
        return Wave(w.y,299792458*(w.x-w.x[len(w)//2]))
    return JointSpectrum(w,xs=299792458*(w.xs-w.xs[len(w.xs)//2]),ys=299792458*(w.ys-w.ys[len(w.ys)//2]))
def hermitegaussmode(n,x,peaknorm=False):
    cc = [1,0.857764,0.810358,0.78268,0.763404,0.74873,0.736941,0.727123,0.71873,0.711415,0.704941,0.699142,0.693895,0.689107,0.684708,0.680642,0.676863,0.673335,0.670029,0.666918,0.663983,0.661204,0.658567,0.656059,0.653667,0.651383,0.649197,0.647101,0.645089,0.643155,0.641292,0.639496,0.637763,0.636088,0.634468,0.632899,0.631379,0.629904,0.628473,0.627082,0.62573,0.624414,0.623133,0.621885,0.620669,0.619482,0.618324,0.617193,0.616089,0.615009,0.613953,0.61292,0.611909,0.610919,0.609949,0.608999,0.608067,0.607154,0.606258,0.605378,0.604515,0.603668,0.602835,0.602017,0.601213,0.600423,0.599646,0.598882,0.59813,0.59739,0.596662,0.595945,0.595239,0.594544,0.593859,0.593184,0.592518,0.591863,0.591216,0.590579,0.58995,0.58933,0.588718,0.588115,0.587519,0.586931,0.58635,0.585777,0.585211,0.584651,0.584099,0.583553,0.583014,0.582482,0.581955,0.581435,0.58092,0.580411,0.579908,0.579411]
    c = 1/cc[n] if peaknorm else 1
    from scipy.special import hermite as H
    return c * H(n)(x) * np.exp(-0.5*x*x) / np.sqrt(2.**n * np.math.factorial(n))
def hermitegausstemporalmode(n,dt,num=501): # dt = FWHM pulse width of mode 0
        σ = 0.5/sqrt(log(2))*dt
        xs = np.linspace(-dt*sqrt(16+n),+dt*sqrt(16+n),num)
        return Wave(hermitegaussmode(n,xs/σ),xs)
def hermitegaussspectralmode(n,λ,dt,dλ=30,num=501,ghz=False,debug=False):
    def energy(x): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # in Hz
        σf = 0.5/sqrt(log(2))*ΔfFWHM
        return 1/(2*pi*σf) * hermitegaussmode(n,(1/λ-x)*1e9*299792458/σf)
    f,df = 1/λ,dλ/λ**2
    xs = np.linspace(f-df,f+df,num)
    fs = (xs-1/λ)*299792458
    w = Wave(energy(xs)*1e9*299792458*sqrt(2*pi),xs)
    if debug and 0==n:
        b = fjcahop(n,λ,λ,dt,L=0,dλ1=dλ,dλ3=dλ,num1=num,num3=num)
        print(b[:,b.ny//2].area(),w.area())
        Wave.plots(w,b[:,b.ny//2],c='0k',l='04')
    return Wave(w.y,fs) if ghz else w
def efficiency(w1,w0):
    return w1.sqr().volume()/w0.sqr().volume()
def integrate(f,g): # computes ∫ f(x,y) g(y,z) dy
    assert f.dy==g.dx and np.allclose(f.ys,g.xs)
    # return g.dx * np.array(f) @ np.array(g)
    return JointSpectrum(g.dx * np.array(f) @ np.array(g),xs=f.xs,ys=g.ys)
def innerproduct(u,g): # computes ∫ u(y) g(y,z) dy if g is 2D else ∫ u(y) g(y) dy 
    def area(a,b): # ∫ a(x) b(x) dx
        assert np.allclose(a.x,b.x)
        return (a*b).area()
    return area(u,g) if 1==len(g.shape) else Wave([area(u,b) for b in g.xwaves()],g.ys)
def schmidtK(ww):
    # u,s,vT = np.linalg.svd(ww, full_matrices=False, compute_uv=True, hermitian=False)
    # v = np.array(vT).transpose()
    s = np.linalg.svd(ww, full_matrices=False, compute_uv=False, hermitian=False)
    s = -np.sort(-s/np.sqrt(np.sum(s**2)))
    return 1/np.sum(np.abs(s)**4)
def schmidtnumber(ww,res=400,keep=None,invspace=False):
    modes = [m for m,f,g in schmidtdecomposition(ww,res=res,keep=keep,invspace=invspace)]
    return 1/sum([np.abs(m)**4 for m in modes])
def schmidtdecomposition(ww,res=400,keep=None,invspace=False):
    import pyqentangle
    f,x0,x1,y0,y1 = ww, ww.xs[0], ww.xs[-1], ww.ys[0], ww.ys[-1]
    if invspace: f,x0,x1,y0,y1 = lambda x,y:ww(1/x,1/y), 1/ww.xs[-1], 1/ww.xs[0], 1/ww.ys[-1], 1/ww.ys[0]
    return pyqentangle.continuous_schmidt_decomposition(f,x0,x1,y0,y1, nb_x1=res, nb_x2=res, keep=keep)
def discretedeltafunction(x,x0,dx): # constant area=1
    return np.maximum(0,1-np.abs((x-x0)/dx))/dx
def discretedeltafunctiontest():
    xs = np.linspace(780-5,780+5,21)
    dx = xs[1]-xs[0]
    ddf = discretedeltafunction
    # print(ddf(780,780.1,dx),ddf(780+dx,780.1,dx),ddf(780,780.1,dx)+ddf(780+dx,780.1,dx))
    ps = [778+p*np.pi/4 for p in np.arange(5)]
    ws = [Wave(ddf(xs,p,dx),xs) for p in ps]
    Wave.plots(*[w.rename(f"{w.area():g} area") for w in ws],m='o',ylim=(-1,4),grid=1)
def nptophat(x,x0,x1):
    return np.heaviside(x-x0,0.5)-np.heaviside(x-x1,0.5)
def tophat(x,x0,x1,dx): # guaranteed constant area=x1-x0 on dx spaced grid
    tri = 0.5 + 0.5*(x1-x0)/dx - np.abs(x-0.5*(x0+x1))/dx
    return np.maximum(0,np.minimum(1,tri))
def tophattest(Δx=np.exp(1)):
    xs = np.linspace(780-6,780+6,25)
    ps = [778+p*np.pi/4 for p in np.arange(6)]
    us = [Wave(nptophat(xs,p,p+Δx),xs) for p in ps]
    vs = [Wave(  tophat(xs,p,p+Δx,xs[1]-xs[0]),xs) for p in ps]
    us = [2+u.rename(f"{u.area():.4f} area numpy").setplot(m='D') for u in us]
    vs = [v.rename(f"{v.area():.4f} area tophat").setplot(m='o') for v in vs]
    Wave.plots(*us,*vs,c='012345',ylim=(-1,4),grid=1,seed=1,corner='lower left',fontsize=8)
def hermitetest():
    from scipy.special import hermite as H
    def u(n,x):
        return H(n)(x) * np.exp(-0.5*x*x) / np.sqrt(2**n * np.math.factorial(n))
    x = np.linspace(-5,5,101)
    ws = [Wave(H(n)(x),x,n) for n in range(5)]
    Wave.plots(*ws,xlim=(-5,5),ylim=(-20,20),grid=1)
    us = [Wave(u(n,x),x,n) for n in range(5)]
    Wave.plots(*us,xlim=(-5,5),ylim=(-2,2),grid=1)
def gaussianproducttest():
    print('θ,σ,ρ',list2str(gaussianproduct(0,10,1, 0,10,1,degrees=1)))
    print('θ,σ,ρ',list2str(gaussianproduct(0,10,1, +90,10,1,degrees=1)))
    print('θ,σ,ρ',list2str(gaussianproduct(+45,10,1, 0,1000,1000,degrees=1)))
    print('θ,σ,ρ',list2str(gaussianproduct(0,10,1, 0,inf,inf,degrees=1)))
    print('θ,σ,ρ',list2str(gaussianproduct(+45,10,1, -45,10,1,degrees=1)))
    print('θ,σ,ρ',list2str(gaussianproduct(-5,10,1, +5,10,1,degrees=1)))
    print('θ,σ,ρ',list2str(gaussianproduct(-5,inf,1, +5,inf,1,degrees=1)))
def ellipticalgaussianmodetest(a=2,b=0.7,c=0.5,num=3):
    p = ellipticalgaussianpurity(a,b,c)
    α1,α2,µ = np.sqrt(2*a*p), np.sqrt(2*c*p), np.sign(b)*np.sqrt((1-p)/(1+p))
    xs,ys = np.linspace(-4,4,101),np.linspace(-4,4,101)
    yy,xx = np.meshgrid(ys,xs)
    zz = exp(-a*xx**2) * exp(+2*b*xx*yy) * exp(-c*yy**2)
    f0 = JointSpectrum(zz,xs=xs,ys=ys)
    f0.plot(legendtext=f'rotated elliptical gaussian\nP={p:g}',save='rotated elliptical gaussian')
    def u(n,x):
        from scipy.special import hermite as H
        return H(n)(x) * np.exp(-0.5*x*x) / np.sqrt(2**n * np.math.factorial(n))
    def λn(n,µ):
        return (1-µ**2) * µ**(2*n)
    fs = [u(n,α1*f0.xx) * u(n,α2*f0.yy) for n in range(num)]
    λs = [λn(n,µ) for n in range(num)]
    f = sum(np.sqrt(λ)*f for λ,f in zip(λs,fs))
    f.plot(legendtext=f'sum of first {num} schmidt modes\nP={sum(λ**2 for λ in λs):g}',save=f'sum of first {num} schmidt modes')
    print('p',p,sum(λ**2 for λ in λs))
def phasematchangletest():
    θ = fjsa(780,2340,0.0002,L=10,sell='ktpwg',Type='yzy',plotangles=0,getangle=1)
    print('θ',θ,phasematchangle(780,2340,sell='ktpwg',Type='yzy',degrees=1))
    θ = fjsa(2340,780,0.0002,L=10,sell='ktpwg',Type='yzy',plotangles=0,getangle=1)
    print('θ',θ,phasematchangle(2340,780,sell='ktpwg',Type='yzy',degrees=1))
    ϕ = fjca(2340,780,0.0002,L=10,sell='ktpwg',Type='zzz',plotangles=1,getangle=1)
    print('ϕ',ϕ,frequencyconversionangle(2340,1170,sell='ktpwg',Type='zzz',degrees=1))
    ϕ = fjca(2340,780,0.0002,L=3.5,sell='ktpwg',Type='zzz',plotangles=0,getangle=1)
    print('ϕ',ϕ,frequencyconversionangle(2340,1170,sell='ktpwg',Type='zzz',degrees=1))
def abc(θ,σ,ρ): # https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
    a = 0.5*cos(θ)**2/σ**2 + 0.5*sin(θ)**2/ρ**2
    b = 0.25*sin(2*θ)/ρ**2 - 0.25*sin(2*θ)/σ**2
    c = 0.5*sin(θ)**2/σ**2 + 0.5*cos(θ)**2/ρ**2
    return (a,b,c)
def θσρ(a,b,c):
    θ = -0.5*np.arctan2(2*b,a-c)
    assert -pi/2<=θ<=+pi/2, f"b{b:g} a-c{a-c:g} θ{θ:g}"
    # print(f"b:{b:g} a-c:{a-c:g} θ:{θ:g}")
    σ = np.sqrt(0.5/(a*cos(θ)**2 - 2*b*cos(θ)*sin(θ) + c*sin(θ)**2))
    ρ = np.sqrt(0.5/(a*sin(θ)**2 + 2*b*cos(θ)*sin(θ) + c*cos(θ)**2))
    # return (θ,σ,ρ)
    return normalizeθσρ(θ,σ,ρ)
def normalizeθσρ(θ,σ,ρ):
    def θnorm(θ): # restrict to ±π, e.g. 135° → -45°
        return (θ+0.5*pi)%pi - 0.5*pi
    θ,σ,ρ = np.where(σ>ρ,(θnorm(θ),σ,ρ),(θnorm(θ+0.5*pi),ρ,σ))
    # θ,σ,ρ = np.where(np.allclose(σ,ρ)&(np.abs(θ)>=0.25*pi),(θnorm(θ+0.5*pi),ρ,σ),(θnorm(θ),σ,ρ))
    θ,σ,ρ = np.where(np.allclose(σ,ρ),(0,σ,σ),(θnorm(θ),σ,ρ))
    return θ,σ,ρ
def ellipticalgaussianpurity(a,b,c):
    return np.sqrt(1-b**2/(a*c))
def apodizedmaxpurity(θ): # θ = phase matching angle
    r = np.abs(0.5*sin(2*θ)) # r = p/q = optimal bandwidth ratio
    return ellipticalgaussianpurity(a=r+sin(θ)**2, b=-r-sin(θ)*cos(θ), c=r+cos(θ)**2)
def fcapodizedmaxpurity(ϕ): # ϕ = phase matching angle
    r = np.abs(0.5*sin(2*ϕ)) # r = p/q = optimal bandwidth ratio
    return ellipticalgaussianpurity(a=r+sin(ϕ)**2, b=+r-sin(ϕ)*cos(ϕ), c=r+cos(ϕ)**2)
def fcspdcpurity(P1,P2): # P1 = JSA purity, P2 = JCA purity
    return np.sqrt(1-(1-P1*P1)*(1-P2*P2)/(1+P1*P2)**2)
# wavelength based joint amplitudes
def fλjsa(λ1,λ2,dt,L=10,sell='ktpwg',Type='yzy',dλ1=30,dλ2=300,num1=201,num2=501,apodized=True): # shape=(num1,num2)
    from sellmeier import polingperiod
    def period(x,y):
        return polingperiod(w1=x,w2=y,sell=sell,Type=Type)
    def phasematching(x,y):
        if apodized:
            return exp(-0.193*( 2*pi * (1/period(x,y)-1/period(λ1,λ2)) * L*1e3/2 )**2)
        return sinc( 2*pi * (1/period(x,y)-1/period(λ1,λ2)) * L*1e3/2 )
    def energy(x,y): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # print('ΔfFWHM (GHz)=',ΔfFWHM*1e-9) # I = exp(-4*log(2)*(f-f0)**2/ΔfFWHM**2)
        return exp(-2*log(2)*((1/abs(λ1)-1/abs(x)+1/abs(λ2)-1/abs(y))*1e9*299792458/ΔfFWHM)**2)
    xs,ys = np.linspace(λ1-dλ1,λ1+dλ1,num1),np.linspace(λ2-dλ2,λ2+dλ2,num2)
    yy,xx = np.meshgrid(ys,xs)
    zz = phasematching(xx,yy) * energy(xx,yy)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    return w
def fλjca(λ1,λ3,dt,L=10,sell='ktpwg',Type='yzy',dλ1=300,dλ3=30,num1=501,num3=201,apodized=True):
    from sellmeier import polingperiod
    def period(x,y):
        return polingperiod(w1=x,w2=1/(1/y-1/x),sell=sell,Type=Type)
    def phasematching(x,y):
        if apodized:
            return exp(-0.193*( 2*pi * (1/period(x,y)-1/period(λ1,λ3)) * L*1e3/2 )**2)
        return sinc( 2*pi * (1/period(x,y)-1/period(λ1,λ3)) * L*1e3/2 )
    def energy(x,y,norm=False): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9)
        # if norm: return energy(x,y)/(1e-99+sum(energy(x,yi) for yi in ys))/Δx
        if norm:
            return energy(x,y,0)/(1e-99+sum(energy(xi,λ3,0) for xi in xs))/Δx
        return exp(-2*log(2)*((1/abs(λ3)-1/abs(y)-1/abs(λ1)+1/abs(x))*1e9*299792458/ΔfFWHM)**2)
    xs,ys = np.linspace(λ1-dλ1,λ1+dλ1,num1),np.linspace(λ3-dλ3,λ3+dλ3,num3)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = phasematching(xx,yy) * energy(xx,yy,norm=1)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    return w
def fλfilter(λ1,λ3,Δλ3=np.inf,dλ1=300,dλ3=30,num1=501,num3=201):
    def energy(x,y):
        λ2 = 1/( 1/λ3-1/λ1 )
        y0 = 1/( 1/x+1/λ2 )
        return discretedeltafunction(y,y0,Δy)/Δx
    def filter(x,y):
        return 1 if np.isposinf(Δλ3) else tophat(y,λ3-Δλ3/2,λ3+Δλ3/2,Δy)
    xs,ys = np.linspace(λ1-dλ1,λ1+dλ1,num1),np.linspace(λ3-dλ3,λ3+dλ3,num3)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = energy(xx,yy) * filter(xx,yy)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    return w
def fpure(λ1,λ3,Δλ1,Δλ3,dλ1=300,dλ3=30,num1=501,num3=201):
    ...
def fjsabest(λ1,λ2,dt,sell='ktpwg',Type='yzy',dλ=30,num1=1001,num2=1001):
    dλ1,dλ2 = dλ,dλ*(λ2/λ1)**2
    θ = phasematchangle(λ1,λ2,sell=sell,Type=Type)
    # P = apodizedmaxpurity(θ)
    # print(' P',P)
    # def energy(x,y): return exp(-2*log(2)*((1/λ1-x+1/λ2-y)*1e9*299792458/ΔfFWHM)**2) # E = exp(-2*log(2)*(f-f0)**2/ΔfFWHM**2)
    ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
    p = 2*log(2)*(1e9*299792458/ΔfFWHM)**2
    q = p/np.abs(0.5*sin(2*θ))
    f1,f2,df1,df2 = 1/λ1,1/λ2,dλ1/λ1**2,dλ2/λ2**2
    xs,ys = np.linspace(f1-df1,f1+df1,num1),np.linspace(f2-df2,f2+df2,num2)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = exp(-p*(xx-1/λ1+yy-1/λ2)**2) * exp(-q*((xx-1/λ1)*sin(θ)+(yy-1/λ2)*cos(θ))**2)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    return w

# frequecy based joint amplitudes
def fjsa(λ1,λ2,dt,L=10,sell='ktpwg',Type='yzy',dλ1=30,dλ2=300,num1=201,num2=501,apodized=True,getangle=False,plotangles=False): # shape=(num1,num2)
    from sellmeier import polingperiod
    def period(x,y):
        return polingperiod(w1=1/x,w2=1/y,sell=sell,Type=Type)
    def phasematching(x,y):
        if apodized:
            return exp(-0.193*( 2*pi * (1/period(x,y)-1/period(1/λ1,1/λ2)) * L*1e3/2 )**2)
        return sinc( 2*pi * (1/period(x,y)-1/period(1/λ1,1/λ2)) * L*1e3/2 )
    def energy(x,y): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # print('ΔfFWHM (GHz)=',ΔfFWHM*1e-9) # I = exp(-4*log(2)*(f-f0)**2/ΔfFWHM**2)
        return exp(-2*log(2)*((1/abs(λ1)-1/abs(x)+1/abs(λ2)-1/abs(y))*1e9*299792458/ΔfFWHM)**2)
    # xs,ys = np.linspace(λ1-dλ1,λ1+dλ1,num1),np.linspace(λ2-dλ2,λ2+dλ2,num2)
    f1,f2,df1,df2 = 1/λ1,1/λ2,dλ1/λ1**2,dλ2/λ2**2
    xs,ys = np.linspace(f1-df1,f1+df1,num1),np.linspace(f2-df2,f2+df2,num2)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = phasematching(xx,yy) * energy(1/xx,1/yy)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    if getangle or plotangles:
        def angle(dx,dy):
            return 180/np.pi*np.arctan2(-dy,dx)
        dp1,dp2 = 1,1
        dp1,dp2 = 0.5*df1*dp1/np.sqrt(dp1**2+dp2**2),0.5*df1*dp2/np.sqrt(dp1**2+dp2**2)
        u0 = Wave([f2,f2+dp2],[f1,f1+dp1],f'increasing energy',c='k',l='3') # direction of increasing pump energy: (x,y) = (dp1,dp2)
        u1 = Wave([f2+dp1,f2-dp1],[f1-dp2,f1+dp2],f'constant energy {angle(dp2,-dp1):g}°',c='w',l='3') # direction of constant energy: (x,y) = (dp2,-dp1) or (-dp2,dp1)
        dq1 = 1/period(f1+Δx,f2)-1/period(f1,f2)
        dq2 = 1/period(f1,f2+Δx)-1/period(f1,f2)
        dq1,dq2 = 0.5*df1*dq1/np.sqrt(dq1**2+dq2**2),0.5*df1*dq2/np.sqrt(dq1**2+dq2**2)
        v0 = Wave([f2,f2+dq2],[f1,f1+dq1],f'increasing Δk',c='k',l='1') # direction of increasing pump energy: (x,y) = (dq1,dq2)
        v1 = Wave([f2+dq1,f2-dq1],[f1-dq2,f1+dq2],f'constant Δk {angle(dq2,-dq1):.1f}°',c='w',l='1') # direction of constant energy: (x,y) = (dq2,-dq1) or (-dq2,dq1)
        if plotangles: Wave.plots(u0,u1,v0,v1,x='1/λ₁ (1/nm)',y='1/λ₂ (1/nm)',image=w,save=f'JSA angles, {λ1}λ1 {λ2}λ2 {sell} {Type} {L:g}mm {dt:g}ns')
    return angle(dq2,-dq1) if getangle else w
def fjca(λ1,λ3,dt,L=10,sell='ktpwg',Type='yzy',dλ1=300,dλ3=30,num1=501,num3=201,apodized=True,getangle=False,plotangles=False):
    from sellmeier import polingperiod
    def period(x,y):
        return polingperiod(w1=1/x,w2=1/(y-x),sell=sell,Type=Type)
    def phasematching(x,y):
        if 0==L:
            return 1
        if apodized:
            return exp(-0.193*( 2*pi * (1/period(x,y)-1/period(1/λ1,1/λ3)) * L*1e3/2 )**2)
        return sinc( 2*pi * (1/period(x,y)-1/period(1/λ1,1/λ3)) * L*1e3/2 )
    def energy(x,y,norm=False): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9)
        # if norm: return energy(x,y)/(1e-99+sum(energy(x,yi) for yi in ys))/Δx
        if norm:
            return energy(x,y,0)/(1e-99+sum(energy(xi,1/λ3,0) for xi in xs))/Δx
            # return energy(x,y,0)/(1e-99+sum(energy(xi,y,0) for xi in xs))/Δx
        return exp(-2*log(2)*((+1/λ1-x-1/λ3+y)*1e9*299792458/ΔfFWHM)**2)
    f1,f3,df1,df3 = 1/λ1,1/λ3,dλ1/λ1**2,dλ3/λ3**2
    xs,ys = np.linspace(f1-df1,f1+df1,num1),np.linspace(f3-df3,f3+df3,num3)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = phasematching(xx,yy) * energy(xx,yy,norm=1)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    if getangle or plotangles:
        def angle(dx,dy):
            return 180/np.pi*np.arctan2(-dy,dx)
        dp1,dp3 = -1,1 # dp1,dp3 = +f1-(f1+Δx)-f3+f3,+f1-f1-f3+(f3+Δx)
        dp1,dp3 = 0.5*df1*dp1/np.sqrt(dp1**2+dp3**2),0.5*df1*dp3/np.sqrt(dp1**2+dp3**2)
        u0 = Wave([f3,f3+dp3],[f1,f1+dp1],f'increasing energy',c='k',l='3') # direction of increasing escort energy: (x,y) = (dp1,dp3)
        u1 = Wave([f3+dp1,f3-dp1],[f1-dp3,f1+dp3],f'constant energy {angle(dp3,-dp1):g}°',c='w',l='3') # direction of constant energy: (x,y) = (dp3,-dp1) or (-dp3,dp1)
        dq1 = 1/period(f1+Δx,f3)-1/period(f1,f3)
        dq3 = 1/period(f1,f3+Δx)-1/period(f1,f3)
        dq1,dq3 = 0.5*df1*dq1/np.sqrt(dq1**2+dq3**2),0.5*df1*dq3/np.sqrt(dq1**2+dq3**2)
        v0 = Wave([f3,f3+dq3],[f1,f1+dq1],f'increasing Δk',c='k',l='1') # direction of increasing escort energy: (x,y) = (dq1,dq3)
        v1 = Wave([f3+dq1,f3-dq1],[f1-dq3,f1+dq3],f'constant Δk {angle(dq3,-dq1):.1f}°',c='w',l='1') # direction of constant energy: (x,y) = (dq3,-dq1) or (-dq3,dq1)
        if plotangles: Wave.plots(u0,u1,v0,v1,x='1/λ₁ (1/nm)',y='1/λ₃ (1/nm)',image=w,save=f'JCA angles, {λ1}λ1 {λ3}λ3 {sell} {Type} {L:g}mm {dt:g}ns')
    return angle(dq3,-dq1) if getangle else w
def fjcahop(n,λ1,λ3,dt,L=10,sell='ktpwg',Type='yzy',dλ1=300,dλ3=30,num1=501,num3=201): # higher order pump
    from sellmeier import polingperiod
    def period(x,y):
        return polingperiod(w1=1/x,w2=1/(y-x),sell=sell,Type=Type)
    def phasematching(x,y):
        return 1 if 0==L else exp(-0.193*( 2*pi * (1/period(x,y)-1/period(1/λ1,1/λ3)) * L*1e3/2 )**2)
    def energy(x,y): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # in Hz
        σf = 0.5/sqrt(log(2))*ΔfFWHM
        return 1/(2*pi*σf) * hermitegaussmode(n,(+1/λ1-x-1/λ3+y)*1e9*299792458/σf)
        # return exp(-2*log(2)*((+1/λ1-x-1/λ3+y)*1e9*299792458/ΔfFWHM)**2)
    f1,f3,df1,df3 = 1/λ1,1/λ3,dλ1/λ1**2,dλ3/λ3**2
    xs,ys = np.linspace(f1-df1,f1+df1,num1),np.linspace(f3-df3,f3+df3,num3)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = phasematching(xx,yy)*energy(xx,yy)*1e9*299792458*sqrt(2*pi)
    return JointSpectrum(zz,xs=xs,ys=ys)
def phasematchangle(λ1,λ2,sell='ktpwg',Type='yzy',temp=20,npy=None,npz=None,degrees=False):
    from sellmeier import qpmwavelengths,groupindex
    # θ = atan( (k'(ωp)-k'(ωs)) / (k'(ωp)-k'(ωi)) )
    # k'(ω) = 1/vg = inverse group velocity = c * group index
    λ1,λ2,λ3 = qpmwavelengths(λ1,λ2)
    ngs = [groupindex(λ,sell+s,temp) for λ,s in zip((λ1,λ2,λ3),Type)]
    θ = np.arctan2( ngs[2]-ngs[0] , ngs[2]-ngs[1] )
    θ = (θ+2*np.pi)%(2*np.pi)
    return θ*180/np.pi if degrees else θ
def phasematchangleplot(sell='ktpwg',Type='yzy',temp=20,npy=None,npz=None,x0=500,x1=2700,dx=10):
    from sellmeier import polingperiod
    from wavedata import wrange,Wave,Wave2D
    x,y = wrange(x0,x1,dx),wrange(x0,x1,dx) # Wave(phasematchangle(x,x,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz,degrees=1),x).plot()
    yy,xx = np.meshgrid(y,x)
    zz = phasematchangle(yy,xx,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz,degrees=1)
    ww = Wave2D(zz,xs=x,ys=y)
    ws = [ww.contour(θ).rename(f"{θ:g}°").setplot(c='k',l=l) for θ,l in zip([180,135,90],'130')]
    qq = np.log(np.abs(1./polingperiod(yy,xx,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz)))
    vv = Wave2D((qq-np.nanmin(qq))**3,xs=x,ys=y)
    xy = Wave((x0,x1),(x0,x1),c='k',lw=0.2)
    Wave.plots(*ws,xy,image=vv,
        x=f'λ{Type[1]} (nm)',y=f'λ{Type[0]} (nm)',xlim=(x0,x1),ylim=(x0,x1),
        aspect=1,colormap='terrain_r',save=f'phasematchangleplot {sell} {Type}')
def frequencyconversionangle(λ1,λe,sell='ktpwg',Type='yzy',temp=20,npy=None,npz=None,degrees=False): # λ2 = escort (pump) wavelength
    from sellmeier import qpmwavelengths,groupindex
    λ1,λ2,λ3 = qpmwavelengths(λ1,λe)
    print('λ1,λ2,λ3',λ1,λ2,λ3)
    ngs = [groupindex(λ,sell+s,temp) for λ,s in zip((λ1,λ2,λ3),Type)]
    θ = np.arctan2( ngs[1]-ngs[0] , ngs[2]-ngs[1] )
    # θ0,Δθ0 = np.pi/8,np.pi*3/180 # phase wrap-around location
    # θ = (θ+np.pi+θ0)%np.pi
    # θ = np.where(θ<Δθ0,np.nan,θ) - θ0
    return θ*180/np.pi if degrees else θ
def frequencyconversionangleplot(sell='ktp',Type='yzy',temp=20,npy=None,npz=None,x0=500,x1=2700,dx=10,λfc=None):
    from sellmeier import polingperiod
    from wavedata import wrange,Wave,Wave2D
    x,y = wrange(x0,x1,dx),wrange(x0,x1,dx) # 
    # Wave(frequencyconversionangle(x,x,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz,degrees=1),x).plot(m='o')
    # w0 = Wave(frequencyconversionangle(2500,x,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz,degrees=1),x)#.plot(m='o')
    # w0 = Wave(frequencyconversionangle(x,2500,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz,degrees=1),x).plot(m='o')
    yy,xx = np.meshgrid(y,x)
    zz = frequencyconversionangle(yy,xx,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz,degrees=1)
    ww = Wave2D(zz,xs=x,ys=y)
    ws = [ww.contour(θ).rename(f"{θ:g}°").setplot(c='k',l=l) for θ,l in zip([180,135,90,45,0,-45,-90],'1304130')]
    ws = [w for w in ws if 1<len(w) or not np.isnan(w[0])]
    # for w in ws: print(w[:3],len(w),bool(w[0]))
    qq = np.log(np.abs(1./polingperiod(yy,xx,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz)))
    vv = Wave2D((qq-np.nanmin(qq))**3,xs=x,ys=y)
    xy = Wave((x0,x1),(x0,x1),c='k',lw=0.2)
    ws = ws+[Wave(np.where(1/λfc-1/x>0,1/(1/λfc-1/x),nan),x,f"{λfc}nm λfc",c='r')] if λfc else ws
    Wave.plots(*ws,xy,image=ww if Type[0]==Type[1] else vv,
        x=f'escort λ{Type[1]} (nm)',y=f'λ{Type[0]} (nm)',xlim=(x0,x1),ylim=(x0,x1),
        aspect=1,colormap=None if Type[0]==Type[1] else 'terrain_r',save=f'frequencyconversionangleplot {sell} {Type}'+f", {λfc}nm λfc"*bool(λfc))
    # ww.plot(x=f'escort λ{Type[1]} (nm)',y=f'λ{Type[0]} (nm)',xlim=(x0,x1),ylim=(x0,x1),aspect=1)
def angleplot():
    # from sellmeier import phasematchangle,frequencyconversionangle
    # u = fjsa(780,2340,0.0002,L=10,sell='ktp',Type='xzx')
    # u.plot(aspect=1,fewerticks=1,show=1)
    # v = fjca(2340,780,0.0002,L=1,sell='ktp',Type='zzz')
    # v.plot(aspect=1,fewerticks=1,show=1)
    # θ = fjsa(780,2340,0.0002,L=1,sell='ktp',Type='xzx',plotangles=1,getangle=1)
    θ = fjsa(780,2340,0.0002,L=10,sell='ktp',Type='xzx',plotangles=1,getangle=1)
    print('θ',θ,phasematchangle(780,2340,sell='ktp',Type='xzx',degrees=1))
    # ϕ = fjca(2340,780,0.0002,L=1,sell='ktp',Type='zzz',plotangles=1,getangle=1)
    ϕ = fjca(2340,780,0.0002,L=10,sell='ktp',Type='zzz',plotangles=1,getangle=1)
    print('ϕ',ϕ,frequencyconversionangle(2340,1170,sell='ktp',Type='zzz',degrees=1))
    # ϕ = fjca(2340,780,0.0002,L=10,sell='ktp',Type='zxx',plotangles=1,getangle=1)
    # print('ϕ',ϕ,frequencyconversionangle(2340,1170,sell='ktp',Type='zxx',degrees=1))
    # ϕ = fjca(2340,780,0.0002,L=10,sell='ktp',Type='xzx',plotangles=1,getangle=1)
    # print('ϕ',ϕ,frequencyconversionangle(2340,1170,sell='ktp',Type='xzx',degrees=1))
# @memory.cache
def fcigar(λ1,λ2,θ,Δλ1=10,Δλ2=1,dλ1=30,dλ2=30,num1=201,num2=201,norm=0): # θ in degrees
    def gauss(x,y,norm):
        u = +(x-f1)*cos(θ/180*pi)+(y-f2)*sin(θ/180*pi)
        v = -(x-f1)*sin(θ/180*pi)+(y-f2)*cos(θ/180*pi)
        # if norm: return gauss(x,y,0)/(1e-99+sum(gauss(f1,yi,0) for yi in ys))/Δx
        if norm:
            return gauss(x,y,0)/(1e-99+sum(gauss(xi,f2,0) for xi in xs))/Δx
        return exp(-u**2/Δf1**2)*exp(-v**2/Δf2**2)
    f1,f2,df1,df2,Δf1,Δf2 = 1/λ1,1/λ2,dλ1/λ1**2,dλ2/λ2**2,Δλ1/λ1**2,Δλ2/λ2**2
    xs,ys = np.linspace(f1-df1,f1+df1,num1),np.linspace(f2-df2,f2+df2,num2)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = gauss(xx,yy,norm)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    return w
def frandom(λ1=780,λ2=780,dλ1=30,dλ2=30,num1=201,num2=201,maxpurity=1,norm=1):
    def r():
        return np.random.rand()
    while 1:
        θ,Δλ1,Δλ2 = 180*r()-90,20*r()+0.1,20*r()+0.1
        if Δλ1>Δλ2: break
    f = fcigar(λ1,λ2,θ,Δλ1,Δλ2,dλ1,dλ2,num1,num2,norm=1)
    p = 1/schmidtnumber(f)
    return (f,p,θ,Δλ1,Δλ2) if p<=maxpurity else frandom(λ1,λ2,dλ1,dλ2,num1,num2,maxpurity,norm)
def feffcigar(θ=+45,ϕ=-45,Δλ1=1,Δλ2=0.1,Δλλ1=1,Δλλ2=0.1,num=201,plot=0):
    u = fcigar(780,780,θ=θ,Δλ1=Δλ1,Δλ2=Δλ2,num1=num,num2=num)
    v = fcigar(780,780,θ=ϕ,Δλ1=Δλλ1,Δλ2=Δλλ2,num1=num,num2=num)
    w = integrate(u,v)
    if plot:
        u.plot(); print('  u purity',1/schmidtK(u))
        v.plot(); print('  v purity',1/schmidtK(v))
        w.plot(); print('  w purity',1/schmidtK(w))
    return 1/schmidtK(w)
def feffcigartest():
    # feffcigar(θ=+45,ϕ=-45,Δλ1=10,Δλ2=1,plot=1)
    ϕs = np.linspace(-90,+90,37)
    # Wave([feffcigar(+45,ϕ) for ϕ in ϕs],ϕs).plot()
    w0 = Wave([feffcigar(+45,ϕ,Δλλ1=10,Δλλ2=1) for ϕ in ϕs],ϕs,0)
    w1 = Wave([feffcigar(+50,ϕ,Δλλ1=10,Δλλ2=2) for ϕ in ϕs],ϕs,1)
    w2 = Wave([feffcigar(+50,ϕ,Δλλ1=5,Δλλ2=2) for ϕ in ϕs],ϕs,2)
    Wave.plots(w0,w1,w2)
    print(w0)
    print(w1)
    print(w2)
def θϕplot(Δλ1=10,Δλ2=3,Δλλ1=3,Δλλ2=1,num=201):
    save = f"θϕplot {Δλ1:g} {Δλ2:g} {Δλλ1:g} {Δλλ2:g}"+(201!=num)*f" {num}"
    print(save)
    θs,ϕs = np.linspace(0,90,72+1),np.linspace(10,80,7+1)
    ws = [Wave([feffcigar(θ,ϕ,Δλ1=Δλ1,Δλ2=Δλ2,Δλλ1=Δλλ1,Δλλ2=Δλλ2,num=num) for θ in θs],θs,f"ϕ={ϕ:g}°") for ϕ in track(ϕs)]
    Wave.plots(*ws,l='432100123',x='θ (°)',y='P',grid=1,seed=1,xlim=(0,90),ylim=(0.89,1.01),save=save)
# @memory.cache
def examplehighpe(θ=2,ϕ=5):
    # u = fjsa(780,2340,0.00125,L=6.5,sell='ktp',Type='yzy',plotangles=1)
    # v = fjca(2340,780,0.0002,L=2,sell='ktp',Type='zzz',plotangles=1)
    # w = integrate(u,v)
    # w.plot(); print(1/schmidtnumber(w)) # 0.96998
    # u = fcigar(780,780,1.73, 3.13,0.19,num1=1001,num2=1001,norm=1).plot()
    # v = fcigar(780,780,4.64,12.64,0.59,num1=1001,num2=1001,norm=1).plot()
    # w = integrate(u,v).plot()
    # p = 1/schmidtnumber(w)
    # e = efficiency(w,u)
    # print('p',p,'e',e)
    u = fcigar(780,780,θ,1.5,0.1,num1=401,num2=401,norm=1)
    v = fcigar(780,780,ϕ,6.5,0.3,num1=401,num2=401,norm=1)
    w = integrate(u,v)
    p0 = 1/schmidtnumber(u)
    p = 1/schmidtnumber(w)
    e = efficiency(w,u)
    # u.plot(); v.plot(); w.plot()
    # print(1/schmidtnumber(u),1/schmidtnumber(v),'θ',θ,'ϕ',ϕ,'p',p,'e',e)
    return (1-p)/(1-p0),e
def scanhighpe():
    θs,ϕs = np.linspace(1,89,88+1),np.linspace(5,85,16+1)
    θs,ϕs = np.linspace(5,85,16+1),np.linspace(5,85,16+1)
    # θs,ϕs = np.linspace(0.2,8.8,44),np.linspace(5,15,3)
    rs = [Wave([examplehighpe(θ,ϕ)[0] for θ in θs],θs,f"ϕ={ϕ:g}°") for ϕ in track(ϕs)]
    Wave.plots(*rs,l='432100123',x='θ (°)',y='R',grid=1,seed=1,xlim=(0,90),ylim=(None,None),save='scanhighpe r vs θ,ϕ')
    es = [Wave([examplehighpe(θ,ϕ)[1] for θ in θs],θs,f"ϕ={ϕ:g}°") for ϕ in track(ϕs)]
    Wave.plots(*es,l='432100123',x='θ (°)',y='E',grid=1,seed=1,xlim=(0,90),ylim=(None,None),save='scanhighpe e vs θ,ϕ')
def puritysearch():
    fs = []
    pmax = 0
    for i in range(1000):
        u,up,θ,Δu1,Δu2 = frandom(maxpurity=0.7)
        v,vp,ϕ,Δv1,Δv2 = frandom(maxpurity=0.7)
        w = integrate(u,v)
        p = 1/schmidtnumber(w)
        e = efficiency(w,u)
        # f = (f'* {p:.3f}','*'+list2str((up,θ,Δu1,Δu2),f='{:6.2f}'),'*'+list2str((vp,ϕ,Δv1,Δv2),f='{:6.2f}'),'*'+list2str((Δu1/Δu2,Δv1/Δv2,Δu1/Δv1),f='{:6.2f}'))
        f = (f'* {p:.3f}','*'+list2str((up,θ,Δu1,Δu2),f='{:6.2f}'),'*'+list2str((vp,ϕ,Δv1,Δv2),f='{:6.2f}'),f'* {e:.3f}e {up:.3f}pu {vp:.3f}pv {p*e:.3f}pe')
        fs += [f]
        if p>pmax:
            print(f"{i:2d}",*f)
            pmax = p
        else:
            print(f"{i:2d}",*f)
            # print(i)
    print()
    for f in sorted(fs):
        print(*f)
def purityvsjcapulse():
    u = fjsa(780,2340,0.0025,L=13,sell='ktp',Type='yzy') # 
    u = fjsa(780,2340,0.0025/2,L=13/2,sell='ktp',Type='yzy') # 
    print('  u purity',1/schmidtnumber(u))
    def jcapurity(dt=0.0002):
        v = fjca(2340,780,dt,L=2,sell='ktp',Type='zzz')
        return 1/schmidtnumber(v)
    def finalpurity(dt=0.0002,eff=False):
        v = fjca(2340,780,dt,L=2,sell='ktp',Type='zzz')
        w = integrate(u,v)
        if eff:
            return efficiency(w,u)
        return 1/schmidtnumber(w)
    dts = np.linspace(0.00001,0.0003,30)
    w0 = Wave([jcapurity(dt) for dt in dts],1000*dts,'Pjca')
    w1 = Wave([finalpurity(dt) for dt in dts],1000*dts,'Peff')
    w2 = Wave([finalpurity(dt,eff=1) for dt in dts],1000*dts,'η')
    Wave.plots(w0,w1,w2,x='pulse width (ps)',y='purity',grid=1,save='purity vs jca pulse example 2')
def upconversionidentitymatrix(aspect=1,plotit=1):
    # upconversion "identity matrix"
    u = fjsa(780,2340,0.0002,L=10,sell='ktp',Type='xzx').plot(aspect=1/aspect,show=plotit)
    print('  u purity',1/schmidtnumber(u))
    v = ffilter(2340,780).plot(aspect=aspect,show=plotit)
    print('  v purity',1/schmidtnumber(v))
    w = integrate(u,v)
    print('  w purity',1/schmidtnumber(w))
    print('efficiency',efficiency(w,u))
    w.plot(aspect=1,show=plotit)

def ffilter(λ1,λ3,Δλ3=np.inf,dλ1=300,dλ3=30,num1=201,num3=201,δfuncwidth=1,halfpass=False,gaussδfunc=False,gaussfilter=False):
    f1,f3,df1,df3,Δf3 = 1/λ1,1/λ3,dλ1/λ1**2,dλ3/λ3**2,Δλ3/λ3**2
    def energy(x,y,norm=1):
        f2 = f3 - f1
        x0 =  y - f2
        if norm:
            return energy(x,y,0)/(1e-99+sum(energy(xi,f3,0) for xi in xs))/Δx
        if gaussδfunc:
            return exp(-(+f1-x-f3+y)**2/(δfuncwidth*Δx)**2)
        return discretedeltafunction(x,x0,δfuncwidth*Δx)
    def filter(x,y):
        if halfpass:
            return tophat(y,f3,f3+Δf3/2,Δy) # half-pass
        if gaussfilter:
            return exp(-0.5*(y-f3)**2/Δf3**2)
        return 1 if np.isposinf(Δf3) else tophat(y,f3-Δf3/2,f3+Δf3/2,Δy)
    xs,ys = np.linspace(f1-df1,f1+df1,num1),np.linspace(f3-df3,f3+df3,num3)
    Δx,Δy = xs[1]-xs[0],ys[1]-ys[0]
    yy,xx = np.meshgrid(ys,xs)
    zz = energy(xx,yy,norm=1) * filter(xx,yy)
    w = JointSpectrum(zz,xs=xs,ys=ys)
    return w
def jcanormalizationplot(u,v): # u=jsa, v=jca
    def waveefficiency(self,original):
        return self.sqr().area()/original.sqr().area()
    def waveintegrate(f,g): # computes ∫ f(y) g(y) dy or ∫ f(y) g(y,z) dy
        if isinstance(g,Wave):
            assert f.dx()==g.dx()
            return (f.yarray()*g.yarray()).sum()*f.dx()
        assert isinstance(g,Wave2D)
        return Wave([waveintegrate(f,w) for w in g.xwaves()],g.ys)
    u0 = u[u.xlen()//2-1,:] # middlish slice (column) of jsa, i.e. fjsa(ωs=ωs0,wi)
    v0 = v[:,v.ylen()//2+10] # middlish slice (row) of jca
    assert all(u.ys==v.xs), 'axes mismatch'
    # print('sum(v0)*v0.dx()',sum(v0)*v0.dx())
    w0 = waveintegrate(u0,v)
    w1 = Wave(w0.y,w0.x-w0.x[w0.len()//2]+u0.x[u0.len()//2]) # shift center to plot overlapped with u0
    eff = waveefficiency(w0,u0) # print('w0 efficiency',eff)
    Wave.plots(u0,1e-6*v0,w1,x='$f_s$ (1/nm)',y='amplitude',mf='0',m='o',lw=1,ylim=(0,2*u0.max()),fewerticks=1,legendtext=f'η={eff:g}')
    # u1,u2 = u0.downsample(2),u0[::2]; print('u1 efficiency',waveefficiency(u1,u0),'u2 efficiency',waveefficiency(u2,u0))
    # Wave.plots(u0,u1,u2,1e-6*v0,w1,x='$f_s$ (1/nm)',y='amplitude',mf='0',m='o',lw=1,ylim=(0,2*u0.max()),fewerticks=1,legendtext=f'η={eff:g}')
def tophatfilter(aspect=1,plotit=1,Δλ=2): # 2nm tophat filter
    u = fjsa(780,2340,0.0001,L=10,sell='ktp',Type='xzx',dλ1=30,dλ2=400,num1=101,num2=201).plot(aspect=1/aspect,show=plotit)
    print('  u purity',1/schmidtnumber(u))
    # v = ffilter(2340,780,Δλ,dλ1=400,dλ3=30,num1=401,num3=301,δfuncwidth=20,halfpass=0)#.plot(aspect=aspect,show=plotit)
    v = ffilter(2340,780,Δλ,dλ1=400,dλ3=30,num1=201,num3=301,δfuncwidth=1,halfpass=0).plot(aspect=aspect,show=plotit)
    print('  v purity',1/schmidtnumber(v))
    w = integrate(u,v)
    print('  w purity',1/schmidtnumber(w))
    print('efficiency',efficiency(w,u))
    w.plot(aspect=1,show=plotit)
    jcanormalizationplot(u,v)
def cwupconversionpump(aspect=1,plotit=1):
    # cw upconversion pump
    u = fjsa(780,2340,0.0002,L=10,sell='ktp',Type='xzx').plot(aspect=1/aspect,show=plotit)
    print('  u purity',1/schmidtnumber(u))
    v = fjca(2340,780,0.02,L=0,sell='ktp',Type='zzz').plot(aspect=aspect,show=plotit)
    print('  v purity',1/schmidtnumber(v))
    w = integrate(u,v)
    print('  w purity',1/schmidtnumber(w))
    print('efficiency',efficiency(w,u))
    w.plot(aspect=1,show=plotit)
def pulsedupconversionpump(aspect=1,plotit=1):
    # pulsed upconversion pump
    u = fjsa(780,2340,0.0002,L=10,sell='ktp',Type='xzx').plot(aspect=1/aspect,fewerticks=1,show=plotit)
    print('  u purity',1/schmidtnumber(u))
    v = fjca(2340,780,0.0002,L=1,sell='ktp',Type='zzz').plot(aspect=aspect,fewerticks=1,show=plotit)
    print('  v purity',1/schmidtnumber(v))
    w = integrate(u,v)
    print('  w purity',1/schmidtnumber(w))
    print('efficiency',efficiency(w,u))
    w.plot(aspect=1,fewerticks=1,show=plotit)
def feff(λ1=780,λ2=2340,res=1001,dλ=40,dt1=0.001,dt2=0.001,L1=10,L2=10,type1='yzy',type2='zzz',sell='ktpwg',plot=False,plotλ=False):
    if plotλ:
        u = fλjsa(λ1,λ2,dt1,L=L1,sell=sell,Type=type1,dλ1=dλ,dλ2=dλ*(λ2/λ1)**2,num1=res,num2=res)
        v = fλjca(λ2,λ1,dt2,L=L2,sell=sell,Type=type2,dλ1=dλ*(λ2/λ1)**2,dλ3=dλ,num1=res,num3=res)
    else:
        u = fjsa(λ1,λ2,dt1,L=L1,sell=sell,Type=type1,dλ1=dλ,dλ2=dλ*(λ2/λ1)**2,num1=res,num2=res)
        v = fjca(λ2,λ1,dt2,L=L2,sell=sell,Type=type2,dλ1=dλ*(λ2/λ1)**2,dλ3=dλ,num1=res,num3=res)
    w = integrate(u,v)
    return u,v,w
def ktpwgexample(res=1001,dλ=40,dt1=0.001,dt2=0.001,L1=10,L2=10,plot=False,plotλ=False):
    if plotλ:
        u = fλjsa(780,2340,dt1,L=L1,sell='ktpwg',Type='yzy',dλ1=dλ,dλ2=dλ*10,num1=res,num2=res)
        v = fλjca(2340,780,dt2,L=L2,sell='ktpwg',Type='zzz',dλ1=dλ*10,dλ3=dλ,num1=res,num3=res)
    else:
        u = fjsa(780,2340,dt1,L=L1,sell='ktpwg',Type='yzy',dλ1=dλ,dλ2=dλ*10,num1=res,num2=res)
        v = fjca(2340,780,dt2,L=L2,sell='ktpwg',Type='zzz',dλ1=dλ*10,dλ3=dλ,num1=res,num3=res)
    if plot or plotλ:
        w = integrate(u,v)
        # u,v,w = u.clip(0,1e-6*u.max()),v.clip(0,1e-6*v.max()),w.clip(0,1e-6*w.max())
        # u,v,w = log(u+1e-99),log(v+1e-99),log(w+1e-99)
        # u,v,w = (u>0),(v>0),(w>0)
        pu,pv,pw,ii,ηη = 1/schmidtnumber(u),1/schmidtnumber(v),1/schmidtnumber(w),w.indistinguishability(),efficiency(w,u)
        print('Pjsa,Pjca,Peff,Ieff,ηeff',list2str((pu,pv,pw,ii,ηη)))
        save = f'ktpwgexample {1000*dt1:g}ps,{1000*dt2:g}ps {L1}mm,{L2}mm '
        s,ss = 'λ' if plotλ else 'f','nm' if plotλ else '1/nm'
        u.plot(x=f'${s}_s$ ({ss})',y=f'${s}_i$ ({ss})',fewerticks=1,aspect=(.1 if plotλ else 1),legendtext=f"P={pu:g}",save=save+(f'{s}i vs {s}s'))
        v.plot(x=f'${s}_i$ ({ss})',y=f'${s}_f$ ({ss})',fewerticks=1,aspect=(10 if plotλ else 1),legendtext=f"P={pv:g}",save=save+(f'{s}f vs {s}i'))
        w.plot(x=f'${s}_s$ ({ss})',y=f'${s}_f$ ({ss})',fewerticks=1,aspect=1,legendtext=f"P={pw:g}\nI={ii:g}\nη={ηη:g}",save=save+(f'{s}f vs {s}s'))
        return
    def f(dt,L=5):
        v = fjca(2340,780,dt,L=L,sell='ktp',Type='zzz',dλ1=dλ*10,dλ3=dλ,num1=res,num3=res)
        w = integrate(u,v)
        # u.plot(aspect=1,fewerticks=1,show=plot); print('  u purity',1/schmidtnumber(u))
        # v.plot(aspect=1,fewerticks=1,show=plot); print('  v purity',1/schmidtnumber(v))
        # w.plot(aspect=1,fewerticks=1,show=plot); print('  w purity',1/schmidtnumber(w)); print('efficiency',efficiency(w,u))
        Pjca,Peff,Ieff,ηeff = 1/schmidtnumber(v),1/schmidtnumber(w),w.indistinguishability(),efficiency(w,u)
        return Pjca,Peff,Ieff,ηeff
    # f(0.001)
    dts = np.linspace(0.05,3,60)
    Ls = (5,10,20)
    for L in Ls:
        Pjcas,Peffs,Ieffs,ηeffs = zip(*[f(0.001*dt,L) for dt in dts])
        Pjcaw,Peffw,Ieffw,ηeffw = [Wave(y,dts) for y,s in zip((Pjcas,Peffs,Ieffs,ηeffs),('Pjca','Peff','Ieff','ηeff'))]
        Wave.plots(Pjcaw,Peffw,Ieffw,ηeffw,x='pulse width (ps)',seed=0,xlim=(0,3),ylim=(0,1.1),scale=(2,1),save=f'ktpwg 780y+2340zJSA 2340z→780zJCA {L}mm {res} {dλ}span')
def telcomexample():
    u = fλjsa(1560,780,0.0005,L=10,sell='ktpwg',Type='yzy',dλ1=40,dλ2=40,num1=1001,num2=1001,apodized=0)
    u.plot(x='λy (nm)',y='λz (nm)',legendtext=f"P={1/schmidtnumber(u):g}",aspect=1,save='1560y+780z 0.5ps 10mm ktpwg JSA')
    u = fjsa(1560,780,0.0005,L=10,sell='ktpwg',Type='yzy',dλ1=40,dλ2=40,num1=1001,num2=1001,apodized=0)
    u.plot(x='fy (1/nm)',y='fz (1/nm)',legendtext=f"P={1/schmidtnumber(u):g}",aspect=0.25,fewerticks=1,save='1560y+780z 0.5ps 10mm ktpwg JSA - freq axes')
def schmidttest():
    f = fcigar(780,780,30,Δλ1=10,Δλ2=1,dλ1=30,dλ2=30,num1=201,num2=201,norm=1)#.plot()
    print(schmidtK(f))
    print(schmidtnumber(f,res=201))
@memory.cache
def filtereddegen(Δλ=np.inf,res=4001,dλ=5,dt=0.0059,L=10,jplot=False):
    u = fjsa(780,780,dt,L=L,sell='ktpwg',Type='yzy',dλ1=dλ,dλ2=dλ,num1=res,num2=res)#.plot()
    v = ffilter(780,780,Δλ3=Δλ,dλ1=dλ,dλ3=dλ,num1=res,num3=res,δfuncwidth=1,halfpass=False,gaussδfunc=False)
    w = integrate(u,v) # 
    print('filtereddegen','dt',dt,'L',L,'P',1/schmidtK(w))#,1/schmidtnumber(w))
    if jplot:
        save = f'filtereddegen {1000*dt:g}ps {L}mm '
        u.plot(x=f'$f_s$ (1/nm)',y=f'$f_i$ (1/nm)',fewerticks=1,aspect=1,legendtext=f"P={1/schmidtK(u):g}",save=save+(f'fi vs fs'))
        v.plot(x=f'$f_i$ (1/nm)',y=f'$f_f$ (1/nm)',fewerticks=1,aspect=1,legendtext=f"P={1/schmidtK(v):g}",save=save+(f'ff vs fi'))
        w.plot(x=f'$f_s$ (1/nm)',y=f'$f_f$ (1/nm)',fewerticks=1,aspect=1,legendtext=f"P={1/schmidtK(w):g}\nI={w.indistinguishability():g}\nη={efficiency(w,u):g}",save=save+(f'ff vs fs'))
        return
    # Δλs = np.geomspace(0.01,1,13)
    Δλs = np.linspace(0.005,0.5,100)
    def pη(Δλfwhm):
        v = ffilter(780,780,Δλ3=Δλfwhm,dλ1=dλ,dλ3=dλ,num1=res,num3=res,δfuncwidth=1,halfpass=False,gaussδfunc=False)
        w = integrate(u,v)
        return 1/schmidtK(w),efficiency(w,u)
    ps,ηs = zip(*[pη(Δλfwhm) for Δλfwhm in Δλs])
    wp,wη = Wave([1]+list(ps),[0]+list(Δλs),'P'),Wave([0]+list(ηs),[0]+list(Δλs),'η')
    Wave.plots(wp,wη,xlim=(0,wp.x[-1]),ylim=(0,1),x='tophat filter bandwidth FWHM (nm)',grid=1,
        save=f'P,η vs tophat filter bandwidth, 780 degen ktpwg {1000*dt:g}ps {L:g}mm')
    # Δλfwhm,p0,η0 = wp.maxloc(),wp.max(),wη(wp.maxloc())
    def info(P=0.939):
        Δλfwhm = wp.xaty(P)
        p0,η0 = wp(Δλfwhm),wη(Δλfwhm)
        print(f"Δλfwhm,p0,η0: {Δλfwhm:.3f}, {p0:.5f}, {η0:.5f}")
    for P in (0.9,0.939,0.99,0.999):
        info(P)
    return Wave(wp.y,wη.y)
    # v = ffilter(780,780,Δλ3=Δλfwhm,dλ1=dλ,dλ3=dλ,num1=res,num3=res,δfuncwidth=1,halfpass=False,gaussδfunc=False)
    # w = integrate(u,v)
    # w.plot() # print(1/schmidtK(w))
def filtereddegencompare():
    w0 = filtereddegen(dt=0.00059* 5,L= 5).rename(f'L={ 5}mm')
    w1 = filtereddegen(dt=0.00059*10,L=10).rename(f'L={10}mm')
    w2 = filtereddegen(dt=0.00059*20,L=20).rename(f'L={20}mm')
    Wave.plots(w2,w1,w0,x='η',y='P',l='023',xlim=(0,1),ylim=(-0.0,1.05),grid=1,save='filtereddegencompare')
def cascadedpurity(λ1=780,λ2=2340,sell='ktpwg',type1='yzy',type2='zzz'):
    θ0 = fjsa(λ1,λ2,0.0002,L=10,sell='ktpwg',Type='yzy',plotangles=0,getangle=1)/180*pi
    θ = phasematchangle(λ1,λ2,sell=sell,Type=type1)
    Pjsa = apodizedmaxpurity(θ)
    ϕ0 = fjca(λ2,λ1,0.0002,L=10,sell='ktpwg',Type='zzz',plotangles=0,getangle=1)/180*pi
    ϕ = frequencyconversionangle(λ1=λ2,λe=1/(1/λ1-1/λ2),sell=sell,Type=type2)
    Pjca = fcapodizedmaxpurity(ϕ)
    P = fcspdcpurity(Pjsa,Pjca)
    print(f"θ0 {θ0:g} θ {θ:g} Pjsa {Pjsa:g} ϕ0 {ϕ0:g} ϕ {ϕ:g} Pjca {Pjca:g} P {P}")
    def cascadedpurityplot(dt1=1.1,L1=10,L2=10,λ1=780,λ2=2340,type1='yzy',type2='zzz',sell='ktpwg',res=1001):
        # degenηvp = filtereddegen(dt=0.00059*L1,L=L1).swapxy()
        degenηvp = filtereddegen(dt=0.00059* 5,L= 5).swapxy()
        # degenηvp.plot(y='η',x='P')
        # print(degenηvp(0.939))
        # print(degenηvp(0.99))
        dts = wrange(0.2,8,0.2)
        # u = fjsa(λ1,λ2,dt1,L=L1,sell=sell,Type=type1,dλ1=dλ,dλ2=dλ*(λ2/λ1)**2,num1=res,num2=res)
        # v = fjca(λ2,λ1,dt2,L=L2,sell=sell,Type=type2,dλ1=dλ*(λ2/λ1)**2,dλ3=dλ,num1=res,num3=res)
        # w = integrate(u,v)
        us,vs,ws = zip(*[feff(λ1=λ1,λ2=λ2,res=res,dλ=40,dt1=0.001*dt1,dt2=0.001*dt,L1=L1,L2=L2,type1=type1,type2=type2,sell=sell) for dt in (dts)])
        Pjsa = Wave([1/schmidtK(u) for u in us],dts,'Pjsa')
        Pjca = Wave([1/schmidtK(v) for v in vs],dts,'Pjca')
        Peff = Wave([1/schmidtK(w) for w in ws],dts,'Peff')
        Ieff = Wave([w.indistinguishability() for w in ws],dts,'Ieff')
        ηeff = Wave([efficiency(w,u) for w,u in zip(ws,us)],dts,'ηeff')
        ηfil = Wave([degenηvp(0.99*p) for p in Peff],dts,'ηfil')
        print(f'γ = {ηeff(Ieff.maxloc())/ηfil(Ieff.maxloc()):g} at Imax')
        text = f'$τ_1$={dt1:g}ps\n$L_1$={L1:g}mm\n$L_2$={L2:g}mm'
        text += f'\nγ={ηeff(Ieff.maxloc())/ηfil(Ieff.maxloc()):.1f} at max Ieff'
        save = f'cascaded {λ1}+{λ2}{type1}, {λ2}→{λ1}{type2} τ1={dt1:g}ps L1={L1:g}mm L2={L2:g}mm {res}res'
        Wave.plots(Pjca,Peff,Ieff,ηeff,ηfil,x='$τ_2$ (ps)',y='',legendtext=text,grid=1,
            xlim=(0,dts[-1]),ylim=(-0.0,1.0),clip=0,
            seed=4,c='31204',scale=(2,1),save=save) # seed 4 rygb seed 2 rybg
        x = Ieff.maxloc()
    dt1,L1=0.57, 5 # γ = 13.6409; γ = 7.44274; γ = 1.66183
    dt1,L1=1.13,10 # γ = 17.2636; γ = 13.6629; γ = 6.23854
    dt1,L1=1.70,20 # γ = 11.7542; γ = 9.7756; γ = 6.42317
    cascadedpurityplot(dt1=dt1,L1=L1,L2= 5,λ1=780,λ2=2340,type1='zyy',type2='yzy',res=1001)
    cascadedpurityplot(dt1=dt1,L1=L1,L2=10,λ1=780,λ2=2340,type1='zyy',type2='yzy',res=1001)
    cascadedpurityplot(dt1=dt1,L1=L1,L2=20,λ1=780,λ2=2340,type1='zyy',type2='yzy',res=1001)
def typeIIpurityvsλ():
    def Pvλ():
        λis = np.linspace(1800,3800,21)
        def P(λ1,λ2):
            u = fjsabest(λ1,λ2,dt=0.001)
            return 1/schmidtK(u)
        u0 = Wave([P(780,λi) for λi in λis],λis,780)
        u1 = Wave(apodizedmaxpurity(phasematchangle(780,λis,sell='ktpwg',Type='yzy')),λis,780)
        v0 = Wave([P(810,λi) for λi in λis],λis,810)
        v1 = Wave(apodizedmaxpurity(phasematchangle(810,λis,sell='ktpwg',Type='yzy')),λis,810)
        Wave.plots(u0,u1,v0,v1,l='2323')
    # Pvλ()
    def PηI(λ1,λ2,dt=0.001,dλ=30): # cw upconversion
        u = fjsabest(λ1,λ2,dt)
        v = fjca(λ2,λ1,dt=0.1,L=0,sell='ktpwg',Type='zzz',dλ1=dλ*(λ2/λ1)**2,dλ3=dλ,num1=1001,num3=1001)
        w = integrate(u,v)
        print(' *  λ1,λ2',λ1,λ2)
        print('P',1/schmidtK(w))
        print('η',efficiency(w,u))
        print('I',w.indistinguishability())
    PηI(780,2340)
    PηI(810,3000)
    PηI(2340,780)
    PηI(2340,810)
    PηI(3000,780)
    def plotex():
        λ1,λ2,dλ = 810,3000,30
        u = fjsabest(λ1,λ2,dt=0.001,dλ=30); u.plot()
        v = fjca(λ2,λ1,dt=0.1,L=0,sell='ktpwg',Type='zzz',dλ1=dλ*(λ2/λ1)**2,dλ3=dλ,num1=1001,num3=1001); v.plot()
        w = integrate(u,v); w.plot()
    # plotex()
def gaussfilteredjsa(Δf=12e9,λ1=1560,λ2=1560,dt=0.1,L=30,sell='lnwg',Type='zzz',dλ1=0.5,dλ2=0.5,num1=1001,num2=1001,plot=False):
    def gaussfilter(x,x0,xfwhm):
        # exp(2*a*(xfwhm/2)**2)=0.5 → a = 2*ln(0.5)/xfwhm**2
        a = 2*log(2)/xfwhm**2
        return exp(-a*(x-x0)**2)
    # print(gaussfilter(1561,1560,2)**2) # 0.5
    u = fjsa(λ1=λ1,λ2=λ2,dt=dt,L=L,sell=sell,Type=Type,dλ1=dλ1,dλ2=dλ2,num1=num1,num2=num2)
    # def rescale(ww):
    #     ww.xs = (ww.xs-1/1560)*1e9*299792458*1e-9 # 1/nm to GHz
    #     ww.ys = (ww.ys-1/1560)*1e9*299792458*1e-9
    #     return ww
    # u = rescale(u)
    u = u.invnm2ghz()
    w = u * gaussfilter(u.xx,0,12) * gaussfilter(u.yy,0,12)
    if plot:
        u.plot(x='Δf (GHz)',y='Δf (GHz)',legendtext=f"P = {1/schmidtK(u):g}")
        w.plot(x='Δf (GHz)',y='Δf (GHz)',legendtext=f"P = {1/schmidtK(w):g}")
        Δλ = 1e9*Δf*(1560e-9)**2/299792458
        print(f"Δf:{Δf*1e-9:g}GHz, Δλ:{Δλ:g}nm")
    return w.ghz2invnm(λ1,λ2)
def hermitegaussmodeplot():
    if 1: # hermitegaussmode test
        xs = np.linspace(-20,20,100001)
        ws = [Wave(hermitegaussmode(n,xs),xs,f"n={n:g}") for n in range(100)]
        print(list2str([w.max() for w in ws],sep=','))
        xs = np.linspace(-20,20,1001)
        ws = [Wave(hermitegaussmode(n,xs,1),xs,f"n={n:g}") for n in range(0,100,25)]
        Wave.plots(*ws,lw=1,scale=(2,1))
    if 1: # hermitegausstemporalmode test
        Wave.plots(*[hermitegausstemporalmode(n,dt=0.1,num=501) for n in range(0,4+1,1)][::-1],x='time (ns)',y='relative amplitude',save='hermitegausstemporalmodeplot')
        # Wave.plots(*[hermitegausstemporalmode(n,dt=0.1,num=501) for n in [125,5,0]])
    if 1: # hermitegaussspectralmode test
        ws = [hermitegaussspectralmode(n,λ=1560,dt=0.1,dλ=0.2,num=501,ghz=0) for n in range(0,4+1,1)][::-1] # print(ws[-1].area())
        # ws = [hermitegaussspectralmode(n,λ=1560,dt=10,dλ=0.002,num=501,ghz=0) for n in range(0,4+1,1)][::-1] # print(ws[-1].area())
        # ws = [hermitegaussspectralmode(n,λ=1560,dt=0.01,dλ=2,num=501,ghz=0) for n in range(0,4+1,1)][::-1] # print(ws[-1].area())
        # ws = [hermitegaussspectralmode(n,λ=1560,dt=0.01,dλ=2,num=5001,ghz=0) for n in range(0,4+1,1)][::-1] # print(ws[-1].area())
        Wave.plots(*[w/ws[0].max() for w in ws],x='Δf (GHz)',y='relative amplitude')
        # Wave.plots(*ws,x='Δf (GHz)',y='relative amplitude',scale=(1,1),save='hermitegaussspectralmodeplot')
def higherorderpumptest(dλ=50,num1=201,num3=501):
    u = fjca(1560,780,dt=0.0001,L=0,sell='ktpwg',Type='yzy',dλ1=4*dλ,dλ3=dλ,num1=num1,num3=num3)
    v = fjcahop(0,1560,780,dt=0.0001,L=0,sell='ktpwg',Type='yzy',dλ1=4*dλ,dλ3=dλ,num1=num1,num3=num3)
    vv = fjcahop(3,1560,780,dt=0.0001,L=0,sell='ktpwg',Type='yzy',dλ1=4*dλ,dλ3=dλ,num1=num1,num3=num3)
    # print(u.shape[1]),len(u[u.xs,:])),len(u.ys))
    uy,vy,vvy = [w[w.nx//2,:] for w in (u,v,vv)]
    Wave.plots(uy,vy,vvy,l='023',scale=(3,1))
    print(uy[u.ny//2]/vy[v.ny//2])
    # u.plot(aspect=1)
    # v.plot(aspect=1)
    # vv.plot(aspect=1)
def innerproducttest():
    dλ=200;num1=501;num3=101
    v = fjcahop(0,1560,1560,dt=0.0001,L=0,sell='ktpwg',Type='yzy',dλ1=dλ,dλ3=dλ,num1=num1,num3=num3)
    print(v.nx,len(v[:,v.ny//2]))
    a = hermitegaussspectralmode(0,1560,dt=0.0001,dλ=200,num=501)
    Wave.plots(v[:,v.ny//2],a,l='0123')
    aa = v[:,v.ny//2:v.ny//2+2].T()
    bb = aa.integrate(v)
    print('aa.shape',aa.shape,'v.shape',v.shape,'bb.shape',bb.shape)
    b = innerproduct(a,v)
    print(innerproduct(a,v[:,v.ny//2]))
    Wave.plots(bb[0,:],b,c='0k',l='04')
def modeupconversionplot(ninput,nescort,λ=1560,dt=0.1,dλ=0.2,num1=1001,num3=201):
    a = hermitegaussspectralmode(ninput,λ,dt=dt,dλ=dλ,num=num1)
    ts = (1,0.1)
    bs = [fjcahop(nescort,λ,λ,dt=t,L=0,dλ1=dλ,dλ3=dλ,num1=num1,num3=num3) for t in ts]
    cs = [innerproduct(a,b) for b in bs]
    ηs = [(c**2).area()/(a**2).area() for c in cs]
    ws = [invnm2ghz(c)/a.max() for c in cs]
    ws = [c.rename(f'output, {t:g}ns escort, η={η:.3f}') for c,t,η in zip(ws,ts,ηs)]
    w0 = (invnm2ghz(a)/a.max()).rename('input, 0.1ns FWHM')
    Wave.plots(w0,*ws,x='Δf (GHz)',y='relative amplitude',c='120',l='022',scale=(1.5,1),fork=0)
def lntelcom(λ=1560,dt=0.1,dλ=0.5,num1=1001,num3=101):
    a = hermitegaussspectralmode(0,λ,dt=dt,dλ=dλ,num=num1)
    bb = fjcahop(0,λ,λ,dt=1*dt,L=0,dλ1=dλ,dλ3=dλ,num1=num1,num3=num3)
    # print('bb[:,bb.ny//2].area()',bb[:,bb.ny//2].area())
    c = innerproduct(a,bb)
    # Wave.plots(bb[:,bb.ny//2],a,l='0123')
    Wave.plots(a,c,l='0123')

    # print(c.sum()/a.sum())
    # print(c.area()/a.area())
    print((c**2).area()/(a**2).area())

    u = gaussfilteredjsa(Δf=12e9,λ1=1560,λ2=1560,dt=0.1,L=30,sell='lnwg',Type='zzz',dλ1=0.5,dλ2=0.5,num1=num1,num2=num1) # u.fplot()
    v0 = fjca(λ,λ,dt=1*dt,L=0,dλ1=dλ,dλ3=dλ,num1=num1,num3=num3)
    v = fjcahop(0,λ,λ,dt=1*dt,L=0,dλ1=dλ,dλ3=dλ,num1=num1,num3=num3)
    vv = fjcahop(3,λ,λ,dt=1*dt,L=0,dλ1=dλ,dλ3=dλ,num1=num1,num3=num3)
    w0 = u.integrate(v0) #.fplot()
    w = u.integrate(v) #.fplot()
    ww = u.integrate(vv) #.fplot()
    print('η',efficiency(w0,u))
    print('η',efficiency(w,u))
    print('η',efficiency(ww,u))



if __name__ == '__main__':
    # Wave2D
    print(datetime.datetime.now())
    # discretedeltafunctiontest()
    # tophattest()
    # hermitetest()
    # angleplot()
    # feffcigartest()
    # θϕplot(10,1,3,1); θϕplot(10,3,10,1); θϕplot(10,3,3,1); θϕplot(10,3,10,1)
    # puritysearch()
    # scanhighpe()
    # purityvsjcapulse()
    # telcomexample()
    # ktpwgexample(1001,dλ=40,plot=1) # Pjsa,Pjca,Peff,Ieff,ηeff 0.245779 0.904167 0.949629 0.969756 0.131137
    # ktpwgexample(1001,dλ=40,plotλ=1) # Pjsa,Pjca,Peff,Ieff,ηeff 0.245778 0.904163 0.949626 0.969739 0.01457
    # from sellmeier import jsiplot
    # jsiplot(w1=780,w2=2340,sell='ktpwg',Type='yzy',lengthinmm=10,dw=None,num=2001,dt=0.001,schmidt=1,schmidtres=400,apodized=1,indistinguishability=1,intensity=1,plot=0) # τ = 0.001, K = 4.1014565771196985, purity = 24.381582035479287%, 0.9999999999999996
    # ktpwgexample(res=1001,dλ=40,dt2=0.01,L2=0,plot=1) # Pjsa,Pjca,Peff,Ieff,ηeff 0.245779 0.0033583 0.247309 0.731864 0.989706
    # ktpwgexample(res=1001,dλ=40,dt2=0.0001,L2=0,plot=1) # Pjsa,Pjca,Peff,Ieff,ηeff 0.245779 0.160166 0.951406 0.457857 0.0813983
    # schmidttest()
    # filtereddegen(Δλ=0.1,jplot=1)
    # filtereddegen(Δλ=0.02,jplot=1)
    # filtereddegen()
    # filtereddegencompare()
    # gaussianproducttest()
    # ellipticalgaussianmodetest()
    # phasematchangletest()
    # upconversionidentitymatrix(aspect=1,plotit=1)
    # tophatfilter(aspect=1,plotit=1)
    # cwupconversionpump(aspect=1,plotit=1)
    # pulsedupconversionpump(aspect=1,plotit=1)
    # cascadedpurity(λ1=780,λ2=2340)
    # cascadedpurity(λ1=810,λ2=2340)
    # typeIIpurityvsλ()
    # gaussfilteredjsa(plot=1)
    # hermitegaussmodeplot()
    # higherorderpumptest()
    # innerproducttest()
    # lntelcom()
    modeupconversionplot(ninput=0,nescort=0)

    print(datetime.datetime.now())


