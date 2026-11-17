import numpy as np
from threadpoolctl import threadpool_limits

def run(X,y,method):
 n=X.shape[1]; w=np.zeros(n); L=np.linalg.norm(X,2)**2; a=1/L; t=.25*a*np.max(np.abs(X.T@y)); z=w.copy(); q=1
 for k in range(400):
  old=w.copy(); u=(z if method=='FISTA' else w); v=u-a*(X.T@(X@u-y)); av=np.abs(v)
  if method=='MESFM':
   w=np.where(av>t,np.sign(v)*(av-t/(1+(av/(t+1e-8))**2.5)),0)
  else: w=np.sign(v)*np.maximum(av-t,0)
  if method=='FISTA':
   qn=(1+np.sqrt(1+4*q*q))/2; z=w+(q-1)/qn*(w-old);q=qn
 if method=='MESFM':
  peak=np.max(np.abs(w)); supp=np.where(np.abs(w)>.2*peak)[0] if peak>0 else []
  if len(supp):
   v=np.zeros(n);v[supp]=np.linalg.lstsq(X[:,supp],y,rcond=None)[0];w=v
 return w

with threadpool_limits(limits=1):
 for sigma in [0,0.05,0.15]:
  result={k:[] for k in ['ISTA','FISTA','MESFM']};supports={k:0 for k in result}
  for seed in range(20):
   rng=np.random.RandomState(seed);m=40;n=100;X=rng.randn(m,n)/np.sqrt(m); truth=np.zeros(n);supp=rng.choice(n,5,replace=False);truth[supp]=rng.choice([-1,1],5)*rng.uniform(1.5,3.2,5);y=X@truth+rng.normal(0,sigma,m)
   for k in result:
    w=run(X,y,k);result[k].append(np.linalg.norm(w-truth)/np.linalg.norm(truth)*100);supports[k]+=set(np.flatnonzero(np.abs(w)>1e-8))==set(supp)
  for k in result:print(sigma,k,round(np.mean(result[k]),2),round(np.std(result[k],ddof=1),2),supports[k],'/20')
