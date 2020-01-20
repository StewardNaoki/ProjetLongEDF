#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pulp import *
from random import *
import csv
import pandas as pd


# In[3]:


c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
c6=[]

b1=[]
b2=[]
b3=[]
b4=[]
b5=[]


a11=[]
a12=[]
a13=[]
a14=[]
a15=[]
a16=[]

a21=[]
a22=[]
a23=[]
a24=[]
a25=[]
a26=[]


a31=[]
a32=[]
a33=[]
a34=[]
a35=[]
a36=[]

a41=[]
a42=[]
a43=[]
a44=[]
a45=[]
a46=[]

a51=[]
a52=[]
a53=[]
a54=[]
a55=[]
a56=[]

Vars=[]
nb_exemple=1000
for i in range(nb_exemple):
#### constante de la fonction de cout

    c1.append(uniform(0.010,0.015))
    c2.append(uniform(0.006,0.010))
    c3.append(uniform(0.007,0.012))
    c4.append(uniform(0.001,0.003))
    c5.append(uniform(0.003,0.007))
    c6.append(uniform(0.0005,0.0015))
    

    
### Contrainte 1    
    a11.append(uniform(0.05,0.15))
    a12.append(uniform(0.15,0.25))
    a13.append(uniform(0.135,0.165))
    a14.append(0)
    a15.append(uniform(0.030,0.050))
    a16.append(0)
    
    b1.append(uniform(7,9))
    
### contraine 2    
    a21.append(uniform(0.06,0.1))
    a22.append(uniform(0.08,0.12))
    a23.append(uniform(0.08,0.13))
    a24.append(uniform(0.007,0.012))
    a25.append(uniform(0.007,0.012))
    a26.append(0)
    
    b2.append(uniform(4,8))
### contraine 3

    a31.append(uniform(0.0005,0.0015))
    a32.append(uniform(0.003,0.007))
    a33.append(uniform(0.001,0.005))
    a34.append(uniform(0.08,0.12))
    a35.append(uniform(0.135,0.165))
    a36.append(0)
    
    b3.append(uniform(0.5,3.5))
### contrainte 4

    a41.append(uniform(0.001,0.003))
    a42.append(uniform(0.003,0.007))
    a43.append(uniform(0.004,0.01))
    a44.append(uniform(0.001,0.003))
    a45.append(uniform(0.005,0.01))
    a46.append(0)
    b4.append(uniform(0.1,0.9))
### contrainte 5
    a51.append(1)
    a52.append(1)
    a53.append(1)
    a54.append(1)
    a55.append(1)
    a56.append(1)
    b5.append(100)


# In[10]:


for i in range(nb_exemple):
    prob = LpProblem("TheCEIProblem",LpMinimize)
    x1=LpVariable("Var1",0,None,LpInteger)
    x2=LpVariable("Var2",0)
    x3=LpVariable("Var3",0)
    x4=LpVariable("Var4",0)
    x5=LpVariable("Var5",0)
    x6=LpVariable("Var6",0)
    prob += c1[i]*x1 + c2[i]*x2+ c3[i]*x3+c4[i]*x4 + c5[i]*x5+ c6[i]*x6
    prob += x1+ x2+x3+ x4+x5+x6== 100 
    prob += a11[i]*x1 + a12[i]*x2+a13[i]*x3+a14[i]*x4+a15[i]*x5+a16[i]*x6 >= b1[i] 
    prob += a21[i]*x1 + a22[i]*x2+a23[i]*x3+a24[i]*x4+a25[i]*x5+a26[i]*x6 >= b2[i]
    prob += a31[i]*x1 + a32[i]*x2+a33[i]*x3+a34[i]*x4+a35[i]*x5+a36[i]*x6 >= b3[i]
    prob += a41[i]*x1 + a42[i]*x2+a43[i]*x3+a44[i]*x4+a45[i]*x5+a46[i]*x6 >= b4[i]

    prob.solve()
    
    for v in prob.variables():
        Vars.append(v.varValue)
    


# In[11]:


liste=[]

ind=0
for i in range(nb_exemple):
    liste.append(i)
output=pd.DataFrame(columns = ['Var0', 'Var1','Var2','Var3','Var4','Var5'],index=liste)

i=0
ind=0
while i < len(Vars):
    output['Var0'][ind]=Vars[i]
    i=i+6
    ind=ind+1
   


i=1
ind=0
while i < len(Vars):
    output['Var1'][ind]=Vars[i]
    i=i+6
    ind=ind+1    
    
    
i=2
ind=0
while i < len(Vars):
    output['Var2'][ind]=Vars[i]
    i=i+6
    ind=ind+1
    
    
i=3
ind=0
while i < len(Vars):
    output['Var3'][ind]=Vars[i]
    i=i+6
    ind=ind+1
    
    
    
i=4
ind=0
while i < len(Vars):
    output['Var4'][ind]=Vars[i]
    i=i+6
    ind=ind+1
    
    
i=5
ind=0
while i < len(Vars):
    output['Var5'][ind]=Vars[i]
    i=i+6
    ind=ind+1
output.to_csv('output.csv', index=False, header=False)


# In[13]:


inputB1=pd.DataFrame(columns = ['B1'],index=liste)
i=0
ind=0
while i < len(b1):
    inputB1['B1'][ind]=b1[i]
    i=i+1
    ind=ind+1
inputB1.to_csv('inputB1.csv', index=False, header=False)

#### 2eme contrainte

inputB2=pd.DataFrame(columns = ['B2'],index=liste)
i=0
ind=0
while i < len(b2):
    inputB2['B2'][ind]=b2[i]
    i=i+1
    ind=ind+1
inputB2.to_csv('inputB2.csv', index=False, header=False)

#3eme contrainte

inputB3=pd.DataFrame(columns = ['B3'],index=liste)
i=0
ind=0
while i < len(b3):
    inputB3['B3'][ind]=b3[i]
    i=i+1
    ind=ind+1
inputB3.to_csv('inputB3.csv', index=False, header=False)

## 4eme contrainte

inputB4=pd.DataFrame(columns = ['B4'],index=liste)
i=0
ind=0
while i < len(b4):
    inputB4['B4'][ind]=b4[i]
    i=i+1
    ind=ind+1
inputB4.to_csv('inputB4.csv', index=False, header=False)
### contrainte 5
inputB5=pd.DataFrame(columns = ['B5'],index=liste)
i=0
ind=0
while i < len(b5):
    inputB5['B5'][ind]=b5[i]
    i=i+1
    ind=ind+1
inputB5.to_csv('inputB5.csv', index=False, header=False)


# In[ ]:





# In[14]:


inputA1=pd.DataFrame(columns = ['a11', 'a12','a13','a14','a15','a16'],index=liste)
i=0
ind=0
while i < len(a11):
    inputA1['a11'][ind]=a11[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a12):
    inputA1['a12'][ind]=a12[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a13):
    inputA1['a13'][ind]=a13[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a14):
    inputA1['a14'][ind]=a14[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a15):
    inputA1['a15'][ind]=a15[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a16):
    inputA1['a16'][ind]=a16[i]
    i=i+1
    ind=ind+1
inputA1.to_csv('inputA1.csv', index=False, header=False)


# In[15]:


inputA2=pd.DataFrame(columns = ['a21', 'a22','a23','a24','a25','a26'],index=liste)
i=0
ind=0
while i < len(a21):
    inputA2['a21'][ind]=a21[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a22):
    inputA2['a22'][ind]=a22[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a23):
    inputA2['a23'][ind]=a23[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a24):
    inputA2['a24'][ind]=a24[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a25):
    inputA2['a25'][ind]=a25[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a26):
    inputA2['a26'][ind]=a26[i]
    i=i+1
    ind=ind+1
inputA2.to_csv('inputA2.csv', index=False, header=False)


# In[16]:


inputA3=pd.DataFrame(columns = ['a31', 'a32','a33','a34','a35','a36'],index=liste)
i=0
ind=0
while i < len(a31):
    inputA3['a31'][ind]=a31[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a32):
    inputA3['a32'][ind]=a32[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a33):
    inputA3['a33'][ind]=a33[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a34):
    inputA3['a34'][ind]=a34[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a35):
    inputA3['a35'][ind]=a35[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a36):
    inputA3['a36'][ind]=a36[i]
    i=i+1
    ind=ind+1
inputA3.to_csv('inputA3.csv', index=False, header=False)


# In[17]:


inputA4=pd.DataFrame(columns = ['a41', 'a42','a43','a44','a45','a46'],index=liste)
i=0
ind=0
while i < len(a41):
    inputA4['a41'][ind]=a41[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a42):
    inputA4['a42'][ind]=a42[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a43):
    inputA4['a43'][ind]=a43[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a44):
    inputA4['a44'][ind]=a44[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a45):
    inputA4['a45'][ind]=a45[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a46):
    inputA4['a46'][ind]=a46[i]
    i=i+1
    ind=ind+1
inputA4.to_csv('inputA4.csv', index=False, header=False)


# In[18]:


inputA5=pd.DataFrame(columns = ['a51', 'a52','a53','a54','a55','a56'],index=liste)
i=0
ind=0
while i < len(a51):
    inputA5['a51'][ind]=a51[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a52):
    inputA5['a52'][ind]=a52[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a53):
    inputA5['a53'][ind]=a53[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a54):
    inputA5['a54'][ind]=a54[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a55):
    inputA5['a55'][ind]=a55[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a56):
    inputA5['a56'][ind]=a56[i]
    i=i+1
    ind=ind+1
inputA5.to_csv('inputA5.csv', index=False, header=False)


# In[19]:


inputC=pd.DataFrame(columns = ['c1', 'c2','c3','c4','c5','c6'],index=liste)
i=0
ind=0
while i < len(c1):
    inputC['c1'][ind]=c1[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(c2):
    inputC['c2'][ind]=c2[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(c3):
    inputC['c3'][ind]=c3[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(c4):
    inputC['c4'][ind]=c4[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(a15):
    inputC['c5'][ind]=c5[i]
    i=i+1
    ind=ind+1
i=0
ind=0
while i < len(c6):
    inputC['c6'][ind]=c6[i]
    i=i+1
    ind=ind+1
inputC.to_csv('inputC.csv', index=False, header=False)


# In[21]:


output.head(10)


# In[ ]:





# In[ ]:




