"""Editorial check of the manuscript equations against a token recurrence.

CPU-only; no accelerator timing, no new performance claim, no changes to
historical experiment evidence.
"""
from pathlib import Path
from paper_paths import check_output
import json
import numpy as np

def recurrence(Q,K,V,g,b,w,S,do,ds):
    state=S.copy()
    outputs=[]
    for i in range(len(Q)):
        dec=np.exp(g[i])[:,None]*state
        r=w[i]*V[i]-dec.T@(b[i]*K[i])
        state=dec+np.outer(K[i],r)
        outputs.append(state.T@Q[i])
    return np.sum(np.asarray(outputs)*do)+np.sum(state*ds)

def backward(Q,K,V,g,b,w,S,do,ds):
    gam=np.exp(np.cumsum(g,axis=0))
    E=b*K*gam; Y=K/gam; Z=w*V; Qg=Q*gam
    M=np.eye(len(Q))+np.tril(E@Y.T,-1)
    A=np.linalg.inv(M)
    R=A@(Z-E@S)
    lam=gam[-1]
    F=Y*lam
    C=np.tril(Qg@Y.T)
    rb=C.T@do+F@ds
    H=np.tril(do@R.T)
    fb=R@ds.T
    P=A.T@rb
    D=-np.tril(P@R.T,-1)
    qgb=do@S.T+H@Y
    eb=-P@S.T+D@Y
    yb=H.T@Qg+D.T@E+fb*lam
    sb=Qg.T@do+lam[:,None]*ds-E.T@P
    qb=gam*qgb; kb=gam*b*eb+yb/gam
    bb=gam*K*eb; vb=w*P; wb=V*P
    gb=Qg*qgb+E*eb-Y*yb
    gb[-1]+=lam*((S*ds).sum(1)+(Y*fb).sum(0))
    gb=np.cumsum(gb[::-1],axis=0)[::-1]
    # Centered vs uncentered consumer for arbitrary dimensions.
    center=(np.max(np.log(gam),axis=0)+np.min(np.log(gam),axis=0))/2
    alpha=gam/np.exp(center); zeta=1/alpha
    uq=H@(K*zeta); ue=D@(K*zeta)
    uy=H.T@(Q*alpha)+D.T@(b*K*alpha)
    centered=[alpha*uq,alpha*b*ue+zeta*uy,
              alpha*K*ue,(Q*alpha)*uq+(b*K*alpha)*ue-(K*zeta)*uy]
    Uq=H@Y; Ue=D@Y; Uy=H.T@Qg+D.T@E
    normal=[gam*Uq,gam*b*Ue+Uy/gam,gam*K*Ue,Qg*Uq+E*Ue-Y*Uy]
    centerr=max(np.max(np.abs(x-y)) for x,y in zip(centered,normal))
    return [qb,kb,vb,gb,bb,wb,sb],float(centerr),P,R

rng=np.random.default_rng(79)
records=[]
for n,k,v in [(1,3,2),(4,3,5),(7,5,3)]:
    for family in ["delta","gdn","kda","gdn2"]:
        Q=rng.normal(size=(n,k))*.3; K=rng.normal(size=(n,k))*.2
        V=rng.normal(size=(n,v)); S=rng.normal(size=(k,v))*.2
        g=rng.uniform(-.3,.08,size=(n,k))
        if family=="delta": g[:]=0
        if family=="gdn": g[:]=g[:,:1]
        beta=rng.uniform(.1,.9,size=(n,1)); beta[0]=0
        b=np.broadcast_to(beta,(n,k)).copy()
        w=np.broadcast_to(beta,(n,v)).copy()
        if family=="gdn2":
            b=rng.uniform(-.1,.9,size=(n,k))
            w=rng.uniform(-.1,.9,size=(n,v))
        inputs=[Q,K,V,g,b,w,S]
        do=rng.normal(size=(n,v)); ds=rng.normal(size=(k,v))
        bars,ce,P,R=backward(*inputs,do,ds)
        maxerr=0
        # Independent complex-step differentiation of the token recurrence.
        for a,x in enumerate(inputs):
            for idx in np.ndindex(x.shape):
                pert=[z.astype(complex) for z in inputs]
                pert[a][idx]+=1e-25j
                val=recurrence(*pert,do,ds).imag/1e-25
                maxerr=max(maxerr,abs(val-bars[a][idx]))
        be=0.
        if family!="gdn2":
            db=bars[4].sum(1)+bars[5].sum(1)
            be=float(np.max(np.abs(beta[:,0]*db-(P*R).sum(1))))
        scalar_gate_error=0.
        if family=='gdn':
            # New manuscript accumulation, checked against the full gate
            # derivative already validated by independent complex-step above.
            gamma=np.exp(np.cumsum(g[:,0])); lam=gamma[-1]
            erase_beta=bars[4].sum(1)
            local=(bars[0]*Q-bars[1]*K).sum(1)+2*beta[:,0]*erase_beta
            state_endpoint=lam*np.sum(S*ds)
            key_endpoint=lam*np.sum((K/gamma[:,None])*(R@ds.T))
            local[-1]+=state_endpoint+key_endpoint
            merged=np.cumsum(local[::-1])[::-1]
            scalar_gate_error=float(np.max(np.abs(merged-bars[3].sum(1))))
        records.append(dict(shape=[n,k,v],family=family,
                            max_gradient_abs=float(maxerr),center_abs=ce,beta_abs=be,
                            scalar_gate_completion_abs=scalar_gate_error))
        assert max(maxerr,ce,be,scalar_gate_error)<1e-11,records[-1]
def backward_partitioned(inputs, do, ds, lengths):
    """Chain chunk pullbacks; obtain boundary states from the token recurrence."""
    Q,K,V,g,b,w,S=inputs
    assert sum(lengths)==len(Q) and min(lengths)>0
    starts=np.cumsum([0]+list(lengths))
    state=S.copy()
    incoming=[]
    for left,right in zip(starts[:-1],starts[1:]):
        incoming.append(state.copy())
        for i in range(left,right):
            dec=np.exp(g[i])[:,None]*state
            correction=w[i]*V[i]-dec.T@(b[i]*K[i])
            state=dec+np.outer(K[i],correction)
    bars=[np.zeros_like(x) for x in inputs]
    terminal=ds.copy()
    for c in reversed(range(len(lengths))):
        left,right=starts[c:c+2]
        local=[x[left:right] for x in inputs[:-1]]+[incoming[c]]
        got,_,_,_=backward(*local,do[left:right],terminal)
        for a in range(6): bars[a][left:right]=got[a]
        terminal=got[-1]
    bars[-1]=terminal
    return bars

multichunk=[]
for lengths,k,v in [([2,1,4,2],3,4),([3,5],5,2),([1,1,3],2,3)]:
    n=sum(lengths)
    for family in ["delta","gdn","kda","gdn2"]:
        Q=rng.normal(size=(n,k))*.3; K=rng.normal(size=(n,k))*.2
        V=rng.normal(size=(n,v)); S=rng.normal(size=(k,v))*.2
        g=rng.uniform(-.3,.08,size=(n,k))
        if family=="delta":g[:]=0
        if family=="gdn":g[:]=g[:,:1]
        beta=rng.uniform(.1,.9,size=(n,1)); beta[::3]=0
        b=np.broadcast_to(beta,(n,k)).copy()
        w=np.broadcast_to(beta,(n,v)).copy()
        if family=="gdn2":
            b=rng.uniform(-.1,.9,size=(n,k))
            w=rng.uniform(-.1,.9,size=(n,v))
        inputs=[Q,K,V,g,b,w,S]
        do=rng.normal(size=(n,v)); ds=rng.normal(size=(k,v))
        bars=backward_partitioned(inputs,do,ds,lengths)
        single,_,_,_=backward(*inputs,do,ds)
        partitionerr=max(float(np.max(np.abs(x-y))) for x,y in zip(bars,single))
        maxerr=0.
        for a,x in enumerate(inputs):
            for idx in np.ndindex(x.shape):
                pert=[z.astype(complex) for z in inputs]
                pert[a][idx]+=1e-25j
                ref=recurrence(*pert,do,ds).imag/1e-25
                maxerr=max(maxerr,float(abs(ref-bars[a][idx])))
        shareerr=0.
        for i in range(n):
            if family!="gdn2":
                pert=[z.astype(complex) for z in inputs]
                pert[4][i]+=1e-25j; pert[5][i]+=1e-25j
                ref=recurrence(*pert,do,ds).imag/1e-25
                shareerr=max(shareerr,float(abs(ref-bars[4][i].sum()-bars[5][i].sum())))
            if family=="gdn":
                pert=[z.astype(complex) for z in inputs]
                pert[3][i]+=1e-25j
                ref=recurrence(*pert,do,ds).imag/1e-25
                shareerr=max(shareerr,float(abs(ref-bars[3][i].sum())))
        row=dict(shape=[n,k,v],chunk_lengths=lengths,family=family,
                 max_gradient_abs=maxerr,partition_invariance_abs=partitionerr,
                 shared_parameter_abs=shareerr)
        assert max(maxerr,partitionerr,shareerr)<1e-11,row
        multichunk.append(row)

gramerr=0
for n,k in [(1,2),(4,3),(7,5)]:
    K=rng.normal(size=(n,k)); G=rng.normal(size=(n,n))
    beta=rng.normal(size=n); beta[0]=0
    J=beta[:,None]*G*(K@K.T)
    L=G@K; R=G.T@(beta[:,None]*K)
    new=beta*(K*L).sum(1)-(K*R).sum(1)
    gramerr=max(gramerr,float(np.max(np.abs(new-(J.sum(1)-J.sum(0))))))
assert gramerr<1e-11
result={"purpose":"editorial equation check, not accelerator experimental evidence",
        "full_gradient_checks":records,"multichunk_gradient_checks":multichunk,
        "gram_max_abs":gramerr,"status":"PASS"}
result['scalar_gate_completion_max_abs']=max(r['scalar_gate_completion_abs'] for r in records)
check_output("math_check.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"status":"PASS","single_chunk_cases":len(records),
                  "multichunk_cases":len(multichunk),
                  "max_gradient_abs":max(r["max_gradient_abs"] for r in records+multichunk),
                  "max_partition_abs":max(r["partition_invariance_abs"] for r in multichunk),
                  "max_shared_parameter_abs":max(r["shared_parameter_abs"] for r in multichunk),
                  "gram_max_abs":gramerr,
                  "scalar_gate_completion_max_abs":result['scalar_gate_completion_max_abs']}))
