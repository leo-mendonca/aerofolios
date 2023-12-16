import sympy

def teste():
    a,b,c=sympy.symbols("a,b,c")
    x,y,z=sympy.symbols("x,y,z")
    f=(a*x+b*y)*(a*x+c*y)

if __name__=="__main__":
    alfa,beta=sympy.symbols("alfa,beta")
    q, r, s = sympy.symbols("q,r,s")
    aj, bj, cj, dj, ej, fj = sympy.symbols("aj,bj,cj,dj,ej,fj")
    ai, bi, ci, di, ei, fi = sympy.symbols("ai,bi,ci,di,ei,fi")
    Mj = aj + bj * alfa + cj * beta + dj * alfa ** 2 + ej * beta ** 2 + fj * alfa * beta
    Mi = ai + bi * alfa + ci * beta + di * alfa ** 2 + ei * beta ** 2 + fi * alfa * beta
    dM = q + r * alfa + s * beta
    produto=Mi*Mj*dM
    combinacoes=[]
    for i in range(6):
        for j in range(6):

            combinacoes.append(alfa**i*beta**j)
    simplificado=sympy.collect(produto.expand(), combinacoes[::-1])
    resposta=ai*aj*q + \
             alfa**5*di*dj*r + \
             alfa**2*beta**2*(bi*ej*r + bi*fj*s + bj*ei*r + bj*fi*s + ci*dj*s + ci*fj*r + cj*di*s + cj*fi*r + di*ej*q + dj*ei*q + fi*fj*q) + \
             alfa*(ai*aj*r + ai*bj*q + aj*bi*q + alfa**2*beta**2*di*ej*r + alfa**2*beta**2*di*fj*s + alfa**2*beta**2*dj*ei*r + alfa**2*beta**2*dj*fi*s + alfa**2*beta**2*fi*fj*r + alfa**2*beta*bi*dj*s + alfa**2*beta*bi*fj*r + alfa**2*beta*bj*di*s + alfa**2*beta*bj*fi*r + alfa**2*beta*ci*dj*r + alfa**2*beta*cj*di*r + alfa**2*beta*di*fj*q + alfa**2*beta*dj*fi*q) + \
             beta*(ai*aj*s + ai*cj*q + aj*ci*q + alfa**2*beta**2*di*ej*s + alfa**2*beta**2*dj*ei*s + alfa**2*beta**2*ei*fj*r + alfa**2*beta**2*ej*fi*r + alfa**2*beta**2*fi*fj*s + alfa*beta**2*bi*ej*s + alfa*beta**2*bj*ei*s + alfa*beta**2*ci*ej*r + alfa*beta**2*ci*fj*s + alfa*beta**2*cj*ei*r + alfa*beta**2*cj*fi*s + alfa*beta**2*ei*fj*q + alfa*beta**2*ej*fi*q) + \
             alfa**2*(ai*bj*r + ai*dj*q + aj*bi*r + aj*di*q + alfa**2*beta*di*dj*s + alfa**2*beta*di*fj*r + alfa**2*beta*dj*fi*r + bi*bj*q) + \
             beta**2*(ai*cj*s + ai*ej*q + aj*ci*s + aj*ei*q + alfa*beta**2*ei*ej*r + alfa*beta**2*ei*fj*s + alfa*beta**2*ej*fi*s + ci*cj*q) + \
             alfa*beta*(ai*bj*s + ai*cj*r + ai*fj*q + aj*bi*s + aj*ci*r + aj*fi*q + bi*cj*q + bj*ci*q) + \
             alfa**3*(ai*dj*r + aj*di*r + bi*bj*r + bi*dj*q + bj*di*q) + \
             beta**3*(ai*ej*s + aj*ei*s + ci*cj*s + ci*ej*q + cj*ei*q) + \
             alfa ** 2 * beta * (ai * dj * s + ai * fj * r + aj * di * s + aj * fi * r + bi * bj * s + bi * cj * r + bi * fj * q + bj * ci * r + bj * fi * q + ci * dj * q + cj * di * q) + \
             alfa * beta ** 2 * (ai * ej * r + ai * fj * s + aj * ei * r + aj * fi * s + bi * cj * s + bi * ej * q + bj * ci * s + bj * ei * q + ci * cj * r + ci * fj * q + cj * fi * q) + \
             alfa**4*(bi*dj*r + bj*di*r + di*dj*q) + \
             beta**4*(ci*ej*s + cj*ei*s + ei*ej*q) + \
             beta**5*ei*ej*s + \
             0
    resposta_corrigida=ai*aj*q + \
             alfa*(ai*aj*r + ai*bj*q + aj*bi*q) + \
             beta*(ai*aj*s + ai*cj*q + aj*ci*q) + \
             alfa**2*(ai*bj*r + ai*dj*q + aj*bi*r + aj*di*q + bi*bj*q) + \
             beta**2*(ai*cj*s + ai*ej*q + aj*ci*s + aj*ei*q + ci*cj*q) + \
             alfa*beta*(ai*bj*s + ai*cj*r + ai*fj*q + aj*bi*s + aj*ci*r + aj*fi*q + bi*cj*q + bj*ci*q) + \
             alfa**3*(ai*dj*r + aj*di*r + bi*bj*r + bi*dj*q + bj*di*q) + \
             beta**3*(ai*ej*s + aj*ei*s + ci*cj*s + ci*ej*q + cj*ei*q) + \
             alfa**2*beta*(ai*dj*s + ai*fj*r + aj*di*s + aj*fi*r + bi*bj*s + bi*cj*r + bi*fj*q + bj*ci*r + bj*fi*q + ci*dj*q + cj*di*q) + \
             alfa*beta**2*(ai*ej*r + ai*fj*s + aj*ei*r + aj*fi*s + bi*cj*s + bi*ej*q + bj*ci*s + bj*ei*q + ci*cj*r + ci*fj*q + cj*fi*q) + \
             alfa**4*(bi*dj*r + bj*di*r + di*dj*q) + \
             beta**4*(ci*ej*s + cj*ei*s + ei*ej*q) + \
             alfa**3*beta*( bi*dj*s + bi*fj*r + bj*di*s + bj*fi*r + ci*dj*r + cj*di*r + di*fj*q + dj*fi*q) + \
             alfa**2*beta**2*(bi*ej*r + bi*fj*s + bj*ei*r + bj*fi*s + ci*dj*s + ci*fj*r + cj*di*s + cj*fi*r + di*ej*q + dj*ei*q + fi*fj*q) + \
             alfa*beta**3*( + bi*ej*s + bj*ei*s + ci*ej*r + ci*fj*s + cj*ei*r + cj*fi*s + ei*fj*q + ej*fi*q) + \
             alfa**5*di*dj*r + \
             beta**5*ei*ej*s + \
             alfa ** 4 * beta(di * dj * s + di * fj * r + dj * fi * r) + \
             alfa ** 3 * beta ** 2 * (di * ej * r + di * fj * s + dj * ei * r + dj * fi * s + fi * fj * r) + \
             alfa ** 2 * beta ** 3 * (di * ej * s + dj * ei * s + ei * fj * r + ej * fi * r + fi * fj * s) + \
             alfa * beta ** 4 * (ei * ej * r + ei * fj * s + ej * fi * s)

    resposta2=ai*aj*q + \
              alfa**5*di*dj*r + \
              alfa**4*(bi*dj*r + bj*di*r + di*dj*q) + \
              alfa**3*(ai*dj*r + aj*di*r + bi*bj*r + bi*dj*q + bj*di*q) + \
              alfa**2*beta**2*(alfa*di*ej*r + alfa*di*fj*s + alfa*dj*ei*r + alfa*dj*fi*s + alfa*fi*fj*r + beta*di*ej*s + beta*dj*ei*s + beta*ei*fj*r + beta*ej*fi*r + beta*fi*fj*s + bi*ej*r + bi*fj*s + bj*ei*r + bj*fi*s + ci*dj*s + ci*fj*r + cj*di*s + cj*fi*r + di*ej*q + dj*ei*q + fi*fj*q) + alfa**2*beta*(ai*dj*s + ai*fj*r + aj*di*s + aj*fi*r + alfa**2*di*dj*s + alfa**2*di*fj*r + alfa**2*dj*fi*r + alfa*bi*dj*s + alfa*bi*fj*r + alfa*bj*di*s + alfa*bj*fi*r + alfa*ci*dj*r + alfa*cj*di*r + alfa*di*fj*q + alfa*dj*fi*q + bi*bj*s + bi*cj*r + bi*fj*q + bj*ci*r + bj*fi*q + ci*dj*q + cj*di*q) + alfa**2*(ai*bj*r + ai*dj*q + aj*bi*r + aj*di*q + bi*bj*q) + alfa*beta**2*(ai*ej*r + ai*fj*s + aj*ei*r + aj*fi*s + beta**2*ei*ej*r + beta**2*ei*fj*s + beta**2*ej*fi*s + beta*bi*ej*s + beta*bj*ei*s + beta*ci*ej*r + beta*ci*fj*s + beta*cj*ei*r + beta*cj*fi*s + beta*ei*fj*q + beta*ej*fi*q + bi*cj*s + bi*ej*q + bj*ci*s + bj*ei*q + ci*cj*r + ci*fj*q + cj*fi*q) + alfa*beta*(ai*bj*s + ai*cj*r + ai*fj*q + aj*bi*s + aj*ci*r + aj*fi*q + bi*cj*q + bj*ci*q) + alfa*(ai*aj*r + ai*bj*q + aj*bi*q) + beta**5*ei*ej*s + beta**4*(ci*ej*s + cj*ei*s + ei*ej*q) + beta**3*(ai*ej*s + aj*ei*s + ci*cj*s + ci*ej*q + cj*ei*q) + beta**2*(ai*cj*s + ai*ej*q + aj*ci*s + aj*ei*q + ci*cj*q) + beta*(ai*aj*s + ai*cj*q + aj*ci*q)

    resposta3= ai*aj*q + \
               alfa**5*di*dj*r + \
               alfa**4*(bi*dj*r + bj*di*r + di*dj*q) + \
               alfa**3*(ai*dj*r + aj*di*r + bi*bj*r + bi*dj*q + bj*di*q) + \
               alfa**2*beta**2*(alfa*di*ej*r + alfa*di*fj*s + alfa*dj*ei*r + alfa*dj*fi*s + alfa*fi*fj*r + beta*di*ej*s + beta*dj*ei*s + beta*ei*fj*r + beta*ej*fi*r + beta*fi*fj*s + bi*ej*r + bi*fj*s + bj*ei*r + bj*fi*s + ci*dj*s + ci*fj*r + cj*di*s + cj*fi*r + di*ej*q + dj*ei*q + fi*fj*q) + alfa**2*beta*(ai*dj*s + ai*fj*r + aj*di*s + aj*fi*r + alfa**2*di*dj*s + alfa**2*di*fj*r + alfa**2*dj*fi*r + alfa*bi*dj*s + alfa*bi*fj*r + alfa*bj*di*s + alfa*bj*fi*r + alfa*ci*dj*r + alfa*cj*di*r + alfa*di*fj*q + alfa*dj*fi*q + bi*bj*s + bi*cj*r + bi*fj*q + bj*ci*r + bj*fi*q + ci*dj*q + cj*di*q) + alfa**2*(ai*bj*r + ai*dj*q + aj*bi*r + aj*di*q + bi*bj*q) + alfa*beta**2*(ai*ej*r + ai*fj*s + aj*ei*r + aj*fi*s + beta**2*ei*ej*r + beta**2*ei*fj*s + beta**2*ej*fi*s + beta*bi*ej*s + beta*bj*ei*s + beta*ci*ej*r + beta*ci*fj*s + beta*cj*ei*r + beta*cj*fi*s + beta*ei*fj*q + beta*ej*fi*q + bi*cj*s + bi*ej*q + bj*ci*s + bj*ei*q + ci*cj*r + ci*fj*q + cj*fi*q) + alfa*beta*(ai*bj*s + ai*cj*r + ai*fj*q + aj*bi*s + aj*ci*r + aj*fi*q + bi*cj*q + bj*ci*q) + alfa*(ai*aj*r + ai*bj*q + aj*bi*q) + beta**5*ei*ej*s + beta**4*(ci*ej*s + cj*ei*s + ei*ej*q) + beta**3*(ai*ej*s + aj*ei*s + ci*cj*s + ci*ej*q + cj*ei*q) + beta**2*(ai*cj*s + ai*ej*q + aj*ci*s + aj*ei*q + ci*cj*q) + beta*(ai*aj*s + ai*cj*q + aj*ci*q)

