using DelimitedFiles


function rot2d(theta)
    M=[
        cos(theta)  -sin(theta);
        sin(theta)  cos(theta)
    ]
    return M
end


function main()

    S=-0.7
    if S<0
        println("\tIN POINTING SPINS")
    else
        println("\tOUT POINTING SPINS")
    end
    Q=[1,0,-2]
    P0=1.0
    Pi=[P0,0,0]
    F2Q=0.565

    SDIG = 3
    gyro=-1.913
    r0=2.818

    a=4.960
    b=4.960
    c=13.599

    adir = a*[1,0,0]
    bdir = b*[-1/2,sqrt(3)/2,0]
    cdir = c*[0,0,1]
    tocart_dir(v)=adir*v[1] .+ bdir*v[2] .+ cdir*v[3]

    arec=(2pi)*(1/a)*[1,1/sqrt(3),0]
    brec=(2pi)*(1/b)*[0,2/sqrt(3),0]
    crec=(2pi)*(1/c)*[0,0,1]
    tocart_rec(v)=arec*v[1] .+ brec*v[2] .+ crec*v[3]

    QC=tocart_rec(Q)

    cr_scattlen = 3.635
    o_scattlen  = 5.803

    # structure from the Bilbao server fractional coordinates
    Cr_01_2=tocart_dir([0.000000,  0.000000,   0.152400])
    Cr_02_1=tocart_dir([0.000000,  0.000000,   0.347600])
    Cr_03_2=tocart_dir([0.000000,  0.000000,   0.652400])
    Cr_04_1=tocart_dir([0.000000,  0.000000,   0.847600])
    Cr_05_1=tocart_dir([0.333333,  0.666667,   0.014267])
    Cr_06_2=tocart_dir([0.333333,  0.666667,   0.319067])
    Cr_07_1=tocart_dir([0.333333,  0.666667,   0.514267])
    Cr_08_2=tocart_dir([0.333333,  0.666667,   0.819067])
    Cr_09_1=tocart_dir([0.666667,  0.333333,   0.180933])
    Cr_10_2=tocart_dir([0.666667,  0.333333,   0.485733])
    Cr_11_1=tocart_dir([0.666667,  0.333333,   0.680933])
    Cr_12_2=tocart_dir([0.666667,  0.333333,   0.985733])

    O01=tocart_dir([0.305600,  0.000000,   0.250000])
    O02=tocart_dir([0.694400,  0.694400,   0.250000])
    O03=tocart_dir([0.000000,  0.305600,   0.250000])
    O04=tocart_dir([0.638933,  0.666667,   0.916667])
    O05=tocart_dir([0.972267,  0.333333,   0.583333])
    O06=tocart_dir([0.027733,  0.361067,   0.916667])
    O07=tocart_dir([0.333333,  0.972267,   0.916667])
    O08=tocart_dir([0.361067,  0.027733,   0.583333])
    O09=tocart_dir([0.666667,  0.638933,   0.583333])
    O10=tocart_dir([0.694400,  0.000000,   0.750000])
    O11=tocart_dir([0.000000,  0.694400,   0.750000])
    O12=tocart_dir([0.305600,  0.305600,   0.750000])
    O13=tocart_dir([0.027733,  0.666667,   0.416667])
    O14=tocart_dir([0.333333,  0.361067,   0.416667])
    O15=tocart_dir([0.638933,  0.972267,   0.416667])
    O16=tocart_dir([0.361067,  0.333333,   0.083333])
    O17=tocart_dir([0.666667,  0.027733,   0.083333])
    O18=tocart_dir([0.972267,  0.638933,   0.083333])

    Crs=[Cr_01_2,Cr_02_1,Cr_03_2,Cr_04_1,Cr_05_1,Cr_06_2,Cr_07_1,Cr_08_2,Cr_09_1,Cr_10_2,Cr_11_1,Cr_12_2]
    Os=[O01,O02,O03,O04,O05,O06,O07,O08,O09,O10,O11,O12,O13,O14,O15,O16,O17,O18]

    NCr=sum([exp(1im*dot(QC,Cr)) for Cr in Crs])*cr_scattlen
    NO=sum([exp(1im*dot(QC,O)) for O in Os])*o_scattlen
    NQ=NCr+NO

    MQCrs=[
        -exp(1im*dot(QC,Cr_01_2)),
        +exp(1im*dot(QC,Cr_02_1)),
        -exp(1im*dot(QC,Cr_03_2)),
        +exp(1im*dot(QC,Cr_04_1)),
        +exp(1im*dot(QC,Cr_05_1)),
        -exp(1im*dot(QC,Cr_06_2)),
        +exp(1im*dot(QC,Cr_07_1)),
        -exp(1im*dot(QC,Cr_08_2)),
        +exp(1im*dot(QC,Cr_09_1)),
        -exp(1im*dot(QC,Cr_10_2)),
        +exp(1im*dot(QC,Cr_11_1)),
        -exp(1im*dot(QC,Cr_12_2))
    ]
    println("SPINS: $([sign(real(m)) for m in MQCrs])")
    MQ=gyro*r0*S*sum(MQCrs)

    println("RE(NQ): $(real(NQ))")
    println("IM(NQ): $(imag(NQ))")
    println()
    println("RE(MQ): $(real(MQ))")
    println("IM(MQ): $(imag(MQ))")
    println()
    MQhat=MQ/sqrt(sum(MQ .* MQ))
    gg=imag(dot(MQ,MQhat))/NQ
    println("γ: $(round(real(gg), digits=SDIG))")
    println()

    MQvec=[0.0,0.0,MQ]
    Qnorm=sqrt(dot(QC,QC))
    Qhat=QC/Qnorm
    MpQ=cross(Qhat,cross(MQvec,Qhat))

    x=tocart_rec(Qhat)
    y=tocart_rec([0,0,-6])
    z=cross(x,y)
    y=cross(z,x)
    # z=bdir
    # y=cross(z,x)
    # z=cross(x,y)
    x=x/sqrt(dot(x,x))
    y=y/sqrt(dot(y,y))
    z=z/sqrt(dot(z,z))
    tocart_mupad(v)=x*v[1] .+ y*v[2] .+ z*v[3]
    MpQp=tocart_mupad(MpQ)

    # vup=inv(Mdir)*z
    # vup=vup/sqrt(dot(vup,vup))
    # println("Vertical axis: $(round.(vup,digits=SDIG))")
    # println()

    println("MQ xtal axes: $(round.(MQvec,digits=SDIG))")
    println("MperpQ xtal axes: $(round.(MpQ,digits=SDIG))")
    println("MperpQ npol axes: $(round.(MpQp,digits=SDIG))")
    println()

    IPi=conj(NQ)*NQ+sum(conj(MpQp) .* MpQp)+dot(Pi,conj(NQ)*MpQp+conj(MpQp)*NQ)+1im*dot(Pi,cross(conj(MpQp),MpQp))
    IPiPf=Pi*(conj(NQ)*NQ-sum(conj(MpQp) .* MpQp))+1im*cross(conj(NQ)*MpQp-conj(MpQp)*NQ, Pi)+(conj(MpQp)*sum(MpQp .* Pi)+sum(conj(MpQp) .* Pi)*MpQp)+(conj(NQ)*MpQp+conj(MpQp)*NQ)-1im*cross(conj(MpQp),MpQp)
    # IPi=conj(NQ)*NQ+sum(MpQp .* conj(MpQp))+2*sum(Pi .* real(MpQp*conj(NQ)))+1im*sum(Pi .* cross(conj(MpQp),MpQp))
    # IPiPf=Pi*(NQ*conj(NQ)-sum(MpQp .* conj(MpQp)))+2*real(MpQp*conj(NQ)+MpQp*sum(Pi .* conj(MpQp)))+2*cross(Pi,imag(MpQp*conj(NQ)))+1im*cross(MpQp,conj(MpQp))
    Pf=real(IPiPf/IPi)

    println("Pi: $(Pi)")
    println("I(Pi): $(round(IPi,digits=SDIG))")
    println("Pf: $(round.(Pf,digits=SDIG))")

    # open("./cr2o3_bm.dat", "w") do io
    #     writedlm(io, [arec brec crec])
    #     writedlm(io, [real.(QC),real.(MpQ),real.(x),real.(y),real.(z)])
    #     writedlm(io, [inv(Mmupad)*Pi,inv(Mmupad)*Pf])
    # end

    return nothing
end


main()