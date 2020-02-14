module KLSTM

using Flux, LinearAlgebra

export Model, E_Beta, multiplyGaussian, sumGaussians, conditionedGaussian, runForward, filtering, smoothing, smooth_all, g_predict, run_all, klstm_optimizer, train_model!, loss_model

#Constructor
mutable struct Model
    B #Emission matrix
    R #Emission noise
    # Prior hyperparameters of covariance noise
    prior_alfa_Q
    prior_beta_Q
    # Posterior hyperparamters of covariance noise, starts with the prior value and updated at each iteration
    alfa_Q
    beta_Q
    d1 #Dimensionality of observations
    d2 #Dimension of output of dense layer
    d3 #Dimension of LSTM output

    version::Integer #version 1 refers to first variant that models velocity, version 2 refers to second variant that directly models position

    #Neural network architecture
    InputLayer #Input layer
    Lstm #LSTM layer
    Affine #Affine mapping as output layer
    f #Function of stacked neural networks
    ps #NN parameters

    function Model(B, R, prior_alfa_Q, prior_beta_Q, d1, d2, d3, version)
        #Define nerual networks
        InputLayer, Lstm, Affine = Dense(d1,d2,relu), LSTM(d2, d3), Dense(d3, d1)
        f = Chain(InputLayer,Lstm,Affine)
        new(B, R, prior_alfa_Q, prior_beta_Q, prior_alfa_Q, prior_beta_Q, d1, d2, d3, version, InputLayer, Lstm, Affine, f, Flux.params(InputLayer, Lstm, Affine))
    end
end

E_Beta(alfa,beta) = diagm(0 => beta ./ (alfa .- 1.0)) #Mean of Beta dist. R.V.

# Kalman Filtering and Smoothing Functions

multiplyGaussian(A,m,V) = A*m, A*V*transpose(A)

sumGaussians(m1,m2,V1,V2) = m1+m2, V1+V2

function conditionedGaussian(m,V,B,o,R)
    K = V*transpose(B)*inv(B*V*transpose(B).+R)
    m_new = m .+ K*(o .- B*m)
    V_new = V .- K*B*V
    return m_new, V_new
end

function runForward(A,B,Q,R,mh_old,Vh_old)
    mh_1, Vh_1 = multiplyGaussian(A,mh_old,Vh_old)
    mh_pred, Vh_pred = sumGaussians(mh_1, zeros(length(mh_old)), Vh_1, Q)
    mo_pred1, Vo_pred1 = multiplyGaussian(B,mh_pred,Vh_pred)
    mo_pred, Vo_pred = sumGaussians(mo_pred1, zeros(length(mo_pred1)), Vo_pred1, R)
end

function filtering(A,B,Q,R,mh_old,Vh_old,obs_new)
    mh_1, Vh_1 = multiplyGaussian(A,mh_old,Vh_old)
    mh_pred, Vh_pred = sumGaussians(mh_1, zeros(length(mh_old)), Vh_1, Q)
    mh, Vh = conditionedGaussian(mh_pred,Vh_pred,B,obs_new,R)
end

function smoothing(A,B,Q,R,f,F,k,K)
    P1 = F
    P21 = A*F
    P12 = transpose(P21)
    P2 = P21*transpose(A)+Q
    G = P12*inv(P2.+10e-8)
    m = f .- G*(A*f-k)
    V = G*K*transpose(G) .+ P1.-G*P21
    C = transpose(G*K) .+ k*transpose(m)
    return m, V, C
end

function smooth_all(A_list,B,Q,R,mh_list,Vh_list,T)
    kh_list, Kh_list, C_list = [Flux.Tracker.data(mh_list[end])], [Flux.Tracker.data(Vh_list[end])], []
    for t=(T-1):-1:1
        kh, Kh, C = smoothing(Flux.Tracker.data(A_list[t]),B,Q,R,Flux.Tracker.data(mh_list[t]),Flux.Tracker.data(Vh_list[t]),kh_list[end],Kh_list[end])
        push!(kh_list,kh)
        push!(Kh_list,Kh)
        push!(C_list,C)
    end
    kh_list = reverse(kh_list)
    Kh_list = reverse(Kh_list)
    C_list = reverse(C_list)
    return kh_list, Kh_list, C_list
end

# Function g_predict uses past observations and function f to predict new A and makes prediction based on it
#version 1 refers to first variant that models velocity, version 2 refers to second variant that directly models position
function g_predict(model,mh_old,Vh_old,o_old,Q)
    if model.version == 1
        A_base = zeros(2*model.d1,2*model.d1)
        A_base[1:model.d1,1:model.d1] = diagm(0 => ones(model.d1))
        A_base[1:model.d1,model.d1+1:2*model.d1] = diagm(0 => ones(model.d1))
        A_base[model.d1+1:2*model.d1,model.d1+1:2*model.d1] = diagm(0 => ones(model.d1))
        a = [Float32.(Flux.Tracker.TrackedArray(ones(model.d1)));model.f(o_old)]
        A_new = A_base .* a
        return runForward(A_new,model.B,Q,model.R,mh_old,Vh_old), A_new
    end
    if model.version == 2
        A_base = diagm(0 => ones(model.d1))
        A_new = A_base .* model.f(o_old)
        return runForward(A_new,model.B,Q,model.R,mh_old,Vh_old), A_new
    end
end

# run_all function realizes the overall process
function run_all(model,data,m_p,V_p,T)
    Q = E_Beta(model.alfa_Q,model.beta_Q)
    Flux.reset!(model.Lstm) #reset LSTM network for new sequence
    mh, Vh = conditionedGaussian(m_p,V_p,model.B,data[1],model.R)
    mh_list, Vh_list = [Tracker.TrackedArray(mh)], [Tracker.TrackedArray(Vh)]
    mo_list, Vo_list, A_list = [], [], [] #prediction list for mean, covariance and A
    for t=2:T
        (mo, Vo), A_t = g_predict(model,mh,Vh,data[t-1],Q)
        mh, Vh = filtering(A_t,model.B,Q,model.R,mh,Vh,data[t])
        push!(mo_list,mo)
        push!(Vo_list,Vo)
        push!(mh_list,mh)
        push!(Vh_list,Vh)
        push!(A_list,A_t)
    end
    return mo_list, Vo_list, mh_list, Vh_list,A_list
end

#Training constructor
mutable struct klstm_optimizer

    N::Integer #number of training sequences
    kappa::Float64 #hyperparamter for stochastic variational inference
    #state priors
    m_p
    V_p
    opt::RMSProp
    mc #keeps the iteration number

    function klstm_optimizer(N,kappa,m_p,V_p,opt)
        new(N,kappa,m_p,V_p,opt,1)
    end

end

function train_model!(model,optimizer::klstm_optimizer,trainset,num_epochs)
    loss_use(data,y) = loss_model(model, optimizer, data, y)
    Flux.reset!(model.Lstm)
    for epoch=1:num_epochs
        for n=1:optimizer.N
            data = trainset[n]
            T = length(data)
            Flux.train!(loss_use, model.ps, [(data,data[2:T])], optimizer.opt)
        end
    end
end

Nlogpds(d,m,V,x) = -0.5*(d*log(2*pi) + log(det(V)) + transpose(x.-m)*inv(V)*(x.-m)) #cost function

function loss_model(model, optimizer, data, y)
    T = length(data)
    x = data[1:T-1]
    ro = (optimizer.mc+1)^optimizer.kappa #learning rate
    Q = E_Beta(model.alfa_Q,model.beta_Q)
    #update local variables
    mo_list, Vo_list, mh_list, Vh_list, A_list = run_all(model,data,optimizer.m_p,optimizer.V_p,T) #run the model
    l = sum(-Nlogpds.(model.d1, mo_list, Vo_list, y)) #loss for training neural network
    kh_list, Kh_list, C_list = smooth_all(A_list,model.B,Q,model.R,mh_list,Vh_list,T) #smoothing
    #update global variables
    #update Q
    if model.version == 2
        beta_cof_Q = zeros(model.d1)
    end
    if model.version == 1
        beta_cof_Q = zeros(2*model.d1)
    end
    for t=2:T
        beta_cof_Q = beta_cof_Q .+ diag(kh_list[t]*transpose(kh_list[t]) .+ Kh_list[t])
        beta_cof_Q = beta_cof_Q .+ diag(-2 .* C_list[t-1]*transpose(Flux.Tracker.data(A_list[t-1]))
            .+ Flux.Tracker.data(A_list[t-1])*(kh_list[t-1]*transpose(kh_list[t-1]).+Kh_list[t-1])*transpose(Flux.Tracker.data(A_list[t-1])))
    end
    model.alfa_Q = (1-ro).*model.alfa_Q .+ ro.*(model.prior_alfa_Q .+ optimizer.N .* (T-1)./2.0)
    model.beta_Q = (1-ro).*model.beta_Q .+ ro.*(model.prior_beta_Q .+ optimizer.N .* 0.5.*beta_cof_Q)
    #for learning rate
    optimizer.mc += 1
    return l
end

end
